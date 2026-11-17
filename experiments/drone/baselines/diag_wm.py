import sys, time, numpy as np, torch
from reallab.drone.api import make
from experiments.drone.baselines import wm
from experiments.drone.baselines.explore import Explorer
torch.set_num_threads(3)
env = make("site", purpose="oracle_train", tag="diag", pilot="neural_wm_dry")
def fly(pol, seed, plan=False):
    obs, _ = env.reset(seed=seed); mem = {k: [obs[k]] for k in ("pos", "lin_vel", "quat", "ang_vel")}; acts = []
    while True:
        a = pol.plan(obs) if plan else pol.act(obs)
        obs, r, te, tr, info = env.step(a); acts.append(a)
        for k in mem: mem[k].append(obs[k])
        if te or tr: break
    d = {**{k: np.stack(v) for k, v in mem.items()}, "actions": np.stack(acts), "end_cause": "collision" if info["collision"] else "other"}
    return d, info
logs = []
for i in range(5):
    d, info = fly(Explorer(seed=i), 1_000_500 + i); logs.append(d); print("explorer", len(d["actions"]), info["collision"])
def rollout_err(model, z, H=30):
    F = np.stack([wm.feat({k: z[k][t] for k in ("pos", "lin_vel", "quat", "ang_vel")}) for t in range(len(z["actions"]) + 1)])
    A = wm.cmd_to_an(z["actions"]); PA = np.concatenate([np.zeros((1, 4), np.float32), A[:-1]])
    out = {h: [] for h in (1, 5, 10, 20, 30)}
    for t0 in range(0, len(A) - H, 2):
        f = torch.as_tensor(F[t0])[None]
        for h in range(H):
            x = torch.cat([f, torch.as_tensor(A[t0 + h])[None], torch.as_tensor(PA[t0 + h])[None]], -1)
            f = f + model.forward_all(x).mean(0)
            if h + 1 in out: out[h + 1].append(np.linalg.norm(f[0, :3].detach().numpy() - F[t0 + h + 1, :3]))
    return {h: float(np.mean(v)) for h, v in out.items() if v}
for hold in (4,):
    tr = [l for i, l in enumerate(logs) if i != hold]
    X, Y, _ = wm.transitions(tr); m, _ = wm.fit(X, Y)
    print("held-out life pos err by horizon", rollout_err(m, logs[hold]))
X, Y, _ = wm.transitions(logs); m, _ = wm.fit(X, Y)
print("explorer max |cmd|", np.abs(np.concatenate([l["actions"] for l in logs])).max(0), "max tilt-ish", max(np.abs(l["quat"][:, :2]).max() for l in logs))
for name, kw in [("base", {}), ("nov.5", dict(nov_pen=0.5)), ("nov.5_tilt.5", dict(nov_pen=0.5, tilt_max=0.5)), ("nov2_tilt.5_unc1", dict(nov_pen=2.0, tilt_max=0.5, unc_pen=1.0))]:
    res = []
    for e in range(3):
        pl = wm.Planner(m, seed=e, **kw); pl.reset(); t0 = time.time()
        d, info = fly(pl, 1_000_600 + e, plan=True)
        res.append((len(d["actions"]), d["pos"][-1].round(1).tolist(), d["lin_vel"][-1].round(1).tolist(), d["quat"][-1][:3].round(2).tolist()))
    print(name, res, "%.0f ms/step" % ((time.time() - t0) / len(d["actions"]) * 1e3), flush=True)
