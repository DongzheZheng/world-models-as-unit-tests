"""Open-loop prediction error of the neural world model vs horizon, on held-out transitions."""
import glob, gzip, json
import numpy as np, torch
from experiments.drone.baselines import wm

torch.set_num_threads(4)
H = 45
HS = [1, 2, 5, 10, 15, 20, 30, 45]


def load_eval_logs(d):
    out = []
    for f in sorted(glob.glob(d + "/*.jsonl.gz")):
        rows = [json.loads(l) for l in gzip.open(f, "rt")]
        obs = [rows[0]["obs"]] + [r["obs"] for r in rows if r["type"] == "step"]
        out.append({k: np.array([o[k] for o in obs]) for k in ("pos", "lin_vel", "quat", "ang_vel")} |
                   {"actions": np.array([r["action"] for r in rows if r["type"] == "step"])})
    return out


def errors(model, logs, stride=3):
    acc = {h: dict(pos=[], vel=[], tilt=[], angvel=[], pos_cv=[], pos_hold=[]) for h in range(1, H + 1)}
    with torch.no_grad():
        for z in logs:
            T = len(z["actions"])
            F = np.stack([wm.feat({k: z[k][t] for k in ("pos", "lin_vel", "quat", "ang_vel")}) for t in range(T + 1)])
            A = wm.cmd_to_an(z["actions"]); PA = np.concatenate([np.zeros((1, 4), np.float32), A[:-1]])
            starts = list(range(0, T - 1, stride))
            f = torch.as_tensor(F[starts])
            for h in range(1, H + 1):
                ok = [i for i, s in enumerate(starts) if s + h <= T]
                if not ok:
                    break
                idx = torch.as_tensor(ok); st = np.array(starts)[ok]
                f = f if h == 1 else f
                x = torch.cat([f[idx] if False else f, torch.as_tensor(A[np.minimum(np.array(starts) + h - 1, T - 1)]),
                               torch.as_tensor(PA[np.minimum(np.array(starts) + h - 1, T - 1)])], -1)
                f = f + model.forward_all(x).mean(0)
                fn = f.numpy()[ok]; tr = F[st + h]
                acc[h]["pos"] += list(np.linalg.norm(fn[:, :3] - tr[:, :3], axis=1))
                acc[h]["vel"] += list(np.linalg.norm(fn[:, 3:6] - tr[:, 3:6], axis=1))
                up_p, up_t = fn[:, 12:15], tr[:, 12:15]
                c = (up_p * up_t).sum(1) / (np.linalg.norm(up_p, axis=1) * np.linalg.norm(up_t, axis=1) + 1e-9)
                acc[h]["tilt"] += list(np.arccos(np.clip(c, -1, 1)))
                acc[h]["angvel"] += list(np.linalg.norm(fn[:, 15:18] - tr[:, 15:18], axis=1))
                f0 = F[st]
                acc[h]["pos_cv"] += list(np.linalg.norm(f0[:, :3] + f0[:, 3:6] * h / 30.0 - tr[:, :3], axis=1))
                acc[h]["pos_hold"] += list(np.linalg.norm(f0[:, :3] - tr[:, :3], axis=1))
    res = {}
    for h in HS:
        if acc[h]["pos"]:
            res[h] = {k: float(np.mean(v)) for k, v in acc[h].items()} | {"n": len(acc[h]["pos"])}
    return res


def load_model(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    m = wm.Ensemble(b["k"], b["hidden"]); m.xtrain = torch.zeros_like(b["state"]["xtrain"]); m.load_state_dict(b["state"])
    return m.eval()


out = {"units": "horizon in agent steps (1/30 s); pos m, vel m/s, tilt rad (angle between up vectors), angvel rad/s; pos_cv = constant-velocity "
                "extrapolation baseline, pos_hold = stay-put baseline"}
final = load_model("runs/drone/baselines_rl/neural_wm/model_final.pt")
ev = load_eval_logs("runs/drone/baselines_rl/evlogs/neural_wm_5lives")
out["final_model_on_50_eval_episodes_never_trained_on"] = errors(final, ev)
out["n_eval_episodes"] = len(ev); out["n_eval_transitions"] = int(sum(len(z["actions"]) for z in ev))
logs = [wm.load_npz(f"runs/drone/blackbox/neural_wm/life_{i}.npz") for i in range(1, 6)]
X, Y, _ = wm.transitions(logs[:4])
m4, _ = wm.fit(X, Y, seed=5)
out["lives1to4_model_on_held_out_life5"] = errors(m4, logs[4:5], stride=1)
json.dump(out, open("runs/drone/baselines_rl/neural_wm_fidelity.json", "w"), indent=1)
for k in ("final_model_on_50_eval_episodes_never_trained_on", "lives1to4_model_on_held_out_life5"):
    print(k)
    for h, v in out[k].items():
        print(f"  h={h:3d} pos {v['pos']:.3f} (cv {v['pos_cv']:.3f}, hold {v['pos_hold']:.3f}) vel {v['vel']:.3f} tilt {v['tilt']:.3f} angvel {v['angvel']:.2f} n={v['n']}")
