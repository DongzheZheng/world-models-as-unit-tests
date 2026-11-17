# Fit the code-world parameters to logged segments by batched CEM (one candidate parameter set per batch slice).
#   python experiments/station/code_wm/fit_world.py --logs <npz...> --out runs/station/code_wm/fit.json [--iters 40]
import argparse, json, time, math
import numpy as np
import torch
from worlds.station import station_world as W

# (name, lo, hi) in log10 space for positive parameters; stability: k/m <= 2e5, cd/m <= 800
SPACE = [("m", -3.0, -0.5), ("k", 2.0, 4.3), ("cd", -2.0, 1.5), ("kb", -2.0, 2.0), ("cv", -2.0, 1.0), ("fc", -3.0, 0.0),
         ("cpost", -2.0, 1.5), ("kp", 1.5, 3.5), ("kd", 0.0, 2.0), ("fmax", 0.3, 2.0), ("mt", -2.0, 0.5),
         ("lscale", -0.5, 0.7), ("loff", -3.0, 0.0), ("vread", -0.3, 0.1)]


def load_segments(files, H, stride, t_max=None):
    segs = []
    for f in files:
        E = np.load(f)
        T = len(E["actions"])
        for t0 in range(1, T - H + 1, stride):
            segs.append(dict(f=f, t0=t0, pts=E["points"][t0 - 1:t0 + H + 1, :, :2], tool=E["tool_pos"][t0 - 1:t0 + H + 1],
                             tv=E["tool_vel"][t0:t0 + H + 1], load=E["load"][t0:t0 + H + 1], act=E["actions"][t0:t0 + H],
                             posts=E["posts"][0], ps=E["post_sides"][0], target=E["target"][0], prev_act=E["actions"][t0 - 1]))
    return segs


def make_batch(segs, P, dev):
    """Replicate each segment for each of P candidates: batch index = cand * S + seg."""
    S = len(segs)
    f = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float32, device=dev)
    x0 = f([s["pts"][1] for s in segs]); xm = f([s["pts"][0] for s in segs])
    v0 = (x0 - xm) / W.DT
    v0[:, -1] = f([s["tv"][0] for s in segs])
    st = dict(x=x0.repeat(P, 1, 1), v=v0.repeat(P, 1, 1), ref=f([s["prev_act"] for s in segs]).repeat(P, 1),
              posts=f([s["posts"] for s in segs]).repeat(P, 1, 1),
              post_sides=torch.as_tensor(np.stack([s["ps"] for s in segs]), device=dev).long().repeat(P, 1),
              target=f([s["target"] for s in segs]).repeat(P, 1), t=torch.zeros(P * S, dtype=torch.long, device=dev),
              load=f([s["load"][0] for s in segs]).repeat(P))
    st["done"] = torch.zeros(P * S, dtype=torch.bool, device=dev); st["success"] = st["done"].clone(); st["overload"] = st["done"].clone()
    st["max_load"] = st["load"].clone()
    acts = f([s["act"] for s in segs]).repeat(P, 1, 1)               # (PS,H,2)
    tgt_x = f([s["pts"][2:] for s in segs]).repeat(P, 1, 1, 1)       # (PS,H,20,2)
    tgt_l = f([s["load"][1:] for s in segs]).repeat(P, 1)             # (PS,H)
    tgt_v = f([s["tv"][1:] for s in segs]).repeat(P, 1, 1)
    return st, acts, tgt_x, tgt_l, tgt_v


def rollout_err(st, acts, tgt_x, tgt_l, tgt_v, par_vec_dict, horizons=(5, 20, 60)):
    st = dict(st)
    st["par"] = par_vec_dict
    H = acts.shape[1]
    ex, el, ev = [], [], []
    for h in range(H):
        st, _, _ = W.step(st, acts[:, h])
        # never stop: rules do not apply during fitting (done flags reset)
        st["done"] = torch.zeros_like(st["done"])
        ex.append((st["x"] - tgt_x[:, h]).norm(dim=-1).mean(-1))
        el.append((st["load"] - tgt_l[:, h]).abs())
        ev.append(((st["par"]["vread"] * st["v"][:, -1]) - tgt_v[:, h]).norm(dim=-1))
    ex = torch.stack(ex, 1); el = torch.stack(el, 1); ev = torch.stack(ev, 1)
    ex = torch.nan_to_num(ex, nan=1.0, posinf=1.0); el = torch.nan_to_num(el, nan=50.0, posinf=50.0)
    ev = torch.nan_to_num(ev, nan=1.0, posinf=1.0)
    return ex, el, ev


def vec_to_par(z, S, dev):
    out = {}
    for j, (n, lo, hi) in enumerate(SPACE):
        out[n] = (10 ** z[:, j]).float().repeat_interleave(S).reshape(-1, 1).to(dev)
    return out


def score(ex, el, ev):
    # metres -> /0.005, newton -> /0.5 (clipped), m/s -> /0.02
    return (ex.mean(1) / 0.005 + torch.clamp(el, max=10).mean(1) / 0.5 + ev.mean(1) / 0.02)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--pop", type=int, default=160)
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--max_segs", type=int, default=72)
    ap.add_argument("--init", default="")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0); rng = np.random.default_rng(0)
    segs = load_segments(args.logs, args.H, args.stride)
    if len(segs) > args.max_segs:
        segs = [segs[i] for i in sorted(rng.choice(len(segs), args.max_segs, replace=False))]
    S = len(segs)
    print("segments", S, flush=True)
    lo = torch.tensor([s[1] for s in SPACE]); hi = torch.tensor([s[2] for s in SPACE])
    mu = torch.tensor([math.log10(W.DEFAULT_PAR[s[0]]) for s in SPACE])
    if args.init:
        ini = json.load(open(args.init))["par"]
        mu = torch.tensor([math.log10(ini[s[0]]) for s in SPACE])
    sd = (hi - lo) / 4
    t0 = time.time()
    best = (1e9, mu.clone())
    for it in range(args.iters):
        z = mu + sd * torch.randn(args.pop, len(SPACE))
        z[0] = best[1]
        z = torch.maximum(torch.minimum(z, hi), lo)
        # stability constraints
        jm, jk, jcd = 0, 1, 2
        z[:, jk] = torch.minimum(z[:, jk], z[:, jm] + math.log10(2e5))
        z[:, jcd] = torch.minimum(z[:, jcd], z[:, jm] + math.log10(800))
        st, acts, tx, tl, tv = make_batch(segs, args.pop, dev)
        with torch.no_grad():
            ex, el, ev = rollout_err(st, acts, tx, tl, tv, vec_to_par(z, S, dev))
        sc = score(ex, el, ev).reshape(args.pop, S).mean(1).cpu()
        order = torch.argsort(sc)
        if sc[order[0]] < best[0]:
            best = (float(sc[order[0]]), z[order[0]].clone())
        elite = z[order[: max(8, args.pop // 8)]]
        mu = 0.3 * mu + 0.7 * elite.mean(0); sd = 0.3 * sd + 0.7 * elite.std(0) + 0.01
        e_best = order[0]
        exb = ex.reshape(args.pop, S, -1)[e_best].mean().item(); elb = el.reshape(args.pop, S, -1)[e_best].mean().item()
        print(f"it {it} best {best[0]:.3f} cur {sc[order[0]]:.3f}  pos err {exb*1000:.2f} mm  load err {elb:.3f} N  {time.time()-t0:.0f}s", flush=True)
    par = {s[0]: float(10 ** best[1][j]) for j, s in enumerate(SPACE)}
    print(json.dumps(par, indent=1))
    json.dump(dict(par=par, score=best[0], segments=S, logs=args.logs, H=args.H, wall_s=time.time() - t0), open(args.out, "w"), indent=1)
