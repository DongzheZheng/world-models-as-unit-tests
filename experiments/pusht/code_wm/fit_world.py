"""Fit the code world's parameters to black-box lives (cross-entropy search over open-loop segment replays).

python experiments/pusht/code_wm/fit_world.py <cfg.json>
cfg: {"lives": ["runs/pusht/blackbox/code_wm/life_1.npz", ...], "H": 10, "out": "runs/pusht/code_wm/fit.json",
      "holdout": [...optional lives for the fidelity check...]}
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from worlds.pusht import pusht_world as W

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

# block / contact parameters searched: name -> (low, high, log-scale)
SPACE = dict(
    com_u=(-50.0, 50.0, False), com_v=(-10.0, 110.0, False), k_gyr=(15.0, 90.0, True),
    restitution=(0.0, 0.8, False), mu=(0.0, 1.5, False), bias=(0.02, 0.5, True),
    lin_damp=(0.1, 20.0, True), ang_damp=(0.1, 20.0, True),
    lin_coul=(0.1, 300.0, True), ang_coul=(0.01, 10.0, True),
)


def load(paths):
    L = []
    for p in paths:
        d = np.load(p)
        L.append(dict(p=d["agent_pos"], pv=d["agent_vel"], q=d["block_pose"], a=d["actions"], name=Path(p).stem,
                      src=str(p)))
    return L


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def segments(L, H, contact_only=True):
    """Start indices (life, t) of H-step windows in which the block moves (or the pusher is near it)."""
    S = []
    for li, d in enumerate(L):
        T = len(d["a"])
        q = d["q"]
        for t in range(1, T - H + 1):
            dq = np.abs(q[t + H, :2] - q[t, :2]).sum() + 60 * abs(wrap(q[t + H, 2] - q[t, 2]))
            near = np.linalg.norm(d["p"][t:t + H + 1] - q[t:t + H + 1, :2], axis=1).min() < 160
            if (not contact_only) or dq > 0.5 or (near and dq > 0.05):
                S.append((li, t))
    return S


def batch(L, S, H):
    f = lambda k, off=0, n=1: torch.tensor(np.stack([L[li][k][t + off:t + off + n] for li, t in S]),
                                           dtype=torch.float32, device=DEV)
    p0, pv0, q0, qm1 = f("p")[:, 0], f("pv")[:, 0], f("q")[:, 0], f("q", -1)[:, 0]
    A = f("a", 0, H)
    Q = f("q", 1, H)
    P = f("p", 1, H)
    return p0, pv0, q0, qm1, A, Q, P


def init_twist(q0, qm1, params):
    """Initial COM velocity / angular velocity from the previous logged pose (finite difference)."""
    cu, cv = params["com_u"], params["com_v"]
    comb = torch.stack([torch.as_tensor(cu).expand(q0.shape[0]), torch.as_tensor(cv).expand(q0.shape[0])], -1)[:, None]
    c0 = W.to_world(q0, comb)[:, 0]
    c1 = W.to_world(qm1, comb)[:, 0]
    return (c0 - c1) / W.DT, wrap(q0[:, 2] - qm1[:, 2]) / W.DT


@torch.no_grad()
def rollout(params, p0, pv0, q0, qm1, A, H):
    v0, w0 = init_twist(q0, qm1, params)
    s = W.make_state(p0, pv0, q0, v0, w0)
    Qs, Ps = [], []
    for h in range(H):
        s = W.physics(s, A[:, h], params)
        Qs.append(s["q"]); Ps.append(s["p"])
    return torch.stack(Qs, 1), torch.stack(Ps, 1)


def pose_err(Qp, Q):
    dpos = (Qp[..., :2] - Q[..., :2]).norm(dim=-1)
    dth = wrap(Qp[..., 2] - Q[..., 2]).abs()
    return dpos, dth


def expand(data, n):
    return [x.repeat(n, *([1] * (x.dim() - 1))) for x in data]


def cem(loss_fn, space, base, n_pop=256, iters=25, elite=0.1, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    names = list(space)
    lo = torch.tensor([space[k][0] for k in names]); hi = torch.tensor([space[k][1] for k in names])
    lg = torch.tensor([space[k][2] for k in names])
    tf = lambda x: torch.where(lg, torch.log(x), x)
    itf = lambda z: torch.where(lg, torch.exp(z), z)
    zlo, zhi = tf(lo), tf(hi)
    mu = (zlo + zhi) / 2
    if base:
        mu = torch.stack([tf(torch.tensor(float(base[k]))) if k in base else mu[i] for i, k in enumerate(names)])
    sd = (zhi - zlo) / 4
    best = (float("inf"), None)
    hist = []
    for it in range(iters):
        z = mu + sd * torch.randn(n_pop, len(names), generator=g)
        z[0] = mu
        z = torch.maximum(torch.minimum(z, zhi), zlo)
        x = itf(z)
        losses = loss_fn({k: x[:, i] for i, k in enumerate(names)})
        order = torch.argsort(losses)
        ne = max(4, int(elite * n_pop))
        el = z[order[:ne]]
        mu, sd = el.mean(0), el.std(0) + 1e-4 * (zhi - zlo)
        if losses[order[0]] < best[0]:
            best = (float(losses[order[0]]), {k: float(x[order[0], i]) for i, k in enumerate(names)})
        hist.append(float(losses[order[0]]))
    unc = {k: [float(itf(mu - 2 * sd)[i]), float(itf(mu + 2 * sd)[i])] for i, k in enumerate(names)}
    return best, unc, hist


def batched_loss(L, S, H, n_pop, w_th=60.0, chunk=None):
    data = batch(L, S, H)
    nS = len(S)

    def loss_fn(prm):
        P = next(iter(prm.values())).shape[0]
        out = torch.zeros(P)
        step = chunk or max(1, 60000 // nS)
        for a in range(0, P, step):
            b = min(P, a + step)
            n = b - a
            p0, pv0, q0, qm1, A, Q, Pp = expand(data, n)
            prm_b = {k: v[a:b].to(DEV).repeat_interleave(nS) for k, v in prm.items()}
            for k, v in W.PARAMS.items():
                prm_b.setdefault(k, torch.tensor(v, device=DEV))
            Qp, Ppred = rollout(prm_b, p0, pv0, q0, qm1, A, H)
            dpos, dth = pose_err(Qp, Q)
            e = (dpos + w_th * dth).mean(-1).reshape(n, nS)
            out[a:b] = e.mean(-1).cpu()
        return out
    return loss_fn


def pusher_loss(L, H):
    S = [(li, t) for li, d in enumerate(L) for t in range(1, len(d["a"]) - H + 1, 2)]
    data = batch(L, S, H)
    nS = len(S)

    def loss_fn(prm):
        P = next(iter(prm.values())).shape[0]
        p0, pv0, q0, qm1, A, Q, Pp = expand(data, P)
        prm_b = {k: v.to(DEV).repeat_interleave(nS) for k, v in prm.items()}
        s = W.make_state(p0, pv0, q0)                 # M1 does not depend on the block (pusher is kinematic)
        err = 0
        for h in range(H):
            s = W.physics(s, A[:, h], prm_b)
            err = err + (s["p"] - Pp[:, h]).norm(dim=-1)
        return (err / H).reshape(P, nS).mean(-1).cpu()
    return loss_fn


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    W.set_object(cfg.get("object", "T"))
    H = cfg.get("H", 10)
    L = load(cfg["lives"])
    t0 = time.time()
    # stage 1: pusher (M1)
    if cfg.get("fit_pusher", True):
        bp, up, hp = cem(pusher_loss(L, 5), dict(kp=(10.0, 1000.0, True), kv=(1.0, 100.0, True)), None,
                         n_pop=128, iters=20)
        print("pusher", bp, up, flush=True)
        W.PARAMS.update(kp=bp[1]["kp"], kv=bp[1]["kv"])
    else:
        bp, up = (float(pusher_loss(L, 5)({k: torch.tensor([W.PARAMS[k]]) for k in ("kp", "kv")})[0]), None), {}
    for k, v in cfg.get("space_override", {}).items():
        SPACE[k] = tuple(v)
    # stage 2: block + contact (M2-M4)
    S = segments(L, H)
    if cfg.get("exclude_wall"):          # drop windows in which the block comes within 15 u of a wall
        from experiments.pusht.code_wm.fidelity import clearance
        cl = [clearance(d["q"]) for d in L]
        S = [(li, t) for li, t in S if cl[li][t - 1:t + H + 1].min() > 15]
    print("segments", len(S), flush=True)
    space = {k: v for k, v in SPACE.items() if k not in cfg.get("fixed", {}) and k in cfg.get("space", SPACE)}
    W.PARAMS.update(cfg.get("fixed", {}))
    bb, ub, hb = cem(batched_loss(L, S, H, cfg.get("pop", 256)), space, cfg.get("init"),
                     n_pop=cfg.get("pop", 256), iters=cfg.get("iters", 25))
    print("block", json.dumps(bb), json.dumps(ub), hb[::5], flush=True)
    params = dict(W.PARAMS); params.update(bb[1])
    alt_base = {k: torch.tensor([v]) for k, v in W.PARAMS.items()}
    carried_loss = float(batched_loss(L, S, H, 1)(alt_base)[0])      # loss with the prior (carried-over) params
    # sensitivity: loss when the COM is forced to the drawing centroid (is the offset real?)
    lf = batched_loss(L, S, H, 1)
    alt = {k: torch.tensor([v]) for k, v in params.items()}
    base_loss = float(lf(alt)[0])
    alt2 = dict(alt); alt2["com_u"] = torch.tensor([W.CENTROID[0]]); alt2["com_v"] = torch.tensor([W.CENTROID[1]])
    cen_loss = float(lf(alt2)[0])
    out = dict(params=params, block_loss_prior_params=carried_loss, pusher_loss_u=bp[0], block_loss=bb[0], block_loss_check=base_loss,
               block_loss_com_at_centroid=cen_loss, ci95=dict(**up, **ub), n_segments=len(S), H=H,
               lives=cfg["lives"], object=W.OBJECT, searched=list(space), wall_s=time.time() - t0)
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["out"]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
