"""Fit the code-world parameters to logged lives by CEM over batched open-loop multi-step prediction.

usage: python experiments/annex/code_wm/fit_world.py <out.json> <life npz>... [--H 20] [--iters 12]
Stage A fits the tool servo to tool_pos (open loop over the whole life from logged commands).
Stage B fits the rod parameters to H-step open-loop replays from many logged readings.
"""
import argparse
import json
import math
import time

import numpy as np
import torch

import importlib
import os

W = importlib.import_module("worlds.annex." + os.environ.get("CW_WORLD", "annex_world"))

ROD_KEYS = ["k_stretch", "c_stretch", "k_bend", "c_bend", "mu", "aniso", "c_visc", "c_quad", "lat_mass", "tip_mass"]
if "mu_lo" in W.PARAMS:
    ROD_KEYS += ["mu_lo", "v_s"]
VEL2 = True
BOUNDS = dict(lat_mass=(0.5, 4.0), tip_mass=(0.2, 6.0), mu_lo=(0.005, 0.5), v_s=(0.01, 2.0),k_stretch=(2e4, 1.0e6), c_stretch=(1.0, 400.0), k_bend=(0.05, 4000.0), c_bend=(1e-4, 20.0),
              mu=(1e-3, 0.3), aniso=(0.01, 5.0), c_visc=(1e-3, 5.0), c_quad=(1e-3, 20.0), mu_post=(0.0, 1.0),
              servo_w=(30.0, 2000.0), servo_z=(0.2, 3.0), servo_amax=(30.0, 2000.0))


def load(paths):
    return [dict(np.load(p)) for p in paths]


def segs(lives, H, every, skip_contact=False):
    """Start readings (life, t) with t>=1 and t+H <= T."""
    out = []
    for li, L in enumerate(lives):
        T = len(L["actions"])
        for t in range(2, T - H + 1, every):
            # planar model: skip windows where the rod leaves the table (life 2: z up to 0.27 m in a violent whip)
            if L["points"][t - 2:t + H + 1, :, 2].max() > 0.008:
                continue
            out.append((li, t))
    return out


def batch_state(lives, starts, params, dev):
    obs, prev, prev2 = [], [], []
    for li, t in starts:
        L = lives[li]
        obs.append({k: L[k][t] for k in ("points", "tool_pos", "tool_vel", "post", "zone", "target", "t")})
        prev.append({"points": L["points"][t - 1]})
        prev2.append({"points": L["points"][max(t - 2, 0)]})
    return W.init_from_obs(obs, prev, params, dev, prev2_obs_batch=prev2 if VEL2 else None)


def seg_loss(lives, starts, params, H, dev, reps):
    """params: dict of (reps,) tensors; every start is evaluated for every candidate. Returns (reps,) loss."""
    S = len(starts)
    st = batch_state(lives, starts * reps, {k: v.repeat_interleave(S) for k, v in params.items()}, dev)
    A = torch.tensor(np.stack([lives[li]["actions"][t:t + H] for li, t in starts]), dtype=torch.float32, device=dev).repeat(reps, 1, 1)
    Ptrue = torch.tensor(np.stack([lives[li]["points"][t + 1:t + H + 1, :, :2] for li, t in starts]), dtype=torch.float32, device=dev).repeat(reps, 1, 1, 1)
    P, _, _ = W.rollout_points(st, A)
    err = (P - Ptrue).norm(dim=-1).mean(-1)              # (B,H) mean node error
    w = torch.linspace(0.3, 1.0, H, device=dev)
    e = (err * w).mean(-1)
    e = torch.nan_to_num(e, nan=10.0, posinf=10.0)
    return e.reshape(reps, S).mean(-1)


def tool_loss(lives, params, dev, reps):
    tot = 0
    for L in lives:
        st = batch_state([L], [(0, 1)] * reps, params, dev)
        st = W.init_from_obs([{k: L[k][0] for k in ("points", "tool_pos", "tool_vel", "post", "zone", "target", "t")}] * reps, None, params, dev)
        A = torch.tensor(L["actions"], dtype=torch.float32, device=dev).unsqueeze(0).repeat(reps, 1, 1)
        _, TL, _ = W.rollout_points(st, A)
        tt = torch.tensor(L["tool_pos"][1:], dtype=torch.float32, device=dev)
        tot = tot + torch.nan_to_num((TL - tt).norm(dim=-1).mean(-1), nan=10.0)
    return tot / len(lives)


def cem(lossfn, keys, init, iters, pop, dev, elite=8, sig0=0.7):
    lo = torch.tensor([math.log(BOUNDS[k][0] + 1e-6) for k in keys], device=dev)
    hi = torch.tensor([math.log(BOUNDS[k][1]) for k in keys], device=dev)
    mu = torch.tensor([math.log(max(init[k], BOUNDS[k][0] + 1e-6)) for k in keys], device=dev)
    sig = torch.full_like(mu, sig0)
    best, bestl = mu.clone(), float("inf")
    for it in range(iters):
        X = mu + sig * torch.randn(pop, len(keys), device=dev)
        X[0] = best
        X = torch.maximum(torch.minimum(X, hi), lo)
        prm = {k: X[:, i].exp() for i, k in enumerate(keys)}
        l = lossfn(prm, pop)
        idx = torch.argsort(l)[:elite]
        if l[idx[0]] < bestl:
            bestl, best = float(l[idx[0]]), X[idx[0]].clone()
        mu = X[idx].mean(0)
        sig = X[idx].std(0).clamp_min(0.03) * 0.9 + 0.02
        print(f"  it {it:2d} best {bestl:.5f} " + " ".join(f"{k}={math.exp(best[i]):.4g}" for i, k in enumerate(keys)), flush=True)
    return {k: float(math.exp(best[i])) for i, k in enumerate(keys)}, bestl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("lives", nargs="+")
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--init", default="")
    ap.add_argument("--skip_tool", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    t0 = time.time()
    lives = load(a.lives)
    prm = dict(W.PARAMS)
    if a.init:
        prm.update(json.load(open(a.init))["params"])
    if not a.skip_tool:
        print("stage A: tool servo")
        tk = ["servo_w", "servo_z", "servo_amax"]
        fixed = {k: v for k, v in prm.items() if k not in tk}
        res, l = cem(lambda p, n: tool_loss(lives, {**{k: torch.full((n,), v, device=dev) for k, v in fixed.items()}, **p}, dev, n),
                     tk, prm, 8, 32, dev)
        prm.update(res)
        print("tool mean error [m]", l)
    starts = segs(lives, a.H, a.every)
    print("stage B: rod,", len(starts), "segments")
    fixed = {k: v for k, v in prm.items() if k not in ROD_KEYS}
    res, l = cem(lambda p, n: seg_loss(lives, starts, {**{k: torch.full((n,), v, device=dev) for k, v in fixed.items()}, **p}, a.H, dev, n),
                 ROD_KEYS, prm, a.iters, a.pop, dev)
    prm.update(res)
    # per-start error report (best params)
    one = {k: torch.full((1,), v, device=dev) for k, v in prm.items()}
    for li, t in starts[:: max(1, len(starts) // 40)]:
        e = seg_loss(lives, [(li, t)], one, a.H, dev, 1)
        print(f"   life {li} t {t:3d} err {float(e):.4f}")
    json.dump(dict(params=prm, loss=l, H=a.H, lives=a.lives, wall_s=time.time() - t0), open(a.out, "w"), indent=1)
    print("done", time.time() - t0, prm)
