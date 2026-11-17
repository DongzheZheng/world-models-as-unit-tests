"""Evaluate the primitive-MPC controller inside the code world only (imagined episodes, zero real cost).
python experiments/pusht/code_wm/eval_world.py <cfg.json>
cfg: {"n_ep": 32, "seed": 0, "planner": {...Planner kwargs...}, "out": "runs/pusht/code_wm/wm_eval_x.json",
      "params_jitter": 0.0}   (params_jitter: relative noise on the world params of the *imagined reality* only)"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pusht.code_wm.controller import DEV, Planner
from worlds.pusht import pusht_world as W


def init_states(n, rng, lo=100, hi=412):
    """Imagined initial states: pusher anywhere, block well inside the table, any angle, no overlap."""
    P, Q = [], []
    while len(P) < n:
        p = rng.uniform(50, 462, 2)
        q = np.array([*rng.uniform(lo, hi, 2), rng.uniform(0, 2 * math.pi)])
        cw = W.to_world(torch.tensor(q[None], dtype=torch.float32), torch.tensor(W.VERTS, dtype=torch.float32)[None])[0]
        if float(cw.min()) < W.WALL_LO or float(cw.max()) > W.WALL_HI:
            continue
        d, _ = W.t_sdf(W.to_body(torch.tensor(q[None], dtype=torch.float32),
                                 torch.tensor(p[None, None], dtype=torch.float32))[:, 0])
        cov = float(W.coverage(torch.tensor(q[None], dtype=torch.float32))[0])
        if float(d[0]) > 25 and cov < 0.5:
            P.append(p); Q.append(q)
    return np.array(P), np.array(Q)


def run(cfg):
    W.set_object(cfg.get("planner", {}).get("object", "T"))
    rng = np.random.default_rng(cfg.get("seed", 0))
    n = cfg.get("n_ep", 32)
    P0, Q0 = init_states(n, rng, *cfg.get("block_range", (100, 412)))
    f = lambda x: torch.tensor(x, dtype=torch.float32, device=DEV)
    s = W.make_state(f(P0), torch.zeros(n, 2, device=DEV), f(Q0))
    real_params = dict(W.PARAMS)
    real_params.update(cfg.get("reality_override", {}))      # imagined reality differs from the planner's model
    jit = cfg.get("params_jitter", 0.0)
    if jit:
        g = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 0) + 1)
        for k in ("com_u", "com_v", "k_gyr", "lin_damp", "ang_damp"):
            scale = 3.0 if k.startswith("com") else abs(real_params[k]) * jit
            real_params[k] = (torch.tensor(real_params[k]) + scale * torch.randn(n, generator=g)).to(DEV)
    planner = Planner(**cfg.get("planner", {}))
    queues = [[] for _ in range(n)]
    done = np.zeros(n, bool); succ = np.zeros(n, bool); steps = np.zeros(n, int)
    maxcov = W.coverage(s["q"]).cpu().numpy()
    prev_q = s["q"].clone()
    t0 = time.time()
    for t in range(300):
        need = [i for i in range(n) if not done[i] and not queues[i]]
        if need:
            idx = torch.tensor(need, device=DEV)
            sub = {k: v[idx] for k, v in s.items()}
            # the controller sees what the real policy would see: pose, pusher state; twist by finite difference
            comb = torch.tensor([[[W.PARAMS["com_u"], W.PARAMS["com_v"]]]], device=DEV)
            sub["v"] = (W.to_world(sub["q"], comb)[:, 0] - W.to_world(prev_q[idx], comb)[:, 0]) / W.DT
            sub["w"] = ((sub["q"][:, 2] - prev_q[idx][:, 2] + math.pi) % (2 * math.pi) - math.pi) / W.DT
            for i, (A, el, sc) in zip(need, planner.plan(sub, 300 - t)):
                queues[i] = list(A[:el])
        act = np.stack([queues[i].pop(0) if (not done[i] and queues[i]) else s["p"][i].cpu().numpy()
                        for i in range(n)])
        prev_q = s["q"].clone()
        s = W.physics(s, f(act), real_params)
        cov = W.coverage(s["q"]).cpu().numpy()
        live = ~done
        steps[live] += 1
        maxcov = np.where(live, np.maximum(maxcov, cov), maxcov)
        newly = live & (cov > 0.95)
        succ |= newly
        done |= newly
        if done.all():
            break
    wall = time.time() - t0
    g = torch.tensor(W.GOAL, device=DEV)
    C = torch.tensor(W.CENTROID, device=DEV)
    cpos = (W.to_world(s["q"], C.expand(n, 1, 2))[:, 0] - W.to_world(g[None], C[None, None])[0, 0]).norm(dim=-1).cpu().numpy()
    cang = np.degrees(np.abs(((s["q"][:, 2] - g[2] + math.pi) % (2 * math.pi) - math.pi).cpu().numpy()))
    res = dict(n_ep=n, success_rate=float(succ.mean()), mean_max_coverage=float(maxcov.mean()),
               mean_steps=float(steps.mean()), wm_steps=int(planner.wm_steps), wall_s=wall,
               wm_steps_per_s=planner.wm_steps / wall, planner=cfg.get("planner", {}), params_jitter=jit,
               reality_override=cfg.get("reality_override", {}), block_range=cfg.get("block_range"),
               per_episode=[dict(success=bool(a), max_cov=float(b), steps=int(c), final_centroid_err=float(e), final_ang_err_deg=float(f))
                            for a, b, c, e, f in zip(succ, maxcov, steps, cpos, cang)])
    return res


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    res = run(cfg)
    print(json.dumps({k: v for k, v in res.items() if k != "per_episode"}, indent=1))
    if cfg.get("out"):
        Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg["out"]).write_text(json.dumps(res, indent=1))
