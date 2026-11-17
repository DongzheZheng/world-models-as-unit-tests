"""Screen candidate programs in the other pilot's code world (imagined episodes only).
python pilots/aspire_pusht/cw_screen.py <candidates.json> <n> <seed0>
candidates.json: [{"name": ..., "version": 16, "kwargs": {...}}, ...]. Appends to runs/pusht/aspire_style/cw_screen.jsonl."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def init_features(seeds, obj="L"):
    import torch
    from experiments.pusht.code_wm import screen as S
    from worlds.pusht import pusht_world as W
    W.set_object(obj)
    goal = np.array(W.GOAL)
    out = {}
    for s in seeds:
        p, q = S._init_state(np.random.default_rng(s), W, *S.RANGES[obj])
        cw = W.to_world(torch.tensor(q[None], dtype=torch.float32), torch.tensor(W.VERTS, dtype=torch.float32)[None])[0].numpy()
        wall = float(min(cw.min() - W.WALL_LO, W.WALL_HI - cw.max()))
        eth = abs((goal[2] - q[2] + math.pi) % (2 * math.pi) - math.pi)
        out[s] = dict(eth_deg=math.degrees(eth), wall=wall, dist=float(np.linalg.norm(q[:2] - goal[:2])))
    return out


if __name__ == "__main__":
    from experiments.pusht.code_wm.screen import run_in_world
    cands = json.load(open(sys.argv[1])); n, seed0 = int(sys.argv[2]), int(sys.argv[3])
    feats = init_features(range(seed0, seed0 + n))
    log = Path("runs/pusht/aspire_style/cw_screen.jsonl"); log.parent.mkdir(parents=True, exist_ok=True)
    for c in cands:
        t0 = time.time()
        eps = run_in_world(f"pilots.aspire_pusht.policy_v{c['version']}:make_policy", c.get("kwargs", {}), "L", n, seed0,
                           n_workers=12, batch=max(1, n // 12 + (n % 12 > 0)))
        wall = time.time() - t0
        succ = np.array([e["success"] for e in eps]); mc = np.array([e["max_coverage"] for e in eps])
        eth = np.array([feats[e["seed"]]["eth_deg"] for e in eps]); wd = np.array([feats[e["seed"]]["wall"] for e in eps])
        bins = {}
        for lo, hi in [(0, 45), (45, 90), (90, 135), (135, 181)]:
            m = (eth >= lo) & (eth < hi); bins[f"eth{lo}-{hi}"] = [int(m.sum()), round(float(succ[m].mean()), 3) if m.any() else None]
        near = wd < 40
        rec = dict(name=c["name"], version=c["version"], kwargs=c.get("kwargs", {}), n=n, seed0=seed0,
                   success=float(succ.mean()), max_coverage=float(mc.mean()), n_steps=float(np.mean([e["n_steps"] for e in eps])),
                   success_away_from_walls=float(succ[~near].mean()), n_near_wall=int(near.sum()), success_near_wall=float(succ[near].mean()) if near.any() else None,
                   by_initial_angle_error=bins, imagined_steps=int(sum(e["n_steps"] for e in eps)), wall_s=round(wall, 1),
                   errors=sum(e["error"] is not None for e in eps), first_error=next((e["error"] for e in eps if e["error"]), None))
        print(json.dumps({k: rec[k] for k in ("name", "success", "max_coverage", "n_steps", "success_away_from_walls", "success_near_wall", "by_initial_angle_error", "wall_s", "errors")}))
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
