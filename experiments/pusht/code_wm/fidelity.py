"""Open-loop fidelity of the code world on held-out lives: replay the logged commands from a logged state and compare
block position / angle after 5, 10 and 20 steps.   python experiments/pusht/code_wm/fidelity.py <cfg.json>
cfg: {"params": "runs/pusht/code_wm/fit_x.json" | null (use pusht_world.PARAMS), "test": [npz...], "out": "...json",
      "label": "..."}"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.pusht.code_wm import fit_world as F
from worlds.pusht import pusht_world as W


def clearance(q):
    """Min distance [u] of the block outline's vertices to the walls, per logged pose (numpy)."""
    V = np.array(W.VERTS, float)
    c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
    X = q[:, None, 0] + c[:, None] * V[None, :, 0] - s[:, None] * V[None, :, 1]
    Y = q[:, None, 1] + s[:, None] * V[None, :, 0] + c[:, None] * V[None, :, 1]
    return np.minimum(np.minimum(X - W.WALL_LO, W.WALL_HI - X), np.minimum(Y - W.WALL_LO, W.WALL_HI - Y)).min(1)


def evaluate(params, lives, horizons=(5, 10, 20), wall=None):
    """wall=None: all windows; "clear": windows whose block stays > 15 u from the walls; "near": the others."""
    L = F.load(lives)
    H = max(horizons)
    S = F.segments(L, H)
    if wall:
        cl = [clearance(d["q"]) for d in L]
        ok = lambda li, t: cl[li][t:t + H + 1].min() > 15
        S = [(li, t) for li, t in S if ok(li, t) == (wall == "clear")]
    data = F.batch(L, S, H)
    p0, pv0, q0, qm1, A, Q, P = data
    prm = {k: torch.tensor(float(v), device=F.DEV) for k, v in params.items()}
    Qp, Pp = F.rollout(prm, p0, pv0, q0, qm1, A, H)
    dpos, dth = F.pose_err(Qp, Q)
    # baseline: "block stays where it is"
    dpos0, dth0 = F.pose_err(q0[:, None].expand_as(Q), Q)
    res = dict(n_segments=len(S))
    for h in horizons:
        e, a = dpos[:, h - 1].cpu().numpy(), np.degrees(dth[:, h - 1].cpu().numpy())
        e0, a0 = dpos0[:, h - 1].cpu().numpy(), np.degrees(dth0[:, h - 1].cpu().numpy())
        res[f"h{h}"] = dict(pos_err_mean_u=float(e.mean()), pos_err_median_u=float(np.median(e)),
                            ang_err_mean_deg=float(a.mean()), ang_err_median_deg=float(np.median(a)),
                            static_baseline_pos_mean_u=float(e0.mean()), static_baseline_ang_mean_deg=float(a0.mean()))
    return res


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    W.set_object(cfg.get("object", "T"))
    params = dict(W.PARAMS)
    if cfg.get("params"):
        params.update(json.loads(Path(cfg["params"]).read_text())["params"])
    res = dict(label=cfg.get("label", ""), params_from=cfg.get("params"), test_lives=cfg["test"],
               **evaluate(params, cfg["test"], wall=cfg.get("wall")))
    print(json.dumps(res, indent=1))
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
