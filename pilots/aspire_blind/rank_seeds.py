"""Rank snapshot seeds by planned path length and total wrap (static geometry) to choose stress-test lives;
also fit load vs speed from my own life logs."""
import importlib
import sys
import numpy as np
from pilots.aspire_blind import geom

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    mod = importlib.import_module(f"pilots.aspire_blind.{sys.argv[1]}")
    d = np.load("runs/station/observation/layouts.npz")
    tool, posts, ps, tgt = d["tool_pos"], d["posts"], d["post_sides"], d["target"]
    rows = []
    for i in range(200):
        pts, S, mode = mod.plan(tool[i], posts[i], ps[i], tgt[i])
        rope = np.vstack([[0.05, 0.0], pts])
        w = np.abs(geom.winding(rope, posts[i])).sum()
        rows.append((i, S[-1], w, min(np.min(np.linalg.norm(pts - p, axis=1)) for p in posts[i])))
    rows = np.array(rows)
    print("top by length:\n", rows[np.argsort(-rows[:, 1])][:6])
    print("top by total wrap:\n", rows[np.argsort(-rows[:, 2])][:6])
    print("score len*wrap:\n", rows[np.argsort(-(rows[:, 1] * rows[:, 2]))][:6])
    for k in sys.argv[2:]:
        L = np.load(f"runs/station/blackbox/aspire_blind/life_{k}.npz", allow_pickle=True)
        tp, ld = L["tool_pos"], L["load"]
        sp = np.r_[0, np.linalg.norm(np.diff(tp, axis=0), axis=1) / 0.05]
        # moving rope length proxy: count of points that moved in last step
        mv = np.r_[0, (np.linalg.norm(np.diff(L["points"][:, :, :2], axis=0), axis=2) > 1e-4).sum(1)]
        A = np.c_[np.ones(len(sp)), sp, sp * mv][1:]
        coef, *_ = np.linalg.lstsq(A, ld[1:], rcond=None)
        print("life", k, "fit load = a + b*speed + c*speed*n_moving:", coef, "resid std", np.std(A @ coef - ld[1:]))
        print("   speed pct", np.percentile(sp[1:], [10, 50, 90]), "n moving pct", np.percentile(mv[1:], [10, 50, 90]))
