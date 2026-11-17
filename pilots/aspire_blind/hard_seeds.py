"""List geometrically hard snapshot seeds for a policy version (static)."""
import importlib
import sys
import numpy as np
from pilots.aspire_blind import geom

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    ver = sys.argv[1] if len(sys.argv) > 1 else "policy_v1"
    mod = importlib.import_module(f"pilots.aspire_blind.{ver}")
    d = np.load("runs/station/observation/layouts.npz")
    tool, posts, ps, tgt = d["tool_pos"], d["posts"], d["post_sides"], d["target"]
    rows = []
    for i in range(200):
        pts, S = mod.corridor_path(tool[i], posts[i], ps[i], tgt[i])
        clr = min(np.min(np.linalg.norm(pts - p, axis=1)) for p in posts[i])
        turn = np.abs(np.diff(np.sign(np.diff(pts[:, 1])))).sum() / 2
        alt = int(np.sum(ps[i][1:] != ps[i][:-1]))
        rows.append((i, clr, S[-1], alt))
    rows = np.array(rows)
    for i in np.where(rows[:, 1] < 0.06)[0]:
        print("low clr seed", i, "posts", posts[i].ravel(), "sides", ps[i], "tool", tool[i], "target", tgt[i], "clr %.3f" % rows[i, 1])
    hard = rows[(rows[:, 3] == 2)]
    hard = hard[np.argsort(-hard[:, 2])]
    print("alternating sides, longest paths (seed, clr, len):")
    print(hard[:10, :3])
