"""Static check of a policy's plan() on the 200 snapshots: mode, clearance, winding of path, length.
Also replays logged observations of a life through the policy (what it WOULD have commanded at t=0)."""
import importlib
import sys
import time
import numpy as np
from pilots.aspire_blind import geom

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    ver = sys.argv[1]
    mod = importlib.import_module(f"pilots.aspire_blind.{ver}")
    d = np.load("runs/station/observation/layouts.npz")
    tool, posts, ps, tgt = d["tool_pos"], d["posts"], d["post_sides"], d["target"]
    modes, clr, wmin, lens = [], [], [], []
    t0 = time.time()
    for i in range(200):
        pts, S, mode = mod.plan(tool[i], posts[i], ps[i], tgt[i])
        modes.append(mode)
        clr.append(min(np.min(np.linalg.norm(pts - p, axis=1)) for p in posts[i]))
        rope = np.vstack([[0.05, 0.0], pts])
        wmin.append((geom.winding(rope, posts[i]) * ps[i]).min())
        lens.append(S[-1])
    clr, wmin, lens = map(np.array, (clr, wmin, lens))
    print("time per plan %.3f s" % ((time.time() - t0) / 200))
    print("modes", {m: modes.count(m) for m in set(modes)})
    print("clearance min %.3f  n<0.06 %d" % (clr.min(), (clr < 0.06).sum()))
    print("anchor+path winding min %.2f n<1.8 %d" % (wmin.min(), (wmin < 1.8).sum()))
    print("length mean %.3f max %.3f" % (lens.mean(), lens.max()))
    g = [i for i, m in enumerate(modes) if m != "corridor"]
    print("non-corridor seeds", g, clr[g], wmin[g], lens[g])
