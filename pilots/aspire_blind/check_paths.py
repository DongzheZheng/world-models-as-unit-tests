"""Static check of a policy version's planned path on the 200 snapshots: clearance and path winding."""
import importlib
import sys
import numpy as np
from pilots.aspire_blind import geom

if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "policy_v1"
    mod = importlib.import_module(f"pilots.aspire_blind.{ver}")
    d = np.load("runs/station/observation/layouts.npz")
    tool, posts, ps, tgt = d["tool_pos"], d["posts"], d["post_sides"], d["target"]
    clr, wmin, lens, infeas = [], [], [], 0
    for i in range(200):
        pts, S = mod.corridor_path(tool[i], posts[i], ps[i], tgt[i])
        clr.append(min(np.min(np.linalg.norm(pts - p, axis=1)) for p in posts[i]))
        w = geom.winding(pts, posts[i]) * ps[i]
        wmin.append(w.min())
        lens.append(S[-1])
    clr, wmin, lens = map(np.array, (clr, wmin, lens))
    print("path clearance to post centres: min %.3f, n<0.06: %d" % (clr.min(), (clr < 0.06).sum()))
    print("path signed winding min-over-posts: min %.2f  n<1.57: %d  n<2.0: %d" % (wmin.min(), (wmin < 1.57).sum(), (wmin < 2.0).sum()))
    print("path length: mean %.3f max %.3f" % (lens.mean(), lens.max()))
    print("low-winding seeds", np.where(wmin < 1.8)[0], wmin[wmin < 1.8])
    print("low clearance seeds", np.where(clr < 0.06)[0])
