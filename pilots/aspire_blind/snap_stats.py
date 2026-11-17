"""Static statistics of the 200 scene snapshots (no motion)."""
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    d = np.load("runs/station/observation/layouts.npz")
    P, tool, posts, ps, tgt, load = d["points"], d["tool_pos"], d["posts"], d["post_sides"], d["target"], d["load"]
    print("keys", list(d.keys()))
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)            # (200,19)
    print("segment length 3d: mean %.4f std %.5f min %.4f max %.4f" % (seg.mean(), seg.std(), seg.min(), seg.max()))
    print("per-index mean seg", seg.mean(0))
    print("total length mean %.4f std %.5f" % (seg.sum(1).mean(), seg.sum(1).std()))
    chord = np.linalg.norm(P[:, -1] - P[:, 0], axis=1)
    print("chord p1-p20 mean %.4f min %.4f max %.4f" % (chord.mean(), chord.min(), chord.max()))
    print("p1 xyz mean", P[:, 0].mean(0), "std", P[:, 0].std(0), "min", P[:, 0].min(0), "max", P[:, 0].max(0))
    print("p20 xyz mean", P[:, -1].mean(0), "std", P[:, -1].std(0))
    print("tool - p20 xy max abs", np.abs(tool - P[:, -1, :2]).max())
    print("z per index mean", P[:, :, 2].mean(0))
    print("z min max", P[:, :, 2].min(), P[:, :, 2].max())
    print("tool min", tool.min(0), "max", tool.max(0))
    print("posts x range", posts[..., 0].min(), posts[..., 0].max(), "y", posts[..., 1].min(), posts[..., 1].max())
    print("posts sorted by x?", np.all(np.diff(posts[..., 0], axis=1) > 0))
    print("posts mean per idx", posts.mean(0))
    pd = np.linalg.norm(posts[:, :, None] - posts[:, None], axis=3)
    print("min post spacing", pd[pd > 0].min())
    print("target min", tgt.min(0), "max", tgt.max(0), "mean", tgt.mean(0))
    print("post_sides combos", np.unique(ps, axis=0, return_counts=True))
    print("load: mean %.4f std %.5f min %.4f max %.4f" % (load.mean(), load.std(), load.min(), load.max()))
    # load vs geometry
    for name, v in [("chord", chord), ("p20z", P[:, -1, 2]), ("tool x", tool[:, 0]), ("dist tool-p1 xy", np.linalg.norm(tool - P[:, 0, :2], axis=1))]:
        print("corr load,", name, np.corrcoef(load, v)[0, 1])
    # straightness: max lateral deviation from chord
    W = np.array([np.abs(np.cross(np.r_[(P[i, -1, :2] - P[i, 0, :2]), 0] / max(1e-9, np.linalg.norm(P[i, -1, :2] - P[i, 0, :2])),
                                  np.c_[P[i, :, :2] - P[i, 0, :2], np.zeros(20)])[:, 2]).max() for i in range(200)])
    print("max lateral deviation from chord: mean %.4f max %.4f" % (W.mean(), W.max()))
    # sides at reset
    from reallab.station import api
    s0 = np.array([api.sides(P[i], posts[i]) for i in range(200)])
    w0 = np.array([api.winding(P[i], posts[i]) for i in range(200)])
    print("initial sides counts", np.unique(s0, return_counts=True), "already ok all 3:", np.sum(np.all(s0 == ps, 1)))
    print("initial winding abs max", np.abs(w0).max())
    # distance of target from posts, from p1
    print("target-p1 dist mean", np.linalg.norm(tgt - P[:, 0, :2], axis=1).mean(), "min", np.linalg.norm(tgt - P[:, 0, :2], axis=1).min(), "max", np.linalg.norm(tgt - P[:, 0, :2], axis=1).max())
    # does a post lie between anchor and tool?
    for i in range(5):
        print("seed", i, "p1", P[i, 0], "p20", P[i, -1], "tool", tool[i], "posts", posts[i].ravel(), "sides", ps[i], "target", tgt[i], "load", load[i])
        print("   pts xy", P[i, :, :2].ravel()[:12], "z", P[i, :, 2])
