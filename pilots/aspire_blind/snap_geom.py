"""Run the waypoint geometry on the 200 snapshot situations (static checks only)."""
import numpy as np
from pilots.aspire_blind import geom

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    d = np.load("runs/station/observation/layouts.npz")
    P, tool, posts, ps, tgt = d["points"], d["tool_pos"], d["posts"], d["post_sides"], d["target"]
    anchor = P[0, 0, :2]
    Lt, Wmin, clr, tool_clr, rope_maxx = [], [], [], [], []
    for i in range(200):
        # taut rope estimate: anchor -> hug points (r=0.034) -> target
        hug = geom.waypoints(anchor, posts[i], ps[i], tgt[i], c=0.034)
        rope = np.vstack([anchor, hug])
        Lt.append(geom.polyline_len(rope))
        w = geom.winding(geom.densify(rope), posts[i]) * ps[i]
        Wmin.append(w.min())
        tp = np.vstack([tool[i], geom.waypoints(tool[i], posts[i], ps[i], tgt[i], c=0.07)])
        tool_clr.append(geom.min_clearance(tp, posts[i]))
        rope_maxx.append(P[i, :, 0].max())
    Lt, Wmin, tool_clr = map(np.array, (Lt, Wmin, tool_clr))
    print("taut length est: mean %.3f max %.3f  n>0.95: %d  n>0.99: %d" % (Lt.mean(), Lt.max(), (Lt > 0.95).sum(), (Lt > 0.99).sum()))
    print("signed winding of hug path (min over posts): min %.2f  n<pi/2: %d" % (Wmin.min(), (Wmin < np.pi / 2).sum()))
    print("tool path min clearance to post centre (c=0.07): min %.3f  n<0.05: %d" % (tool_clr.min(), (tool_clr < 0.05).sum()))
    print("rope max x at reset: max %.3f" % max(rope_maxx), " posts min x %.3f" % posts[..., 0].min())
    print("target x - last post x: min %.3f" % (tgt[:, 0] - posts[:, :, 0].max(1)).min())
    bad = np.where((Wmin < np.pi / 2) | (tool_clr < 0.05) | (Lt > 0.95))[0]
    print("problem seeds", bad, Lt[bad], Wmin[bad], tool_clr[bad])
    # typical seeds for a first life
    med = np.argsort(np.abs(Lt - np.median(Lt)))[:10]
    print("median-length seeds", med, Lt[med], ps[med].tolist())
