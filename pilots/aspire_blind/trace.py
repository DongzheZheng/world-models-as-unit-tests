"""Read a black-box life log and print the per-step execution trace (diagnosis only)."""
import sys
import numpy as np
from pilots.aspire_blind import geom

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    path = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    d = np.load(path, allow_pickle=True)
    P, tp, tv, ld, act, posts, ps = d["points"], d["tool_pos"], d["tool_vel"], d["load"], d["actions"], d["posts"], d["post_sides"]
    T = len(act)
    print("seed", d["seed"], "success", d["success"], "overload", d["overload"], "n_steps", d["n_steps"], "max_load", d["max_load"],
          "final_dist", d["final_dist"], "sides_ok", d["sides_ok"])
    print("posts", posts[0].ravel(), "sides", ps[0], "target", d["target"][0])
    print("p1 displacement max over episode: %.4f" % np.max(np.linalg.norm(P[:, 0, :2] - P[0, 0, :2], axis=1)))
    segl = np.linalg.norm(np.diff(P, axis=1), axis=2)
    print("segment length over time: min %.4f max %.4f ; total length range %.4f..%.4f" % (segl.min(), segl.max(), segl.sum(1).min(), segl.sum(1).max()))
    print(" t   cmd            tool           |v|    lag    load   sides  W                 p1            z(max)")
    for t in list(range(0, T, every)) + [T - 1]:
        w = geom.winding(P[t + 1, :, :2], posts[0])
        lag = np.linalg.norm(act[t] - tp[t + 1])
        print("%3d %s %s %.3f %.3f %6.2f %s %s %s %.4f" % (t + 1, act[t], tp[t + 1], np.linalg.norm(tv[t + 1]), lag, ld[t + 1],
                                                     d["sides_now"][t + 1], w, P[t + 1, 0, :2], P[t + 1, :, 2].max()))
