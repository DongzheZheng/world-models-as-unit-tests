"""Read one of my own black-box lives (or an eval log): decimated trace + load/speed statistics.
usage: python pilots/aspire_station/an_life.py <npz path> [every]"""
import sys
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=170)
    path = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    d = np.load(path, allow_pickle=True)
    pts, ld, tp, tv, a = d["points"], d["load"], d["tool_pos"], d["tool_vel"], d["actions"]
    posts = d["posts"][0]
    T = len(a)
    print("posts", posts.ravel(), "sides", d["post_sides"][0], "target", d["target"][0], "T", T,
          {k: d[k].item() for k in ("success", "overload", "max_load", "final_dist", "sides_ok") if k in d.files})
    sp = np.linalg.norm(tv, axis=1)
    for t in list(range(0, T + 1, every)) + list(range(max(0, T - 4), T + 1)):
        dp = np.min(np.linalg.norm(pts[t][:, None, :2] - posts[None], axis=2), axis=0)
        dtool = np.min(np.linalg.norm(tp[t] - posts, axis=1))
        act = a[min(t, T - 1)]
        print(f"t{t:4d} tool {tp[t]} spd {sp[t]:.3f} load {ld[t]:6.2f} lag {np.linalg.norm(act - tp[t]):.3f} "
              f"sides {d['sides_now'][t]} cable-post {dp} tool-post {dtool:.3f}")
    m = sp[1:] > 0.01
    r = ld[1:][m] / sp[1:][m]
    print("load/speed ratio percentiles", np.percentile(r, [5, 25, 50, 75, 95]).round(1))
    for lo, hi in ((0, .02), (.02, .04), (.04, .06), (.06, .08), (.08, .1), (.1, .2)):
        mm = (sp[1:] >= lo) & (sp[1:] < hi)
        if mm.any():
            print(f"speed [{lo},{hi}) n {mm.sum():3d} load mean {ld[1:][mm].mean():.2f} max {ld[1:][mm].max():.2f}")
