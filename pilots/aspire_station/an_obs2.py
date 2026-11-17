"""Print the per-step trace of observation episodes (decimated)."""
import sys
import numpy as np

OBS = "runs/station/observation"

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    seeds = [int(s) for s in sys.argv[1:]] or [903]
    every = 4
    for s in seeds:
        d = np.load(f"{OBS}/episode_{s}.npz", allow_pickle=True)
        pts, ld, tp, tv, a = d["points"], d["load"], d["tool_pos"], d["tool_vel"], d["actions"]
        posts = d["posts"][0]
        print(f"=== ep {s} posts {posts.ravel()} sides {d['post_sides'][0]} target {d['target'][0]}")
        T = len(a)
        for t in list(range(0, T, every)) + [T - 3, T - 2, T - 1, T]:
            if t < 0 or t > T:
                continue
            sp = np.linalg.norm(tv[t])
            # chord length anchor->tool and min post-distance of cable points
            dpost = np.min(np.linalg.norm(pts[t][:, None, :2] - posts[None], axis=2), axis=0)
            # z max (lifted?)
            act = a[t] if t < T else a[-1]
            lag = np.linalg.norm(act - tp[t])
            print(f"t{t:4d} tool {tp[t]} spd {sp:.3f} load {ld[t]:6.2f} cmd {act} lag {lag:.3f} sides {d['sides_now'][t]} "
                  f"dpost {dpost} zmax {pts[t][:, 2].max():.4f}")
