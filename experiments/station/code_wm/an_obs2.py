# Per-step look at observation episodes: load vs. rope geometry and tool motion.
import numpy as np, sys
D = "runs/station/observation/"
np.set_printoptions(precision=4, suppress=True, linewidth=200)
seeds = [int(s) for s in sys.argv[1:]] or [903, 906]
every = 5
for s in seeds:
    E = np.load(D + f"episode_{s}.npz")
    pts, tp, tv, ld, A = E["points"], E["tool_pos"], E["tool_vel"], E["load"], E["actions"]
    posts = E["posts"][0]
    print(f"=== seed {s} posts {posts.tolist()} sides {E['post_sides'][0]} target {E['target'][0]}  outcome succ={E['success']} T={len(A)}")
    seg = np.linalg.norm(np.diff(pts[..., :2], axis=1), axis=2)
    for t in list(range(0, len(A), every)) + [len(A) - 3, len(A) - 2, len(A) - 1, len(A)]:
        if t < 0 or t > len(A):
            continue
        a = A[t] if t < len(A) else A[-1]
        e2e = np.linalg.norm(tp[t] - pts[t, 0, :2])
        dpost = np.linalg.norm(pts[t, :, None, :2] - posts[None], axis=2).min()
        dtool = np.linalg.norm(tp[t][None] - posts, axis=1).min()
        # rope point speeds
        pv = np.linalg.norm(pts[t, :, :2] - pts[max(t - 1, 0), :, :2], axis=1) / 0.05
        print(f"t={t:3d} cmd={a} tool={tp[t]} lag={np.linalg.norm(a - tp[t]):.4f} |v|={np.linalg.norm(tv[t]):.3f} load={ld[t]:6.3f} "
              f"e2e={e2e:.3f} poly={seg[t].sum():.4f} segmax={seg[t].max():.4f} dRopePost={dpost:.3f} dToolPost={dtool:.3f} "
              f"ropev mean={pv.mean():.3f} z={pts[t,:,2].max():.4f} sides={E['sides_now'][t]}")
