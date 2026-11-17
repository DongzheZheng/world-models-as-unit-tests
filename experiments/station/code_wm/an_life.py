# Print a compact per-step table of a black-box log.
import numpy as np, sys
np.set_printoptions(precision=4, suppress=True, linewidth=220)
f = sys.argv[1]; t0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0; t1 = int(sys.argv[3]) if len(sys.argv) > 3 else 10**6
every = int(sys.argv[4]) if len(sys.argv) > 4 else 1
E = np.load(f)
pts, tp, tv, ld, A = E["points"][..., :2], E["tool_pos"], E["tool_vel"], E["load"], E["actions"]
posts = E["posts"][0]
print("posts", posts.tolist(), "sides", E["post_sides"][0], "target", E["target"][0], "T", len(A), "succ", E["success"], "over", E["overload"])
seg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
for t in range(max(t0, 1), min(t1, len(A)) + 1, every):
    sp = np.linalg.norm(pts[t] - pts[t - 1], axis=1) / 0.05
    disp = np.linalg.norm(tp[t] - tp[t - 1]) / 0.05
    print(f"t={t:3d} cmd=({A[t-1][0]:.4f},{A[t-1][1]:.4f}) tool=({tp[t][0]:.4f},{tp[t][1]:.4f}) lag={np.linalg.norm(A[t-1]-tp[t]):.4f} "
          f"vread={np.linalg.norm(tv[t]):.3f} vdisp={disp:.3f} load={ld[t]:6.3f} sum|v|={sp[1:].sum():.3f} nmov={(sp>0.005).sum():2d} "
          f"e2e={np.linalg.norm(pts[t,19]-pts[t,0]):.3f} poly={seg[t].sum():.4f} segmax={seg[t].max():.4f} dpost={np.linalg.norm(pts[t,:,None]-posts[None],axis=2).min():.3f} sides={E['sides_now'][t]}")
