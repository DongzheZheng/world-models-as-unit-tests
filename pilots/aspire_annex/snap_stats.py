import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
d = np.load("runs/annex/observation/layouts.npz")
P, tool, post, tgt = d["points"], d["tool_pos"], d["post"], d["target"]
print("zone", d["zone"][0])
for name, a in [("tool", tool), ("post", post), ("target", tgt)]:
    print(name, "min", a.min(0), "max", a.max(0), "mean", a.mean(0))
seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
print("seg len 3d mean/min/max", seg.mean(), seg.min(), seg.max(), "rest 0.7/19=", 0.7/19)
print("total arc 3d", seg.sum(1).min(), seg.sum(1).max())
print("z: min", P[:, :, 2].min(), "max", P[:, :, 2].max(), "per-index mean z", P[:, :, 2].mean(0))
print("p20 - tool", np.abs(P[:, 19, :2] - tool).max(), "p20 z", P[:, 19, 2].min(), P[:, 19, 2].max())
print("p1 xy min", P[:, 0, :2].min(0), "max", P[:, 0, :2].max(0))
d1 = np.linalg.norm(P[:, 0, :2] - tgt, axis=1); print("p1-target dist min/mean/max", d1.min(), d1.mean(), d1.max())
dpost = np.linalg.norm(post - tool, axis=1); print("tool-post dist", dpost.min(), dpost.max())
dtp = np.linalg.norm(tgt - post, axis=1); print("post-target dist", dtp.min(), dtp.max())
dtt = np.linalg.norm(tgt - tool, axis=1); print("tool-target dist", dtt.min(), dtt.max())
# chord end-to-end
ch = np.linalg.norm(P[:, 0, :2] - P[:, 19, :2], axis=1); print("end-to-end chord", ch.min(), ch.mean(), ch.max())
# shape: angles between consecutive segments
v = np.diff(P[:, :, :2], axis=1); ang = np.arctan2(v[..., 1], v[..., 0]); turn = np.diff(np.unwrap(ang, axis=1), axis=1)
print("turning per joint abs mean", np.abs(turn).mean(), "max", np.abs(turn).max(), "total turn range", turn.sum(1).min(), turn.sum(1).max())
# min distance of points to post
dp = np.linalg.norm(P[:, :, :2] - post[:, None], axis=2); print("min point-post dist", dp.min(1).min(), dp.min(1).mean())
print("direction angle of tool->post (deg)", np.degrees(np.arctan2(*(post - tool).T[::-1])).min(), np.degrees(np.arctan2(*(post - tool).T[::-1])).max())
print("dir tool->target (deg)", np.degrees(np.arctan2(*(tgt - tool).T[::-1])).min(), np.degrees(np.arctan2(*(tgt - tool).T[::-1])).max())
# is post between tool and target? angle between tool->post and tool->target
a1 = np.arctan2(*(post - tool).T[::-1]); a2 = np.arctan2(*(tgt - tool).T[::-1]); print("angle post vs target from tool (deg)", np.degrees(a2 - a1).min(), np.degrees(a2 - a1).max())
# lateral offset of post from tool->target line
u = (tgt - tool) / dtt[:, None]; w = post - tool; lat = w[:, 0]*u[:, 1] - w[:, 1]*u[:, 0]; along = (w*u).sum(1)
print("post along / lateral wrt tool->target", along.min(), along.max(), lat.min(), lat.max())
for i in range(5):
    print("seed", i, "tool", tool[i], "post", post[i], "tgt", tgt[i]); print(P[i])
