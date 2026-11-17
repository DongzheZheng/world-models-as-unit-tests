import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
d = np.load("runs/annex/observation/layouts.npz")
P, tool, post, tgt = d["points"], d["tool_pos"], d["post"], d["target"]
u = (tgt - post) / 0.12
for name, ref in [("tool0", tool), ("zone_centre", np.array([0.425, 0.0])), ("origin", np.zeros(2)), ("front_mid", np.array([0.55, 0.0]))]:
    w = post - ref; w = w / np.linalg.norm(w, axis=1)[:, None]
    ang = np.degrees(np.arccos(np.clip((w * u).sum(1), -1, 1)))
    print(name, "angle(post-ref, target-post) deg: max", ang.max(), "mean", ang.mean())
# distances of target from zone front-edge nearest point
near = np.clip(tgt, [0.30, -0.20], [0.55, 0.20]); print("dist target to nearest zone point", np.linalg.norm(tgt - near, axis=1).min(), np.linalg.norm(tgt - near, axis=1).max())
nearp = np.clip(post, [0.30, -0.20], [0.55, 0.20]); print("dist post to nearest zone point", np.linalg.norm(post - nearp, axis=1).min(), np.linalg.norm(post - nearp, axis=1).max())
print("radius of post from zone centre", np.linalg.norm(post - [0.425, 0], axis=1).min(), np.linalg.norm(post - [0.425, 0], axis=1).max())
print("radius of post from tool0", np.linalg.norm(post - tool, axis=1).min(), np.linalg.norm(post - tool, axis=1).max())
# is tip initially resting? check rope roughly smooth; rope side relative to tool: mean point minus tool
c = P[:, :, :2].mean(1) - tool; print("rope centroid relative to tool: mean", c.mean(0), "x range", c[:, 0].min(), c[:, 0].max())
# direction of rope at tool end (from p19 to p20)
v = P[:, 19, :2] - P[:, 18, :2]; print("last-seg angle deg min/max", np.degrees(np.arctan2(v[:, 1], v[:, 0])).min(), np.degrees(np.arctan2(v[:, 1], v[:, 0])).max())
