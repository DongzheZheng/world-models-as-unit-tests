# Static statistics of the 200 scene snapshots (no lab calls).
import numpy as np

if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    for k in d.files:
        print(k, d[k].shape)
    P, tool, post, tgt = d["points"], d["tool_pos"], d["post"], d["target"]
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print("zone", d["zone"][0])
    print("seed0 points\n", P[0])
    print("tool0", tool[0], "post0", post[0], "target0", tgt[0])
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)            # (200,19) 3d
    seg2 = np.linalg.norm(np.diff(P[:, :, :2], axis=1), axis=2)
    print("seg3d mean/min/max", seg.mean(), seg.min(), seg.max(), "expected", 0.7 / 19)
    print("seg2d mean/min/max", seg2.mean(), seg2.min(), seg2.max())
    print("arc3d per scene: min/mean/max", seg.sum(1).min(), seg.sum(1).mean(), seg.sum(1).max())
    z = P[:, :, 2]
    print("z overall min/max", z.min(), z.max())
    print("z per index mean", z.mean(0))
    print("z per index std", z.std(0))
    print("p20 - tool", np.abs(P[:, 19, :2] - tool).max(), "p20 z", P[:, 19, 2].min(), P[:, 19, 2].max())
    print("tool range", tool.min(0), tool.max(0))
    print("post range", post.min(0), post.max(0))
    print("target range", tgt.min(0), tgt.max(0))
    dtp = np.linalg.norm(tgt - post, axis=1)
    print("|target-post| range", dtp.min(), dtp.mean(), dtp.max())
    dtt = np.linalg.norm(tgt - tool, axis=1)
    print("|target-tool| range", dtt.min(), dtt.mean(), dtt.max())
    dpt = np.linalg.norm(post - tool, axis=1)
    print("|post-tool| range", dpt.min(), dpt.mean(), dpt.max())
    d1t = np.linalg.norm(P[:, 0, :2] - tgt, axis=1)
    print("initial tip dist range", d1t.min(), d1t.mean(), d1t.max())
    # straightness: end-to-end distance vs arc
    e2e = np.linalg.norm(P[:, 0, :2] - P[:, 19, :2], axis=1)
    print("end-to-end 2d range", e2e.min(), e2e.mean(), e2e.max())
    # turning angles
    v = np.diff(P[:, :, :2], axis=1)
    ang = np.arctan2(v[..., 1], v[..., 0])
    turn = np.angle(np.exp(1j * np.diff(ang, axis=1)))
    print("turn angle abs mean/max", np.abs(turn).mean(), np.abs(turn).max())
    print("total turning per scene min/mean/max", np.abs(turn).sum(1).min(), np.abs(turn).sum(1).mean(), np.abs(turn).sum(1).max())
    print("direction p20->p1 angle (deg) range", np.degrees(np.arctan2(-v.sum(1)[:, 1], -v.sum(1)[:, 0])).min(),
          np.degrees(np.arctan2(-v.sum(1)[:, 1], -v.sum(1)[:, 0])).max())
    # post relative to the line tool->target
    u = (tgt - tool) / dtt[:, None]
    rel = post - tool
    along = (rel * u).sum(1)
    perp = rel[:, 0] * u[:, 1] - rel[:, 1] * u[:, 0]
    print("post along tool->target (frac)", (along / dtt).min(), (along / dtt).mean(), (along / dtt).max())
    print("post perp to tool->target", perp.min(), np.abs(perp).mean(), perp.max())
    # min dist from any point to post
    dp = np.linalg.norm(P[:, :, :2] - post[:, None], axis=2).min(1)
    print("min point-post dist", dp.min(), dp.mean(), dp.max())
    # are points within post radius
    print("points x range", P[:, :, 0].min(), P[:, :, 0].max(), "y", P[:, :, 1].min(), P[:, :, 1].max())
    for i in range(5):
        print(i, "tool", tool[i], "post", post[i], "tgt", tgt[i], "p1", P[i, 0], "e2e", e2e[i], "zmax", z[i].max())
    # correlations
    print("corr target x vs y", np.corrcoef(tgt.T)[0, 1])
    print("target polar from tool: r", dtt.min(), dtt.max(), "angle deg", np.degrees(np.arctan2(tgt[:, 1] - tool[:, 1], tgt[:, 0] - tool[:, 0])).min(), np.degrees(np.arctan2(tgt[:, 1] - tool[:, 1], tgt[:, 0] - tool[:, 0])).max())
    print("tool unique?", np.unique(tool.round(4), axis=0).shape)
    print("tool_vel", np.abs(d["tool_vel"]).max() if "tool_vel" in d.files else None)
