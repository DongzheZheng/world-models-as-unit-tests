"""For each snapshot: can a STRAIGHT rod (0.70 m) from a tool position in the zone put p1 in the target disc while
clearing the post?  Prints feasible fraction and the best margins."""
import numpy as np

if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    post, tgt, tool0 = d["post"], d["target"], d["tool_pos"]
    xs = np.linspace(0.30, 0.55, 26)
    ys = np.linspace(-0.20, 0.20, 41)
    T = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    nf, margins = 0, []
    for i in range(len(post)):
        v = tgt[i] - T
        r = np.linalg.norm(v, axis=1)
        u = v / r[:, None]
        tip = T + 0.70 * u                                  # rod pointing straight at the target centre
        dt = np.abs(r - 0.70)                               # tip distance to target centre
        # distance from post to segment T..tip
        w = post[i] - T
        s = np.clip((w * u).sum(1), 0, 0.70)
        dp = np.linalg.norm(T + s[:, None] * u - post[i], axis=1)
        ok = (dt <= 0.08) & (dp >= 0.04)
        # also allow aiming off-centre: angle offsets
        best = 0
        for off in np.radians(np.linspace(-12, 12, 49)):
            c, sn = np.cos(off), np.sin(off)
            uu = np.stack([u[:, 0] * c - u[:, 1] * sn, u[:, 0] * sn + u[:, 1] * c], 1)
            tp = T + 0.70 * uu
            s2 = np.clip((w * uu).sum(1), 0, 0.70)
            dp2 = np.linalg.norm(T + s2[:, None] * uu - post[i], axis=1)
            dt2 = np.linalg.norm(tp - tgt[i], axis=1)
            ok2 = (dt2 <= 0.08) & (dp2 >= 0.04)
            ok = ok | ok2
            m = np.where((dp2 >= 0.04), 0.10 - dt2, -1).max()
            best = max(best, m)
        nf += ok.any()
        margins.append(best)
    margins = np.array(margins)
    print("feasible straight-rod rest poses:", nf, "/", len(post))
    print("best tip margin (0.10 - dist) quantiles", np.quantile(margins, [0, 0.1, 0.5, 0.9, 1]))
