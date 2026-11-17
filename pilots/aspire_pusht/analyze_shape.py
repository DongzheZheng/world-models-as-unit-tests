"""Shape-aware trace analysis (statistics and linear fits only, no rollouts).
python pilots/aspire_pusht/analyze_shape.py <version> <shape> <npz>..."""
import sys, importlib
import numpy as np
import shapely.geometry as sg

def main():
    m = importlib.import_module(f"pilots.aspire_pusht.policy_v{sys.argv[1]}")
    shape = sys.argv[2]; files = sys.argv[3:]
    card = {"goal_pose": [300, 200, np.pi/2], "object": "L"} if shape == "L" else {"goal_pose": [256, 256, np.pi/4], "block": {"shape": "T"}}
    R_ = []
    for f in files:
        d = np.load(f); P, Q, C, A = d["block_pose"], d["agent_pos"], d["coverage"], d["actions"]
        pol = m.make_policy(); pol.bind(card); pol.reset()
        # replay to get the program's mode/contact trace
        for t in range(len(A)):
            pol.act(dict(agent_pos=Q[t], agent_vel=d["agent_vel"][t], block_pose=P[t], goal_pose=d["goal_pose"][t], coverage=float(C[t]), t=t))
        segs = []
        for t, (mode, i, s) in enumerate(pol.trace):
            if not segs or segs[-1][0] != (mode, i): segs.append([(mode, i), t, 1, s])
            else: segs[-1][2] += 1
        print(f, "success", bool(d["success"]), "max %.3f" % d["max_coverage"], "n", int(d["n_steps"]))
        g = np.array(card["goal_pose"])
        for (mode, i), t0, n, s in segs:
            if n >= 2:
                e = P[t0]; ep = np.linalg.norm(g[:2] - e[:2]); eth = np.degrees(m.wrap(g[2] - e[2]))
                print(f"  t={t0:3d} {mode:8s} c={i:2d} face={pol.F[i]} n={n:3d} s={s:.2f} cov={C[t0]:.3f} |ep|={ep:5.1f} eth={eth:+6.1f}")
        for t in range(len(A)):
            pg = pol.poly(P[t]); dist = pg.distance(sg.Point(Q[t]))
            Rt = m.rot(P[t, 2]).T
            qb, qb1 = Rt @ (Q[t] - P[t, :2]), Rt @ (Q[t+1] - P[t, :2])
            mv = qb1 - qb; L = np.linalg.norm(mv)
            if dist < 16.5 and L > 0.5:
                n = mv / L; p = qb + n * 15.0
                R_.append((p, n, L, m.wrap(P[t+1, 2] - P[t, 2]), Rt @ (P[t+1, :2] - P[t, :2])))
    X = np.array([[(p[0]*n[1]-p[1]*n[0])*L, -n[1]*L, n[0]*L] for p, n, L, _, _ in R_]); y = np.array([r[3] for r in R_])
    def fit(ix):
        c, *_ = np.linalg.lstsq(X[ix], y[ix], rcond=None); return np.array([c[1]/c[0], c[2]/c[0]]), c[0], 1 - np.var(y[ix]-X[ix]@c)/np.var(y[ix])
    com, k, r2 = fit(np.arange(len(y)))
    rng = np.random.default_rng(0)
    bs = np.array([fit(rng.integers(0, len(y), len(y)))[0] for _ in range(300)])
    print(f"COM fit n={len(y)}: ({com[0]:.1f}, {com[1]:.1f}) k={k:.5f} R2={r2:.2f} bootstrap sd ({bs[:,0].std():.1f}, {bs[:,1].std():.1f})")

if __name__ == "__main__":
    main()
