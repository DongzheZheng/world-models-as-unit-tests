"""Pooled linear fits over lives (closed-form estimates for hand-written rules):
 (a) rotation regression -> centre of mass (COM) estimate, with bootstrap spread;
 (b) translation of the COM point vs pusher displacement in contact (2x2 body-frame map)."""
import sys
import numpy as np
from pilots.aspire_pusht.analyze import per_step
from pilots.aspire_pusht.policy_v1 import wrap

def rows(files):
    out = []
    for f in files:
        r = per_step(np.load(f))
        for x in r:
            mvb = x["qb1"] - x["qb"]; L = np.linalg.norm(mvb)
            if x["dist"] < 16.5 and L > 0.5:
                n = mvb / L; p = x["qb"] + n * 15.0
                out.append((p, n, L, x["dth"], x["dpos_b"]))
    return out

def com_fit(R):
    X = np.array([[(p[0]*n[1]-p[1]*n[0])*L, -n[1]*L, n[0]*L] for p, n, L, _, _ in R]); y = np.array([r[3] for r in R])
    c, *_ = np.linalg.lstsq(X, y, rcond=None); return c[1]/c[0], c[2]/c[0], c[0], 1-np.var(y-X@c)/np.var(y)

if __name__ == "__main__":
    R = rows(sys.argv[1:])
    cu, cv, k, r2 = com_fit(R)
    rng = np.random.default_rng(0)
    bs = np.array([com_fit([R[i] for i in rng.integers(0, len(R), len(R))])[:2] for _ in range(300)])
    print(f"COM fit n={len(R)}: u={cu:.1f} v={cv:.1f} k={k:.5f} R2={r2:.2f}; bootstrap sd u={bs[:,0].std():.1f} v={bs[:,1].std():.1f}")
    # translation of COM point: d_com = dpos_b + dth * perp(com)
    com = np.array([cu, cv])
    D = np.array([r[4] + r[3]*np.array([-com[1], com[0]]) for r in R]); U = np.array([r[1]*r[2] for r in R])
    M, *_ = np.linalg.lstsq(U, D, rcond=None)
    print("COM translation = U @ M, M=\n", M.T.round(3), " R2=%.2f" % (1-np.var(D-U@M)/np.var(D)))
    # per push-face direction: mean ratio and angle
    for name, nn in [("+u", (1,0)), ("-u", (-1,0)), ("+v", (0,1)), ("-v", (0,-1))]:
        sel = [i for i, r in enumerate(R) if r[1] @ np.array(nn) > 0.9]
        if len(sel) < 3: continue
        a = np.degrees([np.arctan2(R[i][1][0]*D[i][1]-R[i][1][1]*D[i][0], R[i][1]@D[i]) for i in sel])
        g = [np.linalg.norm(D[i])/R[i][2] for i in sel]
        print(f"  push {name}: n={len(sel)} gain={np.median(g):.2f} angle(COM disp vs push) median={np.median(a):.1f}")
