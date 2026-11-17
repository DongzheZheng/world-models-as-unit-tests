"""Offline trace analysis of black-box lives (statistics and linear fits only; no rollouts).
python pilots/aspire_pusht/analyze.py runs/pusht/blackbox/aspire_style/life_1.npz [...]"""
import sys
import numpy as np
import shapely.geometry as sg
from pilots.aspire_pusht.policy_v1 import BAR, STEM, rot, wrap

def poly(pose):
    R = rot(pose[2]); return sg.MultiPolygon([sg.Polygon(v @ R.T + pose[:2]) for v in (BAR, STEM)]).buffer(0)

def per_step(d):
    P, Q, A, C = d["block_pose"], d["agent_pos"], d["actions"], d["coverage"]
    rows = []
    for t in range(len(A)):
        pg = poly(P[t]); dist = pg.distance(sg.Point(Q[t]))
        dpos = P[t+1, :2] - P[t, :2]; dth = wrap(P[t+1, 2] - P[t, 2])
        Rt = rot(P[t, 2]).T
        qb = Rt @ (Q[t] - P[t, :2]); qb1 = Rt @ (Q[t+1] - P[t, :2])
        rows.append(dict(t=t, dist=dist, dpos=dpos, dth=dth, dpos_b=Rt @ dpos, qb=qb, qb1=qb1, cov=C[t+1],
                         speed=np.linalg.norm(Q[t+1]-Q[t]), lag=np.linalg.norm(A[t]-Q[t+1])))
    return rows

if __name__ == "__main__":
    allr = []
    for f in sys.argv[1:]:
        d = np.load(f)
        r = per_step(d); allr += r
        print(f, "success", d["success"], "max_cov %.3f" % d["max_coverage"], "n", d["n_steps"])
        C = d["coverage"]; P = d["block_pose"]
        for t in range(0, len(r), 15):
            x = r[t]; print(f"  t={t:3d} cov={x['cov']:.3f} pose=({P[t,0]:.0f},{P[t,1]:.0f},{P[t,2]:.2f}) dist={x['dist']:5.1f} "
                            f"qb=({x['qb'][0]:6.1f},{x['qb'][1]:6.1f}) dpos_b=({x['dpos_b'][0]:5.2f},{x['dpos_b'][1]:5.2f}) dth={x['dth']:+.4f} spd={x['speed']:.1f} lag={x['lag']:.1f}")
    # contact vs free steps
    cont = [x for x in allr if x["dist"] < 16.5]
    free = [x for x in allr if x["dist"] > 20]
    print("contact steps", len(cont), "free steps", len(free))
    mv_free = np.array([np.linalg.norm(x["dpos"]) + 50*abs(x["dth"]) for x in free])
    print("block motion in free steps: mean %.3f  frac>0.2: %.2f" % (mv_free.mean(), (mv_free > 0.2).mean()))
    # coasting after contact: motion k steps after last contact
    # COM regression: dth = k*(p x n) - k*cu*n_v + k*cv*n_u, with n = pusher displacement direction (body), p = contact pt
    X, y = [], []
    for x in cont:
        mvb = x["qb1"] - x["qb"]; L = np.linalg.norm(mvb)
        if L < 0.5: continue
        n = mvb / L
        # contact point: pusher centre + n * r (approx)
        p = x["qb"] + n * 15.0
        X.append([(p[0]*n[1]-p[1]*n[0]) * L, -n[1] * L, n[0] * L]); y.append(x["dth"])
    X, y = np.array(X), np.array(y)
    if len(y) > 5:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        k = coef[0]; print("rot regression: k=%.5f  com_u=%.1f com_v=%.1f  (n=%d) R2=%.2f" % (
            k, coef[1]/k, coef[2]/k, len(y), 1 - np.var(y - X@coef)/np.var(y)))
    # pusher tracking: speed vs commanded distance
    spd = np.array([x["speed"] for x in free]); lag = np.array([x["lag"] for x in free])
    print("free pusher speed per step mean %.1f, max %.1f; residual to target mean %.1f" % (spd.mean(), spd.max(), lag.mean()))
    # coasting: block motion as a function of steps since last contact
    for f in sys.argv[1:]:
        r = per_step(np.load(f)); last = -99; co = {}
        for x in r:
            if x["dist"] < 16.5: last = x["t"]
            k = x["t"] - last
            if 1 <= k <= 8: co.setdefault(k, []).append(np.linalg.norm(x["dpos"]) + 50*abs(x["dth"]))
        print("coast", f.split("/")[-1], {k: round(float(np.mean(v)), 3) for k, v in sorted(co.items())})
        # translation direction vs push direction in contact (body frame)
        ang = []
        for x in r:
            if x["dist"] < 16.5:
                mv = x["qb1"] - x["qb"]
                if np.linalg.norm(mv) > 1 and np.linalg.norm(x["dpos_b"]) > 0.3:
                    ang.append(np.degrees(np.arctan2(mv[0]*x["dpos_b"][1]-mv[1]*x["dpos_b"][0], mv @ x["dpos_b"])))
        if ang: print("  angle(block disp, pusher disp) deg: mean %.1f sd %.1f n=%d" % (np.mean(ang), np.std(ang), len(ang)))
        # ratio of block displacement to pusher displacement while in contact
        rat = [np.linalg.norm(x["dpos"]) / np.linalg.norm(x["qb1"]-x["qb"]) for x in r if x["dist"] < 16.5 and np.linalg.norm(x["qb1"]-x["qb"]) > 1]
        if rat: print("  |block disp|/|pusher disp| median %.2f" % np.median(rat))
