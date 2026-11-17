"""For each snapshot: rope direction from the tool, and the best cast plan whose direction u is opposite to it."""
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
ZLO, ZHI = np.array([0.30, -0.20]), np.array([0.55, 0.20])
def seg_dist(p, a, b):
    ab = b - a; t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1); return float(np.linalg.norm(p - (a + t * ab)))
def plans(target, post, reach=0.66, clear=0.03, aim_off=0.05):
    out = []
    radial = (target - post) / np.linalg.norm(target - post); lat = np.array([-radial[1], radial[0]])
    for side in (-1, 0, 1):
        A = target + side * aim_off * lat
        for ang in np.radians(np.arange(-80, 80.1, 1.0)):
            u = np.array([np.cos(ang), np.sin(ang)]); F = A - reach * u
            if np.any(F < ZLO) or np.any(F > ZHI): continue
            if seg_dist(post, F, A) < clear: continue
            s = 0.35
            while s > 0.0 and (np.any(F - s * u < ZLO - 1e-9) or np.any(F - s * u > ZHI + 1e-9)): s -= 0.01
            out.append((ang, s, F, A))
    return out
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz"); P = d["points"][:, :, :2]; tool = d["tool_pos"]
    rows = []
    for i in range(200):
        c = P[i].mean(0) - tool[i]; rdir = np.arctan2(c[1], c[0])
        chord = np.linalg.norm(P[i, 0] - P[i, 19])
        want = np.angle(np.exp(1j * (rdir + np.pi)))
        pl = plans(d["target"][i], d["post"][i])
        angs = np.array([p[0] for p in pl]); ss = np.array([p[1] for p in pl])
        good = ss >= 0.15
        mis = np.abs(np.angle(np.exp(1j * (angs - want)))) if len(pl) else np.array([9])
        best = np.argmin(mis + (~good) * 10) if len(pl) else None
        rows.append((i, np.degrees(want), chord, np.degrees(angs[good].min()) if good.any() else np.nan, np.degrees(angs[good].max()) if good.any() else np.nan, np.degrees(mis[best]) if best is not None else np.nan, ss[best] if best is not None else 0))
    R = np.array(rows)
    print("want-angle dist (deg): pct", np.percentile(R[:, 1], [5, 25, 50, 75, 95]))
    print("feasible u window min/max (stroke>=.15) pct of min", np.nanpercentile(R[:, 3], [5, 50, 95]), "max", np.nanpercentile(R[:, 4], [5, 50, 95]))
    print("misalignment of best plan pct", np.nanpercentile(R[:, 5], [10, 25, 50, 75, 90]))
    idx = np.argsort(R[:, 5] - R[:, 2] * 30)[:12]
    for k in idx: print(R[k])
