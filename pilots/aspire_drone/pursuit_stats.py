"""Do other aircraft react to the drone?  For mover samples within distance bins of the drone, the cosine between
the mover's velocity and the direction to the drone, and how often a mover CHANGES velocity (|dv| > 0.5 m/s)
per bin.  Offline statistics of the black-box instrumentation of all mission-C calibration lives."""
import glob

import numpy as np

bins = [0, 2, 3, 5, 8, 15, 99]
cos_by, chg_by, n_by = {b: [] for b in bins[:-1]}, {b: 0 for b in bins[:-1]}, {b: 0 for b in bins[:-1]}
cls = {b: [] for b in bins[:-1]}
for f in sorted(glob.glob("runs/drone/blackbox/aspire_style_C*/life_*.npz")):
    z = np.load(f)
    P, MP, MV = z["pos"], z["movers_true_pos"], z["movers_true_vel"]
    k = int(z["movers_true_count"].max())
    for j in range(k):
        sp = np.linalg.norm(MV[:, j], axis=1)
        for i in range(1, len(P)):
            r = P[i] - MP[i, j]
            d = np.linalg.norm(r)
            b = max(bb for bb in bins[:-1] if d >= bb)
            n_by[b] += 1
            if np.linalg.norm(MV[i, j] - MV[i - 1, j]) > 0.5:
                chg_by[b] += 1
                cls[b].append(np.linalg.norm(MV[i, j]))
            if sp[i] > 0.3:
                cos_by[b].append(float(MV[i, j] @ r / (sp[i] * d)))
for b in bins[:-1]:
    c = np.array(cos_by[b])
    print(f"d in [{b},{bins[bins.index(b)+1]}): samples {n_by[b]:5d} moving {len(c):5d} mean cos(v, to drone) "
          f"{c.mean() if len(c) else float('nan'):+.2f} frac cos>0.7 {np.mean(c > 0.7) if len(c) else float('nan'):.2f} | "
          f"velocity changes per 100 samples {100 * chg_by[b] / max(n_by[b], 1):.1f} (new speed median "
          f"{np.median(cls[b]) if cls[b] else float('nan'):.1f})")
