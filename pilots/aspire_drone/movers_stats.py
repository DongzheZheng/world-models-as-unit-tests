"""Statistics of the other aircraft from mission-C black-box instrumentation (offline, logs only).
For every mover: motion episodes (|v| > 0.3), their start time, duration, speed, direction, and the drone's
relative position/velocity at the start; plus how often observed rows agree with the truth.
   python pilots/aspire_drone/movers_stats.py life_1.npz [life_2.npz ...]"""
import sys

import numpy as np

for path in sys.argv[1:]:
    z = np.load(path)
    P, V = z["pos"], z["lin_vel"]
    MP, MV = z["movers_true_pos"], z["movers_true_vel"]
    k = int(np.max(z["movers_true_count"]))
    n = len(P)
    print("==", path, str(z["end_cause"]), "movers", k)
    for j in range(k):
        sp = np.linalg.norm(MV[:, j], axis=1)
        moving = sp > 0.3
        segs, i = [], 0
        while i < n:
            if moving[i]:
                s = i
                while i < n and moving[i]:
                    i += 1
                segs.append((s, i))
            i += 1
        desc = []
        for s, e in segs:
            vm = MV[s:e, j].mean(0)
            r = MP[s, j] - P[s]
            rv = np.dot(-r, vm) / (np.linalg.norm(r) * np.linalg.norm(vm) + 1e-9)   # >0: moving toward drone
            disp = MP[min(e, n - 1), j] - MP[s, j]
            desc.append(f"[t{s/30:.2f} {(e-s)/30:.2f}s |v| {np.linalg.norm(vm):.1f} v {np.round(vm,1)} disp {np.linalg.norm(disp):.1f} "
                        f"d_drone {np.linalg.norm(r):.1f} toward {rv:+.2f}]")
        dmin = np.linalg.norm(MP[:, j] - P, axis=1)
        print(f" m{j} start {np.round(MP[0, j], 2)} min d {dmin.min():.2f} at t {dmin.argmin()/30:.2f}; {len(segs)} moves")
        for d in desc:
            print("     ", d)
    if "movers" in z.files:
        errs = []
        for i in range(n):
            nm = int(np.asarray(z["n_movers"][i]).reshape(-1)[0])
            for row in z["movers"][i][:nm]:
                dd = np.linalg.norm(MP[i, :k] - row[:3], axis=1)
                jj = int(np.argmin(dd))
                errs.append((dd[jj], np.linalg.norm(MV[i, jj] - row[3:6])))
        errs = np.array(errs)
        if len(errs):
            print(f" observed rows {len(errs)}: pos err median {np.median(errs[:,0]):.3f} p95 {np.percentile(errs[:,0],95):.3f};"
                  f" vel err median {np.median(errs[:,1]):.3f} p95 {np.percentile(errs[:,1],95):.3f}")
