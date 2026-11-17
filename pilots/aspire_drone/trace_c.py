"""Mission-C trace digest: own state, observed movers, and the black-box instrumentation of all other aircraft.
   python pilots/aspire_drone/trace_c.py runs/drone/blackbox/aspire_style_C/life_1.npz [every]"""
import sys

import numpy as np

z = np.load(sys.argv[1])
every = int(sys.argv[2]) if len(sys.argv) > 2 else 6
P, V = z["pos"], z["lin_vel"]
MP, MV, MC = z["movers_true_pos"], z["movers_true_vel"], z["movers_true_count"]
n = len(z["actions"])
print("end", str(z["end_cause"]), "relays", int(z["relays_reached"]), "steps", n, "movers", MC.reshape(-1)[:1])
k = int(np.max(MC))
print("initial mover states:")
for j in range(k):
    print(f"  m{j} pos {np.round(MP[0, j], 2)} vel {np.round(MV[0, j], 2)}")
for i in list(range(0, n + 1, every)) + [n]:
    d = np.linalg.norm(MP[i, :k] - P[i], axis=1)
    j = int(np.argmin(d))
    seen = int(np.asarray(z["n_movers"][i]).reshape(-1)[0]) if "n_movers" in z.files else -1
    print(f"{i/30:5.2f} own {np.round(P[i], 2)} v {np.round(V[i], 1)} | seen {seen} | nearest m{j} d {d[j]:5.2f}"
          f" at {np.round(MP[i, j], 2)} v {np.round(MV[i, j], 1)} | all d {np.round(d, 1)}")
