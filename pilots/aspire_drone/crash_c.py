"""Diagnose the end of a mission-C life: last 1.5 s of own state, nearest movers (truth vs observed), wall clearance.
   python pilots/aspire_drone/crash_c.py runs/drone/blackbox/<pilot>/life_k.npz"""
import sys

import numpy as np

from pilots.aspire_drone.policy_v6 import box_clearance

z = np.load(sys.argv[1])
P, V, MP, MV = z["pos"], z["lin_vel"], z["movers_true_pos"], z["movers_true_vel"]
k = int(np.max(z["movers_true_count"]))
n = len(P) - 1
print("end", str(z["end_cause"]), "relays", int(z["relays_reached"]), "t", n / 30)
for i in range(max(0, n - 45), n + 1, 3):
    d = np.linalg.norm(MP[i, :k] - P[i], axis=1)
    j = np.argsort(d)[:2]
    nm = int(np.asarray(z["n_movers"][i]).reshape(-1)[0])
    seen = [np.round(r[:3], 1).tolist() for r in z["movers"][i][:nm]][:2]
    print(f"{i/30:5.2f} own {np.round(P[i],2)} v {np.round(V[i],1)} wall {box_clearance(P[i,0], P[i,1])[0]:.2f} | "
          + " ".join(f"m{jj} d{d[jj]:.2f} {np.round(MP[i,jj],1)} v{np.round(MV[i,jj],1)}" for jj in j) + f" | seen {nm} {seen}")
