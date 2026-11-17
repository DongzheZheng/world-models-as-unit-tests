"""Read a black-box life and print a per-step execution trace digest (offline, logs only).
   python pilots/aspire_drone/trace.py runs/drone/blackbox/aspire_style/life_1.npz [every]"""
import sys

import numpy as np

from pilots.aspire_drone.policy_v1 import quat_to_R

z = np.load(sys.argv[1])
every = int(sys.argv[2]) if len(sys.argv) > 2 else 6
print({k: (z[k].shape if z[k].ndim else z[k].item()) for k in z.files})
pos, vel, q, w, A = z["pos"], z["lin_vel"], z["quat"], z["ang_vel"], z["actions"]
ri = z["relay_index"].reshape(-1)
rng = z["ranges"]
n = len(A)
print("relays", z["relays"][0].tolist())
print(" t     x      y      z     vx    vy    vz   roll  pitch  yaw | wx    wy    wz  | cmd vp  vq   vr    T  | ri minR")
for i in range(0, n + 1, every):
    R = quat_to_R(q[i])
    roll = np.arctan2(R[2, 1], R[2, 2]); pitch = -np.arcsin(np.clip(R[2, 0], -1, 1)); yaw = np.arctan2(R[1, 0], R[0, 0])
    a = A[min(i, n - 1)]
    print(f"{i/30:4.2f} {pos[i,0]:6.2f} {pos[i,1]:6.2f} {pos[i,2]:5.2f} {vel[i,0]:5.2f} {vel[i,1]:5.2f} {vel[i,2]:5.2f}"
          f" {np.degrees(roll):6.1f} {np.degrees(pitch):6.1f} {np.degrees(yaw):6.1f} | {w[i,0]:5.2f} {w[i,1]:5.2f} {w[i,2]:5.2f}"
          f" | {a[0]:5.2f} {a[1]:5.2f} {a[2]:5.2f} {a[3]:5.3f} | {ri[i]} {rng[i].min():5.2f}")
i = n
print("END", str(z["end_cause"]), "relays_reached", int(z["relays_reached"]), "steps", n, "final pos", pos[-1].round(2))
