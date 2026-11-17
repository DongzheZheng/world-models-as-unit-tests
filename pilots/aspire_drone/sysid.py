"""Offline system identification from black-box logs (statistics / least squares on logged data only).
   python pilots/aspire_drone/sysid.py life_1.npz [life_2.npz ...]
Fits:  (1) body-rate response   w[i+d] ~ g_ax * cmd[i] + b_ax   per axis, for delays d
       (2) thrust:  vertical/thrust-axis specific force  ~ c * T[i-d]^2  -> hover throttle
       (3) residual (disturbance) acceleration in world frame vs position: a_dist = dv/dt - thrust_acc*b3 + g e_z
"""
import sys

import numpy as np

from pilots.aspire_drone.policy_v1 import quat_to_R

G = 9.81
for path in sys.argv[1:]:
    z = np.load(path)
    pos, vel, q, w, A = z["pos"], z["lin_vel"], z["quat"], z["ang_vel"], z["actions"]
    n = len(A)
    print("==", path, str(z["end_cause"]), n)
    # (1) rate response
    for d in (1, 2, 3):
        print(f" rate fit delay {d}:", end="")
        for ax in range(3):
            x = A[: n - d + 1, ax]; y = w[d: n + 1, ax]
            M = np.c_[x, np.ones_like(x)]
            (g_, b_), *_ = np.linalg.lstsq(M, y, rcond=None)
            r = y - M @ [g_, b_]
            print(f"  ax{ax}: gain {g_:.3f} bias {b_:+.3f} (cmd-equiv {-b_/g_:+.3f}) rms {r.std():.3f}", end="")
        print()
    # (2)+(3) specific force
    acc = np.gradient(vel, 1 / 30.0, axis=0)
    Rs = np.array([quat_to_R(qq) for qq in q])
    b3 = Rs[:, :, 2]
    for d in (1, 2, 3):
        idx = np.arange(d, n)
        T = A[idx - d, 3]
        sf = acc[idx] + np.array([0, 0, G])           # specific force (world)
        sf_b3 = np.einsum("ij,ij->i", sf, b3[idx])     # along thrust axis
        c, *_ = np.linalg.lstsq((T ** 2)[:, None], sf_b3, rcond=None)
        r = sf_b3 - c[0] * T ** 2
        print(f" thrust fit d={d}: c={c[0]:.2f}  hover T={np.sqrt(G / c[0]):.4f}  rms {r.std():.2f}")
    d = 2
    idx = np.arange(d, n)
    T = A[idx - d, 3]
    c = np.linalg.lstsq((T ** 2)[:, None], np.einsum("ij,ij->i", acc[idx] + [0, 0, G], b3[idx]), rcond=None)[0][0]
    adist = acc[idx] + np.array([0, 0, G]) - c * (T ** 2)[:, None] * b3[idx]
    print("  t     x      y     z   | a_dist x    y     z   | vel")
    for j in range(0, len(idx), 3):
        i = idx[j]
        print(f"  {i/30:4.2f} {pos[i,0]:6.2f} {pos[i,1]:5.2f} {pos[i,2]:4.2f} | {adist[j,0]:6.2f} {adist[j,1]:6.2f} {adist[j,2]:6.2f} |"
              f" {vel[i,0]:5.2f} {vel[i,1]:5.2f} {vel[i,2]:5.2f}")
