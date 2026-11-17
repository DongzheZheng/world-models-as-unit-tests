"""Quick look at black-box lives: rate-loop response, acceleration residual vs position (numpy only, prints)."""
import sys

import numpy as np

G = 9.81


def R_of(q):
    x, y, z, w = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


for path in sys.argv[1:]:
    z = np.load(path)
    print("=====", path, "end", z["end_cause"], "steps", int(z["n_steps"]), "relays", int(z["relays_reached"]),
          "mode", z["command_mode"], "seed", z["seed"])
    pos, vel, q, w, a = z["pos"], z["lin_vel"], z["quat"], z["ang_vel"], z["actions"]
    N = len(a)
    print("start pos", pos[0], "relays", z["relays"][0].tolist())
    if str(z["command_mode"]) == "ctbr":
        # rate loop: lagged regression w[t+1] ~ c0*w[t] + c1*a[t] + c2*a[t-1] + c3*a[t-2] + b
        for ax in range(3):
            X = np.stack([w[2:N, ax], a[2:N, ax], a[1:N - 1, ax], a[0:N - 2, ax], np.ones(N - 2)], 1)
            y = w[3:N + 1, ax]
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ c
            print(f"rate axis {ax}: w+ = {c[0]:.3f} w + {c[1]:.3f} a0 + {c[2]:.3f} a-1 + {c[3]:.3f} a-2 + {c[4]:+.3f}   rms {res.std():.3f}")
        # steady response: mean (w - 0.9 a) over steps where command constant 3 steps
    R = R_of(q)
    acc = (vel[1:] - vel[:-1]) * 30.0
    zb = R[:-1, :, 2]
    T = a[:, 3].clip(0, 1) if str(z["command_mode"]) == "ctbr" else None
    print("noise est: pos diff-of-diff std", np.std(np.diff(pos, 2, axis=0), 0) / np.sqrt(6),
          "vel ddiff", np.std(np.diff(vel, 2, axis=0), 0) / np.sqrt(6))
    if T is not None:
        f_ds = G * (T / 0.3) ** 2
        # specific thrust estimate from vertical balance f = (acc + g) . zb  ; ratio to datasheet with delays 0, 1
        need = ((acc + np.array([0, 0, G])) * zb).sum(-1)
        for d in (0, 1, 2):
            fd = f_ds[: N - d] if d == 0 else np.r_[f_ds[:1].repeat(d), f_ds[: N - d]]
            m = slice(10, N)
            k = (need[m] * fd[m]).sum() / (fd[m] ** 2).sum()
            print(f"thrust gain (delay {d}): {k:.3f}  rms {np.std(need[m] - k * fd[m]):.3f}")
        res = acc + np.array([0, 0, G]) - zb * (f_ds * 1.0)[:, None]
        # perpendicular-to-thrust residual (independent of thrust gain): horizontal-ish
        print("t     x      y     z    vx    vy    vz   T   | acc_res(x,y,z) [datasheet thrust]   w")
        for i in range(0, N, 10):
            print(f"{i / 30:4.1f} {pos[i, 0]:6.2f} {pos[i, 1]:5.2f} {pos[i, 2]:5.2f} {vel[i, 0]:5.2f} {vel[i, 1]:5.2f} {vel[i, 2]:5.2f} {a[i, 3]:.3f} | "
                  f"{res[max(0, i - 2):i + 3].mean(0).round(2)}  {w[i].round(2)} a={a[i, :3].round(2)}")
    else:
        for i in range(0, N, 15):
            print(f"{i / 30:4.1f} {pos[i].round(2)} {vel[i].round(2)} sp={a[i].round(2)} w={w[i].round(2)}")
    print("end pos", pos[-1], "vel", vel[-1])
