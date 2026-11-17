"""Fit the parameters of world.py to the records of real trials.

HAND-WRITTEN STAND-IN for an agent's own fitting script (see world.py).  Run it in a workspace:

    python fit_world.py trials/trial_1.npz trials/trial_2.npz

Model: (vel[k+1] - vel[k]) / dt = gain * force[k] - friction * vel[k] + drift + noise, the same gain and friction
on both axes, one drift per axis.  Prints the fitted values as JSON, with the standard error of each and a check of
the integration order (position advanced with the new velocity).
"""
from __future__ import annotations

import json
import sys

import numpy as np

DT = 0.1


def fit(files: list[str]) -> dict:
    rows, rhs, order = [], [], []
    for name in files:
        with np.load(name, allow_pickle=False) as d:
            pos, vel, force = d["pos"], d["vel"], np.clip(d["actions"], -1.0, 1.0)
        acc = (vel[1:] - vel[:-1]) / DT
        order.append(np.abs(pos[1:] - pos[:-1] - vel[1:] * DT).max())
        for axis in (0, 1):
            n = len(force)
            onehot = np.zeros((n, 2))
            onehot[:, axis] = 1.0
            rows.append(np.column_stack([force[:, axis], -vel[:-1, axis], onehot]))
            rhs.append(acc[:, axis])
    A, b = np.vstack(rows), np.concatenate(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ sol
    noise = float(resid.std(ddof=A.shape[1]))
    stderr = noise * np.sqrt(np.diag(np.linalg.inv(A.T @ A)))
    names = ("gain", "friction", "drift_x", "drift_y")
    return {"params": {**{k: round(float(v), 3) for k, v in zip(names, sol)}, "noise": round(noise, 3)},
            "standard_error": {k: round(float(v), 3) for k, v in zip(names, stderr)},
            "steps": int(len(b) // 2), "max_abs_position_update_error": float(max(order))}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(json.dumps(fit(sys.argv[1:]), indent=1))
