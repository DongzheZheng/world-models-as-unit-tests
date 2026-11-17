"""Compare world.py with the records of real trials.

HAND-WRITTEN STAND-IN for an agent's own fidelity script (see world.py).  Run it in a workspace:

    python fidelity.py trials/trial_3.npz

For every record: (1) open loop: start the model, without its random term, from the recorded observation at steps
0, 20 and 40, replay the recorded actions, and report the distance between predicted and recorded position after
5, 20 and 40 steps; (2) replay the whole trial from the reset observation and compare where the puck ends.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from world import World


def observation(d, k: int) -> dict:
    return {"pos": d["pos"][k], "vel": d["vel"][k], "target": d["target"][k], "t": int(d["t"][k])}


def replay(d, start: int, steps: int) -> np.ndarray:
    world = World(params={"noise": 0.0})
    obs = world.reset_from(observation(d, start))
    for k in range(start, start + steps):
        obs, _, _ = world.step(d["actions"][k])
    return obs["pos"]


def fidelity(name: str) -> dict:
    with np.load(name, allow_pickle=False) as data:
        d = {k: data[k] for k in data.files}
    n = int(d["n_steps"])
    horizon = {}
    for steps in (5, 20, 40):
        errors = [float(np.linalg.norm(replay(d, s, steps) - d["pos"][s + steps])) for s in (0, 20, 40) if s + steps <= n]
        horizon[f"position_error_after_{steps}_steps_m"] = round(float(np.mean(errors)), 4)
    end = replay(d, 0, n)
    return {"record": name, "seed": int(d["seed"]), **horizon,
            "final_distance_to_target_real_m": round(float(np.linalg.norm(d["pos"][n] - d["target"][n])), 4),
            "final_distance_to_target_model_m": round(float(np.linalg.norm(end - d["target"][n])), 4)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(json.dumps([fidelity(name) for name in sys.argv[1:]], indent=1))
