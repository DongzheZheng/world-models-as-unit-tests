"""Controller for the puck table.

HAND-WRITTEN STAND-IN for the file an agent would write (see world.py).  A position controller with velocity damping
plus a constant force that cancels the drift of the world model.  The two gains were set by hand for a well-damped
response with the fitted gain and friction, and the file was tested in world.py (evaluate_in_model.py); the
cancelling force is computed from the fitted parameters of world.py.
"""
from __future__ import annotations

import numpy as np

from world import PARAMS

KP = 6.0        # N per metre of distance to the target
KD = 3.0        # N per m/s of puck velocity


class Policy:
    def __init__(self):
        self.hold = -np.array([PARAMS["drift_x"], PARAMS["drift_y"]]) / PARAMS["gain"]     # force that cancels M3

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        force = KP * (obs["target"] - obs["pos"]) - KD * obs["vel"] + self.hold
        return np.clip(force, -1.0, 1.0)


def make_policy() -> Policy:
    return Policy()
