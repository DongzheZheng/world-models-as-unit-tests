"""A deliberately poor controller for the puck table, for the `screen` example.

HAND-WRITTEN.  Weaker gains than policy.py and nothing against the drift: the puck settles beside the target centre,
near the edge of the target circle.
"""
from __future__ import annotations

import numpy as np

KP = 3.0
KD = 1.5


class Policy:
    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        return np.clip(KP * (obs["target"] - obs["pos"]) - KD * obs["vel"], -1.0, 1.0)


def make_policy() -> Policy:
    return Policy()
