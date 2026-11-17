"""World model of the puck table.

HAND-WRITTEN STAND-IN for the file an agent would write.  It shows the format of the deliverable and lets the
examples and tests of the harness run without an agent.  It uses only what a workspace contains: the manual and the
records of the real trials of ../replay.json.  The numbers in PARAMS are the output of fit_world.py on the records
of trials 1 and 2.

Mechanisms (each a named line of step()):
  M1 force      the puck accelerates by gain * force; the force is clipped to [-1, 1] N per component (manual)
  M2 friction   it is slowed in proportion to its velocity
  M3 drift      a constant acceleration: the table is not level.  This is the operators' field note
  M4 puffs      a random acceleration, redrawn every step
Integration as seen in the records: velocity first, then position with the new velocity; dt = 0.1 s (manual).
"""
from __future__ import annotations

import zlib

import numpy as np

DT = 0.1                    # manual
MAX_STEPS = 80              # manual
FORCE_LIMIT = 1.0           # manual
TARGET_RADIUS = 0.07        # manual
REST_SPEED = 0.08           # manual

# Fitted parameters.  Evidence: least squares of (vel[k+1] - vel[k]) / dt on [force, -vel, 1] over the 160 steps of
# trials 1 and 2 (scripted pushes and pauses on both axes); standard errors in brackets.  `python fit_world.py
# trials/trial_1.npz trials/trial_2.npz` prints these numbers.
PARAMS = dict(
    gain=1.970,             # M1, m/s^2 per N                                           [0.035]
    friction=0.777,         # M2, 1/s                                                   [0.029]
    drift_x=0.337,          # M3, m/s^2: with zero force the puck speeds up towards +x  [0.030]
    drift_y=-0.221,         #            and -y (trial 1, steps 0-9)                    [0.029]
    noise=0.332,            # M4, m/s^2: standard deviation of the residual of the fit
)


class World:
    """reset(seed) -> obs and step(action) -> (obs, done, info) in the environment's formats."""

    def __init__(self, params: dict | None = None):
        self.params = dict(PARAMS, **(params or {}))
        self._pos = self._vel = self._target = self._rng = None
        self._t = 0

    def _obs(self) -> dict:
        return {"pos": self._pos.copy(), "vel": self._vel.copy(), "target": self._target.copy(), "t": self._t}

    def reset(self, seed: int) -> dict:
        """A start as the manual describes it: puck at rest in [-1, 1]^2, target in [-0.8, 0.8]^2."""
        rng = np.random.default_rng(int(seed))
        self._pos = rng.uniform(-1.0, 1.0, 2)
        self._target = rng.uniform(-0.8, 0.8, 2)
        self._vel = np.zeros(2)
        self._rng, self._t = rng, 0
        return self._obs()

    def reset_from(self, obs: dict) -> dict:
        """Start from a real reset observation."""
        self._pos = np.array(obs["pos"], dtype=float)
        self._vel = np.array(obs["vel"], dtype=float)
        self._target = np.array(obs["target"], dtype=float)
        self._t = int(obs["t"])
        self._rng = np.random.default_rng(zlib.crc32(self._pos.tobytes() + self._target.tobytes()))
        return self._obs()

    def step(self, action):
        p = self.params
        force = np.asarray(action, dtype=float)
        if force.shape != (2,) or not np.all(np.isfinite(force)):
            raise ValueError("the action must be a finite array of shape (2,)")
        force = np.clip(force, -FORCE_LIMIT, FORCE_LIMIT)
        acc = p["gain"] * force                                         # M1
        acc = acc - p["friction"] * self._vel                           # M2
        acc = acc + np.array([p["drift_x"], p["drift_y"]])              # M3
        acc = acc + p["noise"] * self._rng.standard_normal(2)           # M4
        self._vel = self._vel + acc * DT
        self._pos = self._pos + self._vel * DT
        self._t += 1
        distance = float(np.linalg.norm(self._pos - self._target))
        speed = float(np.linalg.norm(self._vel))
        done = self._t >= MAX_STEPS
        info = {"distance": distance, "speed": speed}
        if done:
            info["success"] = bool(distance <= TARGET_RADIUS and speed <= REST_SPEED)
        return self._obs(), done, info
