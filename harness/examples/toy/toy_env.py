"""A toy environment for trying the harness: a puck on a table, pushed by a clipped force.

It follows the environment contract of harness/contract.py.  The manual below is all an agent is told.  The numbers
under "hidden physics" are what it has to find out from its real trials: how strongly the puck answers a force, how
it slows down, a constant pull across the table, and a disturbance.  This file is never copied into a workspace.
"""
from __future__ import annotations

import numpy as np

MANUAL = """\
PUCK TABLE
==========
A puck lies on a flat table.  You push it with a force.  The task: have the puck at rest inside the target circle
when the episode ends.

INTERFACE
---------
    obs = reset(seed)                   the seed fixes where the puck and the target are
    obs, done, info = step(action)
The same seed followed by the same actions gives exactly the same episode.

OBSERVATION (dict)
------------------
    obs["pos"]      (2,) float   position of the puck (x, y), metres
    obs["vel"]      (2,) float   velocity of the puck, metres per second
    obs["target"]   (2,) float   centre of the target circle, metres (constant within an episode)
    obs["t"]        int          steps taken so far (0 after reset)

ACTION
------
(2,) float: the force (fx, fy) in newtons, held for one step.  Each component is clipped to [-1, 1].
Another shape or a non-finite value raises ValueError.

TIME
----
One step is 0.1 s.  Every episode has exactly 80 steps (8 s); `done` is True on the 80th.

SUCCESS
-------
info["success"] on the last step: at that moment the puck is inside the target circle (distance to its centre at
most 0.07 m) and at rest (speed at most 0.08 m/s).  info["distance"] and info["speed"] are given on every step.

START
-----
The puck starts at rest, x and y uniform in [-1, 1].  The target centre has x and y uniform in [-0.8, 0.8].

NOT DOCUMENTED
--------------
How the puck responds to a force.  Field note from the operators: "It never quite stays where you leave it."
"""

# ---- hidden physics (not in the manual) --------------------------------------------------------------------------
_GAIN = 2.0                 # acceleration per unit force, m/s^2 per N
_FRICTION = 0.8             # viscous friction, 1/s
_DRIFT = (0.35, -0.20)      # constant acceleration, m/s^2 (the table is not level)
_NOISE = 0.30               # standard deviation of a random acceleration, redrawn every step, m/s^2

DT = 0.1
MAX_STEPS = 80
FORCE_LIMIT = 1.0
TARGET_RADIUS = 0.07
REST_SPEED = 0.08


class PuckTable:
    manual = MANUAL
    max_steps = MAX_STEPS

    def __init__(self):
        self._pos = self._vel = self._target = self._rng = None
        self._t = 0
        self._running = False

    def task_card(self) -> dict:
        return {
            "dt_s": DT, "max_steps": MAX_STEPS, "force_limit_n": FORCE_LIMIT,
            "target_radius_m": TARGET_RADIUS, "rest_speed_m_per_s": REST_SPEED,
            "start": {"puck_xy_uniform": [-1.0, 1.0], "target_xy_uniform": [-0.8, 0.8], "puck_velocity": [0.0, 0.0]},
            "observation": {"pos": "(2,) m", "vel": "(2,) m/s", "target": "(2,) m", "t": "int, steps taken"},
            "action": "(2,) force in N, each component clipped to [-1, 1]",
            "success": "on the last step: distance <= target_radius_m and speed <= rest_speed_m_per_s",
        }

    def _obs(self) -> dict:
        return {"pos": self._pos.copy(), "vel": self._vel.copy(), "target": self._target.copy(), "t": self._t}

    def reset(self, seed: int) -> dict:
        start = np.random.default_rng([int(seed), 0])
        self._rng = np.random.default_rng([int(seed), 1])
        self._pos = start.uniform(-1.0, 1.0, 2)
        self._target = start.uniform(-0.8, 0.8, 2)
        self._vel = np.zeros(2)
        self._t = 0
        self._running = True
        return self._obs()

    def step(self, action):
        if not self._running:
            raise RuntimeError("no episode is running; call reset(seed)")
        force = np.asarray(action, dtype=float)
        if force.shape != (2,) or not np.all(np.isfinite(force)):
            raise ValueError("the action must be a finite array of shape (2,)")
        force = np.clip(force, -FORCE_LIMIT, FORCE_LIMIT)
        acc = _GAIN * force - _FRICTION * self._vel + np.array(_DRIFT) + _NOISE * self._rng.standard_normal(2)
        self._vel = self._vel + acc * DT
        self._pos = self._pos + self._vel * DT
        self._t += 1
        distance = float(np.linalg.norm(self._pos - self._target))
        speed = float(np.linalg.norm(self._vel))
        done = self._t >= MAX_STEPS
        info = {"distance": distance, "speed": speed}
        if done:
            self._running = False
            info["success"] = bool(distance <= TARGET_RADIUS and speed <= REST_SPEED)
        return self._obs(), done, info

    def close(self) -> None:
        self._running = False


def make_env() -> PuckTable:
    return PuckTable()
