"""Internal: thin adapter around the registered PyFlyt/QuadX-Waypoints-v4 env (no behavioural changes)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

import PyFlyt.gym_envs  # noqa: F401  (registers the env ids)

from ._port_env import quiet

NUM_TARGETS = 4


class WaypointsEnv:
    def __init__(self, flight_mode: int = 0, max_steps: int | None = None):
        self.flight_mode = int(flight_mode)
        kw = {} if self.flight_mode == 0 else dict(flight_mode=self.flight_mode)
        self.env = gym.make("PyFlyt/QuadX-Waypoints-v4", disable_env_checker=True, **kw)
        sp = self.env.unwrapped.action_space
        self.action_low, self.action_high = np.array(sp.low), np.array(sp.high)
        if self.flight_mode != 0:
            self.action_low = np.full(4, -np.inf)
            self.action_high = np.full(4, np.inf)
        self.max_steps = max_steps
        self.n = 0

    @staticmethod
    def _convert(o) -> dict:
        a = np.asarray(o["attitude"], dtype=np.float64)
        return dict(ang_vel=a[0:3].copy(), quat=a[3:7].copy(), lin_vel=a[7:10].copy(), lin_pos=a[10:13].copy(),
                    prev_action=a[13:17].copy(), aux=a[17:21].copy(),
                    target_deltas=np.asarray(o["target_deltas"], dtype=np.float64).reshape(-1, 3).copy())

    def reset(self, seed: int):
        with quiet():
            o, info = self.env.reset(seed=int(seed))
        self.n = 0
        return self._convert(o), {"seed": int(seed)}

    def step(self, action):
        o, r, te, tr, info = self.env.step(np.asarray(action, dtype=np.float64))
        self.n += 1
        timeout_cap = bool(self.max_steps and self.n >= self.max_steps and not (te or tr))
        tr = bool(tr) or timeout_cap
        te = bool(te)
        out = {}
        if te or tr:
            reached = int(info.get("num_targets_reached", 0))
            success = bool(info.get("env_complete", False))
            out = dict(targets_reached=reached, num_targets=NUM_TARGETS, success=success,
                       crashed=bool(info.get("collision", False)),
                       out_of_bounds=bool(info.get("out_of_bounds", False)),
                       timeout=bool(tr and not success), time_s=self.n / 30.0)
        return self._convert(o), float(r), te, tr, out

    def close(self):
        self.env.close()
