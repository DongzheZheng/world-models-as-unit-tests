"""Internal: built-in reference policies (lab-side)."""
from __future__ import annotations

import numpy as np

_LOW = np.array([-np.pi, -np.pi, -np.pi, 0.0])
_HIGH = np.array([np.pi, np.pi, np.pi, 0.8])


class _Hover:
    def __init__(self, thrust: float = 0.30):
        self.a = np.array([0.0, 0.0, 0.0, thrust])

    def reset(self, seed=None):
        pass

    def act(self, obs):
        return self.a.copy()


def hover_policy(thrust: float = 0.30):
    """Crude constant command: zero body rates and a fixed thrust."""
    return _Hover(thrust)


class _Random:
    def __init__(self, seed=0, low=None, high=None):
        self.base = int(seed)
        self.low = _LOW if low is None else np.asarray(low, float)
        self.high = _HIGH if high is None else np.asarray(high, float)
        self.rng = np.random.default_rng(self.base)

    def reset(self, seed=None):
        self.rng = np.random.default_rng([self.base, 0 if seed is None else int(seed)])

    def act(self, obs):
        return self.rng.uniform(self.low, self.high)


def random_policy(seed: int = 0, low=None, high=None):
    """Uniform random command over the documented action bounds."""
    return _Random(seed, low, high)


class _Replay:
    def __init__(self, path: str):
        self.seq = np.load(path).astype(np.float64).reshape(-1, 4)
        self.i = 0

    def reset(self, seed=None):
        self.i = 0

    def act(self, obs):
        a = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return a.copy()


def replay_policy(path: str):
    """Open-loop replay of a (T,4) action sequence stored in a .npy file (last row is held afterwards)."""
    return _Replay(path)


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class _PidWaypoints:
    """Reference for env instances built with flight_mode=7 (PyFlyt's built-in cascaded position controller).
    Setpoint = [x, y, yaw, z] of the next waypoint (world frame)."""

    def reset(self, seed=None):
        pass

    def act(self, obs):
        if len(obs["target_deltas"]) == 0:
            return np.array([*obs["lin_pos"][:2], 0.0, obs["lin_pos"][2]])
        tgt = obs["lin_pos"] + _rot(obs["quat"]) @ obs["target_deltas"][0]
        return np.array([tgt[0], tgt[1], 0.0, tgt[2]])


def pid_waypoints_policy():
    return _PidWaypoints()


class _Straight:
    """Reference for port env built with flight_mode=7: fly the straight line to the goal."""

    def reset(self, seed=None):
        pass

    def act(self, obs):
        g = obs["goal"]
        return np.array([g[0], g[1], 0.0, g[2]])


def pid_straight_policy():
    return _Straight()


# ------------------------------------------------------------------------------------------------ site pilots
class _SiteNominal:
    """Reference autopilot for the site: uses the public command_mode='position' (PyFlyt flight mode 7), flies the
    corridor centreline with a carrot `lead` metres ahead (speed saturates at the autopilot's 2 m/s limit)."""

    def __init__(self, lead=2.5, z=None, hold_centre=True):
        self.lead, self.z, self.hold_centre = lead, z, hold_centre

    def reset(self, seed=None):
        pass

    def aim(self, obs):
        p, tgt = obs["pos"], obs["goal"]
        if abs(p[1]) > 2.0 and abs(tgt[1]) <= 2.0:                     # inside a side aisle, target in the corridor: come back out first
            carrot = np.array([p[0], 0.0, tgt[2]])
        elif abs(tgt[1]) > 2.0 and abs(p[0] - tgt[0]) > 0.8 and abs(p[1]) <= 2.0:           # target is inside a side aisle: line up on the centreline first
            carrot = np.array([p[0] + np.clip(tgt[0] - p[0], -self.lead, self.lead), 0.0, tgt[2]])
        else:
            dx = tgt[0] - p[0]
            if self.hold_centre:
                yc = tgt[1] if abs(dx) < 4.0 else 0.0       # hold the centreline until close to the relay
                carrot = np.array([p[0] + np.clip(dx, -self.lead, self.lead), yc, tgt[2]])
            else:                                            # carrot a fixed distance along the straight line to the relay
                d = tgt[:2] - p[:2]
                n = np.linalg.norm(d)
                c = p[:2] + d * min(1.0, self.lead / max(n, 1e-6))
                carrot = np.array([c[0], c[1], tgt[2]])
        return carrot

    def act(self, obs):
        c = self.aim(obs)
        return np.array([c[0], c[1], 0.0, c[2] if self.z is None else self.z])


def site_nominal_policy(**kw):
    return _SiteNominal(**kw)
