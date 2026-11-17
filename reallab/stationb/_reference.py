"""Lab-side references and runners for station B.  Internal: pilots must not read this module.

uniform    open-loop: shortest route around the posts on the required sides at clearance `clear` from the post centres,
           then to the target, at one constant speed `speed`, with a short arrival slow-down; then hold.
governor   privileged: the same tight route, speed set every step from the TRUE load (read through the lab hook
           bind_real -> env._loads): full speed `vmax` below `lo` N, linearly down to `vmin` at `hi` N, backing off
           along the route above `back_at` N.
naive_router  reallab.station._reference.naive_router (unchanged).
"""
from __future__ import annotations

import numpy as np

from reallab.station._reference import naive_router, route_waypoints  # noqa: F401


def tight_path(obs, clear):
    from reallab.station import _layout
    way = _layout.taut_route(obs["tool_pos"], obs["posts"], obs["post_sides"], obs["target"], R=clear)
    if way is None:
        way = route_waypoints(obs["tool_pos"], obs["posts"], obs["post_sides"], obs["target"], clear + 0.02)
    way = np.asarray(way, float)
    seg = np.linalg.norm(np.diff(way, axis=0), axis=1)
    return way, np.concatenate([[0.0], np.cumsum(seg)])


class _PathFollower:
    clear = 0.05
    arrive_gain = 2.0       # 1/s: speed <= gain * remaining distance (short arrival slow-down)
    arrive_min = 0.02       # m/s

    def bind(self, card):
        pass

    def reset(self):
        self.path = None

    def _init(self, obs):
        self.path, self.cum = tight_path(obs, self.clear)
        self.s = 0.0

    def _advance(self, v):
        rem = self.cum[-1] - self.s
        v = min(v, max(self.arrive_min, self.arrive_gain * rem))
        self.s = min(self.s + v / 20.0, self.cum[-1])

    def _point(self):
        return np.array([np.interp(self.s, self.cum, self.path[:, k]) for k in range(2)])


class Uniform(_PathFollower):
    def __init__(self, clear=0.05, speed=0.1):
        self.clear, self.speed = clear, speed

    def act(self, obs):
        if self.path is None:
            self._init(obs)
        self._advance(self.speed)
        return self._point()


class Governor(_PathFollower):
    def __init__(self, clear=0.05, vmax=0.3, vmin=0.02, lo=3.0, hi=8.0, back_at=9.0, back=0.05):
        self.clear, self.vmax, self.vmin, self.lo, self.hi, self.back_at, self.back = clear, vmax, vmin, lo, hi, back_at, back

    def bind_real(self, env):
        self.env = env

    def act(self, obs):
        if self.path is None:
            self._init(obs)
        load = float(self.env._loads[-1])            # privileged: the true load of the last control period
        if load > self.back_at:
            self.s = max(0.0, self.s - self.back / 20.0)
        else:
            f = np.clip((self.hi - load) / (self.hi - self.lo), 0.0, 1.0)
            self._advance(self.vmin + f * (self.vmax - self.vmin))
        return self._point()


def uniform(**kw):
    return Uniform(**kw)


def governor(**kw):
    return Governor(**kw)


# ---------------------------------------------------------------- lab runners
def _lab_env(cfg):
    from reallab.stationb import api
    return api.RealStationB(cfg["purpose"], cfg["pilot"], cfg["tag"], None, _lab=True,
                            _time_limit=cfg.get("time_limit"))


def lab_evaluate(policy_factory, policy_kwargs, seeds, purpose="smoke", tag="", n_workers=12, log_dir=None,
                 time_limit=None):
    from reallab.stationb.parallel import _run
    cfg = dict(policy_factory=policy_factory, policy_kwargs=policy_kwargs or {}, purpose=purpose, pilot="", tag=tag,
               log_dir=log_dir, lab_env="reallab.stationb._reference:_lab_env", time_limit=time_limit)
    return _run(cfg, seeds, min(n_workers, 12))


# ---------------------------------------------------------------- load-blind geometric reference
def wrap_angle(points, posts, reach=0.06):
    """Total turning (rad) of the tracked-point polyline at the points lying within `reach` of a post centre."""
    p = np.asarray(points, float)[:, :2]
    seg = np.diff(p, axis=0)
    h = np.arctan2(seg[:, 1], seg[:, 0])
    dh = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)          # turning at interior points 1..18
    near = np.min(np.linalg.norm(p[1:-1, None] - np.asarray(posts)[None], axis=2), axis=1) < reach
    return float(dh[near].sum())


class Geometric(_PathFollower):
    """Load-blind: same tight route; speed each step from observable geometry only:
        v = clip(K / ((1 + moving_length) * exp(a * wrap)), vmin, vmax)
    moving_length = length of the polyline whose points moved faster than 0.02 m/s since the last step;
    wrap = total turning of the polyline next to the posts (wrap_angle).  Speed parameters: K, vmin, vmax."""

    def __init__(self, clear=0.05, K=0.2, a=1.0, vmin=0.04, vmax=0.3):
        self.clear, self.K, self.a, self.vmin, self.vmax = clear, K, a, vmin, vmax

    def reset(self):
        super().reset()
        self.prev = None

    def act(self, obs):
        if self.path is None:
            self._init(obs)
        pts = np.asarray(obs["points"], float)[:, :2]
        if self.prev is None:
            mov = 0.0
        else:
            sp = np.linalg.norm(pts - self.prev, axis=1) * 20.0
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            mov = float(seg[(sp[1:] > 0.02) | (sp[:-1] > 0.02)].sum())
        self.prev = pts
        th = wrap_angle(pts, obs["posts"])
        v = float(np.clip(self.K / ((1.0 + mov) * np.exp(self.a * th)), self.vmin, self.vmax))
        self._advance(v)
        return self._point()


def geometric(**kw):
    return Geometric(**kw)
