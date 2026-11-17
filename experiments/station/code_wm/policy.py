"""code_wm controller: route the rope around the pegs on the required sides with a load-governed speed.

Lab policy protocol: reset(), act(obs) -> (2,), optional bind(task_card), set_seed(s).  numpy only.
Designed and tuned inside the code world (worlds/station/station_world.py) via experiments/station/code_wm/screen.py.

Plan (once, from the reset reading):
  * for each peg (sorted by x) two waypoints on a circle of radius `clear` around it, on the required side
    (side +1 = rope passes below the peg = peg on the left of the rope: angles -135 deg / -45 deg; side -1: +135 / +45),
  * then the target; segments that would pass closer than `clear` to a peg get a detour point on that peg's circle
    on the side the segment already passes.
Execution: a reference point advances along the path at speed v, v governed by the load reading (the rope "fights back"
superlinearly with speed: life_1), paused when the tool lags behind (snag on a peg -> avoid the catch-up lurch that
overloaded episode_906), slowed down near the end; finally hold the target until the tool rests there.
"""
from __future__ import annotations

import numpy as np

DT = 0.05


def _seg_point_dist(a, b, c):
    ab = b - a
    t = np.clip(np.dot(c - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    p = a + t * ab
    return np.linalg.norm(c - p), p, t


class RoutePolicy:
    def __init__(self, clear=0.075, v_max=0.10, v_min=0.015, load_soft=4.0, load_hard=7.0, lag_pause=0.02,
                 v_end=0.04, end_dist=0.10, entry_angle=135.0, exit_angle=45.0, x_pre=0.0):
        self.c = dict(clear=clear, v_max=v_max, v_min=v_min, load_soft=load_soft, load_hard=load_hard,
                      lag_pause=lag_pause, v_end=v_end, end_dist=end_dist, entry=entry_angle, exit=exit_angle, x_pre=x_pre)
        self.path = None

    def bind(self, task_card):
        pass

    def set_seed(self, s):
        pass

    def reset(self):
        self.path = None

    # ---------------------------------------------------------------- planning
    def plan(self, obs):
        c = self.c
        posts = np.asarray(obs["posts"], float)
        sides = np.asarray(obs["post_sides"], int)
        tool = np.asarray(obs["tool_pos"], float)
        target = np.asarray(obs["target"], float)
        order = np.argsort(posts[:, 0])
        wps = [tool.copy()]
        for j in order:
            s = sides[j]
            for ang in (c["entry"], c["exit"]):
                a = np.deg2rad(-ang if s > 0 else ang)
                wps.append(posts[j] + c["clear"] * np.array([np.cos(a), np.sin(a)]))
        wps.append(target.copy())
        # detours: repeat until no segment passes too close to a peg
        for _ in range(12):
            changed = False
            out = [wps[0]]
            for a, b in zip(wps[:-1], wps[1:]):
                worst = None
                for j in range(len(posts)):
                    d, p, t = _seg_point_dist(a, b, posts[j])
                    if d < c["clear"] - 1e-3 and 0.0 < t < 1.0 and (worst is None or d < worst[0]):
                        worst = (d, p, j)
                if worst is not None:
                    d, p, j = worst
                    n = p - posts[j]
                    nn = np.linalg.norm(n)
                    if nn < 1e-6:          # straight through the centre: go round on the required side
                        n = np.array([0.0, -1.0 if sides[j] > 0 else 1.0]); nn = 1.0
                    out.append(posts[j] + c["clear"] * 1.05 * n / nn)
                    changed = True
                out.append(b)
            wps = out
            if not changed:
                break
        wps = [np.clip(w, [0.02, -0.38], [1.18, 0.38]) for w in wps]
        P = np.array(wps)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        self.path = P
        self.cum = np.r_[0.0, np.cumsum(seg)]
        self.s = 0.0
        self.high = 0
        self.ref = tool.copy()

    def point(self, s):
        s = np.clip(s, 0.0, self.cum[-1])
        i = min(np.searchsorted(self.cum, s, side="right") - 1, len(self.path) - 2)
        L = max(self.cum[i + 1] - self.cum[i], 1e-9)
        return self.path[i] + (s - self.cum[i]) / L * (self.path[i + 1] - self.path[i])

    # ---------------------------------------------------------------- acting
    def act(self, obs):
        if self.path is None:
            self.plan(obs)
        c = self.c
        tool = np.asarray(obs["tool_pos"], float)
        load = float(obs["load"])
        remaining = self.cum[-1] - self.s
        # load governor: full speed below load_soft, linearly down to v_min at load_hard, stop above
        if load >= c["load_hard"]:
            v = 0.0
        else:
            f = np.clip((c["load_hard"] - load) / max(c["load_hard"] - c["load_soft"], 1e-6), 0.0, 1.0)
            v = c["v_min"] + f * (c["v_max"] - c["v_min"])
        if remaining < c["end_dist"]:
            v = min(v, max(c["v_end"] * remaining / c["end_dist"], 0.01))
        # snag guard: do not run away from a tool that is held back (prevents the catch-up lurch)
        if np.linalg.norm(self.ref - tool) > c["lag_pause"]:
            v = 0.0
        # taut guard: load stays high although we stopped -> the rope is stretched (around pegs / to the clamp):
        # back off along the path to release it (code world seed 12: static 9 N between two pegs)
        self.high = self.high + 1 if load >= c["load_hard"] else 0
        if self.high >= 2:
            v = -c["v_min"] * 2
        self.s = float(np.clip(self.s + v * DT, 0.0, self.cum[-1]))
        self.ref = self.point(self.s)
        return self.ref.copy()


def make_policy(**kw):
    return RoutePolicy(**kw)
