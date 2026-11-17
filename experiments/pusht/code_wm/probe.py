"""Scripted probing policy for the identification lives (bridge side; reads only observations).

The episode is a sequence of phases; each phase is turned into a queue of pusher targets when it starts, using the
observation at that moment:
  ("track", n)                 pusher-tracking probe: n moves of increasing size between free points far from the block
  ("push", face, speed, dist)  go around the block (radius R) to a pre-contact point, then push along the inward normal
                               of a body-frame face point at `speed` u/step for `dist` u (target ramps open-loop)
  ("release", n)               back off 40 u along -push direction, then hold for n steps (watch the block coast)
  ("hold", n)                  hold the pusher still
"""
from __future__ import annotations

import numpy as np

CENTROID = np.array([0.0, 40.714])          # geometric centroid of the drawn T (body frame)
# body-frame contact point and inward push direction for named probe pushes
FACES = {
    "bar_bottom_mid": ((0.0, 0.0), (0.0, 1.0)),
    "bar_bottom_right": ((40.0, 0.0), (0.0, 1.0)),
    "bar_bottom_left": ((-40.0, 0.0), (0.0, 1.0)),
    "bar_right": ((60.0, 15.0), (-1.0, 0.0)),
    "bar_left": ((-60.0, 15.0), (1.0, 0.0)),
    "stem_top": ((0.0, 120.0), (0.0, -1.0)),
    "stem_right": ((15.0, 80.0), (-1.0, 0.0)),
    "stem_left": ((-15.0, 80.0), (1.0, 0.0)),
    "bar_top_right": ((40.0, 30.0), (0.0, -1.0)),
    "bar_top_left": ((-40.0, 30.0), (0.0, -1.0)),
}


# L block (datasheet): leg u[0,40] v[0,110], foot u[40,100] v[0,40]
CENTROID_L = np.array([37.647, 42.647])
FACES_L = {
    "foot_bottom": ((70.0, 0.0), (0.0, 1.0)),
    "leg_bottom": ((20.0, 0.0), (0.0, 1.0)),
    "foot_right": ((100.0, 20.0), (-1.0, 0.0)),
    "foot_top": ((70.0, 40.0), (0.0, -1.0)),
    "foot_top_tip": ((92.0, 40.0), (0.0, -1.0)),
    "leg_right": ((40.0, 85.0), (-1.0, 0.0)),
    "leg_top": ((20.0, 110.0), (0.0, -1.0)),
    "leg_left_high": ((0.0, 95.0), (1.0, 0.0)),
    "leg_left_low": ((0.0, 20.0), (1.0, 0.0)),
    "leg_left_mid": ((0.0, 45.0), (1.0, 0.0)),
}


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def to_world(pose, p):
    return pose[:2] + rot(pose[2]) @ np.asarray(p, float)


def _seg(a, b, max_step):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / max_step)))
    return [a + (b - a) * (i + 1) / n for i in range(n)]


def _clip(p, lo=22.0, hi=490.0):
    return np.clip(p, lo, hi)


def go_around(p, pose, goal, R=150.0, max_step=25.0):
    """Waypoints from p to goal that stay about R away from the block's centroid (out radially, around, in)."""
    c = to_world(pose, CENTROID)
    a0 = np.arctan2(*(p - c)[::-1])
    a1 = np.arctan2(*(goal - c)[::-1])
    r0 = max(np.linalg.norm(p - c), R)
    pts = []
    cur = np.array(p, float)
    out = _clip(c + r0 * np.array([np.cos(a0), np.sin(a0)]))
    pts += _seg(cur, out, max_step); cur = out
    da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    n = max(1, int(np.ceil(abs(da) / np.radians(20))))
    for i in range(1, n + 1):
        a = a0 + da * i / n
        q = _clip(c + R * np.array([np.cos(a), np.sin(a)]))
        pts += _seg(cur, q, max_step); cur = q
    pts += _seg(cur, goal, max_step)
    return pts


class ProbePolicy:
    def __init__(self, phases, object="T"):
        global CENTROID, FACES
        self.phases = [tuple(p) for p in phases]
        if object == "L":
            CENTROID, FACES = CENTROID_L, FACES_L

    def reset(self):
        self.k = 0
        self.queue = []
        self.last = None
        self.push_dir = np.array([1.0, 0.0])
        self.log = []

    def _expand(self, ph, obs):
        p, pose = np.array(obs["agent_pos"]), np.array(obs["block_pose"])
        kind = ph[0]
        if kind == "track":
            c = to_world(pose, CENTROID)
            q = []
            cur = p
            a = np.arctan2(*(p - c)[::-1])
            sizes = [20, 40, 80, 160, 30, 120]
            for i in range(ph[1]):
                # jump along the circle by an angle giving roughly the requested chord, then hold 6 steps
                chord = sizes[i % len(sizes)]
                R = 175.0
                a = a + 2 * np.arcsin(min(chord / (2 * R), 0.5)) * (1 if i % 2 == 0 else -1) * 1.0
                tgt = _clip(c + R * np.array([np.cos(a), np.sin(a)]))
                if i == 0:
                    q += _seg(cur, tgt, 25.0)      # first get onto the circle, gently
                    q += [tgt] * 6
                else:
                    q += [tgt] * 8                 # step command: the whole jump in one target change
                cur = tgt
            return q
        if kind == "push":
            _, face, speed, dist = ph
            if isinstance(face, (list, tuple)):   # pick the candidate whose push points most toward the table centre
                cw = to_world(pose, CENTROID)
                tc = (np.array([256.0, 256.0]) - cw); tc /= np.linalg.norm(tc) + 1e-9
                face = max(face, key=lambda f: float((rot(pose[2]) @ np.array(FACES[f][1])) @ tc))
            self.log.append(face)
            cp, d = FACES[face]
            cpw = to_world(pose, cp)
            dw = rot(pose[2]) @ np.array(d)
            pre = cpw - dw * (15.0 + 30.0)
            self.push_dir = dw
            q = go_around(p, pose, pre)
            q += [pre] * 3
            q += [pre + dw * speed * (i + 1) for i in range(int(dist / speed) + 3)]
            return [_clip(x, 1.0, 511.0) for x in q]
        if kind == "release":
            back = p - self.push_dir * 40.0
            return _seg(p, back, 20.0) + [back] * ph[1]
        if kind == "hold":
            return [p] * ph[1]
        raise ValueError(kind)

    def act(self, obs):
        while not self.queue and self.k < len(self.phases):
            self.queue = list(self._expand(self.phases[self.k], obs))
            self.k += 1
        if self.queue:
            self.last = np.asarray(self.queue.pop(0), float)
        elif self.last is None:
            self.last = np.array(obs["agent_pos"], float)
        return self.last


def probe_policy(phases, object="T"):
    return ProbePolicy(phases, object)
