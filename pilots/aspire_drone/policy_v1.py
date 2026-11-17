"""policy_v1 -- first program for the site task (ctbr mode).

Geometric controller written from the datasheet, with integral action to absorb unknown biases:
  guidance: follow the polyline start -> relay_k taken from obs["relays"]; along-track speed reference, cross-track PD+I
  attitude: thrust-vector -> desired rotation (yaw 0) -> body-rate command proportional to SO(3) error
  thrust:   T = T_hover * sqrt(u / g)  (datasheet: thrust ~ T^2, hover 0.30)
  safety:   soft map-based push away from the canyon walls (site_map geometry, |y| clear half-width)
"""
from __future__ import annotations

import numpy as np

G = 9.81


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def canyon_half_width(x):
    """Clear half-width of the canyon at x from site_map (pinch x in [31, 34] at |y| < 1.6)."""
    if 31.0 <= x <= 34.0:
        return 1.6
    if 10.0 <= x <= 50.0:
        return 2.0
    return 99.0


class Policy:
    def __init__(self, cruise=5.0, t_hover=0.30, kp=4.5, kd=3.6, ki=1.2, kv=2.5, k_att=7.0, k_yaw=3.0,
                 a_max=8.0, i_max=4.0, wall_margin=0.7, k_wall=8.0, ramp_s=1.5):
        self.cruise, self.t_hover = cruise, t_hover
        self.kp, self.kd, self.ki, self.kv = kp, kd, ki, kv
        self.k_att, self.k_yaw, self.a_max, self.i_max = k_att, k_yaw, a_max, i_max
        self.wall_margin, self.k_wall, self.ramp_s = wall_margin, k_wall, ramp_s
        self.reset()

    def reset(self):
        self.start = None
        self.integ = np.zeros(3)
        self.dbg = {}

    def _segment(self, obs):
        k = int(obs["relay_index"][0])
        relays = np.asarray(obs["relays"], float)
        goal = relays[min(k, len(relays) - 1)]
        prev = self.start if k == 0 else relays[k - 1]
        return prev, goal

    def act(self, obs):
        dt = 1.0 / 30.0
        p = np.asarray(obs["pos"], float)
        v = np.asarray(obs["lin_vel"], float)
        t = float(obs["t"][0])
        R = quat_to_R(np.asarray(obs["quat"], float))
        if self.start is None:
            self.start = p.copy()
        prev, goal = self._segment(obs)
        d = goal - prev
        d /= max(np.linalg.norm(d), 1e-6)
        rel = p - prev
        e_perp = rel - np.dot(rel, d) * d
        v_along = np.dot(v, d)
        v_perp = v - v_along * d
        vref = self.cruise * min(1.0, (t + 0.3) / self.ramp_s)
        a = self.kv * (vref - v_along) * d - self.kp * e_perp - self.kd * v_perp
        # integral: cross-track position error + along-track speed error (absorbs wind, drag, thrust-scale errors)
        self.integ += dt * (-e_perp + 0.3 * (vref - v_along) * d)
        self.integ = np.clip(self.integ, -self.i_max / self.ki, self.i_max / self.ki)
        a = a + self.ki * self.integ
        # map-based wall push
        hw = canyon_half_width(p[0] + 0.3 * v[0])
        lim = hw - self.wall_margin
        if abs(p[1]) > lim:
            a[1] -= np.sign(p[1]) * self.k_wall * (abs(p[1]) - lim)
        ah = a[:2]
        n = np.linalg.norm(ah)
        if n > self.a_max:
            a[:2] = ah * self.a_max / n
        a[2] = np.clip(a[2], -6.0, 10.0)
        f = a + np.array([0.0, 0.0, G])
        b3d = f / np.linalg.norm(f)
        b1c = np.array([1.0, 0.0, 0.0])
        b2d = np.cross(b3d, b1c); b2d /= np.linalg.norm(b2d)
        b1d = np.cross(b2d, b3d)
        Rd = np.column_stack([b1d, b2d, b3d])
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        w = -np.array([self.k_att * eR[0], self.k_att * eR[1], self.k_yaw * eR[2]])
        w = np.clip(w, -3.0, 3.0)
        u = float(np.dot(f, R[:, 2]))
        T = self.t_hover * np.sqrt(max(u, 0.2 * G) / G)
        T = float(np.clip(T, 0.05, 0.8))
        self.dbg = dict(a=a, eR=eR, e_perp=e_perp, vref=vref, integ=self.integ.copy())
        return np.array([w[0], w[1], w[2], T])


def make_policy(**kw):
    return Policy(**kw)
