"""policy_v3 -- repairs after life 2 (ctbr mode).  Self-contained.

Life 2 (v2) succeeded in 10.1 s.  Trace showed:
  - the south side aisle (x ~ 37.5..42) also carries a jet, toward +y, ~ +7.5 m/s^2; v2 had no prior for it, the
    observer caught it with lag: 0.45 m drift, then a -1.6 m/s overshoot at the jet exit           -> S6 south prior
  - north jet profile refined (onset 21.2, peak -12 at 22.3..24.6, tail to 26.3)                  -> S4 updated
  - rate response is a pure 2-step delay: w[i+2] = g*cmd[i] + b, rms 0.06 rad/s; refined g, b      -> S1 updated
  - hover throttle 0.329 (life 1) / 0.339 (life 2) -> C = 88 (hover 0.334), observer absorbs rest  -> S2 updated
  - jet entries need ~0.3 s to tilt 50 deg -> prior evaluated 0.2 s ahead                          -> S5 lead
Route geometry always comes from obs["relays"] (start -> relay1 -> relay2 -> relay3).
"""
from __future__ import annotations

import numpy as np

G = 9.81
RATE_GAIN = np.array([0.89, 0.96, 0.99])
RATE_BIAS = np.array([-0.495, 0.540, -0.275])   # actual body rate at zero command [rad/s]
C_THRUST = 88.0                                 # specific force along body z = C * T^2

# crosswind jets of the side aisles: (x knots, a_y knots [m/s^2]) -- identified from lives 1-2
JETS = [
    (np.array([21.2, 22.3, 24.6, 25.3, 26.3]), np.array([0.0, -12.0, -12.0, -4.0, 0.0])),   # north aisle x 22..25.5
    (np.array([37.3, 38.3, 40.9, 41.6, 42.4]), np.array([0.0, 7.5, 7.5, 3.5, 0.0])),        # south aisle x 38..41.5
]
# canyon clear half-width from site_map: walls |y| = 2 for x in [10, 50], pinch |y| = 1.6 for x in [31, 34]
PINCH = (31.0, 34.0, 1.6)
CANYON = (10.0, 50.0, 2.0)


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
    if PINCH[0] <= x <= PINCH[1]:
        return PINCH[2]
    if CANYON[0] <= x <= CANYON[1]:
        return CANYON[2]
    return 99.0


def jet_prior(x, y=0.0):
    a = np.zeros(3)
    if abs(y) < 2.5:        # inside the canyon line of the aisles
        for xs, ays in JETS:
            a[1] += np.interp(x, xs, ays, left=0.0, right=0.0)
    return a


class Policy:
    def __init__(self, cruise=5.5, kp=4.5, kd=3.6, ki=0.8, kv=2.5, k_att=7.0, k_yaw=3.0, ki_att=1.5,
                 a_max=16.0, i_max=3.0, wall_margin=0.7, k_wall=8.0, ramp_s=1.5, dob_alpha=0.3, dob_max=15.0,
                 lead_s=0.2, use_prior=True):
        self.__dict__.update(dict(cruise=cruise, kp=kp, kd=kd, ki=ki, kv=kv, k_att=k_att, k_yaw=k_yaw, ki_att=ki_att,
                                  a_max=a_max, i_max=i_max, wall_margin=wall_margin, k_wall=k_wall, ramp_s=ramp_s,
                                  dob_alpha=dob_alpha, dob_max=dob_max, lead_s=lead_s, use_prior=use_prior))
        self.reset()

    def reset(self):
        self.start = None
        self.integ = np.zeros(3)
        self.integ_att = np.zeros(3)
        self.r_hat = np.zeros(3)
        self.hist = []          # (v, b3, T, x, y) per step
        self.dbg = {}

    def prior(self, x, y):
        return jet_prior(x, y) if self.use_prior else np.zeros(3)

    def _segment(self, obs):
        k = int(obs["relay_index"][0])
        relays = np.asarray(obs["relays"], float)
        goal = relays[min(k, len(relays) - 1)]
        prev = self.start if k == 0 else relays[k - 1]
        return prev, goal

    def _dob(self, p, v):
        """residual disturbance beyond the prior (thrust command acts with a 2-step delay)."""
        if len(self.hist) >= 3:
            v_prev, b3_prev, _, x_prev, y_prev = self.hist[-1]
            T_del = self.hist[-2][2]
            a_meas = (v - v_prev) * 30.0
            a_model = C_THRUST * T_del ** 2 * b3_prev - np.array([0, 0, G])
            res = a_meas - a_model - self.prior(0.5 * (x_prev + p[0]), 0.5 * (y_prev + p[1]))
            self.r_hat = (1 - self.dob_alpha) * self.r_hat + self.dob_alpha * res
            self.r_hat = np.clip(self.r_hat, -self.dob_max, self.dob_max)
        return self.r_hat

    def act(self, obs):
        dt = 1.0 / 30.0
        p = np.asarray(obs["pos"], float)
        v = np.asarray(obs["lin_vel"], float)
        t = float(obs["t"][0])
        R = quat_to_R(np.asarray(obs["quat"], float))
        if self.start is None:
            self.start = p.copy()
        r_hat = self._dob(p, v)
        prev, goal = self._segment(obs)
        d = goal - prev
        d /= max(np.linalg.norm(d), 1e-6)
        rel = p - prev
        e_perp = rel - np.dot(rel, d) * d
        v_along = np.dot(v, d)
        v_perp = v - v_along * d
        vref = self.cruise * min(1.0, (t + 0.3) / self.ramp_s)
        a = self.kv * (vref - v_along) * d - self.kp * e_perp - self.kd * v_perp
        self.integ += dt * (-e_perp + 0.3 * (vref - v_along) * d)
        self.integ = np.clip(self.integ, -self.i_max / self.ki, self.i_max / self.ki)
        a = a + self.ki * self.integ
        hw = canyon_half_width(p[0] + 0.3 * v[0])
        lim = hw - self.wall_margin
        if abs(p[1]) > lim:
            a[1] -= np.sign(p[1]) * self.k_wall * (abs(p[1]) - lim)
        dist = self.prior(p[0] + v[0] * self.lead_s, p[1] + v[1] * self.lead_s) + r_hat
        a = a - dist
        n = np.linalg.norm(a[:2])
        if n > self.a_max:
            a[:2] *= self.a_max / n
        a[2] = np.clip(a[2], -6.0, 12.0)
        f = a + np.array([0.0, 0.0, G])
        b3d = f / np.linalg.norm(f)
        b2d = np.cross(b3d, [1.0, 0.0, 0.0]); b2d /= np.linalg.norm(b2d)
        b1d = np.cross(b2d, b3d)
        Rd = np.column_stack([b1d, b2d, b3d])
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        self.integ_att = np.clip(self.integ_att + dt * eR, -0.3, 0.3)
        w_des = -np.array([self.k_att, self.k_att, self.k_yaw]) * eR - self.ki_att * self.integ_att
        cmd = np.clip((w_des - RATE_BIAS) / RATE_GAIN, -3.1, 3.1)
        u = float(np.dot(f, R[:, 2]))
        T = float(np.clip(np.sqrt(max(u, 0.2 * G) / C_THRUST), 0.05, 0.8))
        self.hist.append((v.copy(), R[:, 2].copy(), T, p[0], p[1]))
        self.dbg = dict(a=a, eR=eR, e_perp=e_perp, r_hat=r_hat.copy(), dist=dist)
        return np.array([cmd[0], cmd[1], cmd[2], T])


def make_policy(**kw):
    return Policy(**kw)
