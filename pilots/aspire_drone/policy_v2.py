"""policy_v2 -- repairs after life 1 (ctbr mode).

Life 1 trace -> fixes (skill ids in skills.json):
  S1 rate_bias_ff        : steady commanded rates (+0.6, -0.57, +0.29) with zero measured rate -> actuation bias;
                           fitted w = 0.9*(cmd) + (-0.52, +0.50, -0.26). Feed-forward cmd += bias/gain, attitude integral.
  S2 hover_throttle      : thrust fit c*T^2 -> hover T = 0.329 (datasheet 0.30).
  S3 disturbance_observer: residual accel a_dist = dv/dt - c T^2 b3 + g e_z estimated online, low-passed, cancelled.
  S4 aisle_jet_prior     : north side aisle (x ~ 21.5..25.5) blew a_dist_y ~ -13 m/s^2 -> map-based feed-forward,
                           evaluated at the predicted position; DOB handles the residual.
  S5 tilt_authority      : a_max 8 -> 16 m/s^2 (tilt up to ~58 deg), rate clip 3.1.
"""
from __future__ import annotations

import numpy as np

from pilots.aspire_drone.policy_v1 import G, canyon_half_width, quat_to_R, vee

RATE_GAIN = np.array([0.90, 0.90, 0.92])
RATE_BIAS = np.array([-0.52, 0.50, -0.26])      # actual body rate at zero command [rad/s]
C_THRUST = 90.7                                 # specific force along b3 = C * T^2  -> hover T 0.329


def jet_prior(x):
    """Known crosswind jets (world accel) as a function of along-canyon position."""
    a = np.zeros(3)
    # north side aisle, x in [22, 25.5]: blew toward -y at ~13 m/s^2
    if 21.4 <= x <= 26.0:
        ramp_in = np.clip((x - 21.4) / 1.0, 0, 1)
        ramp_out = np.clip((26.0 - x) / 0.5, 0, 1)
        a[1] = -13.0 * min(ramp_in, ramp_out)
    return a


class Policy:
    def __init__(self, cruise=5.5, kp=4.5, kd=3.6, ki=0.8, kv=2.5, k_att=7.0, k_yaw=3.0, ki_att=1.5,
                 a_max=16.0, i_max=3.0, wall_margin=0.7, k_wall=8.0, ramp_s=1.5, dob_alpha=0.3, dob_max=15.0,
                 lead_s=0.12, use_prior=True):
        self.__dict__.update(dict(cruise=cruise, kp=kp, kd=kd, ki=ki, kv=kv, k_att=k_att, k_yaw=k_yaw, ki_att=ki_att,
                                  a_max=a_max, i_max=i_max, wall_margin=wall_margin, k_wall=k_wall, ramp_s=ramp_s,
                                  dob_alpha=dob_alpha, dob_max=dob_max, lead_s=lead_s, use_prior=use_prior))
        self.reset()

    def reset(self):
        self.start = None
        self.integ = np.zeros(3)
        self.integ_att = np.zeros(3)
        self.r_hat = np.zeros(3)
        self.hist = []          # (v, b3, T) per step
        self.dbg = {}

    def prior(self, x):
        return jet_prior(x) if self.use_prior else np.zeros(3)

    def _segment(self, obs):
        k = int(obs["relay_index"][0])
        relays = np.asarray(obs["relays"], float)
        goal = relays[min(k, len(relays) - 1)]
        prev = self.start if k == 0 else relays[k - 1]
        return prev, goal

    def _dob(self, p, v):
        """residual disturbance beyond the prior, from the last two steps (command delay ~1-2 steps)."""
        if len(self.hist) >= 3:
            v_prev, b3_prev, _, x_prev = self.hist[-1]
            T_del = self.hist[-2][2]
            a_meas = (v - v_prev) * 30.0
            a_model = C_THRUST * T_del ** 2 * b3_prev - np.array([0, 0, G])
            res = a_meas - a_model - self.prior(0.5 * (x_prev + p[0]))
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
        # disturbance cancellation: prior at the predicted position + observed residual
        dist = self.prior(p[0] + v[0] * self.lead_s) + r_hat
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
        cmd = (w_des - RATE_BIAS) / RATE_GAIN
        cmd = np.clip(cmd, -3.1, 3.1)
        u = float(np.dot(f, R[:, 2]))
        T = float(np.clip(np.sqrt(max(u, 0.2 * G) / C_THRUST), 0.05, 0.8))
        self.hist.append((v.copy(), R[:, 2].copy(), T, p[0]))
        self.dbg = dict(a=a, eR=eR, e_perp=e_perp, r_hat=r_hat.copy(), dist=dist)
        return np.array([cmd[0], cmd[1], cmd[2], T])


def make_policy(**kw):
    return Policy(**kw)
