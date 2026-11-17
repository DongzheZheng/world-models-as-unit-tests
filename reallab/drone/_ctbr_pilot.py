"""Internal (lab-only, PRIVILEGED): cascaded CTBR controller that knows the hidden truth.

position -> acceleration (incl. wind feed-forward and the damaged-rotor thrust deficit) -> attitude -> body rates + thrust.
`wind_ff=False` gives the 'wind-blind' variant (same controller, feed-forward switched off)."""
from __future__ import annotations

import numpy as np

from . import _site

M = 0.027
G = 9.81
K_NOM = M * G / 0.2988 ** 2                  # N per T^2 (healthy), from the hover calibration
K_EFF = K_NOM * (3 + _site.ROTOR_EFF) / 4    # thrust deficit of the damaged airframe at equal throttles


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def wind_accel(p, v):
    """Expected (turbulence-free) acceleration the jets impart on the airframe."""
    f = np.zeros(3)
    for j in (_site.JET1, _site.JET2):
        w = _site._profile(j, p)
        if w > 1e-4:
            f += _site.WIND_COEF * w * (j["U"] * np.array(j["dir"]) - v)
    return f / M


class CtbrPilot:
    def __init__(self, vmax=5.0, wind_ff=True, kv=3.0, kct=1.5, kz=3.0, kr=9.0, ki_r=4.0, a_dec=2.0, a_max=26.0,
                 corner_v=1.0, turn_v=0.8, v_aisle=3.0, dt=1.0 / 30.0, use_damage_ff=True):
        self.vmax, self.wind_ff, self.kv, self.kct, self.kz = vmax, wind_ff, kv, kct, kz
        self.kr, self.ki_r, self.a_dec, self.a_max = kr, ki_r, a_dec, a_max
        self.corner_v, self.turn_v, self.v_aisle, self.dt = corner_v, turn_v, v_aisle, dt
        self.k_eff = K_EFF if use_damage_ff else K_NOM
        self.reset()

    def reset(self, seed=None):
        self.wps = None
        self.i = 0
        self.iR = np.zeros(3)
        self.s = 1.0
        self.prev = None

    # ------------------------------------------------------------------ path
    def _plan(self, obs):
        R = obs["relays"]
        p0 = obs["pos"].copy()
        z = R[0][2]
        if abs(R[1][1]) > 2.0:       # relay 2 sits in a side aisle: line up on the centreline, dive in, come back out
            w1 = np.array([R[1][0], 0.0, z])
            pts = [(R[0], self.vmax, self.vmax), (w1, self.corner_v, self.vmax), (R[1], self.turn_v, self.v_aisle),
                   (w1, self.corner_v, self.v_aisle), (R[2], self.vmax, self.vmax)]
        else:
            pts = [(R[0], self.vmax, self.vmax), (R[1], self.vmax, self.vmax), (R[2], self.vmax, self.vmax)]
        self.wps = [(np.array(p, float), v, c) for p, v, c in pts]
        self.start = p0
        self.i = 0

    def _ref(self, p):
        while self.i < len(self.wps) - 1 and np.linalg.norm(p - self.wps[self.i][0]) < 0.8:
            self.i += 1
        tgt, v_end, v_cap = self.wps[self.i]
        a = self.start if self.i == 0 else self.wps[self.i - 1][0]
        seg = tgt - a
        L = np.linalg.norm(seg)
        th = seg / max(L, 1e-6)
        rem = np.linalg.norm(tgt - p)
        v_des = min(v_cap, np.sqrt(v_end ** 2 + 2 * self.a_dec * max(rem - 0.5, 0.0)))
        d = p - a
        ct = d - np.dot(d, th) * th                     # cross-track error vector
        ct[2] = 0.0
        if np.dot(tgt - p, th) < 0.0:                    # overshot a waypoint (e.g. after a dodge): go back for it
            back = (tgt - p) / max(rem, 1e-6)
            v_ref = min(2.0, 2.0 * rem) * back
            v_ref[2] = self.kz * (tgt[2] - p[2])
            return v_ref, tgt
        v_ref = v_des * th - self.kct * ct
        v_ref[2] = self.kz * (tgt[2] - p[2]) if abs(th[2]) < 0.2 else v_des * th[2]
        return v_ref, tgt

    def _avoid(self, obs, p, v, v_ref, a):
        return v_ref, a

    # ------------------------------------------------------------------ control
    def act(self, obs):
        if self.wps is None or obs["t"][0] < 1e-6:
            self.reset()
            self._plan(obs)
        p, v = obs["pos"], obs["lin_vel"]
        Rm = _rot(obs["quat"])
        v_ref, tgt = self._ref(p)
        v_ref, _ = self._avoid(obs, p, v, v_ref, None)
        a = self.kv * (v_ref - v)
        if self.wind_ff:
            a = a - wind_accel(p, v)
        n = np.linalg.norm(a[:2])
        if n > self.a_max:
            a[:2] *= self.a_max / n
        # thrust vector and magnitude
        f = M * (a + np.array([0.0, 0.0, G]))
        self.s = float(np.clip(self.s + 0.6 * self.dt * (v_ref[2] - v[2]), 0.7, 1.5))
        T = np.sqrt(self.s * np.linalg.norm(f) / self.k_eff)
        # desired attitude (yaw held at 0)
        zd = f / max(np.linalg.norm(f), 1e-6)
        xc = np.array([1.0, 0.0, 0.0])
        yd = np.cross(zd, xc); yd /= np.linalg.norm(yd)
        xd = np.cross(yd, zd)
        Rd = np.stack([xd, yd, zd], axis=1)
        eR = 0.5 * _vee(Rd.T @ Rm - Rm.T @ Rd)
        self.iR = np.clip(self.iR + eR * self.dt, -0.5, 0.5)
        w = -(self.kr * eR + self.ki_r * self.iR)
        w[2] *= 0.5
        w = np.clip(w, -np.pi, np.pi)
        return np.array([w[0], w[1], w[2], float(np.clip(T, 0.0, 0.8))])


def informed_policy(**kw):
    return CtbrPilot(wind_ff=True, **kw)


def blind_policy(**kw):
    return CtbrPilot(wind_ff=False, **kw)


class CtbrAvoidPilot(CtbrPilot):
    """Privileged mission-C controller: the same cascaded controller (with wind feed-forward) plus sampling-based
    avoidance of the other aircraft listed in obs["movers"]: every step choose, among candidate velocity references
    around the nominal one, the cheapest one whose constant-velocity prediction keeps `safe` metres from every
    listed aircraft over `horizon` seconds and stays inside the canyon."""

    VX = (1.0, 0.7, 0.45, 0.2)
    VY = tuple(np.arange(-3.0, 3.01, 1.0))
    VZ = tuple(np.arange(-2.0, 2.01, 1.0))

    def __init__(self, safe=0.8, horizon=1.0, w_y=0.5, w_z=0.5, vy_max=3.0, vz_max=2.0, grow=0.3, **kw):
        super().__init__(**kw)
        self.vy_max, self.vz_max = vy_max, vz_max
        self.safe, self.horizon, self.w_y, self.w_z, self.grow = safe, horizon, w_y, w_z, grow
        self.ts = np.linspace(0.0, horizon, 8)
        self.prev = None
        VY = np.arange(-vy_max, vy_max + 1e-6, 1.0)
        VZ = np.arange(-vz_max, vz_max + 1e-6, 1.0)
        g = np.array([(fx, y, z) for fx in self.VX for y in VY for z in VZ])
        self.grid = g

    def _avoid(self, obs, p, v, v_ref, a):
        rows = [r for r in obs["movers"] if r[6] > 0]
        if not rows:
            return v_ref, a
        cand = np.empty_like(self.grid)
        cand[:, 0] = v_ref[0] * self.grid[:, 0]
        cand[:, 1] = v_ref[1] + self.grid[:, 1]
        cand[:, 2] = v_ref[2] + self.grid[:, 2]
        # candidate paths (C, T, 3)
        path = p[None, None, :] + cand[:, None, :] * self.ts[None, :, None]
        clear = np.full(len(cand), 9.0)
        for r in rows:
            mp = r[None, None, :3] + r[None, None, 3:6] * self.ts[None, :, None]
            d = (np.linalg.norm(path - mp, axis=2) - self.grow * self.ts[None, :]).min(axis=1)
            clear = np.minimum(clear, d)
        ylim = 1.45
        wall = np.minimum(ylim - np.abs(path[:, :, 1]).max(axis=1), np.minimum(path[:, :, 2].min(axis=1) - 1.0, 5.0 - path[:, :, 2].max(axis=1)))
        cost = ((cand[:, 0] - v_ref[0]) ** 2 + self.w_y * (cand[:, 1] - v_ref[1]) ** 2 + self.w_z * (cand[:, 2] - v_ref[2]) ** 2)
        cost = cost + 60.0 * np.maximum(self.safe - clear, 0.0) ** 1 + 60.0 * np.maximum(-wall, 0.0)
        jw = max(_site._profile(_site.JET1, p), _site._profile(_site.JET2, p))     # jets: keep dodges gentle there
        lim = max(0.8, 1.0 - 0.7 * jw)
        over = np.maximum(np.abs(self.grid[:, 1]) - self.vy_max * lim, 0.0) + np.maximum(np.abs(self.grid[:, 2]) - self.vz_max * lim, 0.0)
        cost = cost + 30.0 * over
        if self.prev is not None:
            cost = cost + 0.4 * ((cand[:, 1] - self.prev[1]) ** 2 + (cand[:, 2] - self.prev[2]) ** 2)
        k = int(np.argmin(cost))
        self.prev = cand[k].copy()
        return cand[k], a


def avoid_policy(**kw):
    kw.setdefault("vmax", 4.5)
    return CtbrAvoidPilot(wind_ff=True, **kw)
