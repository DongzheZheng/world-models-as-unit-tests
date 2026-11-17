"""Lab-side privileged reference controller: sampling-based MPC that plans by rolling out copies of the real
physics (incl. the hidden deviations, unless nominal=True).  Not a pilot; pilots must not read this module."""
from __future__ import annotations

import numpy as np
import shapely.geometry as sg

from reallab.pusht._deviations import DEVIATIONS, DEVIATIONS_L, NOMINAL, NOMINAL_L
from reallab.pusht._sim import SimPushT

_KP_LOCAL = np.array([(-60, 0), (60, 0), (60, 30), (-60, 30), (-15, 30), (15, 30), (15, 120), (-15, 120)], float)
_T_POLY = sg.Polygon([(-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30)])


from reallab.pusht._geometry import L_GOAL, L_OUTLINE  # noqa: E402

_L_KP_LOCAL = np.array(L_OUTLINE, float)
_L_POLY = sg.Polygon(_L_KP_LOCAL)
_GOALS = {"T": np.array([256.0, 256.0, np.pi / 4]), "L": np.array(L_GOAL)}


def _kp(x, y, th, kp_local=_KP_LOCAL):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return kp_local @ R.T + np.array([x, y])


class MPCPolicy:
    def __init__(self, nominal: bool = False, K: int = 192, H: int = 14, iters: int = 2, seed: int = 0,
                 sigma: float = 45.0, w_app: float = 0.15, elite: float = 0.08, object: str = "T", dev: dict | None = None, w_ang: float = 0.0, adapt: bool = False):
        self.w_ang, self.adapt, self.scale = w_ang, adapt, 1.0
        self.object = object
        _dev_override = dev
        dev = (NOMINAL if nominal else DEVIATIONS) if object == "T" else (NOMINAL_L if nominal else DEVIATIONS_L)
        self.sim = SimPushT(dev if _dev_override is None else _dev_override, object=object)
        self.kp_local = _KP_LOCAL if object == "T" else _L_KP_LOCAL
        self.poly = _T_POLY if object == "T" else _L_POLY
        self.K, self.H, self.iters, self.sigma, self.w_app, self.elite = K, H, iters, sigma, w_app, elite
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.goal = _GOALS[object]
        self.goal_kp = _kp(*self.goal, self.kp_local)
        self.env = None
        self.mean = None

    def bind_real(self, env):
        self.env = env

    def set_seed(self, s):
        self.rng = np.random.default_rng([self.base_seed, int(s)])

    def reset(self):
        self.mean = None

    # -- rollout cost -------------------------------------------------------------------------------------
    def _cost(self, s0, acts):
        sim = self.sim
        sim.set_full_state(s0)
        H = len(acts)
        J = 0.0
        for t in range(H):
            o, _, _, _, info = sim.step(acts[t])
            kpd = np.linalg.norm(_kp(o[2], o[3], o[4], self.kp_local) - self.goal_kp, axis=1).mean()
            cov = info["coverage"]
            w = 1.0 if t < H - 1 else 4.0
            dth = abs((o[4] - self.goal[2] + np.pi) % (2 * np.pi) - np.pi)
            J += w * (kpd / 50.0 + 2.0 * (1 - cov) + self.w_ang * dth)
            if cov > 0.95:
                J -= 50.0 * (H - t)
                break
        p = self.poly
        # approach term at the end: distance from pusher to the block
        c, s = np.cos(o[4]), np.sin(o[4])
        d = np.array([o[0] - o[2], o[1] - o[3]])
        loc = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        J += self.w_app * max(0.0, p.distance(sg.Point(*loc)) - 15.0) / 10.0
        return J

    def _sample(self, a0, K):
        """K action sequences (K,H,2): half smooth noise around the warm-started mean, half random waypoints."""
        H, rng = self.H, self.rng
        out = np.empty((K, H, 2))
        k1 = K // 2
        n = rng.normal(0, self.sigma * self.scale, (k1, H, 2))
        n = np.cumsum(n, 1) / np.sqrt(np.arange(1, H + 1))[None, :, None] * 0.8 + n * 0.5
        out[:k1] = self.mean[None] + n
        k2 = K - k1
        nw = 3
        wp = a0[None, None] + rng.uniform(-170, 170, (k2, nw, 2)) * max(self.scale, 0.4)
        wp[:, :, :] = np.clip(wp, 20, 492)
        xs = np.linspace(0, nw, H)
        for k in range(k2):
            pts = np.vstack([a0, wp[k]])
            out[k1 + k, :, 0] = np.interp(xs, np.arange(nw + 1), pts[:, 0])
            out[k1 + k, :, 1] = np.interp(xs, np.arange(nw + 1), pts[:, 1])
        out[0] = self.mean
        return np.clip(out, 5, 507)

    def act(self, obs):
        s0 = self.env._sim.get_full_state()
        a0 = s0[:2]
        if self.adapt:
            kpd0 = np.linalg.norm(_kp(s0[4], s0[5], s0[6], self.kp_local) - self.goal_kp, axis=1).mean()
            self.scale = float(np.clip(kpd0 / 60.0, 0.3, 1.0))
        if self.mean is None:
            self.mean = np.tile(a0, (self.H, 1))
        else:
            self.mean = np.vstack([self.mean[1:], self.mean[-1:]])
        for _ in range(self.iters):
            seqs = self._sample(a0, self.K)
            costs = np.array([self._cost(s0, a) for a in seqs])
            idx = np.argsort(costs)[: max(2, int(self.elite * self.K))]
            self.mean = seqs[idx].mean(0) if _ < self.iters - 1 else seqs[idx[0]]
            self.sigma_used = self.sigma
        return self.mean[0].astype(np.float32)


def true_sim_mpc(**kw):
    return MPCPolicy(**kw)
