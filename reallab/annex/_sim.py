"""Lab-side fast episode simulator (no ledger, state save/restore) for the references and the tuning runs.  Internal:
pilots must not read this module.  The goal rules here are the same as api.py (checked in test_api.py).

Sim(phys, vmax).setup(seed) builds the episode exactly as api.reset(seed) does and stores the start state;
Sim.rollout(schedule, T) restores it and runs a command schedule (a function k -> command, or an array (T, 2)).
A rollout reports, for both goal candidates ("wrap" and "pocket"), the first step at which the goal holds at rest.
"""
from __future__ import annotations

import math

import numpy as np

from reallab.annex import _internal, _layout, _scene
from reallab.annex import _limits as S

ZLO, ZHI = np.array(S.ZONE_LOW), np.array(S.ZONE_HIGH)


def at_rest(prev_pts, pts, prev_tool, tool, hz=S.CONTROL_HZ, v=S.V_REST) -> bool:
    lim = v / hz
    dp = np.linalg.norm(pts[:, :2] - prev_pts[:, :2], axis=1).max()
    dt = np.linalg.norm(tool - prev_tool)
    return bool(dp <= lim and dt <= lim)


def in_zone(tool, tol=S.ZONE_TOL) -> bool:
    return bool(np.all(tool >= ZLO - tol) and np.all(tool <= ZHI + tol))


def wrap_ok(pts, post, side, w_min=S.W_MIN) -> bool:
    return side * _layout.winding(pts, post) >= w_min


def pocket_ok(pts, target, r=S.POCKET_R) -> bool:
    return float(np.linalg.norm(pts[0, :2] - target)) <= r


class Sim:
    def __init__(self, phys: dict | None = None, vmax: float = S.TOOL_VMAX, w_min=S.W_MIN, pocket_r=S.POCKET_R,
                 v_rest=S.V_REST, pocket_w=None, no_post=False):
        self.phys = dict(phys or _internal.physical())
        self.scene = _scene.Scene(self.phys, vmax=vmax)
        self.w_min, self.pocket_r, self.v_rest, self.pocket_w = w_min, pocket_r, v_rest, pocket_w
        self.no_post = no_post

    def setup(self, seed: int) -> dict:
        self.ep = _layout.setup(self.scene, seed)
        if self.no_post:
            self.scene.place(_scene.PARK, self.ep["target"])
        self.s0 = self.scene.get_state()
        self.pts0 = self.scene.tracked_points()
        self.tool0 = self.scene.tool_pos()
        return self.ep

    def restore(self):
        self.scene.set_state(self.s0)

    def rollout(self, schedule, T: int, stop_when=("wrap",), record=False, start_state=None, k0=0) -> dict:
        """Run up to T agent steps.  Stops early once every goal named in `stop_when` has succeeded, or once the
        scene is at rest at a step >= `quiet_after` (schedule attribute, default T) - nothing can change after that."""
        sc, ep = self.scene, self.ep
        sc.set_state(self.s0 if start_state is None else start_state)
        prev_p, prev_t = sc.tracked_points(), sc.tool_pos()
        quiet_after = getattr(schedule, "quiet_after", T)
        res = dict(wrap_t=None, pocket_t=None, n=0)
        best_w, md = -1e9, 1e9
        rec = [] if record else None
        for k in range(k0, k0 + T):
            cmd = schedule(k) if callable(schedule) else schedule[k]
            cmd = np.minimum(np.maximum(np.asarray(cmd, float), ZLO), ZHI)
            sc.step(cmd)
            p, t = sc.tracked_points(), sc.tool_pos()
            if rec is not None:
                rec.append((p, t, cmd))
            w = ep["side"] * _layout.winding(p, ep["post"])
            md = min(md, float(np.min(np.linalg.norm(p[:, :2] - ep["post"], axis=1))))
            best_w = max(best_w, w)
            rest = at_rest(prev_p, p, prev_t, t, v=self.v_rest)
            if rest and in_zone(t):
                if res["wrap_t"] is None and w >= self.w_min:
                    res["wrap_t"] = k + 1
                if (res["pocket_t"] is None and np.linalg.norm(p[0, :2] - ep["target"]) <= self.pocket_r
                        and (self.pocket_w is None or w >= self.pocket_w)):
                    res["pocket_t"] = k + 1
            res["n"] = k + 1
            prev_p, prev_t = p, t
            if stop_when and all(res[g + "_t"] is not None for g in stop_when):
                break
            if rest and k + 1 >= quiet_after:
                break
        res["final_w"] = float(ep["side"] * _layout.winding(prev_p, ep["post"]))
        res["best_w"] = float(best_w)
        res["min_post_dist"] = md
        res["tip_dist"] = float(np.linalg.norm(prev_p[0, :2] - ep["target"]))
        res["tip_final"] = prev_p[0, :2].copy()
        res["at_rest"] = bool(rest)
        if rec is not None:
            res["rec"] = rec
        return res


# ---------------------------------------------------------------- parametrised casts
class Cast:
    """Waypoint schedule in the post frame: u = unit vector zone centre -> post, n = u rotated +90 deg times the
    required side.  params = [a_1, b_1, ..., a_m, b_m, d_1, ..., d_{m-1}]: waypoint j = zone centre + a_j u + b_j n
    (clipped to the zone), commanded from step K_{j-1} to K_j where K_0 = 0 and K_j = K_{j-1} + round(50 d_j); the last
    waypoint is held forever."""

    def __init__(self, params, post, side, m):
        p = np.asarray(params, float)
        c = np.array(S.ZONE_CENTRE)
        u = np.asarray(post, float) - c
        u = u / np.linalg.norm(u)
        n = side * np.array([-u[1], u[0]])
        ab = p[:2 * m].reshape(m, 2)
        self.way = np.minimum(np.maximum(c + ab[:, :1] * u + ab[:, 1:] * n, ZLO), ZHI)
        d = np.maximum(1, np.round(p[2 * m:] * S.CONTROL_HZ).astype(int))
        self.K = np.concatenate([[0], np.cumsum(d)])
        self.quiet_after = int(self.K[-1]) + 1

    def __call__(self, k):
        j = int(np.searchsorted(self.K, k, side="right")) - 1
        return self.way[min(j, len(self.way) - 1)]


def cast_box(m, dmax):
    lo = np.array([-0.3, -0.3] * m + [0.02] * (m - 1))
    hi = np.array([0.3, 0.3] * m + [dmax] * (m - 1))
    return lo, hi


class Spin:
    """Circle-then-release schedule in the post frame (u, n as in Cast).  params = [ca, cb, rho, omega, phi0, t_rel,
    fa, fb]: for t = k/50 < t_rel the command is c + rho (cos(phi0 + omega t) u + sin(phi0 + omega t) n), c = zone
    centre + ca u + cb n; afterwards the command is the point zone centre + fa u + fb n (held)."""
    LO = np.array([-0.12, -0.2, 0.03, -40.0, -np.pi, 0.15, -0.3, -0.3])
    HI = np.array([0.12, 0.2, 0.14, 40.0, np.pi, 1.2, 0.3, 0.3])

    def __init__(self, params, post, side):
        p = np.asarray(params, float)
        c0 = np.array(S.ZONE_CENTRE)
        u = np.asarray(post, float) - c0
        u = u / np.linalg.norm(u)
        n = side * np.array([-u[1], u[0]])
        self.u, self.n = u, n
        self.c = c0 + p[0] * u + p[1] * n
        self.rho, self.om, self.phi0, self.t_rel = p[2], p[3], p[4], p[5]
        self.f = np.minimum(np.maximum(c0 + p[6] * u + p[7] * n, ZLO), ZHI)
        self.k_rel = int(round(self.t_rel * S.CONTROL_HZ))
        self.quiet_after = self.k_rel + 1

    def __call__(self, k):
        if k >= self.k_rel:
            return self.f
        a = self.phi0 + self.om * k / S.CONTROL_HZ
        return self.c + self.rho * (np.cos(a) * self.u + np.sin(a) * self.n)


class Spin2(Spin):
    """Spin, then a second point: params = Spin params + [d, ga, gb]: the release point F is held for round(50 d)
    steps, then the command jumps to zone centre + ga u + gb n (held)."""
    LO = np.concatenate([Spin.LO, [0.02, -0.3, -0.3]])
    HI = np.concatenate([Spin.HI, [0.5, 0.3, 0.3]])

    def __init__(self, params, post, side):
        p = np.asarray(params, float)
        super().__init__(p[:8], post, side)
        c0 = np.array(S.ZONE_CENTRE)
        self.k2 = self.k_rel + max(1, int(round(p[8] * S.CONTROL_HZ)))
        self.g = np.minimum(np.maximum(c0 + p[9] * self.u + p[10] * self.n, ZLO), ZHI)
        self.quiet_after = self.k2 + 1

    def __call__(self, k):
        if k >= self.k2:
            return self.g
        return super().__call__(k)


class Shifted:
    """A schedule started at agent step k0 (for a second attempt from a saved state)."""

    def __init__(self, sched, k0):
        self.s, self.k0 = sched, k0
        self.quiet_after = k0 + getattr(sched, "quiet_after", 0)

    def __call__(self, k):
        return self.s(k - self.k0)


def make_schedule(kind: str, params, post, side, m: int = 3):
    if kind == "spin":
        return Spin(params, post, side)
    if kind == "spin2":
        return Spin2(params, post, side)
    return Cast(params, post, side, m)


# ---------------------------------------------------------------- geometry-scheduled script (criterion 5a)
GEO_DIST0 = 0.48


def geo_params(consts, dist):
    """Cast parameters (m = 3, side fixed +1) as closed-form functions of the layout: the post frame already carries
    the bearing; the forward reach a_2 and the two durations depend linearly on the post's distance.
    consts = [a1, b1, a2, b2, a3, b3, d1, d2, k_a2, k_d1, k_d2]."""
    c = np.asarray(consts, float)
    p = c[:8].copy()
    x = dist - GEO_DIST0
    p[2] += c[8] * x
    p[6] += c[9] * x
    p[7] += c[10] * x
    p[6:] = np.maximum(p[6:], 0.02)
    return p


GEO_LO = np.array([-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, 0.02, 0.02, -2.0, -3.0, -3.0])
GEO_HI = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 2.0, 3.0, 3.0])
