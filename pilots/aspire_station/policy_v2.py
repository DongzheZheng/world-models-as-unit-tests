"""aspire_station policy v2 = v1 with the path pulled taut in its homotopy class (radius 0.065 around posts).

v1 description:
aspire_station policy v1: topological path + load-governed carrot follower.

Path: pass every post on the side that produces the required winding (side +1 -> post on the left of travel), at
clearance 0.07 m, corners smoothed, tool kept >= 0.057 m from every post centre.
Speed: an internal reference moves along the path at v(load): vmax below 4 N, linear down to 0 at 7 N, back-off above
8.5 N; it only advances while the tool is within 2 cm of it (no servo wind-up -> no catch-up rush).
"""
import numpy as np

ANCHOR = np.array([0.05, 0.0])
POST_R, TOOL_R = 0.03, 0.015
DT = 0.05


def _resample(P, ds):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(np.ceil(s[-1] / ds)) + 1)
    si = np.linspace(0, s[-1], n)
    return np.stack([np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])], 1)


def _push_out(P, posts, rmin):
    P = P.copy()
    for c in posts:
        d = P[1:-1] - c
        r = np.linalg.norm(d, axis=1)
        m = r < rmin
        if np.any(m):
            P[1:-1][m] = c + d[m] / np.maximum(r[m, None], 1e-9) * rmin
    return P


def plan_path(tool0, posts, sides, target, clr=0.07, ds=0.005, margin=0.012):
    d = target - ANCHOR
    d = d / np.linalg.norm(d)
    left = np.array([-d[1], d[0]])
    order = np.argsort([(c - ANCHOR) @ d for c in posts])
    wps = [tool0] + [posts[j] - sides[j] * left * clr for j in order] + [target]
    P = _resample(np.array(wps, float), ds)
    rmin = POST_R + TOOL_R + margin
    for _ in range(30):
        Q = P.copy()
        Q[1:-1] = 0.5 * P[1:-1] + 0.25 * (P[:-2] + P[2:])
        P = _push_out(Q, posts, rmin)
    P = _push_out(P, posts, rmin)
    # v2: pull the path taut inside its homotopy class (same sides), keeping `pull_r` from every post centre:
    # no unnecessary dips between posts -> least wrap and shortest drag.
    pull_r = max(rmin, clr - 0.005)
    for dsi, it in ((0.03, 1500), (0.01, 800)):
        P = _resample(P, dsi)
        for _ in range(it):
            Q = P.copy()
            Q[1:-1] = 0.5 * (P[:-2] + P[2:])
            P = _push_out(Q, posts, pull_r)
    return _resample(P, ds)


class Policy:
    def __init__(self, vmax=0.10, vmin=0.0, load_lo=4.0, load_hi=7.0, load_back=8.5, back_step=0.004,
                 max_lag=0.02, clr=0.07, final_zone=0.06):
        self.kw = dict(vmax=vmax, vmin=vmin, load_lo=load_lo, load_hi=load_hi, load_back=load_back,
                       back_step=back_step, max_lag=max_lag, final_zone=final_zone)
        self.clr = clr
        self.reset()

    def reset(self):
        self.P = None
        self.s = None
        self.sref = 0.0
        self.trace = []

    def _point(self, sref):
        return np.array([np.interp(sref, self.s, self.P[:, 0]), np.interp(sref, self.s, self.P[:, 1])])

    def act(self, obs):
        k = self.kw
        tool = np.asarray(obs["tool_pos"], float)
        if self.P is None:
            self.P = plan_path(tool, np.asarray(obs["posts"], float), np.asarray(obs["post_sides"]),
                               np.asarray(obs["target"], float), clr=self.clr)
            self.s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(self.P, axis=0), axis=1))])
        load = float(obs["load"])
        send = self.s[-1]
        if load >= k["load_back"]:
            self.sref = max(0.0, self.sref - k["back_step"])
            v = -1.0
        else:
            if load <= k["load_lo"]:
                v = k["vmax"]
            elif load >= k["load_hi"]:
                v = 0.0
            else:
                v = k["vmax"] * (k["load_hi"] - load) / (k["load_hi"] - k["load_lo"])
            v = max(v, k["vmin"])
            rem = send - self.sref
            if rem < k["final_zone"]:
                v = min(v, max(0.012, 0.6 * rem / k["final_zone"] * k["vmax"]))
            lag = np.linalg.norm(tool - self._point(self.sref))
            if lag < k["max_lag"]:
                self.sref = min(send, self.sref + v * DT)
        cmd = self._point(self.sref)
        self.trace.append((self.sref, v, load))
        return cmd


def make_policy(**kw):
    return Policy(**kw)
