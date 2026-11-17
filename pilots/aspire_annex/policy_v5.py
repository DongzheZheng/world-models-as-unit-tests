"""policy_v5: lay a track, then push the cord out tip-first.  Lives 1-3: every dynamic throw ended with a straight cord
pointing back along whichever way its near part ended up (elastic straightening), landing direction uncontrolled; the
cord follows the tool's path when pulled (life 3 reel: tip tracked the tool at 0.1 m/s) and transmits an axial push
(life 3 strike: the in-line near part moved with the tool).  v5 is kinematic: the tool drives a path so that, when the
cord's tip lies on the path at X (near the zone front), the path there heads AWAY from the target; it then reverses
along its own path by E = |A - X|, which slides the cord tip-first along its body and straight out of X towards the aim
point A (beside the post).  Then hold."""
import numpy as np

ZLO, ZHI = np.array([0.30, -0.20]), np.array([0.55, 0.20])
LROPE = 0.70


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def inzone(p, m=0.0):
    return bool(np.all(p >= ZLO + m) and np.all(p <= ZHI - m))


def resample(W, ds=0.004):
    W = np.asarray(W, float)
    out = [W[0]]
    for a, b in zip(W[:-1], W[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / ds)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def arclen(P):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def chaikin(W, it=3):
    W = np.asarray(W, float)
    for _ in range(it):
        Q = [W[0]]
        for a, b in zip(W[:-1], W[1:]):
            Q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        Q.append(W[-1])
        W = np.array(Q)
    return W


def track_after_X(X, u, side, rho, Lb, ds=0.004):
    """From X heading -u: straight for Lb, then constant-curvature arc (radius rho, turning to `side`), total LROPE."""
    pts, h, p = [X.copy()], np.arctan2(-u[1], -u[0]), X.copy()
    n = int(round(LROPE / ds))
    for k in range(n):
        if k * ds >= Lb:
            h += side * ds / rho
        p = p + ds * np.array([np.cos(h), np.sin(h)])
        pts.append(p.copy())
    return np.array(pts)


def plan(target, post, tool0, aim_off=0.05, clear=0.045, a=0.04, emax=0.66, m=-0.003):
    radial = (target - post) / np.linalg.norm(target - post)
    lat = np.array([-radial[1], radial[0]])
    best, bestv = None, -1e9
    for sa in (-1, 0, 1):
        A = target + sa * aim_off * lat
        for xx in np.arange(0.44, 0.5501, 0.01):
            for yy in np.arange(-0.19, 0.1901, 0.02):
                X = np.array([xx, yy])
                E = float(np.linalg.norm(A - X))
                if E > emax:
                    continue
                u = (A - X) / E
                Q = X + a * u
                if not inzone(Q, m):
                    continue
                c = seg_dist(post, X, A + 0.03 * u)
                if c < clear:
                    continue
                v0 = -E + 0.5 * min(c, 0.08)
                if v0 <= bestv:
                    continue
                done = False
                for rho in (0.16, 0.14, 0.12, 0.11, 0.10, 0.09):
                    for Lb in (0.20, 0.16, 0.12, 0.08, 0.04):
                        for side in (-1, 1):
                            T = track_after_X(X, u, side, rho, Lb)
                            if np.all(T >= ZLO + m) and np.all(T <= ZHI - m):
                                bestv, best, done = v0, dict(X=X, Q=Q, u=u, A=A, E=E, clear=c, rho=rho, Lb=Lb, side=side, T=T), True
                                break
                        if done: break
                    if done: break
    if best is None:
        return None
    u, Q = best["u"], best["Q"]
    latu = np.array([-u[1], u[0]])
    pre = np.clip(Q + best["side"] * 0.09 * latu + 0.01 * u, ZLO, ZHI)    # come in from the side away from the loop
    appr = resample(chaikin([tool0, pre, Q, best["X"]], 2))[:-1]
    P = np.vstack([appr, best["T"]])
    S = arclen(P)
    best.update(P=P, sX=float(S[len(appr)]), send=float(S[-1]))
    return best


class Policy:
    def __init__(self, lay_speed=0.5, push_speed=0.5, **kw):
        self.kw, self.lay_speed, self.push_speed = kw, lay_speed, push_speed
        self.reset()

    def reset(self):
        self.p, self.s, self.phase, self.ref = None, 0.0, "lay", None

    def _at(self, s):
        P, S = self.p["P"], self.p["S"]
        i = int(np.clip(np.searchsorted(S, s), 1, len(S) - 1))
        f = (s - S[i - 1]) / max(S[i] - S[i - 1], 1e-12)
        return P[i - 1] + f * (P[i] - P[i - 1])

    def act(self, obs):
        tool = np.asarray(obs["tool_pos"], float)
        if self.p is None:
            self.p = plan(np.asarray(obs["target"], float), np.asarray(obs["post"], float), tool, **self.kw)
            if self.p is None:
                self.phase, self.ref = "hold", tool.copy()
            else:
                self.p["S"] = arclen(self.p["P"])
        if self.phase == "lay":
            self.s = min(self.s + self.lay_speed / 50.0, self.p["send"])
            if self.s >= self.p["send"]:
                self.phase = "push"
            self.ref = self._at(self.s)
        elif self.phase == "push":
            stop = self.p["send"] - self.p["E"]
            self.s = max(self.s - self.push_speed / 50.0, stop)
            if self.s <= stop:
                self.phase = "hold"
            self.ref = self._at(self.s)
        return np.clip(self.ref, ZLO, ZHI).copy()


def make_policy(**kw):
    return Policy(**kw)
