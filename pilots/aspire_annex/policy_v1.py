"""policy_v1: single tethered cast.  Plan (closed-form geometry, no model): pick a final tool point F in the zone and an
aim point A in the target disc such that |A-F| = reach and the segment F->A clears the post; drag the tool slowly back
to S = F - stroke*u, pause, then command F once (max-rate stroke; the reference brakes to stop at F) and hold F."""
import numpy as np

ZLO, ZHI = np.array([0.30, -0.20]), np.array([0.55, 0.20])


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def plan(target, post, reach=0.66, clear=0.06, aim_off=0.04, stroke_max=0.25):
    """Grid search over candidate (F, A) pairs (pure geometry). Returns F, S, A, u."""
    best, bestv = None, -1e9
    radial = (target - post) / np.linalg.norm(target - post)
    lat = np.array([-radial[1], radial[0]])
    for side in (-1, 1):
        A = target + side * aim_off * lat
        for ang in np.radians(np.arange(-60, 60.1, 1.0)):
            u = np.array([np.cos(ang), np.sin(ang)])
            F = A - reach * u
            if np.any(F < ZLO - 1e-9) or np.any(F > ZHI + 1e-9):
                continue
            c = seg_dist(post, F, A)
            if c < clear:
                continue
            # stroke room behind F along -u inside the zone
            s = stroke_max
            while s > 0.05 and (np.any(F - s * u < ZLO) or np.any(F - s * u > ZHI)):
                s -= 0.01
            v = s * 10 + min(c, 0.10) * 5 - abs(ang) * 0.2
            if v > bestv:
                bestv, best = v, (F, F - s * u, A, u, s, c)
    return best


class CastPolicy:
    def __init__(self, reach=0.66, clear=0.06, aim_off=0.04, stroke_max=0.25, prep_speed=0.6, pause=15):
        self.kw = dict(reach=reach, clear=clear, aim_off=aim_off, stroke_max=stroke_max)
        self.prep_speed, self.pause = prep_speed, pause
        self.reset()

    def reset(self):
        self.t = 0
        self.plan = None
        self.phase = "prep"
        self.ref = None

    def act(self, obs):
        if self.plan is None:
            p = plan(np.asarray(obs["target"]), np.asarray(obs["post"]), **self.kw)
            if p is None:   # fallback: straight radial cast from the back of the zone to the front
                tgt = np.asarray(obs["target"])
                F = np.clip(tgt - 0.66 * (tgt - [0.425, 0]) / np.linalg.norm(tgt - [0.425, 0]), ZLO, ZHI)
                u = (tgt - F) / np.linalg.norm(tgt - F)
                p = (F, np.clip(F - 0.2 * u, ZLO, ZHI), tgt, u, 0.2, 0.0)
            self.plan = p
            self.ref = np.asarray(obs["tool_pos"], float).copy()
        F, S = self.plan[0], self.plan[1]
        self.t += 1
        if self.phase == "prep":
            d = S - self.ref
            n = np.linalg.norm(d)
            step = self.prep_speed / 50.0
            if n <= step:
                self.ref = S.copy()
                self.phase, self.t0 = "pause", self.t
            else:
                self.ref = self.ref + d / n * step
            return self.ref.copy()
        if self.phase == "pause":
            if self.t - self.t0 >= self.pause:
                self.phase = "cast"
            return S.copy()
        return F.copy()


def make_policy(**kw):
    return CastPolicy(**kw)
