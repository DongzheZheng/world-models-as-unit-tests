"""policy_v6: snap-back cast (stroke out, haul back).  Life 4 refuted the follow-the-leader idea (the cord does not track
the tool's path; the tip only moves when the cord is taut, so slow work leaves it where it is).  Lives 1-2: a fast pull
moves the whole straight cord at tool speed, but after a plain stop the hairpin runs out of energy within ~0.5 s.
v6 adds a haul: from F at the back of the zone, strike forward along d (towards the aim point A, beside the post) at full
rate, and as soon as the tool is past ~55% of the stroke command F again (maximal reversal), so the tool pulls the
attached leg back while the free leg is still flying forward; the hairpin is pulled through and the cord should end
stretched from F along d with its tip near F + 0.66 d = A.  Then hold F."""
import numpy as np

ZLO, ZHI = np.array([0.30, -0.20]), np.array([0.55, 0.20])


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def plan(target, post, reach=0.66, clear=0.035, aim_off=0.05, stroke_max=0.30):
    radial = (target - post) / np.linalg.norm(target - post)
    lat = np.array([-radial[1], radial[0]])
    best, bestv = None, -1e9
    for sa in (-1, 0, 1):
        A = target + sa * aim_off * lat
        for ang in np.radians(np.arange(-80, 80.1, 1.0)):
            d = np.array([np.cos(ang), np.sin(ang)])
            F = A - reach * d
            if np.any(F < ZLO) or np.any(F > ZHI):
                continue
            c = seg_dist(post, F, A)
            if c < clear:
                continue
            L = stroke_max
            while L > 0.05 and (np.any(F + L * d > ZHI) or np.any(F + L * d < ZLO)):
                L -= 0.01
            v = 3 * min(L, 0.27) + 0.5 * min(c, 0.07) - 0.3 * abs(sa)
            if v > bestv:
                bestv, best = v, dict(F=F, G=F + L * d, A=A, d=d, L=L, clear=c)
    return best


class Policy:
    def __init__(self, prep_speed=0.5, pause=10, switch_frac=0.55, **kw):
        self.kw, self.prep_speed, self.pause, self.switch_frac = kw, prep_speed, pause, switch_frac
        self.reset()

    def reset(self):
        self.p, self.phase, self.ref, self.t = None, "prep", None, 0

    def act(self, obs):
        tool = np.asarray(obs["tool_pos"], float)
        self.t += 1
        if self.p is None:
            self.p = plan(np.asarray(obs["target"], float), np.asarray(obs["post"], float), **self.kw)
            if self.p is None:
                tgt = np.asarray(obs["target"], float)
                d = (tgt - [0.425, 0.0]) / np.linalg.norm(tgt - [0.425, 0.0])
                F = np.clip(tgt - 0.66 * d, ZLO, ZHI)
                self.p = dict(F=F, G=np.clip(F + 0.2 * d, ZLO, ZHI), d=d, L=0.2)
            self.ref = tool.copy()
        F, G = self.p["F"], self.p["G"]
        if self.phase == "prep":
            dv = F - self.ref
            n = np.linalg.norm(dv)
            step = self.prep_speed / 50.0
            if n <= step:
                self.ref, self.phase, self.t0 = F.copy(), "pause", self.t
            else:
                self.ref = self.ref + dv / n * step
            return self.ref.copy()
        if self.phase == "pause":
            if self.t - self.t0 >= self.pause:
                self.phase = "out"
            return F.copy()
        if self.phase == "out":
            if float((tool - F) @ self.p["d"]) >= self.switch_frac * self.p["L"]:
                self.phase = "back"
                return F.copy()
            return G.copy()
        return F.copy()


def make_policy(**kw):
    return Policy(**kw)
