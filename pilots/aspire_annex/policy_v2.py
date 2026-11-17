"""policy_v2: aligned fly-cast.  Life 1 showed: a short (0.17 m) stroke at 52 deg to the rope's lie folds the rope, the loop
dies just past the tool and the (bending-elastic) rope unfolds back along its stem.  v2 picks, among geometrically valid
casts (|A-F| = reach, F->A grazes the post at most), the direction u most opposite to the rope's current lie
(tool->rope centroid) with the longest stroke, so the stroke pulls the rope along its own axis, then casts at full rate."""
import numpy as np

ZLO, ZHI = np.array([0.30, -0.20]), np.array([0.55, 0.20])


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


def plan(target, post, rope_dir, reach=0.66, clear=0.03, aim_off=0.05, stroke_max=0.35, w_align=1.0):
    radial = (target - post) / np.linalg.norm(target - post)
    lat = np.array([-radial[1], radial[0]])
    want = wrap(rope_dir + np.pi)
    best, bestv = None, -1e9
    for side in (-1, 0, 1):
        A = target + side * aim_off * lat
        for ang in np.radians(np.arange(-80, 80.1, 1.0)):
            u = np.array([np.cos(ang), np.sin(ang)])
            F = A - reach * u
            if np.any(F < ZLO) or np.any(F > ZHI):
                continue
            c = seg_dist(post, F, A)
            if c < clear:
                continue
            s = stroke_max
            while s > 0.04 and (np.any(F - s * u < ZLO - 1e-9) or np.any(F - s * u > ZHI + 1e-9)):
                s -= 0.01
            mis = abs(wrap(ang - want))
            v = -w_align * mis + 3.0 * min(s, 0.25) + 0.5 * min(c, 0.06)
            if v > bestv:
                bestv, best = v, dict(F=F, S=F - s * u, A=A, u=u, s=s, clear=c, mis=mis)
    return best


class CastPolicy:
    def __init__(self, reach=0.66, clear=0.03, aim_off=0.05, stroke_max=0.35, prep_speed=0.5, pause=15, w_align=1.0):
        self.kw = dict(reach=reach, clear=clear, aim_off=aim_off, stroke_max=stroke_max, w_align=w_align)
        self.prep_speed, self.pause = prep_speed, pause
        self.reset()

    def reset(self):
        self.t, self.p, self.phase, self.ref = 0, None, "prep", None

    def act(self, obs):
        if self.p is None:
            tool = np.asarray(obs["tool_pos"], float)
            c = np.asarray(obs["points"])[:, :2].mean(0) - tool
            self.p = plan(np.asarray(obs["target"]), np.asarray(obs["post"]), np.arctan2(c[1], c[0]), **self.kw)
            self.ref = tool.copy()
        F, S = self.p["F"], self.p["S"]
        self.t += 1
        if self.phase == "prep":
            d = S - self.ref
            n = np.linalg.norm(d)
            step = self.prep_speed / 50.0
            if n <= step:
                self.ref, self.phase, self.t0 = S.copy(), "pause", self.t
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
