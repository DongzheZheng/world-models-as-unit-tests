"""policy_v3: overcast + reel-in.  Diagnosis from life 1: the rope is a bending-elastic cord on a low-friction table; a
fold that does not unroll past the tool is undone by the rope's own elasticity, so landing precision from the cast alone
is hopeless ("small change changes where it ends up").  v3 (a) casts so the tip should land BEYOND the target on a line
F->A that passes beside the post (|A-F| = reach_cast < full rope chord), with the longest full-rate stroke whose direction
is as opposite as possible to the rope's lie and with little slack from backing up; (b) once the rope has nearly stopped,
reels the tip in slowly with live feedback (tool moves along target - tip) until the tip is inside the disc, then holds."""
import numpy as np

ZLO, ZHI = np.array([0.30, -0.20]), np.array([0.55, 0.20])


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


def plan(target, post, tool0, rope_dir, reach_cast=0.52, clear=0.035, aim_off=0.05, stroke_max=0.32,
         w_align=0.6, w_slack=4.0, w_stroke=3.0):
    radial = (target - post) / np.linalg.norm(target - post)
    lat = np.array([-radial[1], radial[0]])
    rv = np.array([np.cos(rope_dir), np.sin(rope_dir)])
    want = wrap(rope_dir + np.pi)
    best, bestv = None, -1e9
    for side in (-1, 0, 1):
        A = target + side * aim_off * lat
        for R in (reach_cast - 0.04, reach_cast, reach_cast + 0.04):
            for ang in np.radians(np.arange(-80, 80.1, 1.0)):
                u = np.array([np.cos(ang), np.sin(ang)])
                F = A - R * u
                if np.any(F < ZLO) or np.any(F > ZHI):
                    continue
                c = seg_dist(post, F, A + 0.15 * u)
                if c < clear:
                    continue
                s = stroke_max
                while s > 0.06 and (np.any(F - s * u < ZLO - 1e-9) or np.any(F - s * u > ZHI + 1e-9)):
                    s -= 0.01
                S = F - s * u
                slack = max(0.0, float((S - tool0) @ rv))
                mis = abs(wrap(ang - want))
                v = -w_align * mis + w_stroke * s - w_slack * slack * 0.25 + 0.5 * min(c, 0.07) - 2 * abs(R - reach_cast)
                if v > bestv:
                    bestv, best = v, dict(F=F, S=S, A=A, u=u, s=s, clear=c, mis=mis, slack=slack, R=R)
    return best


class Policy:
    def __init__(self, reach_cast=0.52, prep_speed=0.5, pause=10, settle_v=0.12, settle_max=60, reel_speed=0.15,
                 stop_dist=0.035, **plan_kw):
        self.kw = dict(reach_cast=reach_cast, **plan_kw)
        self.prep_speed, self.pause, self.settle_v, self.settle_max = prep_speed, pause, settle_v, settle_max
        self.reel_speed, self.stop_dist = reel_speed, stop_dist
        self.reset()

    def reset(self):
        self.t, self.p, self.phase, self.ref, self.prev, self.best = 0, None, "prep", None, None, 9.0
        self.log = []

    def act(self, obs):
        pts = np.asarray(obs["points"])[:, :2]
        tool = np.asarray(obs["tool_pos"], float)
        tgt = np.asarray(obs["target"], float)
        if self.p is None:
            c = pts.mean(0) - tool
            self.p = plan(tgt, np.asarray(obs["post"]), tool, np.arctan2(c[1], c[0]), **self.kw)
            self.ref = tool.copy()
        vmax = 0.0 if self.prev is None else float(np.linalg.norm(pts - self.prev, axis=1).max() * 50)
        self.prev = pts.copy()
        F, S = self.p["F"], self.p["S"]
        self.t += 1
        tip = pts[0]
        dtip = float(np.linalg.norm(tip - tgt))
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
                self.phase, self.t0 = "cast", self.t
            return S.copy()
        if self.phase == "cast":
            if self.t - self.t0 > 12 and (vmax < self.settle_v or self.t - self.t0 > self.settle_max):
                self.phase, self.ref = "reel", tool.copy()
            return F.copy()
        if self.phase == "reel":
            # tip beyond the target (farther from the tool than the target along the rope's line) -> pull it in
            beyond = float((tip - tgt) @ (tip - tool)) > 0
            if dtip <= self.stop_dist or not beyond or dtip > self.best + 0.01:
                self.phase = "hold"
                return self.ref.copy()
            self.best = min(self.best, dtip)
            g = (tgt - tip) / max(dtip, 1e-9)
            step = min(self.reel_speed / 50.0, dtip)
            new = np.clip(self.ref + g * step, ZLO, ZHI)
            if np.linalg.norm(new - self.ref) < 1e-4:
                self.phase = "hold"
            self.ref = new
            return self.ref.copy()
        return self.ref.copy()


def make_policy(**kw):
    return Policy(**kw)
