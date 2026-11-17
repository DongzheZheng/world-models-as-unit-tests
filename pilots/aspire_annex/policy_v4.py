"""policy_v4: swing-into-the-post.  Lives 1-2 showed the cord moves as a stiff, straight body under a smooth fast pull (all
points at 3.5 m/s, chord 0.698), but an abrupt stop of an AXIAL pull buckles it and kills the energy within 0.1 s
(KE 12.8 -> 1.4), while the sideways part of the motion turns into a swing about the tool (life 2: ~110 deg rotation).
So v4 throws the cord into a rotation about the parked tool and uses the post as the stop: park the tool at T where the
line from T tangent to the post (cord radius included) carries a straight cord's tip (0.68 m) to the target; strike
perpendicular to the cord's current lie, in the rotation sense that reaches the post the short way, with the longest
stroke that ends at T; hold T; optional slow reel-in if the tip rests beyond the target."""
import numpy as np

ZLO, ZHI = np.array([0.30, -0.20]), np.array([0.55, 0.20])
RC = 0.04          # post radius + cord radius
LR = 0.68          # chord of a resting, nearly straight cord


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


def rope_dir(pts, tool):
    w = np.linspace(0.5, 1.5, 20)       # weight the far part a little more
    c = (pts * w[:, None]).sum(0) / w.sum() - tool
    return float(np.arctan2(c[1], c[0]))


def plan(target, post, tool0, r, lr=LR, stroke_max=0.34, w_rot=0.1, perp_tol=np.radians(40)):
    best, bestv = None, -1e9
    xs, ys = np.arange(0.30, 0.5501, 0.01), np.arange(-0.20, 0.2001, 0.01)
    for x in xs:
        for y in ys:
            T = np.array([x, y])
            dv = post - T
            dist = np.linalg.norm(dv)
            if dist <= RC + 0.05 or dist > lr - 0.02:
                continue
            thc = np.arctan2(dv[1], dv[0])
            for s in (-1, 1):
                th = thc + s * np.arcsin(RC / dist)
                tip = T + lr * np.array([np.cos(th), np.sin(th)])
                err = float(np.linalg.norm(tip - target))
                if err > 0.06:
                    continue
                sigma = -s                                   # rotation sense that meets the post on side s
                rot = (sigma * (th - r)) % (2 * np.pi)      # rotation needed (rad, >0)
                # stroke direction: perpendicular to the cord, sense sigma; allow deviation to gain stroke length
                bestL, bestm = 0, None
                for dev in np.radians(np.arange(-40, 41, 5)):
                    m_ang = r + sigma * np.pi / 2 + dev
                    m = np.array([np.cos(m_ang), np.sin(m_ang)])
                    if float(m @ np.array([np.cos(r), np.sin(r)])) > 0.15:    # do not strike towards the cord
                        continue
                    L = stroke_max
                    while L > 0.05 and (np.any(T - L * m < ZLO - 1e-9) or np.any(T - L * m > ZHI + 1e-9)):
                        L -= 0.01
                    val = L * np.cos(dev)                      # perpendicular stroke length
                    if val > bestL:
                        bestL, bestm, bestLL = val, m, L
                if bestm is None:
                    continue
                S = T - bestLL * bestm
                slack = max(0.0, float((S - tool0) @ np.array([np.cos(r), np.sin(r)])))
                v = -10 * err + 3 * min(bestL, 0.3) - w_rot * np.degrees(rot) * 0.1 - 2 * slack
                if v > bestv:
                    bestv = v
                    best = dict(T=T, S=S, m=bestm, L=bestLL, Lperp=bestL, side=s, sigma=sigma, rot=rot, err=err,
                                th=th, tip=tip, slack=slack)
    return best


class Policy:
    def __init__(self, prep_speed=0.5, pause=10, reel=True, reel_speed=0.1, settle_steps=90, stop_dist=0.04, **kw):
        self.kw = kw
        self.prep_speed, self.pause, self.reel, self.reel_speed = prep_speed, pause, reel, reel_speed
        self.settle_steps, self.stop_dist = settle_steps, stop_dist
        self.reset()

    def reset(self):
        self.t, self.p, self.phase, self.ref, self.prev, self.best = 0, None, "prep", None, None, 9.0

    def act(self, obs):
        pts = np.asarray(obs["points"])[:, :2]
        tool = np.asarray(obs["tool_pos"], float)
        tgt = np.asarray(obs["target"], float)
        if self.p is None:
            self.p = plan(tgt, np.asarray(obs["post"]), tool, rope_dir(pts, tool), **self.kw)
            if self.p is None:          # no geometric solution: park near the zone front facing the target
                T = np.clip(tgt - LR * (tgt - tool) / np.linalg.norm(tgt - tool), ZLO, ZHI)
                self.p = dict(T=T, S=tool.copy())
            self.ref = tool.copy()
        vmax = 0.0 if self.prev is None else float(np.linalg.norm(pts - self.prev, axis=1).max() * 50)
        self.prev = pts.copy()
        T, S = self.p["T"], self.p["S"]
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
                self.phase, self.t0 = "strike", self.t
            return S.copy()
        if self.phase == "strike":
            if self.t - self.t0 > self.settle_steps or (self.t - self.t0 > 30 and vmax < 0.05):
                self.phase, self.ref = ("reel" if self.reel else "hold"), T.copy()
            return T.copy()
        if self.phase == "reel":
            beyond = float((tip - tgt) @ (tip - tool)) > 0
            if dtip <= self.stop_dist or not beyond or dtip > self.best + 0.01:
                self.phase = "hold"
                return self.ref.copy()
            self.best = min(self.best, dtip)
            g = (tgt - tip) / max(dtip, 1e-9)
            new = np.clip(self.ref + g * min(self.reel_speed / 50.0, dtip), ZLO, ZHI)
            if np.linalg.norm(new - self.ref) < 1e-4:
                self.phase = "hold"
            self.ref = new
            return self.ref.copy()
        return self.ref.copy()


def make_policy(**kw):
    return Policy(**kw)
