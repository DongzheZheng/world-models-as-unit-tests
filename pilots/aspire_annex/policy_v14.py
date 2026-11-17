"""policy_v14 (extended run): v9 geometry (clearance 0.035) with stroke 0.25 m plus a 'soft catch'.
Seed-62 bracket: L 0.23 -> tip stalls 0.136 m from the target against the post's side; L 0.27 -> tip reaches the target
at 4.5 m/s and the cord snaps taut (chord ~0.61) and recoils.  L 0.25 sits between.  Soft catch (feedback on live
readings): once the tip is ahead of the tool, the chord exceeds 0.52 m and the tip still moves outward faster than
1 m/s, the tool gives forward along the cast line by up to 0.06 m (following the tip), so the snap is softened instead
of recoiling; then it holds.  v9 doc follows: v8 but the cord's lie is measured by the tool->centroid direction, not the near part.
Life x1-2 (v8): near-part misalignment 1.9 deg but centroid misalignment +26.5 deg -> resting direction +139 deg off;
across lives 2, 5, x1-2 the sign of the resting deviation follows the sign of the CENTROID misalignment.
v8 doc follows: v6/v7 snap-back cast + choose the cast line by the cord's near-part lie.
Fit so far (lives 2, 5): the resting direction deviates from the stroke direction d in the SAME sense as the signed
misalignment mis = near-part direction - (d + 180 deg) (life 5: mis -25 -> delta -56).  v8 therefore picks, among the
geometrically valid casts (parked point F with |A-F| = reach, line beside the post, stroke room >= 0.15 m), the one whose
d is most opposite to the near part (measured live: p20 -> p14), re-plans once on arrival with the re-measured lie, and
optionally pre-compensates with a closed-form rule  rest_dir ~= d + k*mis  (k from the life logs; k=0 = no compensation)."""
import numpy as np

ZLO, ZHI = np.array([0.30, -0.20]), np.array([0.55, 0.20])


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def near_dir(pts):
    v = pts.mean(0) - pts[19]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def candidates(target, post, reach=0.66, clear=0.035, aim_offs=(-0.06, -0.03, 0.0, 0.03, 0.06), stroke_max=0.25, Lmin=0.22):
    radial = (target - post) / np.linalg.norm(target - post)
    lat = np.array([-radial[1], radial[0]])
    out = []
    for off in aim_offs:
        A = target + off * lat
        for angd in np.arange(-80, 80.1, 1.0):
            d = np.array([np.cos(np.radians(angd)), np.sin(np.radians(angd))])
            F = A - reach * d
            if np.any(F < ZLO) or np.any(F > ZHI):
                continue
            c = seg_dist(post, F, A)
            if c < clear:
                continue
            L = stroke_max
            while L > 0.05 and (np.any(F + L * d > ZHI) or np.any(F + L * d < ZLO)):
                L -= 0.01
            if L < Lmin:
                continue
            out.append(dict(F=F, G=F + L * d, A=A, d=d, angd=float(angd), L=L, clear=c, off=off))
    return out


def choose(cands, near, k=0.0, here=None, max_move=None):
    """Pick the cast whose PREDICTED resting direction (stroke dir + k*mis) points at its aim point, preferring small
    |mis| and long strokes.  With k=0 this is 'stroke most opposite to the near part'."""
    best, bv = None, -1e9
    for c in cands:
        if here is not None and max_move is not None and np.linalg.norm(c["F"] - here) > max_move:
            continue
        mis = wrapd(near - (c["angd"] + 180.0))
        # stroke direction is chosen so that predicted rest = aim line direction; with k>0 we steer the stroke instead
        v = -abs(mis) * (1.0 if k == 0 else 0.3) + 40 * min(c["L"], 0.27) - 30 * abs(c["off"])
        if v > bv:
            bv, best = v, dict(c, mis=mis)
    if best is not None and k != 0:
        # steer: stroke direction s such that s + k*wrap(near - s - 180) = aim line direction (solved in closed form)
        aim = best["angd"]
        s = (aim - k * wrapd(near - 180.0 - aim) - k * aim) / (1 - k) if abs(1 - k) > 1e-6 else aim
        s = aim + float(np.clip(wrapd(s - aim), -35, 35))
        ds = np.array([np.cos(np.radians(s)), np.sin(np.radians(s))])
        L = best["L"]
        while L > 0.05 and (np.any(best["F"] + L * ds > ZHI) or np.any(best["F"] + L * ds < ZLO)):
            L -= 0.01
        best.update(G=best["F"] + L * ds, d=ds, L=L, stroke_deg=s)
    return best


class Policy:
    def __init__(self, prep_speed=0.5, pause=8, switch_frac=0.55, k=0.0, replan=True, give=0.06, give_chord=0.52, **kw):
        self.give, self.give_chord = give, give_chord
        self.kw, self.prep_speed, self.pause, self.switch_frac, self.k, self.replan = kw, prep_speed, pause, switch_frac, k, replan
        self.reset()

    def reset(self):
        self.p, self.phase, self.ref, self.t, self.cands, self.replanned = None, "prep", None, 0, None, False

    def act(self, obs):
        tool = np.asarray(obs["tool_pos"], float)
        pts = np.asarray(obs["points"])[:, :2]
        self.t += 1
        if self.p is None:
            self.cands = candidates(np.asarray(obs["target"], float), np.asarray(obs["post"], float), **self.kw)
            if not self.cands:
                self.cands = candidates(np.asarray(obs["target"], float), np.asarray(obs["post"], float), Lmin=0.12, **self.kw)
            self.p = choose(self.cands, near_dir(pts), self.k)
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
                self.ref = F.copy()
                if self.replan and not self.replanned and self.cands:
                    self.replanned = True
                    q = choose(self.cands, near_dir(pts), self.k, here=F, max_move=0.08)
                    if q is not None and np.linalg.norm(q["F"] - F) > 0.005:
                        self.p = q
                        return self.ref.copy()
                self.phase, self.t0 = "pause", self.t
            else:
                self.ref = self.ref + dv / n * step
            return self.ref.copy()
        if self.phase == "pause":
            if self.t - self.t0 >= self.pause:
                self.phase = "out"
            return F.copy()
        if self.phase == "out":
            if float((tool - F) @ self.p["d"]) >= self.switch_frac * self.p["L"]:
                self.phase, self.hold, self.prevtip = "back", F.copy(), pts[0].copy()
                return F.copy()
            return G.copy()
        if self.phase == "back":
            tip = pts[0]
            rel = tip - tool
            chord = float(np.linalg.norm(rel))
            vout = float((tip - self.prevtip) @ (rel / max(chord, 1e-9))) * 50.0
            self.prevtip = tip.copy()
            ahead = float(rel @ self.p["d"]) > 0.3
            if np.linalg.norm(tool - F) < 0.01 and ahead and chord > self.give_chord and vout > 1.0:
                u = rel / chord
                self.hold = np.clip(self.hold + min(0.02, max(0.0, self.give - np.linalg.norm(self.hold - F))) * u, ZLO, ZHI)
            return self.hold.copy()
        return F.copy()


def make_policy(**kw):
    return Policy(**kw)
