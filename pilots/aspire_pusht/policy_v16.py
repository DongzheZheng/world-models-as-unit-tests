"""aspire_style Push-T control program v16 (code-as-policy, hand-written; no forward model).

Rule: choose a contact point on the T and a push direction whose unit wrench (n, (p-c)x n / rho) best aligns with the
desired block twist (pose error), walk the pusher around the block to the pre-contact point, push along n by a depth
proportional to the remaining error. Hysteresis keeps the chosen contact while it stays good.
v2 repairs (life 1 trace: 82% of steps spent walking, ~28-step walk per 7-step push, alternating two contacts):
  skill S1 travel-cost in contact choice + persistence while the contact still makes progress;
  skill S2 faster free-space navigation (pusher lags its target by ~60%: command further) and tighter orbit;
  COM from the life-1 rotation regression (1.4, 42.2) ~ nominal (0, 40.7): keep nominal-ish (1, 41.5).
v3 repairs (life 2 trace, success at step 193):
  skill S3 off-centre mass: pooled rotation regression over lives 1-2 puts the COM at u=+5.4 (bootstrap sd 1.9),
     v=42.9 -> COM (5, 42.5) used for the torque sign / lever arm of every contact;
  skill S4 corner slip: pushes within ~7 u of a corner slid off laterally 4 times in a row -> candidate faces trimmed
     away from corners, lateral tolerance 6 -> 8;
  skill S2b pusher response fitted as dq = 0.3 (target - q) + 0.3 v dt -> command further ahead (v_free 90),
     orbit with an 80 u look-ahead on a 112 u circle.
v4 repairs (life 3 trace, success at step 226, 65% of steps walking):
  skill S4b sliding contact: 4 cycles of "push 5 steps / lateral drift > tol / 7-step re-approach" on the same face ->
     while pushing, re-anchor the contact to the pusher's actual position along the same face instead of re-approaching;
  skill S3 update: pooled rotation regression over lives 1-3 (n=184, R2=0.92): COM (3.0, 44.0), sd (1.4, 2.1);
  skill S5 coasting: after release the block keeps moving with per-step decay ~0.63 (~1.7x its last step in total);
     near the goal a push is stopped (pusher backs off) when the remaining error along the contact's wrench is no
     larger than the expected coast, to avoid the overshoot seen at t=222 (0.943 -> 0.916).
v5 repairs (life 4 trace, success at step 169; lives 3-4 spent 83-111 steps going from coverage 0.8 to 0.95, with
  ~2/3 of those steps walking between contacts):
  skill S6 precision phase: when the twist error is small (< 20), weight travel cost more (lam 0.25 -> 0.6) and keep
     the current contact down to alignment 0.3, so fine corrections use nearby faces instead of long walks.
  COM kept at (3, 44) (pooled over 4 lives: (2.6, 43.5), sd (1.4, 1.6)).
v6 (extended run; official evaluation of v5: 10/50, median max coverage 0.86, median 300 steps -> the program runs out
  of time; the 5 official lives all ended just above 0.95 after 105-278 steps, the block standing still during long
  walks): skill S7 tempo -
  free-space commands amplified x2 (pusher closes only ~40% of the gap per step), orbit look-ahead 110 u on a target
  circle 10 u outside the orbit, corridor entry tolerance 4 -> 6 u, corridor steps 1.5x the gap (<= 25 u);
  coarse pushes deeper (k_depth 0.4 -> 0.5, d_max 22 -> 40).
v7 (life x1-2 trace: stuck 272 steps in 'push' on a contact with alignment 0.00, pusher parked 4 u behind it, block
  motionless at coverage 0.82): skill S8 deadlock guard - with the precision travel weight, the travel cost of
  every well-aligned contact exceeded its alignment, so the program kept picking the adjacent point on the same useless
  face (alignment ~0.01) and the coast stop (S5) held the push at -4 u. Repair: contacts below the keep threshold are
  never selected (unless none qualifies), and a stall
  watchdog bans the current contact for 25 steps if the block has not moved for 6 steps while 'pushing'.
v8 (lives x1-3..5 with v7: 3/3 success in 93/115/246 steps; x1-5 lost coverage in pushes started on contacts with
  alignment 0.33-0.55 (0.937 -> 0.85 at t=177-216, 0.78 -> 0.59 at t=84-117)): skill S6b - a NEW contact must have
  alignment >= 0.6 (fallback: best available); an ongoing push may continue down to the keep threshold.
v9 (new object, the L; no L trace yet): skill S9 shape transfer - every rule above is shape-agnostic except the
  polygon, the contact faces and the centre of mass. The program now reads the block shape from the task card
  (bind) and builds the polygon and the reachable faces from it (faces trimmed by the same clearance rules as the T:
  >= ~8 u from convex corners, >= r + 3 from an adjoining concave wall). Centre of mass for the L: the area centroid
  (37.6, 42.6) until an L trace allows the S3 regression. All tuned constants (pusher response, coast factor 1.7,
  thresholds) are carried over unchanged (field note: same batch as the T).
v10 (L life 1 with v9: failure, max coverage 0.945 at t=263; the angle error fell only from +69 to +2.5 deg over 300
  steps in a fixed 4-contact cycle (left side, leg right side, leg top, bottom), each cycle ~60 steps and only 6-10 deg;
  position error was already < 10 u by t=150): skill S10 rotation gain - the rotation regression gives k = 0.00008
  rad per (u of push x u of lever) on the L against 0.00015 on the T with the same tool; per unit of push the L's
  centre of mass translates 0.81 u (T: 0.56) and it turns 0.000068 rad per u*u of lever (T: 0.00014, medians), i.e.
  relative to translation the L turns ~1/1.9 (regression) to ~1/2.9 (medians) as much as the T. The predicted motion
  direction of a contact, which the score compares with the desired twist in the same coverage metric, is therefore
  [n, tau / (g rho)] with g = 2.5 for the L (g = 1 for the T): more turning needs high-lever contacts. COM (S3): the L regression gives (40.0, 51.9), sd (2.3, 2.2), vs the area
  centroid (37.6, 42.6); single-object COM fits proved fragile on the T, so the estimate is shrunk halfway: (39, 47).
  skill S11 relative thresholds: with the corrected motion direction the best achievable alignment on the L is often
  0.55-0.65, so the T's absolute keep/select thresholds (0.45/0.6) would end pushes at once; they become
  min(abs, 0.75 / 0.9 x best achievable), floor 0.2 (unchanged on the T whenever the best alignment is >= 0.67).
  In L life 1 every push ended by falling below the keep threshold (none by slipping), 15 walks of 9-18 steps.
v11 (extended run on the L; official L evaluation of v10: 8/50. L life 2 (seed 29, v10): start eth = +83 deg with the
  body origin only 2 u from its goal at t=19, then 280 steps of short pushes, eth 83 -> 27 deg, coverage <= 0.43):
  skill S12 twist about the centre of mass - the translation error was measured at the body origin (the L's outer
  corner, ~61 u from its COM), so a large angle error looked like 'translation done' although the COM was ~80 u off;
  each push was then judged wrongly. The translation error and the coast estimate now use the COM point
  (goal COM = goal_xy + R(goal_th) com). Rotation phase: while |eth| > 15 deg the translation part of the desired
  twist is weighted 0.5, so pushes that turn the block are not cut short by the translation they cause.
v12 (L life x1-1, seed 29 again, v11: max coverage 0.43 -> 0.78, still a failure; eth 83 -> 19 deg. The first push
  turned the block 19 deg in 9 steps, later pushes only 1-4 deg in 6 steps: push depth was 0.5 x the remaining error
  in coverage units (~8 u -> ~2.4 u/step), while the L needs ~g = 2.5x more push per radian than the T): skill S10b -
  the push depth uses the remaining error with its rotation part scaled by g (push units). COM (S3) from the pooled L
  regression over 3 lives: (37.4, 48.8), sd (1.0, 1.7) -> (37.5, 48.0).
v13 (x1-2, seed 29, v12: success at step 267; x2-1, seed 41, v12: failure 0.862 from eth -135 deg; in both ~80% of
  steps were walks: pushes lasted 2-4 steps and ended because alignment fell below the keep threshold, each followed
  by a 9-22-step walk): skill S13 push commitment - on the L a started push is kept for at least 6 push steps unless
  its alignment falls below 0.1, it is banned, or (rotation phase) its torque no longer has the sign of eth. Turning
  the L needs long pushes; the translation they cause is corrected by the next contact.
v14 (x2-2, seed 41 again, v13: failure 0.882 (v12: 0.862); pushes now ~6 steps. Phase breakdown of the 300 steps:
  76 'radial exit' steps at 10.9 u/step, 37 orbit at 31.5, 48 direct at 29.7, 48 corridor at 8.4, 91 pushing):
  skill S7b tempo of leaving the block - the radial exit commanded a point exactly on the orbit circle (no
  amplification, no tangential progress). Now it commands twice the remaining radial distance and, once beyond
  0.85 of the orbit radius, already advances along the orbit; corridor approach commands 2x (was 1.5x) the gap.
v15 (x3-1, seed 41 a third time, v14: failure but max coverage 0.942 (v13 0.882, v12 0.862); coverage > 0.9 from
  t=231; in the precision phase coverage rose during walks (coasting) and fell during each push: 0.940 -> 0.923,
  0.929 -> 0.919, 0.900 -> 0.888): skill S10c - the g-scaled (push-unit) depth of S10b overshoots near the goal; it is
  used only outside the precision phase, inside it the T-validated depth (0.5 x remaining error, coverage units).
v16 (code-world phase; validation now in the other pilot's code world, worlds/pusht/pusht_world.py, via
  experiments/pusht/code_wm/screen.py): same rules as v15; the per-object constants (COM, rotation gain g, push
  commitment, rotation-phase translation weight and threshold) can be overridden with obj_over={...} so candidate
  programs can be screened; defaults are set from the code-world screening (see the header of the chosen values).
"""
from __future__ import annotations

import numpy as np
import shapely.geometry as sg

R_PUSHER = 15.0
BAR = np.array([(-60, 0), (60, 0), (60, 30), (-60, 30)], float)
STEM = np.array([(-15, 30), (15, 30), (15, 120), (-15, 120)], float)
LEG = np.array([(0, 0), (40, 0), (40, 110), (0, 110)], float)
FOOT = np.array([(40, 0), (100, 0), (100, 40), (40, 40)], float)
SHAPES = {
    "T": dict(rects=(BAR, STEM), com=(3.0, 44.0), faces=[
        ("v", 0.0, (-50, 50), (0, 1)), ("u", 60.0, (6, 24), (-1, 0)), ("u", -60.0, (6, 24), (1, 0)),
        ("v", 30.0, (37, 53), (0, -1)), ("v", 30.0, (-53, -37), (0, -1)), ("u", 15.0, (50, 112), (-1, 0)),
        ("u", -15.0, (50, 112), (1, 0)), ("v", 120.0, (-8, 8), (0, -1))]),
    "L": dict(rects=(LEG, FOOT), com=(37.5, 48.0), rot_gain=2.5, commit=6, faces=[
        ("v", 0.0, (10, 90), (0, 1)),       # bottom (leg + foot), push +v
        ("u", 100.0, (8, 32), (-1, 0)),     # foot right end, push -u
        ("v", 40.0, (60, 92), (0, -1)),     # foot top, push -v (clear of the leg)
        ("u", 40.0, (60, 102), (-1, 0)),    # leg right side, push -u (clear of the foot)
        ("v", 110.0, (8, 32), (0, -1)),     # leg top, push -v
        ("u", 0.0, (8, 102), (1, 0))]),     # leg left side, push +u
}
WALL_LO, WALL_HI = 5.0 + 2.0 + R_PUSHER + 1.0, 506.0 - 2.0 - R_PUSHER - 1.0


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def build_candidates(faces, step=5.0):
    """Body-frame contact points p and push directions n (into the block) on the faces reachable by the disc."""
    P, N, F = [], [], []
    for fi, (ax, val, (a, b), n) in enumerate(faces):
        for s in np.arange(a, b + 1e-6, step):
            P.append((val, s) if ax == "u" else (s, val))
            N.append(n)
            F.append(fi)
    return np.array(P, float), np.array(N, float), np.array(F)


class AspirePolicy:
    R_ORB = 112.0

    def __init__(self, shape="T", com=None, obj_over=None, coast=1.7, rho=60.0, k_depth=0.5, d_min=2.0, d_max=40.0, v_free=120.0, look=110.0, lat_tol=8.0,
                 hyst=0.35, keep=0.45, lam=0.25, lam_prec=0.6, keep_prec=0.3, prec_nt=20.0, select_min=0.6, gap=2.0):
        self.com_override = com
        self.obj_over = obj_over or {}
        self.set_shape(shape)
        self.rho, self.k_depth, self.d_min, self.d_max = rho, k_depth, d_min, d_max
        self.v_free, self.hyst, self.gap = v_free, hyst, gap
        self.keep, self.lam = keep, lam
        self.look, self.lat_tol = look, lat_tol
        self.coast = coast
        self.lam_prec, self.keep_prec, self.prec_nt = lam_prec, keep_prec, prec_nt
        self.select_min = select_min
        self.goal = np.array([256.0, 256.0, np.pi / 4])
        self.reset()

    def set_shape(self, shape):
        sh = SHAPES[shape]
        self.shape, self.rects = shape, sh["rects"]
        self.com = np.array(self.com_override if self.com_override is not None else sh["com"], float)
        self.P, self.N, self.F = build_candidates(sh["faces"])
        sh = dict(sh, **self.obj_over.get(shape, {}))
        if self.com_override is None and "com" in sh:
            self.com = np.array(sh["com"], float)
        self.g = sh.get("rot_gain", 1.0)
        self.commit = sh.get("commit", 0)
        self.wt_rot = sh.get("wt_rot", 0.5)
        self.rot_deg = sh.get("rot_deg", 15.0)
        self.s10c = sh.get("s10c", True)
        for k in ("k_depth", "d_max", "d_min", "v_free", "look", "lat_tol", "keep", "keep_prec", "select_min", "hyst",
                  "lam", "lam_prec", "prec_nt", "coast", "rho"):
            if k in sh:
                setattr(self, k, sh[k])

    def bind(self, card):
        self.goal = np.array(card["goal_pose"], float)
        self.set_shape(card.get("object", card.get("block", {}).get("shape", "T")))

    def reset(self):
        self.cur = None          # index of committed contact
        self.mode = "approach"
        self.prev_pose = None
        self.still = 0
        self.push_steps = 0
        self.ban = {}
        self.t = 0
        self.trace = []          # per-step diagnostics (kept in memory)

    # ---------------------------------------------------------------- geometry
    def poly(self, pose, buf=0.0):
        R = rot(pose[2])
        g = sg.MultiPolygon([sg.Polygon(v @ R.T + pose[:2]) for v in self.rects]).buffer(0)
        return g.buffer(buf) if buf else g

    def scores(self, pose):
        R = rot(pose[2])
        eth = wrap(self.goal[2] - pose[2])
        ep = (self.goal[:2] + rot(self.goal[2]) @ self.com) - (pose[:2] + R @ self.com)   # S12: COM error
        n_w = self.N @ R.T
        lever = self.P - self.com                      # body frame; torque is frame invariant
        tau = lever[:, 0] * self.N[:, 1] - lever[:, 1] * self.N[:, 0]
        w = np.c_[n_w, tau / (self.rho * self.g)]
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        wt = self.wt_rot if abs(eth) > np.radians(self.rot_deg) else 1.0
        tw = np.r_[wt * ep, self.rho * eth]
        nt = np.linalg.norm(np.r_[ep, self.rho * eth])
        s = w @ tw / max(np.linalg.norm(tw), 1e-9)
        # feasibility: pre-contact pusher centre must be inside the walls
        pre = (self.P - self.N * (R_PUSHER + self.gap)) @ R.T + pose[:2]
        ok = np.all((pre > WALL_LO) & (pre < WALL_HI), axis=1)
        s = np.where(ok, s, -2.0)
        return s, nt, ep, eth

    def nav(self, q, tgt, pose):
        """Next pusher target moving from q towards tgt without crossing the block."""
        body = self.poly(pose, R_PUSHER + 3.0)
        seg = sg.LineString([q, tgt]) if np.linalg.norm(tgt - q) > 1e-6 else sg.Point(q)
        if not seg.intersects(body):
            d = tgt - q
            L = np.linalg.norm(d)
            return q + d / max(L, 1e-9) * min(2.0 * L, self.v_free)
        c = np.array(self.poly(pose).centroid.coords[0])
        R_orb = self.R_ORB
        rq = q - c
        dq = np.linalg.norm(rq)
        rt = tgt - c
        pht = np.arctan2(rt[1], rt[0])
        if dq < R_orb - 8.0:                     # S7b: leave fast, start turning once clear of the body
            phq = np.arctan2(rq[1], rq[0])
            r_cmd = dq + 2.0 * (R_orb - dq)
            if dq > 0.85 * R_orb:
                dph = wrap(pht - phq)
                phq = phq + np.sign(dph) * min(abs(dph), 0.5 * self.look / R_orb)
            return c + r_cmd * np.array([np.cos(phq), np.sin(phq)])
        phq = np.arctan2(rq[1], rq[0])
        rt = tgt - c
        pht = np.arctan2(rt[1], rt[0])
        dph = wrap(pht - phq)
        step = min(abs(dph), self.look / R_orb) * np.sign(dph)
        ph = phq + step
        return c + (R_orb + 10.0) * np.array([np.cos(ph), np.sin(ph)])

    # ---------------------------------------------------------------- control
    def act(self, obs):
        pose = np.array(obs["block_pose"], float)
        q = np.array(obs["agent_pos"], float)
        s, nt, ep, eth = self.scores(pose)
        R = rot(pose[2])
        # S4b: re-anchor a sliding contact along its face while pushing
        if self.mode == "push" and self.cur is not None:
            qb = R.T @ (q - pose[:2])
            same = np.flatnonzero(self.F == self.F[self.cur])
            dd = np.linalg.norm(qb - (self.P[same] - self.N[same] * R_PUSHER), axis=1)
            j = same[int(np.argmin(dd))]
            if dd.min() < self.lat_tol + 4.0:
                self.cur = int(j)
        # S1: travel cost (arc around the block centroid + radial distance) to reach each pre-contact point
        R = rot(pose[2])
        pre_all = (self.P - self.N * (R_PUSHER + self.gap)) @ R.T + pose[:2]
        c = pose[:2] + R @ self.com
        aq = np.arctan2(*(q - c)[::-1])
        ap = np.arctan2(pre_all[:, 1] - c[1], pre_all[:, 0] - c[0])
        travel = self.R_ORB * np.abs(wrap(ap - aq)) + np.abs(np.linalg.norm(q - c) - np.linalg.norm(pre_all - c, axis=1))
        prec = nt < self.prec_nt
        sc = s - (self.lam_prec if prec else self.lam) * travel / 100.0
        if self.cur is not None and self.mode == "push":
            sc[self.cur] = s[self.cur]            # already there: no travel cost
        # S8: stall watchdog and forced switch away from a contact below its keep threshold
        if self.prev_pose is not None and self.mode == "push":
            moved = np.linalg.norm(pose[:2] - self.prev_pose[:2]) + 50 * abs(wrap(pose[2] - self.prev_pose[2]))
            self.still = self.still + 1 if moved < 0.2 else 0
            if self.still >= 6 and self.cur is not None:
                self.ban[self.cur] = self.t + 25
                self.still = 0
        else:
            self.still = 0
        for j, until in list(self.ban.items()):
            if self.t < until:
                sc[j] = -np.inf
            else:
                del self.ban[j]
        smax = float(np.max(s))                   # S11: thresholds relative to the best achievable alignment
        kth = max(0.2, min(self.keep_prec if prec else self.keep, 0.75 * smax))
        sel = max(0.2, min(self.select_min, 0.9 * smax))
        cur_ok = self.cur is not None and self.cur not in self.ban and s[self.cur] >= kth
        cur_sc = sc[self.cur] if cur_ok else None
        if np.any(s >= sel):                      # S6b: new contacts must be well aligned
            sc = np.where(s >= sel, sc, -np.inf)
        elif np.any(s >= kth):                    # never select a contact that is below the keep threshold
            sc = np.where(s >= kth, sc, -np.inf)
        if cur_ok:
            sc[self.cur] = cur_sc                 # an ongoing contact may continue down to the keep threshold
        best = int(np.argmax(sc))
        committed = False
        if self.mode == "push" and self.cur is not None and self.push_steps < self.commit and self.cur not in self.ban \
                and s[self.cur] > 0.1:            # S13: push commitment
            lv = self.P[self.cur] - self.com
            tau_c = lv[0] * self.N[self.cur][1] - lv[1] * self.N[self.cur][0]
            committed = abs(eth) <= np.radians(self.rot_deg) or tau_c * eth > 0
        if committed:
            pass
        elif self.cur is None or self.cur in self.ban or s[self.cur] < kth or sc[best] > sc[self.cur] + self.hyst:
            if self.cur != best:
                self.mode = "approach"
            self.cur = best
        i = self.cur
        p_w = R @ self.P[i] + pose[:2]
        n_w = R @ self.N[i]
        touch = p_w - n_w * R_PUSHER                      # pusher centre at contact
        along = (q - touch) @ n_w
        lat = q - touch - along * n_w
        if self.mode == "push" and (np.linalg.norm(lat) > self.lat_tol or along < -20.0):
            self.mode = "approach"
        if self.mode == "approach":
            pre = touch - n_w * self.gap
            launch = touch - n_w * 25.0
            in_corridor = np.linalg.norm(lat) < 6.0 and -30.0 < along < 2.0
            if in_corridor:
                if np.linalg.norm(q - pre) < 5.0:
                    self.mode = "push"
                    tgt = None
                else:
                    dpre = np.linalg.norm(pre - q)
                    tgt = q + (pre - q) / dpre * min(2.0 * dpre, 25.0)
            else:
                tgt = self.nav(q, launch, pose)
        if self.mode == "push":
            rem = nt * max(s[i], 0.0)                     # remaining error along this contact's wrench
            remp = np.linalg.norm(np.r_[ep, self.rho * self.g * eth]) * max(s[i], 0.0)   # S10b: in push units
            depth = float(np.clip(self.k_depth * (rem if (prec and self.s10c) else remp), self.d_min, self.d_max))   # S10c
            if self.prev_pose is not None and nt < 25.0:  # S5: expected coast vs remaining error
                w = np.r_[n_w, ((self.P[i] - self.com) @ np.array([self.N[i][1], -self.N[i][0]])) / (self.rho * self.g)]
                w /= np.linalg.norm(w)
                dcom = pose[:2] + R @ self.com - self.prev_pose[:2] - rot(self.prev_pose[2]) @ self.com
                mv = w @ np.r_[dcom, self.rho * wrap(pose[2] - self.prev_pose[2])]
                if rem <= self.coast * max(mv, 0.0) + 0.5:
                    depth = -4.0
            tgt = touch + n_w * depth
        tgt = np.clip(tgt, WALL_LO, WALL_HI)
        self.prev_pose = pose
        self.push_steps = self.push_steps + 1 if self.mode == "push" else 0
        self.t += 1
        self.trace.append((self.mode, i, float(s[i])))
        return np.asarray(tgt, float)


def make_policy(**kw):
    return AspirePolicy(**kw)
