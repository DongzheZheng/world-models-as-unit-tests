"""aspire_style Push-T control program v4 (code-as-policy, hand-written; no forward model).

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
"""
from __future__ import annotations

import numpy as np
import shapely.geometry as sg

R_PUSHER = 15.0
BAR = np.array([(-60, 0), (60, 0), (60, 30), (-60, 30)], float)
STEM = np.array([(-15, 30), (15, 30), (15, 120), (-15, 120)], float)
WALL_LO, WALL_HI = 5.0 + 2.0 + R_PUSHER + 1.0, 506.0 - 2.0 - R_PUSHER - 1.0


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def build_candidates(step=5.0):
    """Body-frame contact points p and push directions n (into the block), on the T faces reachable by the disc."""
    faces = [  # (fixed coord axis, fixed value, varying range, push dir)
        ("v", 0.0, (-50, 50), (0, 1)),      # bar bottom, push +v
        ("u", 60.0, (6, 24), (-1, 0)),      # bar right end
        ("u", -60.0, (6, 24), (1, 0)),      # bar left end
        ("v", 30.0, (37, 53), (0, -1)),     # bar top right
        ("v", 30.0, (-53, -37), (0, -1)),   # bar top left
        ("u", 15.0, (50, 112), (-1, 0)),    # stem right side
        ("u", -15.0, (50, 112), (1, 0)),    # stem left side
        ("v", 120.0, (-8, 8), (0, -1)),   # stem top
    ]
    P, N, F = [], [], []
    for fi, (ax, val, (a, b), n) in enumerate(faces):
        for s in np.arange(a, b + 1e-6, step):
            P.append((val, s) if ax == "u" else (s, val))
            N.append(n)
            F.append(fi)
    return np.array(P, float), np.array(N, float), np.array(F)


class AspirePolicy:
    R_ORB = 112.0

    def __init__(self, com=(3.0, 44.0), coast=1.7, rho=60.0, k_depth=0.4, d_min=2.0, d_max=22.0, v_free=90.0, look=80.0, lat_tol=8.0,
                 hyst=0.35, keep=0.45, lam=0.25, gap=2.0):
        self.com = np.array(com, float)
        self.rho, self.k_depth, self.d_min, self.d_max = rho, k_depth, d_min, d_max
        self.v_free, self.hyst, self.gap = v_free, hyst, gap
        self.keep, self.lam = keep, lam
        self.look, self.lat_tol = look, lat_tol
        self.coast = coast
        self.P, self.N, self.F = build_candidates()
        self.goal = np.array([256.0, 256.0, np.pi / 4])
        self.reset()

    def bind(self, card):
        self.goal = np.array(card["goal_pose"], float)

    def reset(self):
        self.cur = None          # index of committed contact
        self.mode = "approach"
        self.prev_pose = None
        self.trace = []          # per-step diagnostics (kept in memory)

    # ---------------------------------------------------------------- geometry
    def poly(self, pose, buf=0.0):
        R = rot(pose[2])
        g = sg.MultiPolygon([sg.Polygon(v @ R.T + pose[:2]) for v in (BAR, STEM)]).buffer(0)
        return g.buffer(buf) if buf else g

    def scores(self, pose):
        R = rot(pose[2])
        ep = self.goal[:2] - pose[:2]
        eth = wrap(self.goal[2] - pose[2])
        n_w = self.N @ R.T
        lever = self.P - self.com                      # body frame; torque is frame invariant
        tau = lever[:, 0] * self.N[:, 1] - lever[:, 1] * self.N[:, 0]
        w = np.c_[n_w, tau / self.rho]
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        tw = np.r_[ep, self.rho * eth]
        nt = np.linalg.norm(tw)
        s = w @ tw / max(nt, 1e-9)
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
            return tgt if L <= self.v_free else q + d / L * self.v_free
        c = np.array(self.poly(pose).centroid.coords[0])
        R_orb = self.R_ORB
        rq = q - c
        dq = np.linalg.norm(rq)
        if dq < R_orb - 8.0:
            return c + rq / max(dq, 1e-6) * R_orb
        phq = np.arctan2(rq[1], rq[0])
        rt = tgt - c
        pht = np.arctan2(rt[1], rt[0])
        dph = wrap(pht - phq)
        step = min(abs(dph), self.look / R_orb) * np.sign(dph)
        ph = phq + step
        return c + R_orb * np.array([np.cos(ph), np.sin(ph)])

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
        sc = s - self.lam * travel / 100.0
        if self.cur is not None and self.mode == "push":
            sc[self.cur] = s[self.cur]            # already there: no travel cost
        best = int(np.argmax(sc))
        if self.cur is None or s[self.cur] < self.keep or sc[best] > sc[self.cur] + self.hyst:
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
            in_corridor = np.linalg.norm(lat) < 4.0 and -30.0 < along < 2.0
            if in_corridor:
                if np.linalg.norm(q - pre) < 4.0:
                    self.mode = "push"
                    tgt = None
                else:
                    tgt = pre if np.linalg.norm(q - pre) <= 15 else q + (pre - q) / np.linalg.norm(pre - q) * 15
            else:
                tgt = self.nav(q, launch, pose)
        if self.mode == "push":
            rem = nt * max(s[i], 0.0)                     # remaining error along this contact's wrench
            depth = float(np.clip(self.k_depth * rem, self.d_min, self.d_max))
            if self.prev_pose is not None and nt < 25.0:  # S5: expected coast vs remaining error
                w = np.r_[n_w, ((self.P[i] - self.com) @ np.array([self.N[i][1], -self.N[i][0]])) / self.rho]
                w /= np.linalg.norm(w)
                mv = w @ np.r_[pose[:2] - self.prev_pose[:2], self.rho * wrap(pose[2] - self.prev_pose[2])]
                if rem <= self.coast * max(mv, 0.0) + 0.5:
                    depth = -4.0
            tgt = touch + n_w * depth
        tgt = np.clip(tgt, WALL_LO, WALL_HI)
        self.prev_pose = pose
        self.trace.append((self.mode, i, float(s[i])))
        return np.asarray(tgt, float)


def make_policy(**kw):
    return AspirePolicy(**kw)
