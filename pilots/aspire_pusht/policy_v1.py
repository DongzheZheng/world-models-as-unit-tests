"""aspire_style Push-T control program v1 (code-as-policy, hand-written; no forward model).

Rule: choose a contact point on the T and a push direction whose unit wrench (n, (p-c)x n / rho) best aligns with the
desired block twist (pose error), walk the pusher around the block to the pre-contact point, push along n by a depth
proportional to the remaining error. Hysteresis keeps the chosen contact while it stays good.
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
        ("v", 0.0, (-57, 57), (0, 1)),      # bar bottom, push +v
        ("u", 60.0, (3, 27), (-1, 0)),      # bar right end
        ("u", -60.0, (3, 27), (1, 0)),      # bar left end
        ("v", 30.0, (35, 57), (0, -1)),     # bar top right
        ("v", 30.0, (-57, -35), (0, -1)),   # bar top left
        ("u", 15.0, (50, 117), (-1, 0)),    # stem right side
        ("u", -15.0, (50, 117), (1, 0)),    # stem left side
        ("v", 120.0, (-12, 12), (0, -1)),   # stem top
    ]
    P, N, F = [], [], []
    for fi, (ax, val, (a, b), n) in enumerate(faces):
        for s in np.arange(a, b + 1e-6, step):
            P.append((val, s) if ax == "u" else (s, val))
            N.append(n)
            F.append(fi)
    return np.array(P, float), np.array(N, float), np.array(F)


class AspirePolicy:
    def __init__(self, com=(0.0, 40.7), rho=60.0, k_depth=0.4, d_min=2.0, d_max=15.0, v_free=30.0,
                 hyst=0.15, gap=2.0):
        self.com = np.array(com, float)
        self.rho, self.k_depth, self.d_min, self.d_max = rho, k_depth, d_min, d_max
        self.v_free, self.hyst, self.gap = v_free, hyst, gap
        self.P, self.N, self.F = build_candidates()
        self.goal = np.array([256.0, 256.0, np.pi / 4])
        self.reset()

    def bind(self, card):
        self.goal = np.array(card["goal_pose"], float)

    def reset(self):
        self.cur = None          # index of committed contact
        self.mode = "approach"
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
        R_orb = 81.0 + 10.0 + R_PUSHER + 12.0
        rq = q - c
        dq = np.linalg.norm(rq)
        if dq < R_orb - 8.0:
            return c + rq / max(dq, 1e-6) * R_orb
        phq = np.arctan2(rq[1], rq[0])
        rt = tgt - c
        pht = np.arctan2(rt[1], rt[0])
        dph = wrap(pht - phq)
        step = min(abs(dph), self.v_free / R_orb) * np.sign(dph)
        ph = phq + step
        return c + R_orb * np.array([np.cos(ph), np.sin(ph)])

    # ---------------------------------------------------------------- control
    def act(self, obs):
        pose = np.array(obs["block_pose"], float)
        q = np.array(obs["agent_pos"], float)
        s, nt, ep, eth = self.scores(pose)
        best = int(np.argmax(s))
        if self.cur is None or s[self.cur] < 0.2 or s[best] > s[self.cur] + self.hyst:
            if self.cur != best:
                self.mode = "approach"
            self.cur = best
        i = self.cur
        R = rot(pose[2])
        p_w = R @ self.P[i] + pose[:2]
        n_w = R @ self.N[i]
        touch = p_w - n_w * R_PUSHER                      # pusher centre at contact
        along = (q - touch) @ n_w
        lat = q - touch - along * n_w
        if self.mode == "push" and (np.linalg.norm(lat) > 6.0 or along < -20.0):
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
                    tgt = pre if np.linalg.norm(q - pre) <= 10 else q + (pre - q) / np.linalg.norm(pre - q) * 10
            else:
                tgt = self.nav(q, launch, pose)
        if self.mode == "push":
            depth = float(np.clip(self.k_depth * nt * max(s[i], 0.0), self.d_min, self.d_max))
            tgt = touch + n_w * depth
        tgt = np.clip(tgt, WALL_LO, WALL_HI)
        self.trace.append((self.mode, i, float(s[i])))
        return np.asarray(tgt, float)


def make_policy(**kw):
    return AspirePolicy(**kw)
