"""Internal (lab-only): stochastic patrol aircraft for mission C.  Never to be read by the world-model author.

Design (all values are lab truth):
  5-8 movers (radius 0.35 m, kinematic spheres), each drawn at random from a small zoo of state machines:
    along      patrol along the canyon on a fixed lane, bouncing between random ends, random pauses
    across     shuttle across the canyon (or in/out of a side-aisle mouth) at a fixed x, random pauses at the ends
    dart       hover for an exponential dwell, then dart (3-6 m/s) to a random point nearby
    reactive   patrol; when the pilot comes within a trigger radius it blocks (tracks the pilot's lateral/vertical
               position at its own x), chases (approaches in 3-D) or retreats along the canyon (fixed per mover)
  rare events (per mover per episode): 5 % a dash (x3.5 speed for 0.8 s), 5 % a sudden reversal, at a random time.
Own rng stream, so missions A/B are untouched.  Movers ignore the wind.
"""
from __future__ import annotations

import numpy as np

RADIUS = 0.35
X_LO, X_HI = 10.0, 50.0
Y_LIM = 1.55
Z_LO, Z_HI = 1.2, 4.2
GAPS = {"N": (22.3, 25.2, 1), "S": (38.3, 41.2, -1)}     # side-aisle mouths (x0, x1, side)
P_DASH, P_REV = 0.05, 0.05
KINDS = ["along", "across", "dart", "reactive"]
KIND_P = [0.25, 0.25, 0.20, 0.30]
REACT = ["block", "chase", "shy"]
REACT_P = [0.40, 0.30, 0.30]


class Movers:
    def __init__(self, seed: int, relays: np.ndarray, start: np.ndarray, n_range=(7, 8)):
        self.rng = np.random.default_rng([int(seed), 4242])
        r = self.rng
        self.n = int(r.integers(n_range[0], n_range[1] + 1))
        self.relays, self.start = np.asarray(relays, float), np.asarray(start, float)
        self.m = []
        for i in range(self.n):
            for _ in range(200):
                s = self._draw()
                if np.linalg.norm(s["pos"] - self.start) > 6.0 and all(
                        np.linalg.norm(s["pos"] - q) > 2.5 for q in self.relays):
                    break
            self.m.append(s)
        self.t = 0.0
        self.prev_pos = self.pos.copy()

    # ------------------------------------------------------------------ construction
    @property
    def pos(self):
        return np.array([s["pos"] for s in self.m])

    @property
    def vel(self):
        return np.array([s["vel"] for s in self.m])

    def _draw(self):
        r = self.rng
        kind = str(r.choice(KINDS, p=KIND_P))
        s = dict(kind=kind, vel=np.zeros(3), pause=0.0, mode="patrol", dash_t=-1.0, dash_left=0.0)
        z = float(r.uniform(1.6, 3.6))
        sp = float(np.clip(r.lognormal(np.log(1.7), 0.4), 0.7, 5.0))
        if kind in ("along", "reactive"):
            L = r.uniform(8.0, 22.0)
            x0 = r.uniform(X_LO, X_HI - L)
            s.update(lane=float(r.uniform(-1.2, 1.2)), z0=z, xr=(x0, x0 + L), speed=sp, dir=float(r.choice([-1, 1])))
            s["pos"] = np.array([r.uniform(*s["xr"]), s["lane"], z])
            if kind == "reactive":
                s.update(react=str(r.choice(REACT, p=REACT_P)), trig=float(r.uniform(2.8, 4.0)),
                         axes=str(r.choice(["y", "z", "yz"], p=[0.35, 0.3, 0.35])))
        elif kind == "across":
            if r.random() < 0.5:
                side = "N" if r.random() < 0.5 else "S"
                g = GAPS[side]
                x = r.uniform(g[0] + 0.3, g[1] - 0.3)
                far = 6.5
                yr = (-Y_LIM, far) if g[2] > 0 else (-far, Y_LIM)
            else:
                x = r.uniform(X_LO + 1, X_HI - 1)
                yr = (-Y_LIM, Y_LIM)
            s.update(x0=float(x), yr=yr, z0=z, speed=float(sp * 0.8), dir=float(r.choice([-1, 1])))
            s["pos"] = np.array([x, r.uniform(*yr), z])
        else:   # dart
            s.update(dwell=float(r.exponential(1.6) + 0.3), speed=float(r.uniform(3.0, 6.0)), target=None)
            s["pos"] = np.array([r.uniform(X_LO, X_HI), r.uniform(-Y_LIM, Y_LIM), z])
        if r.random() < P_DASH:
            s["dash_t"] = float(r.uniform(2.0, 14.0))
        if r.random() < P_REV:
            s["rev_t"] = float(r.uniform(2.0, 14.0))
        return s

    # ------------------------------------------------------------------ dynamics
    @staticmethod
    def _clip(p):
        p = p.copy()
        p[1] = np.clip(p[1], -Y_LIM, Y_LIM)
        p[2] = np.clip(p[2], Z_LO, Z_HI)
        return p

    def step(self, dt: float, pilot_pos: np.ndarray):
        r = self.rng
        self.prev_pos = self.pos.copy()
        self.t += dt
        for s in self.m:
            v = np.zeros(3)
            k = s["kind"]
            p = s["pos"]
            if s["dash_t"] >= 0 and self.t >= s["dash_t"]:
                s["dash_t"] = -1.0
                s["dash_left"] = 0.8
            if s.get("rev_t", -1) >= 0 and self.t >= s["rev_t"]:
                s["rev_t"] = -1.0
                if "dir" in s:
                    s["dir"] *= -1
                if k == "dart":
                    s["target"] = None
                    s["dwell"] = 0.0
                if k == "reactive":
                    s["react"] = {"block": "chase", "chase": "shy", "shy": "block"}[s["react"]]
            boost = 3.5 if s["dash_left"] > 0 else 1.0
            s["dash_left"] = max(0.0, s["dash_left"] - dt)
            if s["pause"] > 0:
                s["pause"] -= dt
            elif k in ("along", "reactive"):
                if "seen" not in s or self.t - s["seen_t"] >= 0.4:      # reaction time: the pilot is re-observed every 0.4 s
                    s["seen"], s["seen_t"] = np.asarray(pilot_pos, float).copy(), self.t
                pilot_pos = s["seen"]
                d = np.asarray(pilot_pos) - p
                dist = np.linalg.norm(d)
                if k == "reactive" and dist < s["trig"]:
                    s["mode"] = s["react"]
                elif k == "reactive" and dist > 1.3 * s["trig"]:
                    s["mode"] = "patrol"
                if s["mode"] == "block":
                    tgt = np.array([p[0], pilot_pos[1] if "y" in s["axes"] else p[1], pilot_pos[2] if "z" in s["axes"] else p[2]])
                    dd = tgt - p
                    v = 1.3 * dd / max(np.linalg.norm(dd), 0.3)
                elif s["mode"] == "chase":
                    v = 1.5 * d / max(dist, 0.3)
                elif s["mode"] == "shy":
                    away = np.array([np.sign(p[0] - pilot_pos[0]) or 1.0, 0.0, 0.0])
                    v = 3.5 * away + np.array([0.0, 0.5 * np.sign(p[1] - pilot_pos[1] + 1e-6), 0.0])
                else:
                    v = np.array([s["dir"] * s["speed"] * boost, 0.0, 0.0])
                    tgt_yz = np.array([s["lane"], s["z0"]])
                    v[1:] = 1.5 * (tgt_yz - p[1:])
                    if (p[0] >= s["xr"][1] and s["dir"] > 0) or (p[0] <= s["xr"][0] and s["dir"] < 0):
                        s["dir"] *= -1
                        v[0] *= -1
                        if r.random() < 0.3:
                            s["pause"] = float(r.uniform(0.3, 2.0))
            elif k == "across":
                v = np.array([0.0, s["dir"] * s["speed"] * boost, 1.5 * (s["z0"] - p[2])])
                v[0] = 1.5 * (s["x0"] - p[0])
                if (p[1] >= s["yr"][1] and s["dir"] > 0) or (p[1] <= s["yr"][0] and s["dir"] < 0):
                    s["dir"] *= -1
                    v[1] *= -1
                    s["pause"] = float(r.uniform(0.0, 1.5))
            else:   # dart
                if s["target"] is None:
                    s["dwell"] -= dt
                    if s["dwell"] <= 0:
                        s["target"] = np.array([np.clip(p[0] + r.uniform(-8, 8), X_LO, X_HI), r.uniform(-Y_LIM, Y_LIM),
                                                r.uniform(1.6, 3.6)])
                else:
                    dd = s["target"] - p
                    n = np.linalg.norm(dd)
                    if n < 0.4:
                        s["target"] = None
                        s["dwell"] = float(r.exponential(1.6) + 0.3)
                    else:
                        v = s["speed"] * boost * dd / n
            newp = p + v * dt
            if k in ("along", "reactive"):
                newp[0] = np.clip(newp[0], X_LO, X_HI)
            if not (k == "across" and (s["yr"][1] > Y_LIM or s["yr"][0] < -Y_LIM)):
                newp = self._clip(newp)
            else:
                newp[2] = np.clip(newp[2], Z_LO, Z_HI)
            s["vel"] = (newp - p) / dt
            s["pos"] = newp
