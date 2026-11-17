"""policy_v6 -- mission-generic version of v5 for phase 2 (mission B), written offline with ZERO flights.  ctbr mode.

Unchanged from v5 (validated by lives 2-5 and 50/50 on mission A): S1 rate-bias feed-forward, S2 thrust constant,
S3 disturbance observer, S4/S6 jet profiles on the canyon line, attitude/thrust law, gains, 6 m/s canyon cruise.

New for routes that leave the canyon centreline (mission B: R2 sits inside the north side aisle):
  N1 route planner     the straight relay-to-relay segment may cut through a wall (R1 -> R2 does).  Segments are
                       checked against the site_map boxes inflated by 0.6 m; a blocked one is replaced by an
                       L-shaped path with a corner (B.x, A.y) or (A.x, B.y), whichever leg pair is free.  Corners
                       inside a jet are shifted 0.03 s^2 * a_jet against the jet (|shift| <= 0.8 m), i.e. away
                       from the wall the jet pushes toward.
  N2 corner speeds     along-track speed reference min(cruise, sqrt(v_end^2 + 2 a_b d_remaining)) with
                       v_end = cruise for (near) straight joins, 1.5 m/s for ~90 deg turns, 0.5 m/s for reversals;
                       negative when past the end point (come back).  Legs that are not along the canyon are
                       capped at 3.5 m/s.  Corners switch at 0.4 m along-track, relays only when the lab
                       advances relay_index.  For mission A this reduces exactly to v5's constant cruise.
  N3 jet inside aisles (assumption, never measured) the jet comes out of the aisle, so the same x-profile is assumed
                       to act inside the aisle at 70 % confidence; the observer (S3) corrects the rest.
  N4 more authority    a_max 16 -> 24 m/s^2 and observer clip 15 -> 20 m/s^2: climbing the aisle against a
                       12-18 m/s^2 jet needs > 16 m/s^2 of horizontal acceleration.
  N5 generic walls     the wall push uses distances to all site_map boxes instead of the canyon half-width.
"""
from __future__ import annotations

import numpy as np

G = 9.81
RATE_GAIN = np.array([0.89, 0.96, 0.99])
RATE_BIAS = np.array([-0.495, 0.540, -0.275])   # actual body rate at zero command [rad/s]
C_THRUST = 88.0                                 # specific force along body z = C * T^2

JETS = [   # (x knots, a_y knots [m/s^2], side of the aisle the jet comes from: +1 north / -1 south)
    (np.array([21.2, 21.6, 22.1, 22.6, 23.2, 23.8, 24.3, 24.9, 25.4, 25.9, 26.4, 26.9]),
     np.array([0.0, -2.5, -6.5, -10.0, -13.0, -14.0, -13.0, -9.0, -5.0, -3.0, -1.0, 0.0]), +1),
    (np.array([37.0, 37.6, 38.2, 38.8, 39.5, 40.3, 40.9, 41.4, 42.0, 42.7]),
     np.array([0.0, 2.5, 6.0, 8.5, 9.5, 8.0, 6.0, 4.0, 2.0, 0.0]), -1),
]
AISLE_CONF = 0.7

# site_map() static geometry (public survey): container stacks, all taller than the ceiling -> 2-D rectangles
BOXES = np.array([   # cx, cy, hx, hy
    [16.0, 7.0, 6.0, 5.0],
    [37.75, 7.0, 12.25, 5.0],
    [24.0, -7.0, 14.0, 5.0],
    [45.75, -7.0, 4.25, 5.0],
    [32.5, 1.8, 1.5, 0.2],
    [32.5, -1.8, 1.5, 0.2],
])


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def jet_prior(x, y=0.0):
    a = np.zeros(3)
    for xs, ays, side in JETS:
        v = np.interp(x, xs, ays, left=0.0, right=0.0)
        if abs(y) < 2.5:
            a[1] += v
        elif side * y > 0:          # inside the aisle the jet comes from (assumed, N3)
            a[1] += AISLE_CONF * v
    return a


# ------------------------------------------------------------------ geometry / planning (static, no dynamics)
def box_clearance(px, py):
    """xy distance to the nearest box and outward unit normal (negative distance = inside)."""
    best, nrm = 1e9, np.zeros(2)
    for cx, cy, hx, hy in BOXES:
        dx, dy = px - cx, py - cy
        qx, qy = abs(dx) - hx, abs(dy) - hy
        if qx > 0 or qy > 0:
            ox, oy = max(qx, 0.0), max(qy, 0.0)
            dist = np.hypot(ox, oy)
            n = np.array([np.sign(dx) * ox, np.sign(dy) * oy]) / max(dist, 1e-9)
        else:
            dist = max(qx, qy)
            n = np.array([np.sign(dx), 0.0]) if qx > qy else np.array([0.0, np.sign(dy)])
        if dist < best:
            best, nrm = dist, n
    return best, nrm


def segment_free(a, b, infl=0.6):
    n = max(2, int(np.linalg.norm(b[:2] - a[:2]) / 0.1))
    for s in np.linspace(0, 1, n + 1):
        p = a + s * (b - a)
        if box_clearance(p[0], p[1])[0] < infl:
            return False
    return True


def plan_route(start, relays):
    """[(point, kind, relay_number)], kind in start/corner/relay."""
    wps = [(np.asarray(start, float), "start", -1)]
    prev = wps[0][0]
    for k, r in enumerate(np.asarray(relays, float)):
        if not segment_free(prev, r):
            best = None
            for c in (np.array([r[0], prev[1], r[2]]), np.array([prev[0], r[1], r[2]])):
                shift = float(np.clip(-0.03 * jet_prior(c[0], c[1])[1], -0.8, 0.8))
                for cc in (c + np.array([0.0, shift, 0.0]), c):
                    if segment_free(prev, cc) and segment_free(cc, r):
                        L = np.linalg.norm(cc - prev) + np.linalg.norm(r - cc)
                        if best is None or L < best[0]:
                            best = (L, cc)
                        break
            if best is not None:
                wps.append((best[1], "corner", -1))
        wps.append((r, "relay", k))
        prev = r
    return wps


class Policy:
    def __init__(self, cruise=6.0, side_cruise=3.5, a_brake=4.0, v_turn=1.5, v_rev=0.5, kp=4.5, kd=3.6, ki=0.8,
                 kv=2.5, k_att=7.0, k_yaw=3.0, ki_att=1.5, a_max=24.0, i_max=3.0, wall_margin=0.7, k_wall=8.0,
                 ramp_s=1.5, dob_alpha=0.4, dob_max=20.0, lead_s=0.15, corner_switch=0.4):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self.reset()

    def reset(self):
        self.wps = None
        self.wi = 1
        self.integ = np.zeros(3)
        self.integ_att = np.zeros(3)
        self.r_hat = np.zeros(3)
        self.hist = []
        self.dbg = {}

    # -------------------------------------------------------------- guidance
    def _advance(self, p, k_lab):
        while self.wi < len(self.wps) - 1:
            pt, kind, rn = self.wps[self.wi]
            prev = self.wps[self.wi - 1][0]
            d = pt - prev; d /= max(np.linalg.norm(d), 1e-6)
            if kind == "relay" and k_lab > rn:
                self.wi += 1
            elif kind == "corner" and np.dot(pt - p, d) < self.corner_switch:
                self.wi += 1
            else:
                break

    def _v_end(self, d):
        if self.wi >= len(self.wps) - 1:
            return self.cruise
        nxt = self.wps[self.wi + 1][0] - self.wps[self.wi][0]
        c = np.dot(d, nxt / max(np.linalg.norm(nxt), 1e-6))
        if c > 0.94:
            return self.cruise
        return self.v_turn if c > -0.5 else self.v_rev

    def _dob(self, p, v):
        if len(self.hist) >= 3:
            v_prev, b3_prev, _, x_prev, y_prev = self.hist[-1]
            T_del = self.hist[-2][2]
            a_meas = (v - v_prev) * 30.0
            a_model = C_THRUST * T_del ** 2 * b3_prev - np.array([0, 0, G])
            res = a_meas - a_model - jet_prior(0.5 * (x_prev + p[0]), 0.5 * (y_prev + p[1]))
            self.r_hat = (1 - self.dob_alpha) * self.r_hat + self.dob_alpha * res
            self.r_hat = np.clip(self.r_hat, -self.dob_max, self.dob_max)
        return self.r_hat

    def act(self, obs):
        dt = 1.0 / 30.0
        p = np.asarray(obs["pos"], float)
        v = np.asarray(obs["lin_vel"], float)
        t = float(obs["t"][0])
        R = quat_to_R(np.asarray(obs["quat"], float))
        if self.wps is None:
            self.wps = plan_route(p, obs["relays"])
        r_hat = self._dob(p, v)
        self._advance(p, int(obs["relay_index"][0]))
        goal, prev = self.wps[self.wi][0], self.wps[self.wi - 1][0]
        d = goal - prev
        d /= max(np.linalg.norm(d), 1e-6)
        rel = p - prev
        e_perp = rel - np.dot(rel, d) * d
        v_along = np.dot(v, d)
        v_perp = v - v_along * d
        cruise = self.cruise if abs(d[0]) > 0.9 else self.side_cruise
        d_rem = float(np.dot(goal - p, d))
        v_end = min(self._v_end(d), cruise)
        if d_rem >= 0:
            vref = min(cruise, np.sqrt(v_end ** 2 + 2 * self.a_brake * d_rem))
        else:
            vref = -min(2.0, np.sqrt(2 * self.a_brake * -d_rem)) if v_end < cruise else cruise
        vref = min(vref, self.cruise * min(1.0, (t + 0.3) / self.ramp_s))
        a = self.kv * (vref - v_along) * d - self.kp * e_perp - self.kd * v_perp
        self.integ += dt * (-e_perp + 0.3 * (vref - v_along) * d)
        self.integ = np.clip(self.integ, -self.i_max / self.ki, self.i_max / self.ki)
        a = a + self.ki * self.integ
        clr, nrm = box_clearance(p[0] + 0.3 * v[0], p[1] + 0.3 * v[1])
        if clr < self.wall_margin:            # same law as v5 (|y| > half_width - 0.7), for any wall
            a[:2] += nrm * self.k_wall * (self.wall_margin - clr)
        dist = jet_prior(p[0] + v[0] * self.lead_s, p[1] + v[1] * self.lead_s) + r_hat
        a = a - dist
        n = np.linalg.norm(a[:2])
        if n > self.a_max:
            a[:2] *= self.a_max / n
        a[2] = np.clip(a[2], -6.0, 12.0)
        f = a + np.array([0.0, 0.0, G])
        b3d = f / np.linalg.norm(f)
        b2d = np.cross(b3d, [1.0, 0.0, 0.0]); b2d /= np.linalg.norm(b2d)
        b1d = np.cross(b2d, b3d)
        Rd = np.column_stack([b1d, b2d, b3d])
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        self.integ_att = np.clip(self.integ_att + dt * eR, -0.3, 0.3)
        w_des = -np.array([self.k_att, self.k_att, self.k_yaw]) * eR - self.ki_att * self.integ_att
        cmd = np.clip((w_des - RATE_BIAS) / RATE_GAIN, -3.1, 3.1)
        u = float(np.dot(f, R[:, 2]))
        T = float(np.clip(np.sqrt(max(u, 0.2 * G) / C_THRUST), 0.05, 0.8))
        self.hist.append((v.copy(), R[:, 2].copy(), T, p[0], p[1]))
        self.dbg = dict(a=a, eR=eR, e_perp=e_perp, r_hat=r_hat.copy(), dist=dist, wi=self.wi, vref=vref)
        return np.array([cmd[0], cmd[1], cmd[2], T])


def make_policy(**kw):
    return Policy(**kw)
