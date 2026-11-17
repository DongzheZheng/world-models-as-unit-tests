"""policy_v8 -- repairs after mission-C life 1 (v7, success 9.67 s, but minimum separation 0.66 m).

Life-1 instrumentation (movers_stats.py), 7 other aircraft, observed rows accurate (pos err 5 cm, vel 8 cm/s):
  - 'darters' (4 of 7): hover, then dash at a constant ~5 m/s (4.6-5.2 per aircraft) for 0.1-1.5 s in a random
    direction, every ~2 s; a dash can start next to the drone (m1 started one 1.1 m away -> 0.66 m miss)
  - 'patrols' (3 of 7): constant-speed shuttles, 0.9-1.9 m/s (two cross the canyon along y through the side
    aisles, one creeps along -x down the canyon)
  - v7 passed hovering darters at 0.8-1.3 m: inside the reach of a dash that starts at the wrong moment.
Repairs:
  C6 separation depends on what the mover is doing: hovering (|v| < 0.5 m/s, may dash at any time) -> keep
     radius + 1.7 m (a dash starting at >= ~2 m leaves ~0.3 s to dodge 0.5 m); moving at constant velocity
     (patrol or dash in progress) -> radius + 0.9 m around its CPA
  C7 avoidance acceleration cap 14 -> 20 m/s^2 (dodging a dash that starts close)
  C8 altitude band 0.8-5.0 m (ceiling 6 m = out of bounds)
  C9 replaying life 1 through the acceleration-level law showed it acting only in the last ~0.6 s with 12-19 m/s^2:
     the cross-track damping -kd*v_perp cancels the avoidance velocity, so the required acceleration keeps
     growing.  Avoidance is now velocity-level: an escape velocity v_av = sum 1.5 (D - |m|) / max(t*, 0.3) e
     (|v_av| <= 5 m/s) is TRACKED by the damping term (kd (v_av_perp - v_perp)) and the along-track speed loop;
     the acceleration law is kept only for t* < 0.6 s (emergencies) and for the proximity shell.
History of v7:
policy_v7 -- mission C first program: v6 (validated on A and B) + reactive avoidance of other aircraft.  ctbr mode.

Nothing is known about the other aircraft yet, so the avoidance is pure closed-form geometry on obs["movers"]:
  C1 constant-velocity closest point of approach (CPA): relative position r = p_m - p, relative velocity
     w = v_m - v, t* = clip(-(r.w)/|w|^2, 0, 3 s), miss vector m = r + w t*.
  C2 if |m| < D (= mover radius + 0.65 m) the drone needs a displacement (D - |m|) along e = -m/|m| within t*:
     closed-form kinematic acceleration 2 (D - |m|) / max(t*, 0.35)^2 (capped 14 m/s^2) along e.
     e is biased toward the vertical (the canyon is 4 m wide but 0.8-5.2 m of altitude is usable and the walls
     are 7.8 m tall); for an almost head-on miss the drone climbs if below the mover / at the same height and
     z < 3.5, else descends.
  C3 proximity shell: within D + 0.6 m an additional push straight away from the mover.
  C4 while any threat is active the cross-track reference follows the drone (no PD fighting the avoidance);
     afterwards it decays back to the route at 1.2 m/s.
  C5 if the CPA lies ahead and is unavoidable at the current speed (t* small, |m| tiny) the along-track speed
     reference is cut to 2 m/s, so the time window (19.4 s vs ~9.5 s needed) is spent instead of the airframe.
"""
from __future__ import annotations

import numpy as np

from pilots.aspire_drone.policy_v6 import (C_THRUST, G, RATE_BIAS, RATE_GAIN, box_clearance, jet_prior, plan_route,
                                           quat_to_R, vee)
from pilots.aspire_drone.policy_v6 import Policy as V6

Z_MIN, Z_MAX = 0.8, 5.0


def mover_rows(obs):
    if "movers" not in obs:
        return np.zeros((0, 7))
    M = np.asarray(obs["movers"], float).reshape(-1, 7)
    n = int(np.asarray(obs.get("n_movers", [len(M)])).reshape(-1)[0])
    return M[:n]


class Policy(V6):
    def __init__(self, margin=0.9, margin_hover=1.7, horizon=3.0, a_avoid_max=20.0, vert_bias=2.0, off_decay=1.2, slow_v=2.0, **kw):
        self.margin, self.margin_hover, self.horizon, self.a_avoid_max = margin, margin_hover, horizon, a_avoid_max
        self.vert_bias, self.off_decay, self.slow_v = vert_bias, off_decay, slow_v
        super().__init__(**kw)

    def reset(self):
        super().reset()
        self.off = np.zeros(3)
        self.threat_t = -9.0

    def avoidance(self, p, v, obs):
        """returns (escape velocity, emergency acceleration, slow flag, info)"""
        v_av, a = np.zeros(3), np.zeros(3)
        slow = False
        info = []
        for row in mover_rows(obs):
            pm, vm, rad = row[:3], row[3:6], max(row[6], 0.2)
            r, w = pm - p, vm - v
            D = rad + (self.margin_hover if np.linalg.norm(vm) < 0.5 else self.margin)
            ww = float(w @ w)
            t = float(np.clip(-(r @ w) / ww, 0.0, self.horizon)) if ww > 1e-6 else 0.0
            m = r + w * t
            dm = float(np.linalg.norm(m))
            dist = float(np.linalg.norm(r))
            info.append((round(dist, 2), round(t, 2), round(dm, 2)))
            if dm < D and t < self.horizon:
                if dm > 0.15:
                    e = -m / dm
                else:                                   # head-on: pick vertical side
                    up = (pm[2] >= p[2] and p[2] < 3.5) or p[2] < 1.6
                    e = np.array([0.0, 0.0, 1.0 if up else -1.0])
                e = e * np.array([1.0, 1.0, self.vert_bias])
                if p[2] > Z_MAX - 0.4 and e[2] > 0 or p[2] < Z_MIN + 0.4 and e[2] < 0:
                    e[2] = -0.3 * e[2]
                e /= max(np.linalg.norm(e), 1e-6)
                v_av += 1.5 * (D - dm) / max(t, 0.3) * e
                if t < 0.6:
                    a += min(self.a_avoid_max, 2.0 * (D - dm) / max(t, 0.25) ** 2) * e
                if t < 1.0 and dm < 0.5 * D and float(-(r @ v)) < 0 and np.linalg.norm(vm) >= 0.5:
                    slow = True        # only for moving aircraft: braking near a hovering darter extends exposure
            if dist < D + 0.6:
                a += 8.0 * (D + 0.6 - dist) * (-r / max(dist, 1e-6))
        n = np.linalg.norm(v_av)
        if n > 5.0:
            v_av *= 5.0 / n
        n = np.linalg.norm(a)
        if n > self.a_avoid_max:
            a *= self.a_avoid_max / n
        return v_av, a, slow, info

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
        v_av, a_av, slow, info = self.avoidance(p, v, obs)
        active = np.linalg.norm(a_av) > 0.05 or np.linalg.norm(v_av) > 0.05
        if active:
            self.threat_t = t
            self.off = e_perp.copy()                    # C4: reference follows the drone during avoidance
        elif t - self.threat_t > 0.3:
            n = np.linalg.norm(self.off)
            if n > 0:
                self.off *= max(0.0, n - self.off_decay * dt) / n
        e_ref = e_perp - self.off
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
        if slow:
            vref = min(vref, self.slow_v)
        av_along = float(np.dot(v_av, d))
        av_perp = v_av - av_along * d
        a = self.kv * (vref + av_along - v_along) * d - self.kp * e_ref + self.kd * (av_perp - v_perp)
        if not active:
            self.integ += dt * (-e_ref + 0.3 * (vref - v_along) * d)
            self.integ = np.clip(self.integ, -self.i_max / self.ki, self.i_max / self.ki)
        a = a + self.ki * self.integ + a_av
        if p[2] < Z_MIN:
            a[2] += 6.0 * (Z_MIN - p[2])
        if p[2] > Z_MAX:
            a[2] -= 6.0 * (p[2] - Z_MAX)
        clr, nrm = box_clearance(p[0] + 0.3 * v[0], p[1] + 0.3 * v[1])
        if clr < self.wall_margin:
            a[:2] += nrm * self.k_wall * (self.wall_margin - clr)
        dist = jet_prior(p[0] + v[0] * self.lead_s, p[1] + v[1] * self.lead_s) + r_hat
        a = a - dist
        n = np.linalg.norm(a[:2])
        if n > self.a_max:
            a[:2] *= self.a_max / n
        a[2] = np.clip(a[2], -7.0, 12.0)
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
        self.dbg = dict(a=a, e_perp=e_perp, r_hat=r_hat.copy(), dist=dist, wi=self.wi, vref=vref, a_av=a_av, v_av=v_av,
                        slow=slow, movers=info)
        return np.array([cmd[0], cmd[1], cmd[2], T])


def make_policy(**kw):
    return Policy(**kw)
