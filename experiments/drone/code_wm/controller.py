"""code_wm pilot: a batched (torch) path-following controller for the site task, plus the real-world policy wrapper.

The same `control()` function drives both the code world (thousands of envs on the GPU) and the real drone (B=1, CPU),
so whatever is tuned in imagination is exactly what flies.  Command mode "ctbr": a = (p, q, r, T).

Structure (per env):
  path      polyline start -> relay_1 -> relay_2 -> relay_3 -> (3 m beyond relay_3); segment = current relay index
  outer     a_des = kp*e_perp + kv*(v_des - v) + ki*int(e_perp) - a_comp(world model) ; v_des = speed * tangent
  thrust    F = m (a_des + g z); throttle T = 0.3 sqrt(|F . z_body| / (m g * thrust_gain)) / 1  (+ bias)
  attitude  R_des from F direction and heading 0; rate cmd = -k_att * vee(R_des^T R - R^T R_des)/2 / rate_gain
`prm` is a dict of floats (gains and world-model compensation terms); see DEFAULT.
"""
from __future__ import annotations

import math

import numpy as np
import torch

G = 9.81
DEFAULT = dict(speed=4.0, kp=3.0, kv=3.0, ki=0.6, k_att=7.0, max_tilt=0.75, thrust_gain=1.0, rate_gain=0.9,
               t_bias=0.0, p_bias=0.0, q_bias=0.0, r_bias=0.0, lookahead_corner=1.5, slow_r3=0.0, corner_dist=4.0, corner_speed=3.0, return_fix=0.0,
               av_k=0.0, av_R=2.0, av_T=1.5, av_up=0.0,
               wind_ff=0.0)


def quat_to_R(q: torch.Tensor) -> torch.Tensor:
    """(B,4) quaternion (x,y,z,w) body->world -> (B,3,3) rotation matrix."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def path_nodes(start: torch.Tensor, relays: torch.Tensor) -> torch.Tensor:
    """(B,3), (B,3,3) -> (B,5,3) nodes."""
    d = relays[:, 2] - relays[:, 1]
    ext = relays[:, 2] + 3.0 * d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.cat([start[:, None], relays, ext[:, None]], 1)


def route_nodes(start: torch.Tensor, relays: torch.Tensor, route, return_relay_ids: bool = False):
    """Generic waypoint route.  Items: [x, y, z]; "R1".."R3" (this episode's relay); or {"like": "R2", "y": 0.0}
    (that relay's position with some coordinates overridden, e.g. the aisle entry point in line with a jittered relay).
    Returns (B, len(route)+2, 3): start, route..., 3 m extension beyond the last point (and, optionally, a list giving
    for every node the 1-based relay number it is, 0 otherwise)."""
    pts, rid = [start], [0]
    for it in route:
        if isinstance(it, str):
            pts.append(relays[:, int(it[1:]) - 1]); rid.append(int(it[1:]))
        elif isinstance(it, dict):
            p = relays[:, int(it["like"][1:]) - 1].clone()
            for j, ax in enumerate("xyz"):
                if ax in it:
                    p[:, j] = float(it[ax])
            pts.append(p); rid.append(0)
        else:
            pts.append(torch.as_tensor(it, dtype=start.dtype, device=start.device).expand_as(start)); rid.append(0)
    d = pts[-1] - pts[-2]
    pts.append(pts[-1] + 3.0 * d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)); rid.append(0)
    nodes = torch.stack(pts, 1)
    return (nodes, rid) if return_relay_ids else nodes


def advance(pos, nodes, k, r_adv: float = 0.8, relay_ids=None, relay_index=None):
    """Route mode: move to the next segment once within r_adv of its end point or past it.  Phase-2 fix: if the end
    point is a relay (relay_ids), advance ONLY once the real relay index shows it was reached, never because the drone
    flew past it (control() then turns back to it)."""
    B = pos.shape[0]
    ar = torch.arange(B, device=pos.device)
    a, b = nodes[ar, k], nodes[ar, k + 1]
    seg = b - a
    L = seg.norm(dim=-1).clamp(min=1e-6)
    s = ((pos - a) * seg).sum(-1) / L
    go = ((pos - b).norm(dim=-1) < r_adv) | (s >= L)
    if relay_ids is not None:
        rid = torch.as_tensor(relay_ids, device=pos.device)[k + 1]
        go = torch.where(rid > 0, relay_index.long() >= rid, go)
    return (k + go.long()).clamp(max=nodes.shape[1] - 2)


def avoid_accel(prm: dict, pos, vel, movers):
    """Mission C: repulsive acceleration from the other aircraft in view (movers rows: pos, vel, radius; zero = empty).
    Constant-velocity prediction of both aircraft over av_T seconds; if the predicted closest approach is inside av_R,
    push away from the mover's predicted position at that moment (weight grows quadratically with the intrusion), with
    an extra upward component av_up (the other aircraft were seen only between z 1.6 and 3.6 m)."""
    mp, mv = movers[..., :3], movers[..., 3:6]
    valid = (movers[..., 6] > 0).float()
    dp = pos[:, None] - mp
    dv = vel[:, None] - mv
    tstar = (-(dp * dv).sum(-1) / ((dv * dv).sum(-1) + 1e-6)).clamp(min=0.0)
    tstar = torch.minimum(tstar, torch.as_tensor(prm["av_T"], dtype=pos.dtype, device=pos.device).expand_as(tstar)
                          if not torch.is_tensor(prm["av_T"]) else prm["av_T"].expand_as(tstar))
    c = dp + dv * tstar[..., None]
    dstar = c.norm(dim=-1).clamp(min=1e-3)
    R = prm["av_R"]
    w = ((R - dstar) / R).clamp(min=0) ** 2 * valid
    dirn = c / dstar[..., None]
    up = torch.zeros_like(dirn); up[..., 2] = 1.0
    upc = prm["av_up"][..., None] if torch.is_tensor(prm["av_up"]) else prm["av_up"]   # (B,1) -> (B,1,1)
    acc = (w[..., None] * (dirn + upc * up)).sum(1)
    return prm["av_k"] * acc


def control(prm: dict, pos, vel, quat, nodes, k, integ, dt=1 / 30, wind_est=None, excite=None, movers=None):
    """One controller step.  k (B,) long = current relay index (segment k: node k -> node k+1).
    integ (B,3) integrator state.  wind_est (B,3) optional world-model acceleration to cancel (feed-forward).
    Returns (action (B,4), new integ)."""
    B = pos.shape[0]
    ar = torch.arange(B, device=pos.device)
    a = nodes[ar, k]
    b = nodes[ar, k + 1]
    seg = b - a
    L = seg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    t = seg / L
    s = ((pos - a) * t).sum(-1, keepdim=True).clamp(min=0.0)
    # blend into the next segment near the end of this one (the relay counts within 1 m, no need to hit its centre)
    c = a + s.clamp(max=L) * t
    e = c - pos
    e_perp = e - (e * t).sum(-1, keepdim=True) * t
    e_par = (e * t).sum(-1, keepdim=True) * t
    tn_idx = (k + 2).clamp(max=nodes.shape[1] - 1)
    nb = nodes[ar, tn_idx]
    t2 = nb - b
    t2 = t2 / t2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    lam = ((s - (L - prm["lookahead_corner"])) / prm["lookahead_corner"]).clamp(0, 1)
    tdir = (1 - lam) * t + lam * t2
    tdir = tdir / tdir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    spd = prm["speed"] * torch.ones_like(s)            # gains may be floats or (B,1) tensors
    # slow down before sharp corners: within corner_dist of the segment end, speed -> corner_speed * f(turn angle)
    turn = (1 - (t * t2).sum(-1, keepdim=True)) / 2          # 0 straight .. 1 reversal
    near = ((s - (L - prm["corner_dist"])) / prm["corner_dist"]).clamp(0, 1)
    drop = (1 - prm["corner_speed"] / (spd + 1e-3)).clamp(min=0)
    spd = spd * (1 - near * turn * drop * 2).clamp(min=0.25)
    v_des = spd * tdir
    if prm.get("return_fix", 0.0):
        # Phase-2 fix of the missed-relay deadlock (first noticed on evaluation seed 1038 of mission A, reproduced in the
        # code world): once past the end of the segment without having been advanced (relay missed), fly straight back
        # to the end point instead of balancing the along-track pull against the forward speed reference.
        past = (((pos - a) * t).sum(-1, keepdim=True) > L)
        eb = b - pos
        db = eb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        v_back = torch.minimum(spd, 1.2 * db) * eb / db
        v_des = torch.where(past, v_back, v_des)
        e_perp = torch.where(past, torch.zeros_like(e_perp), e_perp)
        e_par = torch.where(past, eb, e_par)
    integ = (integ + e_perp * dt).clamp(-2.0, 2.0)
    a_des = prm["kp"] * (e_perp + e_par) + prm["kv"] * (v_des - vel) + prm["ki"] * integ
    if wind_est is not None:
        a_des = a_des - prm["wind_ff"] * wind_est
    if movers is not None and prm.get("av_k", 0.0) is not None:
        a_des = a_des + avoid_accel(prm, pos, vel, movers)
    # tilt limit on the horizontal part
    amax = G * torch.tan(torch.as_tensor(prm["max_tilt"], dtype=pos.dtype, device=pos.device))
    ah = a_des[:, :2]
    n = ah.norm(dim=-1, keepdim=True)
    ah = ah * (amax / n.clamp(min=1e-6)).clamp(max=1.0)
    az = a_des[:, 2:].clamp(-0.6 * G, 2.0 * G)
    F = torch.cat([ah, az + G], -1)                     # per unit mass
    R = quat_to_R(quat)
    zb = R[:, :, 2]
    f_along = (F * zb).sum(-1).clamp(min=0.2 * G)
    T = 0.3 * torch.sqrt(f_along / (G * prm["thrust_gain"])) + prm["t_bias"]
    zd = F / F.norm(dim=-1, keepdim=True)
    xc = torch.tensor([1.0, 0.0, 0.0], device=pos.device, dtype=pos.dtype).expand(B, 3)
    yd = torch.cross(zd, xc, dim=-1)
    yd = yd / yd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    xd = torch.cross(yd, zd, dim=-1)
    Rd = torch.stack([xd, yd, zd], -1)
    M = Rd.transpose(1, 2) @ R - R.transpose(1, 2) @ Rd
    eR = 0.5 * torch.stack([M[:, 2, 1], M[:, 0, 2], M[:, 1, 0]], -1)
    w = -prm["k_att"] * eR / prm["rate_gain"]
    w = w - torch.tensor([prm["p_bias"], prm["q_bias"], prm["r_bias"]], device=pos.device, dtype=pos.dtype)
    if excite is not None:
        w = w + excite[:, :3]
        T = T + excite[:, 3]
    w = w.clamp(-math.pi, math.pi)
    T = T.clamp(0.0, 0.8)
    return torch.cat([w, T[:, None]], -1), integ


def excitation(t: float, t_end: float) -> np.ndarray:
    """System-identification dither used in life 1 while hovering: staggered square-ish pulses on p, q, r and T."""
    if t >= t_end:
        return np.zeros(4)
    ph = int(t / 0.2)                      # 0.2 s slots
    pattern = [(0, 1.2), (0, -1.2), (1, 1.2), (1, -1.2), (2, 1.5), (2, -1.5), (3, 0.06), (3, -0.06)]
    ax, amp = pattern[ph % len(pattern)]
    out = np.zeros(4)
    if (ph // len(pattern)) % 2 == 0 or ax == 2:
        out[ax] = amp
    return out


class Pilot:
    """Real-world policy: reset() / act(obs dict) -> np.ndarray (4,).  Hover-ID phase optional (life 1)."""

    def __init__(self, prm: dict | None = None, id_time: float = 0.0, belief: str | None = None, tuned: str | None = None,
                 route=None, r_adv: float = 0.8, excite_scale: float = 1.0):
        """prm: controller gains (or tuned: a tune.py output json whose "controller" is used).
        belief: fitted-world json; its wind_field is used as feed-forward (needs prm wind_ff > 0)."""
        import json
        self.prm = dict(DEFAULT)
        if tuned:
            self.prm.update(json.load(open(tuned))["controller"])
        self.prm.update(prm or {})
        self.id_time = id_time
        self.route, self.r_adv, self.excite_scale = route, r_adv, excite_scale
        self.wind = None
        if belief:
            from worlds.drone.site_world import DATASHEET
            self.wind = dict(DATASHEET)
            self.wind.update(json.load(open(belief))["full"])
        self.reset()

    def reset(self, seed=None):
        self.integ = torch.zeros(1, 3, dtype=torch.float64)
        self.start = None
        self.n = 0
        self.k = None

    def act(self, obs):
        f = lambda k: torch.as_tensor(np.asarray(obs[k], dtype=np.float64))[None]
        pos, vel, quat = f("pos"), f("lin_vel"), f("quat")
        relays = f("relays")
        k = torch.as_tensor([int(np.asarray(obs["relay_index"]).reshape(-1)[0])])
        if self.start is None:
            self.start = pos.clone()
        tt = float(np.asarray(obs["t"]).reshape(-1)[0])
        if self.route is not None:
            nodes, rid = route_nodes(self.start, relays, self.route, return_relay_ids=True)
            if self.k is None:
                self.k = torch.zeros(1, dtype=torch.long)
            gate = rid if self.prm.get("return_fix", 0.0) else None
            self.k = advance(pos, nodes, self.k, self.r_adv, gate, k)
            k = self.k
        else:
            nodes = path_nodes(self.start, relays)
        prm = dict(self.prm)
        wind_est = None
        if self.wind is not None:
            from worlds.drone.site_world import wind_accel_np
            wind_est = torch.as_tensor(wind_accel_np(self.wind, obs["pos"]))[None]
        exc = None
        if tt < self.id_time:
            # hold the start point while the dither runs
            nodes = self.start[:, None].expand(1, 5, 3).clone()
            nodes[:, 1:, 0] += 1e-3 * torch.arange(1, 5)
            k = torch.zeros(1, dtype=torch.long)
            exc = torch.as_tensor(excitation(tt, self.id_time) * self.excite_scale)[None]
            prm["speed"] = 0.0
        mov = f("movers") if ("movers" in obs and float(self.prm.get("av_k", 0.0)) > 0) else None
        act, self.integ = control(prm, pos, vel, quat, nodes, k, self.integ, wind_est=wind_est, excite=exc, movers=mov)
        self.n += 1
        return act[0].numpy()


def make_pilot(**kw):
    return Pilot(**kw)
