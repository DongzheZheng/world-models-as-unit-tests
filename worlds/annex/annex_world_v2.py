"""Annex code world: an elastic rod lying on a slippery table, one end pinned to a servoed tool, a peg, a target.

What it is (diagnosis from life 1, see runs/annex/code_wm/diagnosis.json):
  * 20 tracked points = 20 lumped masses of a planar elastic rod (rest length 0.70 m, straight rest shape),
    joined by 19 stiff but slightly compliant links (rest 0.70/19 m).  Life 1: link length 0.0368 +- 0.0010 m
    even in a 5 m/s fling (max 0.0378), so links are springs (`k_stretch`), not hard constraints.
  * bending elasticity: with the tool held still, kinetic energy rose 0.46 -> 3.4 (per unit node mass) while the
    squared-turning sum fell 3.0 -> 1.5 (life 1, steps 179-203): stored bending energy.  `k_bend`, `c_bend`.
  * point 20 is PINNED (not clamped) to the tool: the last link turned freely over 250 deg (life 1).
  * table friction is weak: a gently moving rod decelerates at ~0.2 m/s^2 (life 1 steps 0-9 and 70-85) and a flung
    rod still moved at ~0.9 m/s 1.8 s after the fling.  Coulomb `mu` (smoothed) + viscous `c_visc`.
  * tool: motion reference with |v|<=4 m/s, |a|<=60 m/s^2 braking to stop at the command (manual), followed by a
    position servo (2nd-order, `servo_w`, `servo_z`, accel limit `servo_amax`); life 1 tool_pos matched the ideal
    reference to ~1 mm, the tool is not noticeably slowed by the rod (rod reaction ignored).
  * peg: rigid cylinder r=0.03 m (manual); rod nodes (radius 0.005, z=0.005 in every reading) are pushed out of
    r_post + r_rod with a stiff penalty and Coulomb friction `mu_post`.  Not yet observed in a life.
All lengths m, time s; forces are per unit node mass (accelerations), so the rod mass never appears.

Functional API (state = dict of tensors with leading batch dim B; nothing is mutated in place):
  init_from_obs(obs_batch, prev_obs_batch=None, params=None, device)  -> state
  step(state, action (B,2)) -> (state, reward (B,), done (B,))           exact lab episode rules
  observe(state) -> list of lab-format obs dicts (numpy)
  rollout_points(state, actions (B,T,2)) -> fast open-loop roll (used by fidelity & planning)
"""
from __future__ import annotations

import math

import numpy as np
import torch

# ---------------------------------------------------------------- public facts (manual / task card)
N = 20
SEG = 0.70 / 19
DT = 0.02
TIME_LIMIT = 236
ZONE = ((0.30, -0.20), (0.55, 0.20))
ZONE_TOL = 0.01
REST_STEP = 0.02 / 50.0
TARGET_R = 0.10
POST_R = 0.03
ROD_R = 0.005          # snapshots: every point at z = 0.0050 m -> radius 5 mm lying on the table
VMAX, AMAX = 4.0, 60.0
G = 9.81

# ---------------------------------------------------------------- fitted parameters (see fit_world.py, life 1)
# values are overwritten by the fit; ranges are used for domain randomisation (uncertainty)
PARAMS = dict(
    k_stretch=4.0e5,   # link spring, accel per unit strain*m  [1/s^2]
    c_stretch=60.0,    # link damper [1/s]
    k_bend=2.0,        # bending stiffness: accel = k_bend * angle / SEG [m/s^2 per rad]
    c_bend=0.05,       # bending damping [m/s per rad/s]
    mu=0.02,           # Coulomb table friction coefficient (decel mu*g)
    aniso=1.0,         # friction across the rod / along the rod
    c_visc=0.0,        # viscous table drag [1/s]
    c_quad=0.0,        # quadratic (air-like) drag on the velocity normal to the rod [1/m]
    lat_mass=1.0,      # effective inertia factor for motion across the rod (rolling)
    tip_mass=1.0,      # relative lumped mass of the free-end node
    mu_lo=0.13,        # v2: friction coefficient at low speed
    v_s=0.2,           # v2: transition speed between mu_lo and mu [m/s]
    mu_post=0.2,       # peg friction
    servo_w=200.0,     # tool servo natural frequency [rad/s]
    servo_z=1.0,       # tool servo damping ratio
    servo_amax=200.0,  # tool accel limit from the 60 N force limit [m/s^2]
)
N_SUB = 40             # substeps per 0.02 s agent step (dt = 0.5 ms; 20 substeps went unstable with the fitted
                       # bending damping: rod never came to rest when held)
import os as _os
# v2: link projection iterations per substep (0 = spring links as in the official world; env CW_PBD for ablation)
PBD_ITERS = int(_os.environ.get("CW_PBD", "3"))
V_EPS = 0.003         # Coulomb smoothing speed [m/s] (below the 0.02 m/s rest threshold, so a resting rod sticks)


def _p(params, key, B, dev):
    v = params.get(key, PARAMS[key]) if params is not None else PARAMS[key]
    t = torch.as_tensor(v, dtype=torch.float32, device=dev)
    return t.expand(B).reshape(B, 1) if t.dim() == 0 else t.reshape(B, 1)


def make_params(B, dev, params=None):
    return {k: _p(params, k, B, dev) for k in PARAMS}


# ---------------------------------------------------------------- state construction / observation
def init_from_obs(obs_batch, prev_obs_batch=None, params=None, device="cuda", prev2_obs_batch=None):
    """Build a state from lab readings.  Node velocities come from (obs - prev_obs)/DT if a previous reading is
    given, else zero (a single reading carries no velocity; life 1 showed the rod may already move at reset)."""
    dev = torch.device(device)
    f = lambda k: torch.tensor(np.stack([np.asarray(o[k], dtype=np.float64) for o in obs_batch]), dtype=torch.float32, device=dev)
    pos = f("points")[..., :2]
    B = pos.shape[0]
    if prev_obs_batch is not None:
        pp = torch.tensor(np.stack([np.asarray(o["points"])[:, :2] for o in prev_obs_batch]), dtype=torch.float32, device=dev)
        vel = (pos - pp) / DT
        if prev2_obs_batch is not None:     # 2nd-order backward difference (less lag under acceleration)
            p2 = torch.tensor(np.stack([np.asarray(o["points"])[:, :2] for o in prev2_obs_batch]), dtype=torch.float32, device=dev)
            vel = (3 * pos - 4 * pp + p2) / (2 * DT)
    else:
        vel = torch.zeros_like(pos)
    tool, tv = f("tool_pos"), f("tool_vel")
    vel[:, -1] = tv
    t = torch.tensor([int(o["t"]) for o in obs_batch], device=dev)
    return dict(pos=pos, vel=vel, tool=tool, tool_v=tv, ref=tool.clone(), ref_v=tv.clone(),
                post=f("post"), target=f("target"), t=t,
                prev_pos=pos.clone(), prev_tool=tool.clone(),
                done=torch.zeros(B, dtype=torch.bool, device=dev), success=torch.zeros(B, dtype=torch.bool, device=dev),
                prm=make_params(B, dev, params))


def observe(state):
    pos = state["pos"].detach().cpu().numpy().astype(np.float64)
    B = pos.shape[0]
    z = np.full((B, N, 1), ROD_R)
    out = []
    tool = state["tool"].cpu().numpy().astype(np.float64)
    tv = state["tool_v"].cpu().numpy().astype(np.float64)
    post = state["post"].cpu().numpy().astype(np.float64)
    tg = state["target"].cpu().numpy().astype(np.float64)
    t = state["t"].cpu().numpy()
    for b in range(B):
        out.append(dict(points=np.concatenate([pos[b], z[b]], 1), tool_pos=tool[b], tool_vel=tv[b], post=post[b],
                        zone=np.array(ZONE, dtype=np.float64), target=tg[b], t=int(t[b])))
    return out


# ---------------------------------------------------------------- mechanisms
def _reference(ref, ref_v, cmd, h):
    """Manual: reference travels towards the command, |v|<=4, |a|<=60, braking to stop at the command."""
    e = cmd - ref
    d = e.norm(dim=-1, keepdim=True)
    u = e / d.clamp_min(1e-9)
    vdes = u * torch.minimum(torch.full_like(d, VMAX), torch.sqrt(2 * AMAX * (d - 0.5 * AMAX * h * h).clamp_min(0)))
    dv = vdes - ref_v
    n = dv.norm(dim=-1, keepdim=True)
    dv = dv * torch.clamp(AMAX * h / n.clamp_min(1e-9), max=1.0)
    ref_v = ref_v + dv
    step = ref_v * h
    # do not overshoot the command
    over = (step.norm(dim=-1, keepdim=True) >= d)
    ref = torch.where(over, cmd, ref + step)
    ref_v = torch.where(over & (d < 1e-6), torch.zeros_like(ref_v), ref_v)
    return ref, ref_v


def _servo(tool, tool_v, ref, ref_v, prm, h):
    w, z, amax = prm["servo_w"], prm["servo_z"], prm["servo_amax"]
    a = w * w * (ref - tool) + 2 * z * w * (ref_v - tool_v)
    a = torch.maximum(torch.minimum(a, amax), -amax)       # per-axis force limit
    tool_v = tool_v + a * h
    return tool + tool_v * h, tool_v


def _perp(v):
    return torch.stack([-v[..., 1], v[..., 0]], -1)


def _rod_accel(pos, vel, post, prm):
    """Accelerations of the 20 nodes (node 20 is overwritten by the tool)."""
    d = pos[:, 1:] - pos[:, :-1]                          # (B,19,2)
    L = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    u = d / L
    dv = vel[:, 1:] - vel[:, :-1]
    # links: spring + damper along the link
    f = (prm["k_stretch"].unsqueeze(-1) * (L - SEG) + prm["c_stretch"].unsqueeze(-1) * (dv * u).sum(-1, keepdim=True)) * u
    acc = torch.zeros_like(pos)
    acc[:, :-1] += f
    acc[:, 1:] -= f
    # bending: E = 0.5*k_bend*SEG * sum(theta^2)-like; force = -(k theta + c theta_dot) dtheta/dp
    a, b = d[:, :-1], d[:, 1:]
    La2, Lb2 = (a * a).sum(-1, keepdim=True), (b * b).sum(-1, keepdim=True)
    cr = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dt_ = (a * b).sum(-1)
    th = torch.atan2(cr, dt_).unsqueeze(-1)              # (B,18,1)
    ga = torch.stack([a[..., 1], -a[..., 0]], -1) / La2  # dtheta/da
    gb = torch.stack([-b[..., 1], b[..., 0]], -1) / Lb2  # dtheta/db
    g0, g1, g2 = -ga, ga - gb, gb                         # wrt p_{i-1}, p_i, p_{i+1}
    thd = (g0 * vel[:, :-2]).sum(-1, keepdim=True) + (g1 * vel[:, 1:-1]).sum(-1, keepdim=True) + (g2 * vel[:, 2:]).sum(-1, keepdim=True)
    m = -(prm["k_bend"].unsqueeze(-1) * SEG * th + prm["c_bend"].unsqueeze(-1) * SEG * thd)
    acc[:, :-2] += m * g0
    acc[:, 1:-1] += m * g1
    acc[:, 2:] += m * g2
    # table friction: Coulomb (smoothed, optionally anisotropic) + viscous
    tang = torch.zeros_like(pos)
    tang[:, :-1] += u
    tang[:, 1:] += u
    tang = tang / tang.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    vt = (vel * tang).sum(-1, keepdim=True) * tang
    vn = vel - vt
    an = prm["aniso"].unsqueeze(-1)
    veff = vt + an * vn
    sp = torch.sqrt((vel * vel).sum(-1, keepdim=True) + V_EPS ** 2)
    # v2: speed-dependent (Stribeck-like) friction: mu at speed, mu_lo below ~v_s (life 5 steps 40-52: a stopped,
    # slightly bent rod started creeping again -> little static holding)
    spd = sp - 0 * sp
    mu_eff = prm["mu"].unsqueeze(-1) + (prm["mu_lo"] - prm["mu"]).unsqueeze(-1) * torch.exp(-(spd / prm["v_s"].unsqueeze(-1)) ** 2)
    acc = acc - mu_eff * G * veff / sp - prm["c_visc"].unsqueeze(-1) * vel
    acc = acc - prm["c_quad"].unsqueeze(-1) * vn.norm(dim=-1, keepdim=True) * vn
    # peg: stiff penalty out of r_post + r_rod, Coulomb friction on the contact
    r = pos - post.unsqueeze(1)
    dist = r.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    pen = (POST_R + ROD_R - dist).clamp_min(0)
    nrm = r / dist
    fn = 3.0e5 * pen - 50.0 * (vel * nrm).sum(-1, keepdim=True).clamp(max=0) * (pen > 0)
    vtan = vel - (vel * nrm).sum(-1, keepdim=True) * nrm
    ff = -prm["mu_post"].unsqueeze(-1) * fn * vtan / torch.sqrt((vtan * vtan).sum(-1, keepdim=True) + V_EPS ** 2)
    acc = acc + fn * nrm + ff
    # inertia: motion across the rod may include rolling of the round rod (effective inertia factor lat_mass);
    # the free-end node may carry a different lumped mass (tip_mass, 0.5 if the points are ends of uniform links)
    at = (acc * tang).sum(-1, keepdim=True) * tang
    acc = at + (acc - at) / prm["lat_mass"].unsqueeze(-1)
    acc = torch.cat([acc[:, :1] / prm["tip_mass"].unsqueeze(-1), acc[:, 1:]], 1)
    return acc


def physics(state, cmd):
    """Advance positions by one agent step (0.02 s) under a (zone-limited) command.  Pure function."""
    prm = state["prm"]
    pos, vel = state["pos"], state["vel"]
    tool, tool_v, ref, ref_v = state["tool"], state["tool_v"], state["ref"], state["ref_v"]
    h = DT / N_SUB
    post = state["post"]
    for _ in range(N_SUB):
        ref, ref_v = _reference(ref, ref_v, cmd, h)
        tool, tool_v = _servo(tool, tool_v, ref, ref_v, prm, h)
        acc = _rod_accel(pos, vel, post, prm)
        vel = vel + acc * h
        vel = torch.cat([vel[:, :-1], tool_v.unsqueeze(1)], 1)
        p_old = pos
        pos = pos + vel * h
        pos = torch.cat([pos[:, :-1], tool.unsqueeze(1)], 1)
        if PBD_ITERS:
            # v2: near-inextensible links by position projection (inelastic: a snapping link loses its relative
            # normal velocity; this is what the official fit tried to mimic with c_stretch pinned at its bound)
            pos = _project_links(pos, prm)
            vel = torch.cat([((pos - p_old) / h)[:, :-1], tool_v.unsqueeze(1)], 1)
    return pos, vel, tool, tool_v, ref, ref_v


def _project_links(pos, prm):
    # inverse masses: free end 1/tip_mass, interior 1, tool node 0 (kinematic)
    B = pos.shape[0]
    w = torch.ones(B, N, 1, device=pos.device)
    w[:, 0] = 1.0 / prm["tip_mass"]
    w[:, -1] = 0.0
    for _ in range(PBD_ITERS):
        for par in (0, 1):                       # red-black Gauss-Seidel over the 19 links
            i = torch.arange(par, N - 1, 2, device=pos.device)
            d = pos[:, i + 1] - pos[:, i]
            L = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            wi, wj = w[:, i], w[:, i + 1]
            corr = (L - SEG) / (wi + wj).clamp_min(1e-9) * d / L
            pos = pos.index_add(1, i, wi * corr).index_add(1, i + 1, -wj * corr)
    return pos


def clip_cmd(a):
    lo = torch.tensor(ZONE[0], device=a.device, dtype=a.dtype)
    hi = torch.tensor(ZONE[1], device=a.device, dtype=a.dtype)
    return torch.maximum(torch.minimum(a, hi), lo)


def step(state, action):
    """One agent step with the exact lab rules.  Finished episodes are frozen.  Returns (state, reward, done)."""
    a = clip_cmd(torch.as_tensor(action, dtype=torch.float32, device=state["pos"].device))
    pos, vel, tool, tool_v, ref, ref_v = physics(state, a)
    live = (~state["done"]).unsqueeze(-1)
    sel = lambda new, old: torch.where(live.reshape(-1, *([1] * (new.dim() - 1))), new, old)
    pos, vel = sel(pos, state["pos"]), sel(vel, state["vel"])
    tool, tool_v, ref, ref_v = sel(tool, state["tool"]), sel(tool_v, state["tool_v"]), sel(ref, state["ref"]), sel(ref_v, state["ref_v"])
    t = state["t"] + (~state["done"]).long()
    rest = ((pos - state["pos"]).norm(dim=-1).amax(-1) <= REST_STEP) & ((tool - state["tool"]).norm(dim=-1) <= REST_STEP)
    d_prev = (state["pos"][:, 0] - state["target"]).norm(dim=-1)
    d = (pos[:, 0] - state["target"]).norm(dim=-1)
    lo = torch.tensor(ZONE[0], device=tool.device) - ZONE_TOL
    hi = torch.tensor(ZONE[1], device=tool.device) + ZONE_TOL
    inside = ((tool >= lo) & (tool <= hi)).all(-1)
    succ = rest & (d <= TARGET_R) & inside & (t >= 1) & ~state["done"]
    trunc = (~succ) & (t >= TIME_LIMIT) & ~state["done"]
    reward = torch.where(state["done"], torch.zeros_like(d), (d_prev - d) + 10.0 * succ.float())
    new = dict(state)
    new.update(pos=pos, vel=vel, tool=tool, tool_v=tool_v, ref=ref, ref_v=ref_v, t=t, prev_pos=state["pos"],
               prev_tool=state["tool"], done=state["done"] | succ | trunc, success=state["success"] | succ)
    return new, reward, new["done"]


@torch.no_grad()
def rollout_points(state, actions):
    """Open-loop physics only (no episode rules): returns positions (B,T,20,2) and tool (B,T,2)."""
    P, TL = [], []
    s = dict(state)
    for k in range(actions.shape[1]):
        pos, vel, tool, tool_v, ref, ref_v = physics(s, clip_cmd(actions[:, k]))
        s.update(pos=pos, vel=vel, tool=tool, tool_v=tool_v, ref=ref, ref_v=ref_v)
        P.append(pos)
        TL.append(tool)
    return torch.stack(P, 1), torch.stack(TL, 1), s


def state_from_log(L, t, params=None, device="cuda", idx=None):
    """State at logged reading t (t>=1 uses backward-difference node velocities) from a black-box log dict."""
    o = {k: L[k][t] for k in ("points", "tool_pos", "tool_vel", "post", "zone", "target", "t")}
    prev = None
    if t >= 1:
        prev = {k: L[k][t - 1] for k in ("points",)}
    return init_from_obs([o], [prev] if prev else None, params, device)
