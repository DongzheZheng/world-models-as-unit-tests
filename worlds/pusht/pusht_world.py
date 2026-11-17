"""Code world of the Push-T rig (pure torch, batched, functional).

state = dict of tensors with leading batch dim B:
  p (B,2) pusher centre, pv (B,2) pusher velocity [u/s]
  q (B,3) block body-frame pose (x, y, theta)    -- the observed pose (body origin = middle of the bar's bottom edge)
  v (B,2) velocity of the block's centre of mass [u/s], w (B,) angular velocity [rad/s]
  t (B,) step counter
step(state, action (B,2) pusher target) -> (next_state, reward (B,), done (B,)); never mutates its input.

Mechanisms (each a named function below; numbers in PARAMS, evidence in the comments next to them):
  M1 pusher_pd       the pusher is a point driven by a PD law toward the held target; it is NOT slowed by the block
                     (kinematic w.r.t. the block)
  M2 contact         disc (radius 15) vs the drawn T (two boxes): velocity-level impulses at the closest point with a
                     restitution e, Coulomb friction mu at the contact and a penetration bias
  M3 block_inertia   the block is a rigid body with an unpublished centre of mass (com_u, com_v) in the body frame and
                     radius of gyration k about it (mass normalised to 1; the pusher is effectively infinitely heavy)
  M4 table_friction  after contact ends the block coasts and stops: viscous decay (lin_damp, ang_damp, 1/s) plus a
                     Coulomb part (lin_coul u/s^2, ang_coul rad/s^2)
  M5 walls           the block's corner points are kept inside the wall rectangle (crude; the controller avoids walls)
"""
from __future__ import annotations

import math

import torch

DT = 0.1                    # one agent step [s] (manual: control_hz = 10)
N_SUB = 10                  # integration substeps per agent step
R_PUSHER = 15.0             # datasheet
GOAL = (256.0, 256.0, math.pi / 4)
# the drawn T: boxes (centre u, centre v, half width, half height) in the body frame (datasheet, exact)
BOXES = ((0.0, 15.0, 60.0, 15.0), (0.0, 75.0, 15.0, 45.0))
WALL_LO, WALL_HI = 7.0, 504.0          # inner faces of the walls (task_card corners 5/506, half thickness 2)

# Fitted parameters (filled in by experiments/pusht/code_wm/fit_world.py; the values below are the fit result).
PARAMS = dict(
    # M1 pusher PD gains [1/s^2, 1/s]: fitted to free-space step responses + ramps; life 1 alone -> (99.8, 19.9),
    # life 2 alone -> (100.4, 20.1); mean pusher position error 0.01-0.02 u over 5-step replays. Pusher velocity is
    # unchanged while it pushes (life 1, steps 68-80: 28.7,-22.5 u/s constant) -> kinematic w.r.t. the block.
    kp=100.0, kv=20.0,
    # M3 centre of mass in the body frame [u]. The drawing's centroid is (0, 40.7); fits give (24.3, 30.2) on life 1,
    # (25.2, 29.1) on life 2, (24.1, 30.0) on lives 1+2. Evidence: a push along the symmetry axis through the stem tip
    # (life 1, steps 68-81) turned the block by -18 deg in 13 steps. Forcing the COM to the centroid raises the
    # 10-step replay loss from 0.41 to 3.06 (life 1).  THIS is what "does not push the way the drawing suggests" means.
    com_u=24.1, com_v=30.0,
    # M3 radius of gyration about the COM [u] (m = 1): 50.4 (life 1), 55.6 (life 2), 53.6 (lives 1+2). For a uniform
    # T about this COM it would be ~ 48; loosely determined (+-5).
    k_gyr=53.6,
    # M2 contact: restitution 0.10 / 0.00 / 0.03 (small, poorly determined); contact friction mu ~ 0 in every fit
    # (0.002 / 0.012 / 0.0002) -> the pusher slides freely along the block; bias = penetration recovery (0.05).
    restitution=0.03, mu=0.0, bias=0.052,
    # M4 table friction: release-and-watch (life 1 steps 81-90: block displacement shrinks ~x0.63 per step) ->
    # viscous decay ~4.6/s linear, ~5/s angular; small Coulomb terms (2.2 u/s^2, 0.01 rad/s^2).
    lin_damp=4.63, ang_damp=4.95, lin_coul=2.19, ang_coul=0.01,
)


# ---------------------------------------------------------------- objects
# Geometry is datasheet (exact). Physics parameters are per object; for the L every mechanism and every value that is
# not about mass distribution is carried over from the T ("same batch" field note), and only the mass-distribution
# parameters were refitted on the L's first life -- see PARAMS_L.
VERTS_T = ((-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30))   # CCW outline
CENTROID_T = (0.0, 40.714)
# L datasheet: leg u in [0,40], v in [0,110]; foot u in [40,100], v in [0,40]; origin = outer corner; goal (300,200,pi/2)
BOXES_L = ((20.0, 55.0, 20.0, 55.0), (70.0, 20.0, 30.0, 20.0))
VERTS_L = ((0, 0), (100, 0), (100, 40), (40, 40), (40, 110), (0, 110))
CENTROID_L = (37.647, 42.647)
GOAL_L = (300.0, 200.0, math.pi / 2)
PARAMS_L = dict(PARAMS)          # carried over from the T; overwritten below by the L fit (PARAMS_L.update)
PARAMS_L.update(
    # L life 1 (scripted probe, 300 steps), 10-step replay loss, all other values carried over from the T.
    # Loss with T values (incl. T's COM): 2.34; mass-only refit (mu kept 0): 0.83; mass + contact mu: 0.217;
    # refitting ALL parameters only reaches 0.195 -> pusher PD, floor friction, restitution and bias transfer.
    # M3 COM: (23.4, 55.6) vs the L's drawn centroid (37.6, 42.6) -> offset (-14, +13) toward the leg; fits with
    # different free sets give com_u 22.8-23.6, com_v 55.5-57.2. COM at the centroid: loss 0.79 vs 0.22.
    com_u=23.4, com_v=55.6,
    # M3 radius of gyration: 99 (83-101 across fits; 90 hit the first search bound) -- much larger than a uniform L
    # (~40): the L turns sluggishly. Loosely determined, partly traded against mu.
    k_gyr=99.0,
    # M2 contact friction: L life 1 alone is flat for mu >= 0.36 (loss 0.236 @0.36, 0.217 @5; 0.652 @0) and its fits
    # ran to the search bound; the wall-free windows of L life 2 have a clear minimum at mu = 0.36 (2.34 vs 3.75 @5,
    # 4.41 @0); a joint fit on both (wall-free) gives 0.36 with COM (23.3, 54.2), k 99.7. -> mu = 0.4, plausible range
    # 0.3-1 (runs/pusht/code_wm/L_mu_scan.json). Unlike the T (mu ~ 0) the pusher does NOT slide freely on the L.
    mu=0.4,
)
PARAMS_T = PARAMS
OBJECTS = dict(T=dict(boxes=BOXES, verts=VERTS_T, centroid=CENTROID_T, goal=GOAL, params=PARAMS_T),
               L=dict(boxes=BOXES_L, verts=VERTS_L, centroid=CENTROID_L, goal=GOAL_L, params=PARAMS_L))
OBJECT = "T"
VERTS, CENTROID = VERTS_T, CENTROID_T


def set_object(name):
    """Switch the module's geometry, goal and default parameters to object `name` ("T" or "L")."""
    global OBJECT, BOXES, VERTS, CENTROID, GOAL, PARAMS
    o = OBJECTS[name]
    OBJECT, BOXES, VERTS, CENTROID, GOAL, PARAMS = name, o["boxes"], o["verts"], o["centroid"], o["goal"], o["params"]


def _prm(params, name, like):
    v = (params or PARAMS).get(name, PARAMS[name])
    return torch.as_tensor(v, dtype=like.dtype, device=like.device)


def rot(th):
    c, s = torch.cos(th), torch.sin(th)
    return c, s


def to_body(q, pts):
    """World points pts (B,...,2) -> body frame of pose q (B,3)."""
    shp = (q.shape[0],) + (1,) * (pts.dim() - 2)
    c, s = rot(q[:, 2].reshape(shp))
    d = pts - q[:, :2].reshape(shp + (2,))
    return torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def to_world(q, pts):
    shp = (q.shape[0],) + (1,) * (pts.dim() - 2)
    c, s = rot(q[:, 2].reshape(shp))
    return torch.stack([c * pts[..., 0] - s * pts[..., 1], s * pts[..., 0] + c * pts[..., 1]], -1) + \
        q[:, :2].reshape(shp + (2,))


def t_sdf(pb):
    """Signed distance from body-frame points pb (...,2) to the T and the outward unit normal (body frame)."""
    best_d, best_n = None, None
    for (cu, cv, hu, hv) in BOXES:
        du, dv = pb[..., 0] - cu, pb[..., 1] - cv
        qu, qv = du.abs() - hu, dv.abs() - hv
        ou, ov = qu.clamp(min=0), qv.clamp(min=0)
        outside = torch.sqrt(ou * ou + ov * ov + 1e-12)
        inside = torch.maximum(qu, qv).clamp(max=0)
        d = torch.where((qu > 0) | (qv > 0), outside, inside)
        # normal: outside -> direction of (ou, ov); inside -> the nearest face
        nu_out, nv_out = ou / outside, ov / outside
        face_u = qu > qv
        nu = torch.where((qu > 0) | (qv > 0), nu_out, face_u.float())
        nv = torch.where((qu > 0) | (qv > 0), nv_out, (~face_u).float())
        n = torch.stack([nu * torch.sign(du + 1e-9), nv * torch.sign(dv + 1e-9)], -1)
        if best_d is None:
            best_d, best_n = d, n
        else:
            take = d < best_d
            best_d = torch.where(take, d, best_d)
            best_n = torch.where(take[..., None], n, best_n)
    return best_d, best_n


# ---------------------------------------------------------------- mechanisms
def pusher_pd(p, pv, target, params, dt):
    """M1: semi-implicit Euler of a = kp (target - p) - kv pv."""
    kp, kv = _prm(params, "kp", p), _prm(params, "kv", p)
    kp, kv = (kp[:, None] if kp.dim() == 1 else kp), (kv[:, None] if kv.dim() == 1 else kv)
    pv = pv + (kp * (target - p) - kv * pv) * dt
    return p + pv * dt, pv


def contact(p, pv, q, v, w, params):
    """M2+M3: one velocity-level contact impulse between the (kinematic) pusher and the block."""
    cu, cv = _prm(params, "com_u", p), _prm(params, "com_v", p)
    k = _prm(params, "k_gyr", p)
    e, mu, beta = _prm(params, "restitution", p), _prm(params, "mu", p), _prm(params, "bias", p)
    pb = to_body(q, p[:, None, :])[:, 0]
    d, nb = t_sdf(pb)
    pen = R_PUSHER - d
    c, s = rot(q[:, 2])
    n = torch.stack([c * nb[:, 0] - s * nb[:, 1], s * nb[:, 0] + c * nb[:, 1]], -1)        # outward, world
    tvec = torch.stack([-n[:, 1], n[:, 0]], -1)
    cpt = p - n * d[:, None]                                                               # contact point, world
    com = to_world(q, torch.stack([cu.expand_as(d), cv.expand_as(d)], -1)[:, None, :])[:, 0]
    r = cpt - com
    vc = v + w[:, None] * torch.stack([-r[:, 1], r[:, 0]], -1)
    u = pv - vc                                                                            # pusher rel. to block
    un, ut = (u * n).sum(-1), (u * tvec).sum(-1)
    rn = r[:, 0] * n[:, 1] - r[:, 1] * n[:, 0]
    rt = r[:, 0] * tvec[:, 1] - r[:, 1] * tvec[:, 0]
    I = k * k
    Kn, Kt = 1.0 + rn * rn / I, 1.0 + rt * rt / I
    # pusher approaching (un < 0) is stopped: un' = -e un + bias pen / h ; Delta un = jn Kn (pusher is kinematic)
    jn = (-(1 + e) * un + beta * pen.clamp(min=0) / (DT / N_SUB)) / Kn
    jn = torch.where(pen > 0, jn.clamp(min=0), torch.zeros_like(jn))
    jt = torch.maximum(torch.minimum(-ut / Kt, mu * jn), -mu * jn)       # Coulomb cone at the contact
    J = -(jn[:, None] * n + jt[:, None] * tvec)                         # impulse on the block (m = 1)
    v = v + J
    w = w + (r[:, 0] * J[:, 1] - r[:, 1] * J[:, 0]) / I
    return v, w


def table_friction(v, w, params, dt):
    """M4: viscous + Coulomb decay of the block's planar twist (about the COM)."""
    ld, ad = _prm(params, "lin_damp", v), _prm(params, "ang_damp", v)
    lc, ac = _prm(params, "lin_coul", v), _prm(params, "ang_coul", v)
    ld, lc = (ld[:, None] if ld.dim() == 1 else ld), (lc[:, None] if lc.dim() == 1 else lc)
    v = v * torch.exp(-ld * dt)
    sp = v.norm(dim=-1, keepdim=True)
    v = v * ((sp - lc * dt).clamp(min=0) / (sp + 1e-9))
    w = w * torch.exp(-ad * dt)
    w = torch.sign(w) * (w.abs() - ac * dt).clamp(min=0)
    return v, w


def integrate(q, v, w, params, dt):
    """Advance the body pose from the COM twist (the COM moves with v; the body origin follows the rotation)."""
    cu, cv = _prm(params, "com_u", q), _prm(params, "com_v", q)
    comb = torch.stack([cu.expand(q.shape[0]), cv.expand(q.shape[0])], -1)[:, None, :]
    com = to_world(q, comb)[:, 0] + v * dt
    th = q[:, 2] + w * dt
    q2 = torch.cat([com, th[:, None]], -1)
    origin = com - (to_world(q2, comb)[:, 0] - com)
    return torch.cat([origin, th[:, None]], -1)


def walls(q, v):
    """M5: push the block back inside the wall rectangle if a corner crossed it (zeroing the outward velocity)."""
    corners = torch.tensor(VERTS, dtype=q.dtype, device=q.device)
    cw = to_world(q, corners.expand(q.shape[0], -1, -1))
    lo = (WALL_LO - cw).clamp(min=0).amax(1)
    hi = (cw - WALL_HI).clamp(min=0).amax(1)
    shift = lo - hi
    v = torch.where(shift != 0, torch.zeros_like(v), v)
    return torch.cat([q[:, :2] + shift, q[:, 2:]], -1), v


# ---------------------------------------------------------------- coverage (reward)
def _goal_points(spacing=5.0, device="cpu", dtype=torch.float32):
    pts = []
    for (cu, cv, hu, hv) in BOXES:
        us = torch.arange(cu - hu + spacing / 2, cu + hu, spacing)
        vs = torch.arange(cv - hv + spacing / 2, cv + hv, spacing)
        U, V = torch.meshgrid(us, vs, indexing="ij")
        pts.append(torch.stack([U.flatten(), V.flatten()], -1))
    gp = torch.cat(pts).to(device=device, dtype=dtype)
    g = torch.tensor(GOAL, device=device, dtype=dtype)[None]
    return to_world(g, gp[None])[0]


_GP = {}


def coverage(q):
    """Approximate area(block ∩ goal)/area(goal): fraction of a 5-u grid on the goal T lying inside the block."""
    key = (q.device, q.dtype, OBJECT)
    if key not in _GP:
        _GP[key] = _goal_points(device=q.device, dtype=q.dtype)
    gp = _GP[key]
    d, _ = t_sdf(to_body(q, gp[None].expand(q.shape[0], -1, -1)))
    return (d < 0).float().mean(-1)


# ---------------------------------------------------------------- step
def make_state(p, pv, q, v=None, w=None):
    B = p.shape[0]
    z = torch.zeros(B, device=p.device, dtype=p.dtype)
    return dict(p=p, pv=pv, q=q, v=torch.zeros_like(p) if v is None else v, w=z if w is None else w,
                t=torch.zeros(B, device=p.device, dtype=torch.long))


def physics(state, action, params=None):
    p, pv, q, v, w = state["p"], state["pv"], state["q"], state["v"], state["w"]
    target = action.clamp(0.0, 512.0)
    h = DT / N_SUB
    for _ in range(N_SUB):
        p, pv = pusher_pd(p, pv, target, params, h)
        v, w = contact(p, pv, q, v, w, params)
        v, w = table_friction(v, w, params, h)
        q = integrate(q, v, w, params, h)
        q, v = walls(q, v)
    return dict(p=p, pv=pv, q=q, v=v, w=w, t=state["t"] + 1)


def step(state, action, params=None):
    s = physics(state, action, params)
    cov = coverage(s["q"])
    done = (cov > 0.95) | (s["t"] >= 300)
    return s, (cov / 0.95).clamp(0, 1), done
