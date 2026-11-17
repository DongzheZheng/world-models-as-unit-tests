"""Code world of the container-yard site (code_wm pilot; pure torch, batched, differentiable in the dynamics).

Picture: a 27 g quadrotor commanded in body rates + collective throttle at 30 Hz flies through a yard whose walls are
axis-aligned boxes (public site map, copied below).  The world is integrated at 120 Hz (4 sub-steps per agent step).

Mechanisms (each a named parameter in PRM; evidence in the comments; DATASHEET = from the manual, not fitted):
  geometry       BOXES, yard bounds, ceiling 6 m, collision = sdf(box) < DRONE_R or z < DRONE_R
  command delay  CMD_DELAY agent steps before a command reaches the flight controller
  rate loop      first-order lag of the body rates toward RATE_GAIN * command (TAU_RATE) + RATE_BIAS (body, rad/s)
  thrust         f = g * THRUST_GAIN * (T/0.3)^2 per unit mass along body z, first-order lag TAU_MOTOR
  drag           a = -DRAG * v + DRAG * WIND0
  aisle jets     y-acceleration fields across the canyon at the two side aisles (wind_field) + OU gusts GUST_STD
  sensor noise   OBS_NOISE (pos m, vel m/s) added to observations only

Functional API:  s = init_state(B, relays, start, device, prm, gen);  s2, r, done = step(s, action, prm)
                 obs(s) -> dict of tensors shaped like the real observation (pos, lin_vel, quat, ang_vel, relays,...)
                 dyn_step(s, a, prm) -> s2  (physics only, differentiable; used for fitting)
"""
from __future__ import annotations

import math

import numpy as np
import torch

DT = 1.0 / 30.0
N_SUB = 4
G = 9.81

# ------------------------------------------------------------------ geometry (public site map, unmetered survey)
# (cx, cy, cz, hx, hy, hz): north wall x 10-22 & 25.5-50 (north aisle x 22-25.5), south wall x 10-38 & 41.5-50
# (south aisle x 38-41.5), canyon |y| < 2, pinch blocks at x 31-34 leave |y| < 1.6.
BOXES = [(16.0, 7.0, 3.885, 6.0, 5.0, 3.885), (37.75, 7.0, 3.885, 12.25, 5.0, 3.885),
         (24.0, -7.0, 3.885, 14.0, 5.0, 3.885), (45.75, -7.0, 3.885, 4.25, 5.0, 3.885),
         (32.5, 1.8, 3.885, 1.5, 0.2, 3.885), (32.5, -1.8, 3.885, 1.5, 0.2, 3.885)]
X_LIM, Y_LIM, CEIL, OOB_MARGIN = (0.0, 60.0), (-12.0, 12.0), 6.0, 2.0
START = (3.0, 0.0, 2.0)
MISSION_A = ((14.0, 0.0, 2.5), (28.0, 0.0, 2.5), (57.0, 0.0, 2.5))
TIME_LIMIT_A = 17.0
RELAY_RADIUS = 1.0
DRONE_R = 0.08        # PRIOR: cf2x arm 0.04 m + prop; collision margin used by the world

# ------------------------------------------------------------------ parameters
DATASHEET = dict(
    CMD_DELAY=0.0,        # agent steps (fractional, 0..2: linear blend of neighbouring commands)
    RATE_GAIN=0.9,        # DATASHEET: settles at 88-92 % of the command
    TAU_RATE=0.0145,      # DATASHEET: ~90 % within one agent step -> tau = (1/30)/ln 10
    RATE_BIAS_P=0.0, RATE_BIAS_Q=0.0, RATE_BIAS_R=0.0,   # rad/s added to the achieved body rates
    THRUST_GAIN=1.0,      # DATASHEET: hover at T = 0.30
    TAU_MOTOR=0.010,      # DATASHEET: 10 ms
    DRAG=0.0,             # 1/s, linear drag per unit mass (datasheet silent)
    WIND0_X=0.0, WIND0_Y=0.0, WIND0_Z=0.0,               # m/s uniform wind (acts through DRAG)
    JET_N_AY=0.0, JET_N_X=23.75, JET_N_SIG=1.5,   # north-aisle jet: y-accel at the mouth (m/s^2), centre (aisle x 22-25.5), spread
    JET_N_GY=0.0, JET_N_GZ=0.0, JET_N_SP=0.0,     # its change with distance into the canyon, with height, spreading rate
    JET_S_AY=0.0, JET_S_X=39.75, JET_S_SIG=1.5,   # south-aisle jet (aisle x 38-41.5 on the site map)
    JET_S_GY=0.0, JET_S_GZ=0.0, JET_S_SP=0.0,
    GUST_STD=0.0, GUST_TAU=1.0,                          # OU gust (m/s^2 of acceleration) std and time constant (s)
    OBS_NOISE_POS=0.0, OBS_NOISE_VEL=0.0,
)
PRM = dict(DATASHEET)     # the fitted world overrides entries here (see the bottom of the file)


def boxes_tensor(device):
    b = torch.tensor(BOXES, dtype=torch.float32, device=device)
    return b[:, :3], b[:, 3:]


def sdf_boxes(p: torch.Tensor, device=None) -> torch.Tensor:
    """Signed distance (B,) from points p (B,3) to the nearest wall box (differentiable in p)."""
    c, h = boxes_tensor(p.device)
    q = (p[:, None, :] - c[None]).abs() - h[None]
    out = q.clamp(min=0).norm(dim=-1)
    ins = q.amax(-1).clamp(max=0)
    return (out + ins).amin(-1)


def _col(v):
    """Per-env parameter (B,) -> (B,1) so it broadcasts against (B,3); scalars pass through."""
    return v[:, None] if torch.is_tensor(v) and v.dim() == 1 else v


def jet_accel(p: torch.Tensor, prm: dict, side: str) -> torch.Tensor:
    """y-acceleration (B,) of one aisle jet.  d = distance from the aisle mouth into the canyon (mouth at |y| = 2 m;
    d < 0 inside the aisle).  Peak AY at the mouth, scaled by (1 + GY*d + GZ*(z - 2.5)); Gaussian in x about the aisle
    centre with spread SIG + SP*d (the jet spreads as it crosses the canyon)."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    j = "JET_" + side + "_"
    d = (2.0 - y) if side == "N" else (y + 2.0)
    sig = (prm[j + "SIG"] + prm[j + "SP"] * d).clamp(min=0.4)
    amp = prm[j + "AY"] * (1 + prm[j + "GY"] * d + prm[j + "GZ"] * (z - 2.5)).clamp(min=0.0)
    return amp * torch.exp(-0.5 * ((x - prm[j + "X"]) / sig) ** 2)


def wind_field(p: torch.Tensor, prm: dict) -> torch.Tensor:
    """External (non-thrust, non-drag) acceleration field (B,3) at positions p (B,3) [m/s^2]:
    uniform wind through drag (DRAG * WIND0) + the two aisle jets (pure y acceleration)."""
    B = p.shape[0]
    w0 = torch.stack([torch.as_tensor(prm[k], device=p.device, dtype=p.dtype).expand(B) for k in ("WIND0_X", "WIND0_Y", "WIND0_Z")], -1)
    zero = torch.zeros_like(p[:, 0])
    return _col(prm["DRAG"]) * w0 + torch.stack([zero, jet_accel(p, prm, "N") + jet_accel(p, prm, "S"), zero], -1)


def wind_accel_np(prm: dict, pos) -> np.ndarray:
    """Expected external acceleration at one position (numpy helper for the real-world controller)."""
    p = torch.as_tensor(np.asarray(pos, dtype=np.float32))[None]
    return wind_field(p, prm)[0].double().numpy()


def quat_to_R(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def R_to_quat(R):
    """(B,3,3) -> (B,4) (x,y,z,w); assumes no 180-degree rotations (fine for flight)."""
    w = 0.5 * torch.sqrt((1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp(min=1e-8))
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    return torch.stack([x, y, z, w], -1)


def _expm_so3(phi):
    """Rodrigues: (B,3) rotation vector -> (B,3,3)."""
    th = phi.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    k = phi / th
    K = torch.zeros(phi.shape[0], 3, 3, device=phi.device, dtype=phi.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - K.transpose(1, 2)
    th = th[..., None]
    I = torch.eye(3, device=phi.device, dtype=phi.dtype)[None]
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def dyn_step(s: dict, a: torch.Tensor, prm: dict, gust_noise: torch.Tensor | None = None) -> dict:
    """Physics of one agent step (no termination logic).  s keys: pos, vel, R, w, f, prev_a, gust."""
    # command delay: CMD_DELAY in [0, 2] agent steps, linear blend of the current and two previous commands
    d = prm["CMD_DELAY"]
    d = d if torch.is_tensor(d) else torch.tensor(float(d), device=a.device)
    d = _col(d.clamp(0, 2))
    d1 = d.clamp(max=1)
    d2 = (d - 1).clamp(min=0)
    u = (1 - d1) * a + (d1 - d2) * s["prev_a"] + d2 * s["prev_a2"]
    rate_cmd = u[:, :3]
    T = u[:, 3].clamp(0, 1)
    bias = torch.stack([torch.as_tensor(prm[k], device=a.device, dtype=a.dtype).expand(a.shape[0])
                        for k in ("RATE_BIAS_P", "RATE_BIAS_Q", "RATE_BIAS_R")], -1)
    f_cmd = G * prm["THRUST_GAIN"] * (T / 0.3) ** 2
    pos, vel, R, w, f = s["pos"], s["vel"], s["R"], s["w"], s["f"]
    gust = s["gust"]
    if gust_noise is not None:
        al = math.exp(-DT / float(prm["GUST_TAU"]))
        gust = al * gust + math.sqrt(1 - al * al) * float(prm["GUST_STD"]) * gust_noise
    h = DT / N_SUB
    aw = 1 - math.exp(-h / float(prm["TAU_RATE"])) if not torch.is_tensor(prm["TAU_RATE"]) else 1 - torch.exp(-h / prm["TAU_RATE"])
    af = 1 - math.exp(-h / float(prm["TAU_MOTOR"])) if not torch.is_tensor(prm["TAU_MOTOR"]) else 1 - torch.exp(-h / prm["TAU_MOTOR"])
    w_tgt = _col(prm["RATE_GAIN"]) * rate_cmd + bias
    for _ in range(N_SUB):
        w = w + aw * (w_tgt - w)
        f = f + af * (f_cmd - f)
        acc = R[:, :, 2] * f[:, None] - _col(prm["DRAG"]) * vel + wind_field(pos, prm) + gust
        acc = acc + torch.tensor([0.0, 0.0, -G], device=a.device, dtype=a.dtype)
        vel = vel + h * acc
        pos = pos + h * vel
        R = R @ _expm_so3(w * h)
    return dict(s, pos=pos, vel=vel, R=R, w=w, f=f, prev_a=a, prev_a2=s["prev_a"], gust=gust)


def init_state(B: int, relays: torch.Tensor, start: torch.Tensor, device, prm: dict | None = None) -> dict:
    """relays (B,3,3), start (B,3): hover-like start (level, zero velocity, thrust at hover)."""
    prm = prm or PRM
    f0 = torch.full((B,), G, device=device)
    return dict(pos=start.clone(), vel=torch.zeros(B, 3, device=device), R=torch.eye(3, device=device).expand(B, 3, 3).clone(),
                w=torch.zeros(B, 3, device=device), f=f0, prev_a=torch.tensor([0, 0, 0, 0.3], device=device).expand(B, 4).clone(),
                prev_a2=torch.tensor([0, 0, 0, 0.3], device=device).expand(B, 4).clone(),
                gust=torch.zeros(B, 3, device=device), relays=relays.clone(), k=torch.zeros(B, dtype=torch.long, device=device),
                t=torch.zeros(B, device=device), done=torch.zeros(B, dtype=torch.bool, device=device),
                cause=torch.zeros(B, dtype=torch.long, device=device))   # 0 running 1 success 2 collision 3 oob 4 timeout


def sample_start(B, device, gen=None, mission=MISSION_A):
    """Episode jitter as stated in the manual: start +-1 m, relays +-0.5 m (uniform; z jitter assumed the same)."""
    st = torch.tensor(START, device=device) + (torch.rand(B, 3, device=device, generator=gen) * 2 - 1) * torch.tensor([1.0, 1.0, 0.3], device=device)
    rl = torch.tensor(mission, device=device)[None] + (torch.rand(B, len(mission), 3, device=device, generator=gen) * 2 - 1) * 0.5
    return st, rl


def step(s: dict, a: torch.Tensor, prm: dict | None = None, time_limit: float = TIME_LIMIT_A, gen=None):
    """Full environment step: physics + relays + termination + reward.  Finished envs are frozen."""
    prm = prm or PRM
    B = a.shape[0]
    gn = torch.randn(B, 3, device=a.device, generator=gen) if prm["GUST_STD"] > 0 else None
    s2 = dyn_step(s, a, prm, gn)
    ar = torch.arange(B, device=a.device)
    nrel = s["relays"].shape[1]
    kk = s["k"].clamp(max=nrel - 1)
    d_prev = (s["pos"] - s["relays"][ar, kk]).norm(dim=-1)
    d_new = (s2["pos"] - s["relays"][ar, kk]).norm(dim=-1)
    reached = d_new < RELAY_RADIUS
    k2 = s["k"] + reached.long()
    success = k2 >= nrel
    p = s2["pos"]
    coll = (sdf_boxes(p) < DRONE_R) | (p[:, 2] < DRONE_R)
    if "mp" in s:                                                     # mission C: other aircraft
        s2 = dict(s2, **movers_step(s, s["pos"], gen))
        coll = coll | mover_hit(s2, p)
    oob = (p[:, 0] < X_LIM[0] - OOB_MARGIN) | (p[:, 0] > X_LIM[1] + OOB_MARGIN) | (p[:, 1].abs() > Y_LIM[1] + OOB_MARGIN) | (p[:, 2] > CEIL)
    t2 = s["t"] + DT
    tout = t2 >= time_limit - 1e-6
    r = (d_prev - d_new) - 0.01 + 20 * reached.float() + 50 * success.float() - 50 * (coll | oob).float()
    cause = torch.where(success, 1, torch.where(coll, 2, torch.where(oob, 3, torch.where(tout, 4, 0))))
    done_new = cause > 0
    live = ~s["done"]
    out = {}
    for key, v in s2.items():
        if torch.is_tensor(v) and v.shape[:1] == (B,) and key not in ("done", "cause", "k", "t", "relays"):
            m = live.view(-1, *([1] * (v.dim() - 1)))
            out[key] = torch.where(m, v, s[key])
        else:
            out[key] = v
    out["k"] = torch.where(live, k2.clamp(max=nrel - 1) if False else k2, s["k"])
    out["t"] = torch.where(live, t2, s["t"])
    out["cause"] = torch.where(live, cause, s["cause"])
    out["done"] = s["done"] | (live & done_new)
    return out, r * live.float(), out["done"]


# ------------------------------------------------------------------ other aircraft (mission C)
# Evidence: code_wm_C lives (black-box instrumentation of the true mover states; see MOVER_PRIOR for the fitted ranges).
# Each mover is a sphere (MOVER_R, manual) following one of three kinematic behaviours, constant speed between stops:
#   0 shuttle_x  back and forth along x between two ends at fixed (y, z), dwelling briefly at each end
#   1 shuttle_y  back and forth across the canyon (y) at fixed (x, z)
#   2 wanderer   hops at high speed to random waypoints inside its own box of the canyon, dwells, picks again; with
#                probability CHASE per hop the waypoint is our drone's current position (clamped into its box)
#   3 guard      a shuttle_x that reacts: once we come within G_TRIG it leaves its lane and flies toward our current
#                position at G_SPEED (3-D pursuit, z capped at G_ZMAX), until we are beyond G_REL; then it returns to
#                its lane and resumes shuttling.  Evidence: life 2 mover 4 (lane 1.78 m/s, z 2.62): triggered at 2.4 m
#                when we passed above, climbed at 1.3 m/s toward us (to z 4.2, y/z only), back to its lane at ~10 m;
#                life 3 mover 2 (lane 2.73 m/s, z 2.11): triggered at 3.2 m, pursued at 1.5 m/s in 3-D (to z 3.4),
#                stopped at ~9 m.  Of 8 close (< 3.2 m) passes of x-shuttles in lives 2-3, 2 pursued, 1 dodged away
#                at 3.5 m/s for 0.8 s (life 3 mover 0; not modelled: it moves away), 5 ignored us -> P_GUARD 0.25.
# All of them start at rest; per-episode parameters are drawn from MOVER_PRIOR (behaviour varies between episodes).
MOVER_R = 0.35
K_MOVERS = 8
TIME_LIMIT_C = 19.4
MOVER_PRIOR = dict(        # ranges cover what code_wm_C lives 1-4 showed (fit_movers.py), widened a little
    n_shuttle_x=(1, 5), n_shuttle_y=(0, 4), n_wanderer=(0, 4),
    sx_y=(-1.3, 1.3), sx_z=(2.0, 3.6), sx_speed=(0.8, 3.8), sx_span=(6.0, 20.0), sx_x=(10.0, 50.0), sx_dwell=(0.0, 1.6),
    sy_x=(12.0, 48.0), sy_halfspan=(1.2, 1.6), sy_z=(2.0, 3.6), sy_speed=(0.6, 3.8), sy_dwell=(0.0, 1.0),
    w_xlen=(7.0, 24.0), w_x=(10.0, 50.0), w_y=(-1.3, 1.3), w_z=(1.6, 3.6), w_speed=(3.0, 6.0), w_dwell=(0.3, 5.0),
    w_chase=(0.0, 0.3), start_wait=(0.0, 2.0),
    p_guard=0.25, g_trig=(1.5, 3.5), g_rel=(5.0, 10.0), g_speed=(1.0, 2.0), g_zmax=5.5,
    p_aisle_sy=0.5, sy_aisle_x=((23.75, 39.75), 0.6), sy_aisle_depth=(2.0, 7.0))


def _u(shape, lo_hi, device, gen):
    lo, hi = lo_hi
    return lo + (hi - lo) * torch.rand(shape, device=device, generator=gen)


def sample_movers(B: int, device, gen=None, prior: dict | None = None) -> dict:
    """Per-env mover population and parameters -> state entries m_* (B, K[, 3])."""
    pr = dict(MOVER_PRIOR); pr.update(prior or {})
    K = K_MOVERS
    ri = lambda lh: torch.randint(int(lh[0]), int(lh[1]) + 1, (B,), device=device, generator=gen)
    nx, ny, nw = ri(pr["n_shuttle_x"]), ri(pr["n_shuttle_y"]), ri(pr["n_wanderer"])
    j = torch.arange(K, device=device)[None]
    kind = torch.where(j < nx[:, None], 0, torch.where(j < (nx + ny)[:, None], 1, 2))
    active = j < (nx + ny + nw).clamp(max=K)[:, None]
    lo = torch.zeros(B, K, 3, device=device); hi = torch.zeros(B, K, 3, device=device)
    # shuttle_x
    span = _u((B, K), pr["sx_span"], device, gen)
    x0 = pr["sx_x"][0] + (pr["sx_x"][1] - pr["sx_x"][0] - span) * torch.rand(B, K, device=device, generator=gen)
    yx, zx = _u((B, K), pr["sx_y"], device, gen), _u((B, K), pr["sx_z"], device, gen)
    # shuttle_y
    xy, hs, zy = _u((B, K), pr["sy_x"], device, gen), _u((B, K), pr["sy_halfspan"], device, gen), _u((B, K), pr["sy_z"], device, gen)
    # wanderer box
    wl = _u((B, K), pr["w_xlen"], device, gen)
    wx0 = pr["w_x"][0] + (pr["w_x"][1] - pr["w_x"][0] - wl) * torch.rand(B, K, device=device, generator=gen)
    k0, k1 = kind == 0, kind == 1
    lo[..., 0] = torch.where(k0, x0, torch.where(k1, xy, wx0)); hi[..., 0] = torch.where(k0, x0 + span, torch.where(k1, xy, wx0 + wl))
    lo[..., 1] = torch.where(k0, yx, torch.where(k1, -hs, torch.full_like(hs, pr["w_y"][0])))
    hi[..., 1] = torch.where(k0, yx, torch.where(k1, hs, torch.full_like(hs, pr["w_y"][1])))
    lo[..., 2] = torch.where(k0, zx, torch.where(k1, zy, torch.full_like(zy, pr["w_z"][0])))
    hi[..., 2] = torch.where(k0, zx, torch.where(k1, zy, torch.full_like(zy, pr["w_z"][1])))
    speed = torch.where(k0, _u((B, K), pr["sx_speed"], device, gen), torch.where(k1, _u((B, K), pr["sy_speed"], device, gen),
                                                                                    _u((B, K), pr["w_speed"], device, gen)))
    dwell = torch.where(k0, _u((B, K), pr["sx_dwell"], device, gen), torch.where(k1, _u((B, K), pr["sy_dwell"], device, gen),
                                                                                    _u((B, K), pr["w_dwell"], device, gen)))
    chase = torch.where(kind == 2, _u((B, K), pr["w_chase"], device, gen), torch.zeros_like(dwell))
    # shuttle_y through a side aisle: x at an aisle centre, span from the far canyon side deep into the aisle
    aisle = k1 & (torch.rand(B, K, device=device, generator=gen) < pr["p_aisle_sy"])
    north = torch.rand(B, K, device=device, generator=gen) < 0.5
    ax = torch.where(north, pr["sy_aisle_x"][0][0], pr["sy_aisle_x"][0][1]) + _u((B, K), (-pr["sy_aisle_x"][1], pr["sy_aisle_x"][1]), device, gen)
    depth = _u((B, K), pr["sy_aisle_depth"], device, gen)
    lo[..., 0] = torch.where(aisle, ax, lo[..., 0]); hi[..., 0] = torch.where(aisle, ax, hi[..., 0])
    lo[..., 1] = torch.where(aisle, torch.where(north, torch.full_like(depth, -1.55), -2.0 - depth), lo[..., 1])
    hi[..., 1] = torch.where(aisle, torch.where(north, 2.0 + depth, torch.full_like(depth, 1.55)), hi[..., 1])
    guard = k0 & (torch.rand(B, K, device=device, generator=gen) < pr["p_guard"])
    kind = torch.where(guard, 3, kind)
    g_trig, g_rel, g_speed = _u((B, K), pr["g_trig"], device, gen), _u((B, K), pr["g_rel"], device, gen), _u((B, K), pr["g_speed"], device, gen)
    mp = lo + (hi - lo) * torch.rand(B, K, 3, device=device, generator=gen)
    tgt = torch.where(torch.rand(B, K, 1, device=device, generator=gen) < 0.5, lo, hi)
    tgt = torch.where((kind == 2)[..., None], lo + (hi - lo) * torch.rand(B, K, 3, device=device, generator=gen), tgt)
    return dict(m_kind=kind, m_active=active, m_lo=lo, m_hi=hi, m_speed=speed, m_dwell=dwell, m_chase=chase, mp=mp,
                mv=torch.zeros(B, K, 3, device=device), m_tgt=tgt, m_wait=_u((B, K), pr["start_wait"], device, gen),
                m_gtrig=g_trig, m_grel=g_rel, m_gspeed=g_speed, m_eng=torch.zeros(B, K, dtype=torch.bool, device=device),
                m_gzmax=torch.full((B, K), float(pr["g_zmax"]), device=device))


def movers_step(s: dict, own_pos: torch.Tensor, gen=None) -> dict:
    """Advance all movers by one agent step (kinematic, not differentiable)."""
    mp, tgt, wait = s["mp"], s["m_tgt"], s["m_wait"]
    d = tgt - mp
    dist = d.norm(dim=-1)
    moving = wait <= 0
    stepl = torch.minimum(s["m_speed"] * DT, dist)
    mp2 = torch.where(moving[..., None], mp + d / dist.clamp(min=1e-6)[..., None] * stepl[..., None], mp)
    v = torch.where(moving[..., None], d / dist.clamp(min=1e-6)[..., None] * s["m_speed"][..., None], torch.zeros_like(mp))
    arrived = moving & (dist - stepl < 1e-4)
    B, K = wait.shape
    dev = mp.device
    # next target: shuttles swap ends, wanderers draw a point in their box (or our position, with prob CHASE)
    lo, hi = s["m_lo"], s["m_hi"]
    other_end = torch.where((tgt - hi).abs().sum(-1, keepdim=True) < 1e-3, lo, hi)
    rnd = lo + (hi - lo) * torch.rand(B, K, 3, device=dev, generator=gen)
    me = torch.maximum(torch.minimum(own_pos[:, None].expand(B, K, 3), hi), lo)
    chase = torch.rand(B, K, device=dev, generator=gen) < s["m_chase"]
    wand = torch.where(chase[..., None], me, rnd)
    new_tgt = torch.where((s["m_kind"] == 2)[..., None], wand, other_end)
    tgt2 = torch.where(arrived[..., None], new_tgt, tgt)
    dw = s["m_dwell"] * (0.4 + 1.2 * torch.rand(B, K, device=dev, generator=gen))
    wait2 = torch.where(arrived, dw, wait - DT)
    v2 = torch.where(arrived[..., None], torch.zeros_like(v), v)
    # guards: engage within G_TRIG, release beyond G_REL; engaged -> fly toward our position; else back to the lane
    gd = s["m_kind"] == 3
    dme = (own_pos[:, None] - mp).norm(dim=-1)
    eng = gd & ((dme < s["m_gtrig"]) | (s["m_eng"] & (dme < s["m_grel"])))
    lane = s["m_lo"][..., 1:]
    me3 = own_pos[:, None].expand(B, K, 3)
    goal = torch.stack([me3[..., 0], me3[..., 1].clamp(-1.6, 1.6), torch.minimum(me3[..., 2], s["m_gzmax"])], -1)
    back = torch.cat([mp2[..., :1], lane], -1)                    # lane point at the current x
    goal = torch.where(eng[..., None], goal, back)
    e = goal - mp
    en = e.norm(dim=-1)
    off = gd & (eng | ((mp[..., 1:] - lane).norm(dim=-1) > 0.02))
    gstep = torch.minimum(s["m_gspeed"] * DT, en)
    gp = mp + e / en.clamp(min=1e-6)[..., None] * gstep[..., None]
    gv = e / en.clamp(min=1e-6)[..., None] * s["m_gspeed"][..., None] * (en > 1e-3)[..., None]
    mp3 = torch.where(off[..., None], gp, mp2)
    v3 = torch.where(off[..., None], gv, v2)
    return dict(mp=mp3, mv=v3, m_tgt=tgt2, m_wait=wait2, m_eng=eng)


def mover_hit(s: dict, own_pos: torch.Tensor) -> torch.Tensor:
    d = (s["mp"] - own_pos[:, None]).norm(dim=-1)
    return ((d < MOVER_R + DRONE_R) & s["m_active"]).any(-1)


def observe_movers(s: dict, own_pos: torch.Tensor, max_range: float = 15.0) -> tuple:
    """(B, 8, 7) rows (pos, vel, radius) of active movers within 15 m, nearest first, zeros elsewhere.  Line of sight
    is not modelled (inside the canyon the walls rarely hide a mover that is within 15 m)."""
    d = (s["mp"] - own_pos[:, None]).norm(dim=-1)
    vis = s["m_active"] & (d < max_range)
    key = torch.where(vis, d, torch.full_like(d, 1e9))
    order = key.argsort(-1)
    g = lambda x: torch.gather(x, 1, order[..., None].expand(-1, -1, x.shape[-1]))
    rows = torch.cat([g(s["mp"]), g(s["mv"]), torch.full_like(d, MOVER_R)[..., None]], -1)
    vs = torch.gather(vis, 1, order)
    return rows * vs[..., None].float(), vs.sum(-1)


def observe(s: dict, prm: dict | None = None, gen=None) -> dict:
    """Observation like the real one (world-frame pos/vel, quat body->world, body rates, relays, relay_index)."""
    prm = prm or PRM
    B = s["pos"].shape[0]
    nz = lambda sd: (torch.randn(B, 3, device=s["pos"].device, generator=gen) * sd) if sd > 0 else 0.0
    o = dict(pos=s["pos"] + nz(prm["OBS_NOISE_POS"]), lin_vel=s["vel"] + nz(prm["OBS_NOISE_VEL"]),
             quat=R_to_quat(s["R"]), ang_vel=s["w"], relays=s["relays"],
             relay_index=s["k"].clamp(max=s["relays"].shape[1] - 1), t=s["t"])
    if "mp" in s:
        o["movers"], o["n_movers"] = observe_movers(s, s["pos"])
    return o


# ------------------------------------------------------------------ fitted world (code_wm lives 1-3, mission A, ctbr)
# Fitted by gradient descent through dyn_step on every 20-step open-loop window of lives 1-3 (fit_world.py).
# Open-loop position error after 1 s, median over windows: datasheet 1.2-1.9 m -> this world 0.11-0.20 m; on lives it had
# not seen: life-1 world on life 2 0.21 m, lives-1+2 world on life 3 0.25 m (datasheet 1.23 / 1.90 m).
FITTED = dict(
    CMD_DELAY=1.04,       # commands act one agent step late: life 1 regression w[t+1] on a[t] ~0, on a[t-1] 0.88; fit 1.00 (l1), 1.04 (l1-3)
    RATE_GAIN=0.89,       # datasheet 0.9 confirmed
    TAU_RATE=0.006,       # faster than datasheet once the delay is modelled (fit, weakly identified)
    RATE_BIAS_P=-0.52,    # FAULT: constant body-rate offsets (rad/s) -- in steady cruise of lives 1-2 the controller had to
    RATE_BIAS_Q=+0.52,    #   command (+0.5, -0.5, +0.27) to hold zero rates; same values fitted on each life separately
    RATE_BIAS_R=-0.26,    #   (l1: -0.50/+0.49/-0.24, l1-2: -0.50/+0.50/-0.25).  Left-down / nose-down / clockwise drift.
    THRUST_GAIN=0.875,    # FAULT: 12.5 % thrust deficit, hover needs T = 0.32 not 0.30 (vertical residual -1.3 m/s^2, life 1)
    DRAG=0.21,            # 1/s; x residual -1.2 m/s^2 at 6 m/s cruise (life 2); trades off against WIND0_X
    WIND0_X=0.82, WIND0_Y=-0.40,   # weak, poorly identified (drag * wind = 0.17, -0.08 m/s^2)
    JET_N_AY=-16.7, JET_N_X=23.82, JET_N_SIG=1.38,  # SITE: jet out of the north aisle pushes toward -y; life 1 crashed into
    JET_N_GY=-0.054, JET_N_GZ=-0.11, JET_N_SP=-0.06,  #   the south wall at x 24.2 (y-accel residual -9..-11 m/s^2 at x 22-24)
    JET_S_AY=+10.4, JET_S_X=39.96, JET_S_SIG=1.23,  # SITE: jet out of the south aisle pushes toward +y; found in life 2
    JET_S_GY=0.019, JET_S_GZ=-0.048, JET_S_SP=0.043,  #   (residual +9 m/s^2 at x 39-41), probed at z 1.5 / y -2.4 in life 3
)
PRM.update(FITTED)
# What is still not captured: within the jets the one-step y-acceleration residual stays ~1.5-2 m/s^2 rms (unsteady or
# structured flow); controller synthesis randomises jet strength (+-30 %), position (+-0.4 m) and adds OU gusts.
# Held out: this lives-1-3 world on lives 4 / 5 (never fitted): 1 s open-loop error 0.16 / 0.23 m (datasheet 1.31 / 2.07 m).
# Refit on all five lives (runs/drone/code_wm/fit_all.json, for phase 2) moves nothing materially: delay 1.03, bias
# (-0.53, +0.53, -0.26), thrust 0.876, north jet -15.5 at x 23.91, south jet +10.1 at x 39.95.
