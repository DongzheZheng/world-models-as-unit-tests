"""Code world of the "station" installation: a rope dragged across a table through a field of pegs.

What it is (inferred from observation logs + own lives, see runs/station/code_wm/diagnosis.json):
  * a ~0.99 m rope lying flat on the table, 20 markers spaced ~0.0523 m; marker 1 clamped at (0.05, 0), marker 20 held by
    the tool (layouts.npz: marker 1 std 0, |marker20 - tool| < 2e-6).
  * the rope slides on the table with friction; only the dragged part moves (overdamped, node speeds decay away from tool).
  * the "load" is the pulling force of the rope on the tool (end-link tension), which grows steeply with drag speed
    (life_1: 0.45 / 0.7 / 1.7 / 3.4 / 6.0 N at 0.02 / 0.05 / 0.10 / 0.15 / 0.20 m/s on the same rope) and when the rope nears
    full extension (life_1 t=400..424: 4 -> 7.5 N at 0.06 m/s as end-to-end distance went 0.91 -> 0.975 m), but is ~0.15 N for
    a nearly straight rope at rest (life_1 t>430) => it is a dynamic (friction + inertia) force, not static tension.
  * pegs: rigid cylinders r=0.03 block rope and tool; rope wrapped on a peg rubs on it (capstan-like extra drag, observation
    fit load/sum|v| grows 3.9 -> 6.1 with wrap angle).
  * tool: reference moves toward the command at <= 0.30 m/s; tool follows the reference with a force-limited PD servo; when
    blocked by a peg the reference keeps going, then the tool lurches (episode_906: lag 0.094 m then 0.74 m/s, load 21 N).

Model: explicit mass-spring-damper chain (tension-only links + weak bending), viscous + regularised Coulomb drag on the
table, extra tangential drag for nodes touching a peg, projection out of pegs, PD-servoed tool mass coupled to the rope.
All parameters live in state["par"] (each (B,1)) so they can be randomised per imagined episode.  Pure torch, batched,
functional: step() never mutates its input.
"""
from __future__ import annotations

import math
import torch

# ------------------------------------------------------------------ documented interface facts (manual / task_card)
DT = 0.05                      # agent step [s]
TIME_LIMIT = 600
LOAD_LIMIT = 10.0
TOL, REST_SPEED = 0.02, 0.02
TABLE_LO, TABLE_HI = (0.0, -0.4), (1.2, 0.4)
REF_SPEED = 0.30               # reference speed limit [m/s] (manual)
TOOL_R, POST_R = 0.015, 0.03   # manual
N = 20
# ------------------------------------------------------------------ geometry measured in the observation material
ANCHOR = (0.05, 0.0)           # layouts.npz points[:,0] = (0.05, 0.00) in 200/200 snapshots, never moves in logs
ROPE_Z = 0.004                 # z of all markers 0.0039..0.0042 in every log
SEG_L0 = 0.0523                # layouts.npz mean marker spacing (0.0519..0.0526), polyline 0.9939 m
ROPE_R = 0.004                 # rope half thickness (z = 0.004 resting on table) -> rope-peg contact radius 0.034

# ------------------------------------------------------------------ fitted parameters (fit_world.py, see diagnosis.json)
# CEM fit (experiments/station/code_wm/fit_world.py, runs/station/code_wm/fit2.json) to 120 twenty-step segments of
# observation episodes 900-907 + life_1 (speed steps 0.02..0.20 m/s, slack push, near-taut pull).  Mean marker error over
# 20 steps 5.4 mm (rope-stays-still baseline ~30 mm), mean load error 0.44 N.  Held out: 908, 909, lives 2-3.
# Model units are effective: the chain is a coarse 20-node stand-in for a continuous rope, so only the products that set
# the response (drag per speed, inertia, load scale) are meaningful, not the individual "physical" values.
DEFAULT_PAR = dict(
    m=0.0962,       # mass per rope node [kg] (effective; sets the superlinear speed dependence of the load)
    k=660.4,        # link stiffness [N/m] (bilateral)
    cd=24.42,       # link damping [N s/m]
    kb=6.254,       # bending stiffness [N/m]
    cv=1.236,       # viscous table drag per node [N s/m]
    fc=0.00219,     # Coulomb table friction per node [N] (fit says ~0: drag is viscous, life_1 speed steps)
    cpost=2.072,    # extra tangential drag of a node touching a peg [N s/m] (capstan-like rubbing)
    kp=1360.7,      # tool servo stiffness [N/m]
    kd=60.40,       # tool servo damping [N s/m]
    fmax=2.062,     # servo force limit [model N] (tool lags/snags under load: episodes 906, 908)
    mt=0.4275,      # tool mass [kg]
    lscale=4.321,   # load reading = lscale * mean |rope force on tool| + loff
    loff=0.2136,
    vread=0.898,    # reported tool_vel = vread * model tool velocity at end of step
)
N_SUB = 40                     # substeps per agent step (explicit integration, dt_s = 1.25 ms; stable for k/m <= 2e5)


def _par_tensor(par, B, dev):
    p = dict(DEFAULT_PAR)
    p.update(par or {})
    out = {}
    for k, v in p.items():
        t = torch.as_tensor(v, dtype=torch.float32, device=dev)
        out[k] = (t.reshape(-1, 1) if t.dim() > 0 else t.expand(1)).expand(B, 1).clone() if t.numel() in (1, B) else t
    return out


def init_from_obs(obs_batch, par=None, device="cuda"):
    """Build a state from a list of lab-format reset observations (dicts) or a dict of stacked arrays."""
    if isinstance(obs_batch, dict):
        ob = obs_batch
    else:
        import numpy as np
        ob = {k: np.stack([o[k] for o in obs_batch]) for k in ("points", "tool_pos", "posts", "post_sides", "target")}
        ob["tool_vel"] = np.stack([o.get("tool_vel", np.zeros(2)) for o in obs_batch])
        ob["load"] = np.array([float(o.get("load", 0.0)) for o in obs_batch])
    f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    x = f(ob["points"])[..., :2].clone()
    B = x.shape[0]
    v = torch.zeros_like(x)
    if "tool_vel" in ob:
        v[:, -1] = f(ob["tool_vel"])
    tool = f(ob["tool_pos"])
    x[:, -1] = tool
    st = dict(x=x, v=v, ref=tool.clone(), posts=f(ob["posts"]), post_sides=torch.as_tensor(ob["post_sides"], device=device).long(),
              target=f(ob["target"]), t=torch.zeros(B, dtype=torch.long, device=device),
              load=f(ob["load"]) if "load" in ob else torch.zeros(B, device=device),
              done=torch.zeros(B, dtype=torch.bool, device=device), success=torch.zeros(B, dtype=torch.bool, device=device),
              overload=torch.zeros(B, dtype=torch.bool, device=device))
    st["max_load"] = st["load"].clone()
    st["par"] = _par_tensor(par, B, device)
    return st


# ------------------------------------------------------------------ side rule (manual, exact)
def winding(x, posts):
    """x (B,20,2), posts (B,3,2) -> W (B,3)."""
    d = x[:, None, :, :] - posts[:, :, None, :]                    # (B,3,20,2)
    a = torch.atan2(d[..., 1], d[..., 0])
    da = torch.remainder(a[..., 1:] - a[..., :-1] + math.pi, 2 * math.pi) - math.pi
    return da.sum(-1)


def sides(x, posts):
    w = winding(x, posts)
    th = math.pi / 2
    return torch.where(w >= th, 1, torch.where(w <= -th, -1, 0)).long()


# ------------------------------------------------------------------ physics
def _substep(x, v, ref, tool_v, posts, p, cmd, dt):
    """One explicit integration substep.  x, v: (B,20,2) (node 0 anchor, node 19 tool)."""
    # reference ramps toward the command at <= REF_SPEED (manual)
    dr = cmd - ref
    dn = dr.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    ref = ref + dr * torch.clamp(REF_SPEED * dt / dn, max=1.0)
    # links: bilateral spring-damper (life_1 t=236-244: pushing the slack rope back at 0.3 m/s gave 4.3 N, so the rope
    # pushes back on the tool too -> not tension-only)
    d = x[:, 1:] - x[:, :-1]                                       # (B,19,2)
    l = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    u = d / l
    ldot = ((v[:, 1:] - v[:, :-1]) * u).sum(-1, keepdim=True)
    ten = p["k"][:, :, None] * (l - SEG_L0) + p["cd"][:, :, None] * ldot   # (B,19,1), >0 = tension
    fl = ten * u                                                    # force on node i (towards i+1)
    F = torch.zeros_like(x)
    F[:, :-1] += fl
    F[:, 1:] -= fl
    # bending energy kb/2 * sum |x_{i-1} - 2 x_i + x_{i+1}|^2 (momentum conserving; the coiled rope relaxes slowly and
    # pushes on the tool at rest: reset load 0.23..0.85 N while markers creep, life_1 t<16 sum|v| ~0.1 m/s, load 0.45)
    b = x[:, :-2] - 2 * x[:, 1:-1] + x[:, 2:]                      # (B,18,2)
    kb = p["kb"][:, :, None]
    F[:, :-2] -= kb * b
    F[:, 1:-1] += 2 * kb * b
    F[:, 2:] -= kb * b
    f_tool = F[:, -1].clone()                                      # rope force on the tool (end link + bending)
    # table friction: viscous + regularised Coulomb (rope nodes 1..18)
    sp = v.norm(dim=-1, keepdim=True)
    F = F - p["cv"][:, :, None] * v - p["fc"][:, :, None] * v / torch.sqrt(sp ** 2 + 0.003 ** 2)
    # peg contact (rope): nodes inside contact radius -> extra tangential drag (capstan-like rubbing)
    rc = POST_R + ROPE_R
    dp = x[:, :, None, :] - posts[:, None, :, :]                   # (B,20,3,2)
    dpn = dp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    touch = (dpn < rc + 0.002).float()
    nrm = dp / dpn
    vt = v[:, :, None, :] - (v[:, :, None, :] * nrm).sum(-1, keepdim=True) * nrm
    F = F - (p["cpost"][:, :, None, None] * touch * vt).sum(2)
    a = F / p["m"][:, :, None]
    v_new = v + a * dt
    # tool: PD servo toward reference, force limited, plus the rope force on the tool
    fs = p["kp"] * (ref - x[:, -1]) - p["kd"] * tool_v
    fsn = fs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    fs = fs * torch.clamp(p["fmax"] / fsn, max=1.0)
    tool_a = (fs + f_tool) / p["mt"]
    tool_v = tool_v + tool_a * dt
    v_new[:, 0] = 0.0
    v_new[:, -1] = tool_v
    x_new = x + v_new * dt
    # projection out of pegs (rope radius rc, tool radius TOOL_R+POST_R); remove inward velocity
    rad = torch.full((1, N, 1, 1), rc, device=x.device)
    rad[:, -1] = TOOL_R + POST_R
    dp = x_new[:, :, None, :] - posts[:, None, :, :]
    dpn = dp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    nrm = dp / dpn
    pen = torch.clamp(rad - dpn, min=0.0)
    x_new = x_new + (pen * nrm).sum(2)
    vin = torch.clamp((v_new[:, :, None, :] * nrm).sum(-1, keepdim=True), max=0.0) * (pen > 0).float()
    v_new = v_new - (vin * nrm).sum(2)
    x_new[:, 0] = x[:, 0]
    tool_v = v_new[:, -1]
    # table bounds for the tool
    lo = torch.tensor(TABLE_LO, device=x.device); hi = torch.tensor(TABLE_HI, device=x.device)
    x_new[:, -1] = torch.maximum(torch.minimum(x_new[:, -1], hi), lo)
    return x_new, v_new, ref, tool_v, f_tool.norm(dim=-1)   # load sample = |rope force on tool|


def step(state, action):
    """state dict, action (B,2) commanded tool position -> (new_state, reward (B,), done (B,)).  Pure function."""
    p = state["par"]
    lo = torch.tensor(TABLE_LO, device=action.device); hi = torch.tensor(TABLE_HI, device=action.device)
    cmd = torch.maximum(torch.minimum(action.float(), hi), lo)
    x, v, ref = state["x"], state["v"], state["ref"]
    tool_v = v[:, -1]
    dt = DT / N_SUB
    acc = torch.zeros(x.shape[0], device=x.device)
    for _ in range(N_SUB):
        x, v, ref, tool_v, ten = _substep(x, v, ref, tool_v, state["posts"], p, cmd, dt)
        acc = acc + ten
    load = p["lscale"][:, 0] * acc / N_SUB + p["loff"][:, 0]
    alive = ~state["done"]
    # freeze finished episodes
    m3 = alive[:, None, None]
    x = torch.where(m3, x, state["x"]); v = torch.where(m3, v, state["v"]); ref = torch.where(alive[:, None], ref, state["ref"])
    load = torch.where(alive, load, state["load"])
    t = state["t"] + alive.long()
    # episode rules (manual, exact)
    tp, tgt = x[:, -1], state["target"]
    d = (tp - tgt).norm(dim=-1)
    d_prev = (state["x"][:, -1] - tgt).norm(dim=-1)
    sn = sides(x, state["posts"]); sp_prev = sides(state["x"], state["posts"])
    n_ok = (sn == state["post_sides"]).sum(-1); n_ok_prev = (sp_prev == state["post_sides"]).sum(-1)
    vr = (p["vread"] * v[:, -1]).norm(dim=-1)
    overload = alive & (load > LOAD_LIMIT)
    success = alive & ~overload & (d <= TOL) & (vr <= REST_SPEED) & (n_ok == 3)
    trunc = alive & ~overload & ~success & (t >= TIME_LIMIT)
    reward = 10 * (d_prev - d) + (n_ok - n_ok_prev).float() - 0.1 * torch.clamp(load / LOAD_LIMIT - 0.5, min=0.0)
    reward = reward + 10 * success.float() - 10 * overload.float()
    reward = torch.where(alive, reward, torch.zeros_like(reward))
    done = state["done"] | overload | success | trunc
    new = dict(state)
    new.update(x=x, v=v, ref=ref, t=t, load=load, done=done, success=state["success"] | success,
               overload=state["overload"] | overload, max_load=torch.maximum(state["max_load"], torch.where(alive, load, state["max_load"])))
    return new, reward, done


def observe(state):
    """Lab-format observations, batched: dict of tensors with a leading batch dim (use observe_list for numpy dicts)."""
    x = state["x"]
    B = x.shape[0]
    pts = torch.cat([x, torch.full((B, N, 1), ROPE_Z, device=x.device)], -1)
    return dict(points=pts, tool_pos=x[:, -1], tool_vel=state["par"]["vread"] * state["v"][:, -1], posts=state["posts"],
                post_sides=state["post_sides"], sides_now=sides(x, state["posts"]), target=state["target"],
                load=state["load"], t=state["t"])


def observe_list(state):
    o = observe(state)
    o = {k: v.detach().cpu().numpy() for k, v in o.items()}
    B = o["points"].shape[0]
    out = []
    for i in range(B):
        d = {k: (o[k][i].astype("float64") if k not in ("post_sides", "sides_now", "t") else o[k][i].astype("int64")) for k in o}
        d["load"] = float(o["load"][i]); d["t"] = int(o["t"][i])
        out.append(d)
    return out


def episode_info(state):
    x = state["x"]
    sn = sides(x, state["posts"])
    return dict(success=state["success"], overload=state["overload"], sides_ok=(sn == state["post_sides"]).sum(-1),
                final_dist=(x[:, -1] - state["target"]).norm(dim=-1), max_load=state["max_load"], n_steps=state["t"])
