"""Imagined lives: roll the code_wm controller through the code world (batched, GPU), with the world's parameters
randomised over what the logs could not pin down (see `randomise`)."""
from __future__ import annotations

import json
import math

import numpy as np
import torch

from experiments.drone.code_wm.controller import DEFAULT, advance, control, excitation, path_nodes, route_nodes
from worlds.drone import site_world as W


def load_world(path):
    prm = dict(W.DATASHEET)
    prm.update(json.load(open(path))["full"] if path else {})
    return prm


def compensation(belief: dict) -> dict:
    """Controller terms that invert the believed aircraft faults (rate bias / gain, thrust gain)."""
    g = float(belief["RATE_GAIN"])
    return dict(rate_gain=g, thrust_gain=float(belief["THRUST_GAIN"]), p_bias=float(belief["RATE_BIAS_P"]) / g,
                q_bias=float(belief["RATE_BIAS_Q"]) / g, r_bias=float(belief["RATE_BIAS_R"]) / g)


def randomise(prm: dict, B: int, device, gen, unc: dict) -> dict:
    """Per-env world parameters.  unc maps a parameter to ("add"|"mul"|"uniform", spread or (lo, hi))."""
    out = dict(prm)
    for k, (kind, v) in unc.items():
        u = torch.rand(B, device=device, generator=gen) * 2 - 1
        if kind == "add":
            out[k] = float(prm[k]) + v * u
        elif kind == "mul":
            out[k] = float(prm[k]) * (1 + v * u)
        elif kind == "uniform":
            out[k] = v[0] + (v[1] - v[0]) * (u + 1) / 2
    return out


def rollout(prm_c: dict, prm_w: dict, belief: dict, B: int = 1024, device="cuda", seed=0, id_time=0.0,
            mission=W.MISSION_A, time_limit=W.TIME_LIMIT_A, record=False, wind_ff=True, route=None, r_adv=0.8,
            movers=None):
    """movers: None (missions A/B) or a MOVER_PRIOR override dict (mission C, other aircraft sampled per env)."""
    g = torch.Generator(device=device).manual_seed(seed)
    start, relays = W.sample_start(B, device, g, mission)
    s = W.init_state(B, relays, start, device, prm_w)
    if movers is not None:
        s.update(W.sample_movers(B, device, g, movers))
    integ = torch.zeros(B, 3, device=device)
    rid = None
    if route is None:
        nodes = path_nodes(start, relays)
    else:
        nodes, rid = route_nodes(start, relays, route, return_relay_ids=True)
    gate = rid if (rid is not None and float(prm_c.get("return_fix", 0.0)) > 0) else None
    kr = torch.zeros(B, dtype=torch.long, device=device)
    hold = start[:, None].expand(B, nodes.shape[1], 3).clone()
    hold[:, 1:, 0] += 1e-3 * torch.arange(1, nodes.shape[1], device=device)
    n_steps = int(round(time_limit * 30))
    traj, mtraj, steps = [], [], 0
    for i in range(n_steps):
        o = W.observe(s, prm_w, g)
        tt = i / 30.0
        prm = dict(prm_c)
        exc = None
        if route is None:
            nd, kk = nodes, o["relay_index"]
        else:
            kr = advance(o["pos"], nodes, kr, r_adv, gate, o["relay_index"])
            nd, kk = nodes, kr
        if tt < id_time:
            nd, kk = hold, torch.zeros_like(kk)
            exc = torch.as_tensor(excitation(tt, id_time), dtype=torch.float32, device=device)[None].expand(B, 4)
            prm["speed"] = 0.0
        we = W.wind_field(o["pos"], belief) if wind_ff else None
        a, integ = control(prm, o["pos"], o["lin_vel"], o["quat"], nd, kk, integ, wind_est=we, excite=exc,
                           movers=o.get("movers") if float(prm_c.get("av_k", 0.0) if not torch.is_tensor(prm_c.get("av_k", 0.0)) else 1.0) > 0 else None)
        s, r, done = W.step(s, a, prm_w, time_limit, g)
        steps += int((~done).sum())
        if record:
            traj.append(s["pos"][:8].cpu().numpy())
            if movers is not None:
                mtraj.append(s["mp"][:8].cpu().numpy())
        if bool(done.all()):
            break
    c = s["cause"].cpu().numpy()
    t = s["t"].cpu().numpy()
    out = dict(cause=c, t=t, k=s["k"].cpu().numpy(), wm_steps=steps, final_pos=s["pos"].cpu().numpy())
    if record:
        out["traj"] = np.stack(traj, 1)
        if mtraj:
            out["mtraj"] = np.stack(mtraj, 1)
    return out


def summarise(r, mask=None):
    c, t = r["cause"], r["t"]
    if mask is not None:
        c, t = c[mask], t[mask]
    return dict(success=float((c == 1).mean()), collision=float((c == 2).mean()), oob=float((c == 3).mean()),
                timeout=float(((c == 4) | (c == 0)).mean()), t_success=float(t[c == 1].mean()) if (c == 1).any() else None)
