"""Screen arbitrary lab-protocol controllers INSIDE the code world (no real episodes).

run_in_world(policy_factory, policy_kwargs, n_episodes, seed0, cap_tics=24000) -> list of
    {"ret": tics survived (lab reward), "capped": bool}
The world (worlds/fps/take_cover_world.py, unchanged, same per-episode randomisation as for my own controller) is
stepped at frame_skip 4 in a batch of episodes; each step a lab-format observation dict is built per episode:
  tic, player {x,y,z,angle,pitch,vx,vy,vz,health,armor,dead,on_ground,...}, objects = VISIBLE objects by the
  world's visibility rule, with stable ids, names "DoomImp" / "DoomImpBall", x,y,z,angle,vx,vy,vz and a crude bbox,
  depth_scan = zeros(64).
Approximations (the world does not model these): depth_scan (zeros; no screened program reads it); bounding boxes
(rough projection only); exploded fireballs are shown for ONE observation with v=0 at their last position (the lab
shows them for a few steps); the player's x is always -416 and vx = 0 (no hit knock-back); damage_taken/hits_taken
tracked from world health.  `bind()` gets the lab's public floor plan (computed once per worker, cached).
usage: python experiments/doom/code_wm/screen.py  -> runs/probe/screen_doom.json (aspire_doom v1..v5)
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from worlds.fps import take_cover_world as W

ACTIONS = ["noop", "MOVE_LEFT", "MOVE_RIGHT"]
IMP_ID0, BALL_ID0 = 1, 100000


def _cached_floor_plan():
    from reallab.doom import api
    if getattr(api, "_screen_cached", False):
        return
    fp = api.floor_plan("take_cover")
    api.floor_plan = lambda scenario: copy.deepcopy(fp)
    api._screen_cached = True


def _bbox(dx, dy, r, h):
    """crude screen box (lab: tan(b) = (159.5 - c)/160); only for completeness."""
    c = 159.5 - 160.0 * dy / max(dx, 1e-3)
    w = max(1.0, 160.0 * 2 * r / max(dx, 1e-3))
    hh = max(1.0, 160.0 * h / max(dx, 1e-3))
    return dict(sx=int(c - w / 2), sy=int(120 - hh / 2), sw=int(w), sh=int(hh))


def run_in_world(policy_factory: str, policy_kwargs: dict, n_episodes: int, seed0: int, cap_tics: int = 24000,
                 device: str = "cpu") -> list[dict]:
    torch.set_num_threads(1)
    _cached_floor_plan()
    mod, fn = policy_factory.split(":")
    factory = getattr(importlib.import_module(mod), fn)
    B = n_episodes
    pols = []
    for i in range(B):
        p = factory(**(policy_kwargs or {}))
        if hasattr(p, "bind"):
            p.bind("take_cover", list(ACTIONS))
        if hasattr(p, "set_seed"):
            p.set_seed(seed0 + i)
        p.reset()
        pols.append(p)
    g = torch.Generator(device=device).manual_seed(seed0)
    s = W.init_state(B, device, g)
    NB = s["ball_on"].shape[1]
    ball_id = np.zeros((B, NB), dtype=np.int64)
    next_id = np.full(B, BALL_ID0, dtype=np.int64)
    prev_on = np.zeros((B, NB), bool)
    prev_bx = np.zeros((B, NB))
    ret = np.zeros(B)
    alive = np.ones(B, bool)
    dmg = np.zeros(B)
    hits = np.zeros(B, dtype=np.int64)
    exploded = [[] for _ in range(B)]   # (id, x, y) shown once with v = 0
    max_steps = cap_tics // W.TICS_PER_STEP

    def obs_dicts(st, idx):
        c = {k: v.cpu().numpy() for k, v in st.items() if k in ("tic", "py", "vy", "imp_on", "imp_y", "ball_on", "bx",
                                                             "by", "bvx", "bvy", "health")}
        out = {}
        for e in idx:
            py = float(c["py"][e])
            objs = []
            for j in np.nonzero(c["imp_on"][e])[0]:
                dx, dy = W.IMP_X - W.PX, c["imp_y"][e, j] - py
                if dx > 0 and abs(dy) < dx * W.FOV_HALF_TAN + W.IMP_R:
                    objs.append(dict(id=int(IMP_ID0 + j), name="DoomImp", x=W.IMP_X, y=float(c["imp_y"][e, j]), z=0.0,
                                     angle=180.0, vx=0.0, vy=0.0, vz=0.0, **_bbox(dx, dy, W.IMP_R, 56)))
            for j in np.nonzero(c["ball_on"][e])[0]:
                dx, dy = c["bx"][e, j] - W.PX, c["by"][e, j] - py
                if dx > 0 and abs(dy) < dx * W.FOV_HALF_TAN + W.BALL_R:
                    vx, vy = float(c["bvx"][e, j]), float(c["bvy"][e, j])
                    objs.append(dict(id=int(ball_id[e, j]), name="DoomImpBall", x=float(c["bx"][e, j]),
                                     y=float(c["by"][e, j]), z=32.0, angle=math.degrees(math.atan2(vy, vx)) % 360,
                                     vx=vx, vy=vy, vz=0.0, **_bbox(dx, dy, W.BALL_R, 8)))
            for bid, x, y in exploded[e]:
                dx, dy = x - W.PX, y - py
                if dx > 0 and abs(dy) < dx * W.FOV_HALF_TAN + W.BALL_R:
                    objs.append(dict(id=int(bid), name="DoomImpBall", x=x, y=y, z=32.0, angle=0.0, vx=0.0, vy=0.0,
                                     vz=0.0, **_bbox(dx, dy, W.BALL_R, 8)))
            objs.sort(key=lambda o: o["id"])
            h = float(c["health"][e])
            out[e] = dict(tic=int(c["tic"][e]), objects=objs, depth_scan=np.zeros(64, np.float32),
                          player=dict(x=W.PX, y=py, z=0.0, angle=0.0, pitch=0.0, vx=0.0, vy=float(c["vy"][e]), vz=0.0,
                                      health=h, armor=0.0, selected_weapon=-1, ammo=-1.0, attack_ready=False,
                                      killcount=0, itemcount=0, dead=h <= 0, on_ground=True,
                                      damage_taken=float(dmg[e]), hits_taken=int(hits[e]), damage_dealt=0.0,
                                      hits_dealt=0))
        return out

    def update_ids(st):
        on = st["ball_on"].cpu().numpy()
        bx = st["bx"].cpu().numpy()
        by = st["by"].cpu().numpy()
        new = on & (~prev_on | (bx > prev_bx + 1.0))      # balls only move -x; a jump back up = relaunch
        for e, j in zip(*np.nonzero(new)):
            ball_id[e, j] = next_id[e]
            next_id[e] += 1
        for e in range(B):
            exploded[e] = []
        gone = prev_on & ~on
        for e, j in zip(*np.nonzero(gone)):
            exploded[e].append((ball_id[e, j], float(prev_bx[e, j] + 0.0), float(prev_by[e, j])))
        prev_on[:] = on
        prev_bx[:] = bx
        prev_by[:] = by

    prev_by = np.zeros((B, NB))
    update_ids(s)
    obs = obs_dicts(s, range(B))
    for t in range(max_steps):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        a = np.zeros(B, dtype=np.int64)
        for e in idx:
            a[e] = int(pols[e].act(obs[e]))
        h0 = s["health"].cpu().numpy().copy()
        s, r, done = W.step(s, torch.as_tensor(a, device=device), g)
        rr = r.cpu().numpy()
        ret[alive] += rr[alive]
        h1 = s["health"].cpu().numpy()
        hit = (h1 < h0) & alive
        dmg[hit] += (h0 - h1)[hit]
        hits[hit] += 1
        alive = alive & s["alive"].cpu().numpy()
        update_ids(s)
        idx = np.nonzero(alive)[0]
        obs = obs_dicts(s, idx)
    return [dict(ret=float(ret[e]), capped=bool(alive[e]), hits=int(hits[e])) for e in range(B)]


def _task(args):
    v, chunk, n, seed = args
    t0 = time.time()
    res = run_in_world(f"pilots.aspire_doom.policy_{v}:make_policy", {}, n, seed)
    return v, chunk, res, time.time() - t0


if __name__ == "__main__":
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    versions, N, CH = ["v1", "v2", "v3", "v4", "v5"], 200, 25
    tasks = [(v, c, CH, 20000 + 1000 * c) for v in versions for c in range(N // CH)]   # same world seeds per version
    t0 = time.time()
    with mp.get_context("spawn").Pool(16) as pool:
        outs = pool.map(_task, tasks, chunksize=1)
    wall = time.time() - t0
    res = {}
    for v in versions:
        eps = [e for vv, c, r, _ in sorted(outs, key=lambda o: (o[0], o[1])) if vv == v for e in r]
        rets = np.array([e["ret"] for e in eps])
        res[v] = dict(n=len(eps), mean=float(rets.mean()), median=float(np.median(rets)),
                      sem=float(rets.std() / len(rets) ** 0.5), frac_ge_750=float((rets >= 750).mean()),
                      frac_at_cap=float(np.mean([e["capped"] for e in eps])), min=float(rets.min()), max=float(rets.max()))
    res["wall_s"] = wall
    res["note"] = ("200 imagined episodes per version in worlds/fps/take_cover_world.py (unchanged), cap 24000 tics; "
                   "world seeds 20000+1000*chunk, identical across versions")
    Path("runs/probe").mkdir(parents=True, exist_ok=True)
    Path("runs/probe/screen_doom.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
