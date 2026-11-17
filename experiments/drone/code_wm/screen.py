"""Screen someone else's controller INSIDE the code world (no real episodes).

run_in_world(policy_factory, policy_kwargs, mission, n_episodes, seed0) -> list of per-episode dicts
  * the world is worlds/drone/site_world.py, unchanged, with the beliefs and randomisation used for code_wm's own
    predictions: mission A -> fit_l123.json + tune_A123 randomisation; mission C -> fit_all.json + tune_C5
    randomisation and mover population prior (MOVER_PRIOR); mission B -> fit_all.json + tune_B randomisation
  * one policy object per imagined episode (numpy, CPU); the world steps all episodes as one batch at 30 Hz
  * observation dict in the lab's format, built from world state each step: pos / lin_vel (with the world's sensor
    noise), quat, ang_vel, t, relay_index, relays, goal (= current relay), movers / n_movers (mission C, from the
    world's observe_movers: nearest first within 15 m, line of sight not modelled), ranges (NOT computed: a constant
    12.0 array -- none of the screened programs reads it; check before reusing for a program that does)
  * success / collision (walls, ground, other aircraft) / out of bounds / timeout exactly as in site_world.step,
    time limits A 17.0 s, B 24.2 s, C 19.4 s
  python experiments/drone/code_wm/screen.py   -> runs/probe/screen_drone.json (v1-v5 on A, v6-v14 on C)
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from experiments.drone.code_wm.imagine import load_world, randomise
from worlds.drone import site_world as W

O = "runs/drone/code_wm/"
SETUP = {
    "A": dict(world=O + "fit_l123.json", cfg=O + "cfg/tune_A123.json", mission=W.MISSION_A, time_limit=17.0, movers=False),
    "B": dict(world=O + "fit_all.json", cfg=O + "cfg/tune_B.json", mission=((17, 0, 2.5), (23.75, 6, 2.5), (57, 0, 2.5)),
              time_limit=24.2, movers=False),
    "C": dict(world=O + "fit_all.json", cfg=O + "cfg/tune_C5.json", mission=W.MISSION_A, time_limit=W.TIME_LIMIT_C, movers=True),
}
CAUSE = {1: "success", 2: "collision", 3: "out_of_bounds", 4: "timeout", 0: "timeout"}


def _factory(spec: str):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def run_in_world(policy_factory: str, policy_kwargs: dict, mission: str, n_episodes: int, seed0: int) -> list[dict]:
    torch.set_num_threads(1)
    st = SETUP[mission]
    cfg = json.load(open(st["cfg"]))
    belief = load_world(st["world"])
    base = dict(belief); base.update(cfg["world_override"])
    unc = {k: (v[0], tuple(v[1]) if isinstance(v[1], list) else v[1]) for k, v in cfg["unc"].items()}
    B, dev = n_episodes, "cpu"
    g = torch.Generator(device=dev).manual_seed(seed0)
    pw = randomise(base, B, dev, g, unc)
    start, relays = W.sample_start(B, dev, g, st["mission"])
    s = W.init_state(B, relays, start, dev, pw)
    if st["movers"]:
        s.update(W.sample_movers(B, dev, g, cfg.get("movers", {})))
    make = _factory(policy_factory)
    pols = [make(**(policy_kwargs or {})) for _ in range(B)]
    for p in pols:
        p.reset()
    n_steps = int(round(st["time_limit"] * 30))
    ranges = np.full((6, 16), 12.0)
    for _ in range(n_steps):
        o = W.observe(s, pw, g)
        on = {k: v.double().cpu().numpy() for k, v in o.items() if torch.is_tensor(v)}
        done = s["done"].cpu().numpy()
        acts = np.zeros((B, 4))
        for i in range(B):
            if done[i]:
                acts[i] = (0, 0, 0, 0.3)
                continue
            k = int(on["relay_index"][i])
            ob = dict(pos=on["pos"][i], lin_vel=on["lin_vel"][i], quat=on["quat"][i], ang_vel=on["ang_vel"][i],
                      t=np.array([on["t"][i]]), relay_index=np.array([float(k)]), relays=on["relays"][i],
                      goal=on["relays"][i][k], ranges=ranges.copy())
            if st["movers"]:
                ob["movers"] = on["movers"][i]
                ob["n_movers"] = np.array([float(on["n_movers"][i])])
            a = np.asarray(pols[i].act(ob), dtype=np.float64).reshape(4)
            acts[i] = a if np.all(np.isfinite(a)) else (0, 0, 0, 0)
        s, _, d = W.step(s, torch.as_tensor(acts, dtype=torch.float32), pw, st["time_limit"], g)
        if bool(d.all()):
            break
    c, t, kk = s["cause"].numpy(), s["t"].numpy(), s["k"].numpy()
    return [dict(seed=seed0, episode=i, success=bool(c[i] == 1), time_s=float(t[i]), relays_reached=int(min(kk[i], 3)),
                 end_cause=CAUSE[int(c[i])]) for i in range(B)]


def summarise(rows):
    n = len(rows)
    ts = [r["time_s"] for r in rows if r["success"]]
    out = dict(n=n, success=sum(r["success"] for r in rows) / n, time_success_s=float(np.mean(ts)) if ts else None)
    for k in ("collision", "timeout", "out_of_bounds"):
        out[k] = sum(r["end_cause"] == k for r in rows) / n
    out["mean_relays_reached"] = float(np.mean([r["relays_reached"] for r in rows]))
    return out


def _job(args):
    ver, mission, n, seed0 = args
    t0 = time.time()
    rows = run_in_world(f"pilots.aspire_drone.policy_{ver}:make_policy", {}, mission, n, seed0)
    sm = summarise(rows)
    sm["wall_s"] = time.time() - t0
    print(mission, ver, json.dumps(sm), flush=True)
    return mission, ver, sm


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(f"v{i}", "A", 100, 5000) for i in range(1, 6)] + [(f"v{i}", "C", 200, 6000) for i in range(6, 15)]
    with mp.get_context("spawn").Pool(14) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    out = {"A": {}, "C": {}}
    for mission, ver, sm in res:
        out[mission][ver] = sm
    out["wall_s"] = time.time() - t0
    out["setup"] = {m: dict(world=SETUP[m]["world"], randomisation=SETUP[m]["cfg"], time_limit=SETUP[m]["time_limit"]) for m in ("A", "C")}
    import os
    os.makedirs("runs/probe", exist_ok=True)
    json.dump(out, open("runs/probe/screen_drone.json", "w"), indent=1)
    print("wall", out["wall_s"])
