"""Screening harness: run any lab-protocol policy inside the code world.

    run_in_world(policy_factory, policy_kwargs, init_obs, device="cuda", par=None, randomize=0.0, seed=0) -> list[dict]

policy_factory: "package.module:function" (same as reallab.station.parallel.evaluate); the factory is called once per
episode (fresh object), bind(task_card) / set_seed(i) are called if present, then reset() and act(obs) per step with
lab-format numpy observations produced by the code world.  One imagined episode per reset reading in `init_obs`
(lab-format dicts, e.g. layouts.npz rows or env.reset() readings).  Returns per episode:
success, overload, sides_ok, final_dist, max_load, n_steps.
par: code-world parameter dict (default: runs/station/code_wm/world_par.json if present, else the world defaults).
randomize: relative log-uniform jitter of the parameters the logs pin down least (per episode), 0 = nominal world.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import torch

from worlds.station import station_world as W

PAR_FILE = Path(__file__).resolve().parents[3] / "runs" / "station" / "code_wm" / "world_par.json"
UNCERTAIN = ("lscale", "cv", "m", "cpost", "fmax", "kb")      # least pinned-down parameters (see diagnosis.json)

TASK_CARD = None


def _task_card():
    # public interface facts (copied from the manual; the world must not import the lab)
    return dict(table=dict(low=[0.0, -0.4], high=[1.2, 0.4], surface_z=0.0), control_hz=20.0, time_limit_steps=600,
                action=dict(low=[0.0, -0.4], high=[1.2, 0.4], tool_speed_limit=0.30, tool_radius=0.015),
                posts=dict(count=3, radius=0.03, height=0.08), points=dict(count=20),
                success=dict(tolerance=0.02, rest_speed=0.02), side_threshold=float(np.pi / 2), load_limit=10.0,
                calibration_lives=5)


def default_par():
    if PAR_FILE.exists():
        return json.loads(PAR_FILE.read_text())["par"]
    return dict(W.DEFAULT_PAR)


def layouts_as_obs(path="runs/station/observation/layouts.npz", idx=None):
    L = np.load(path)
    n = len(L["seeds"])
    idx = range(n) if idx is None else idx
    out = []
    for i in idx:
        out.append(dict(points=L["points"][i], tool_pos=L["tool_pos"][i], tool_vel=np.zeros(2), posts=L["posts"][i],
                        post_sides=L["post_sides"][i].astype(np.int64), target=L["target"][i], load=float(L["load"][i]), t=0))
    return out


def _load(spec):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def run_in_world(policy_factory: str, policy_kwargs: dict, init_obs: list, device="cuda", par=None, randomize=0.0,
                 seed=0, return_traces=False):
    par = dict(par or default_par())
    B = len(init_obs)
    st = W.init_from_obs(init_obs, par=par, device=device)
    if randomize > 0:
        g = torch.Generator(device="cpu").manual_seed(seed)
        for k in UNCERTAIN:
            j = torch.exp((torch.rand(B, 1, generator=g) * 2 - 1) * np.log1p(randomize)).to(device)
            st["par"][k] = st["par"][k] * j
    fac = _load(policy_factory)
    pols = []
    for i in range(B):
        p = fac(**(policy_kwargs or {}))
        if hasattr(p, "bind"):
            p.bind(_task_card())
        if hasattr(p, "set_seed"):
            p.set_seed(i)
        p.reset()
        pols.append(p)
    obs = observe = W.observe_list(st)
    # reset readings: use the given ones exactly (load, sides_now as the lab would report them)
    for i in range(B):
        obs[i]["load"] = float(init_obs[i].get("load", obs[i]["load"]))
    traces = [[] for _ in range(B)] if return_traces else None
    with torch.no_grad():
        for t in range(W.TIME_LIMIT):
            done = st["done"].cpu().numpy()
            if done.all():
                break
            acts = np.zeros((B, 2))
            for i in range(B):
                if not done[i]:
                    a = np.asarray(pols[i].act(obs[i]), dtype=np.float64).reshape(2)
                    if not np.all(np.isfinite(a)):
                        raise ValueError("non-finite action")
                    acts[i] = a
                else:
                    acts[i] = obs[i]["tool_pos"]
            st, _, _ = W.step(st, torch.as_tensor(acts, dtype=torch.float32, device=device))
            obs = W.observe_list(st)
            if return_traces:
                for i in range(B):
                    if not done[i]:
                        traces[i].append((acts[i].copy(), obs[i]["tool_pos"].copy(), obs[i]["load"]))
    info = {k: v.cpu().numpy() for k, v in W.episode_info(st).items()}
    res = [dict(success=bool(info["success"][i]), overload=bool(info["overload"][i]), sides_ok=int(info["sides_ok"][i]),
                final_dist=float(info["final_dist"][i]), max_load=float(info["max_load"][i]), n_steps=int(info["n_steps"][i]))
           for i in range(B)]
    if return_traces:
        return res, traces
    return res


def summarize(res):
    n = len(res)
    s = sum(r["success"] for r in res); o = sum(r["overload"] for r in res)
    wrong = sum((not r["success"]) and (not r["overload"]) and r["sides_ok"] < 3 for r in res)
    late = n - s - o - wrong
    return dict(n=n, success=s / n, overload=o / n, wrong_side=wrong / n, not_at_target=late / n,
                mean_max_load=float(np.mean([r["max_load"] for r in res])), mean_steps=float(np.mean([r["n_steps"] for r in res])))


if __name__ == "__main__":
    import sys, time
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    obs = layouts_as_obs()
    t0 = time.time()
    res = run_in_world("experiments.station.code_wm.policy:make_policy", kw, obs)
    print(json.dumps(summarize(res)), f"{time.time()-t0:.1f}s")
    bad = [(i, r) for i, r in enumerate(res) if not r["success"]][:12]
    for i, r in bad:
        print(i, r)
