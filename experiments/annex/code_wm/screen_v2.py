"""v2 screening harness (extended run): identical protocol, runs policies inside annex_world_v2 with world_params_v2.json.

run_in_world(policy_factory, policy_kwargs, init_obs, device="cuda", params=None, init_vel=None, seed=0)
  policy_factory "package.module:function"; the policy gets reset(), act(obs)->(2,), optional bind(task_card) and
  set_seed(i).  One imagined episode per reset reading in `init_obs` (lab-format dicts, e.g. snapshot readings),
  run with the exact lab episode rules of the world (zone limiting, rest rule, success rule, 236-step limit).
  params: optional dict overriding world parameters (the "imagined reality"); init_vel: optional list of (20,2)
  initial node velocities (a reset reading carries none; default zero).
  Returns per episode dict(success, tip_dist, n_steps).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time

import numpy as np
import torch

from worlds.annex import annex_world_v2 as W

WORLD_PARAMS = "runs/annex/code_wm/world_params_v2.json"


def task_card():
    # public interface facts (same content as the lab's task card, restated so the harness needs no lab import)
    return dict(control_hz=50.0, step_seconds=0.02, time_limit_steps=W.TIME_LIMIT,
                zone=dict(low=list(W.ZONE[0]), high=list(W.ZONE[1])),
                action=dict(low=list(W.ZONE[0]), high=list(W.ZONE[1]), reference_speed_limit=4.0,
                            reference_accel_limit=60.0, servo_force_limit_per_axis=60.0),
                post=dict(count=1, radius=0.03, height=0.08), points=dict(count=20, rest_arc_length=0.70),
                success=dict(target_radius=0.10, rest_speed=0.02, zone_tol=0.01))


def _load(spec):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def world_params(extra=None):
    p = dict(W.PARAMS)
    if os.path.exists(WORLD_PARAMS):
        p.update(json.load(open(WORLD_PARAMS))["params"])
    if extra:
        p.update(extra)
    return p


def run_in_world(policy_factory: str, policy_kwargs: dict, init_obs: list, device="cuda", params=None,
                 init_vel=None, seed=0, verbose=False, record_actions=False) -> list:
    pol = _load(policy_factory)(**(policy_kwargs or {}))
    if hasattr(pol, "bind"):
        pol.bind(task_card())
    prm = world_params(params)
    out = []
    for i, o in enumerate(init_obs):
        o = dict(o)
        o["t"] = 0
        st = W.init_from_obs([o], None, prm, device)
        if init_vel is not None:
            v = torch.tensor(np.asarray(init_vel[i])[None, :, :2], dtype=torch.float32, device=st["pos"].device)
            v[:, -1] = 0
            st["vel"] = v
        if hasattr(pol, "set_seed"):
            pol.set_seed(seed * 100000 + i)
        pol.reset()
        obs = W.observe(st)[0]
        n = 0
        acts = []
        while True:
            a = np.asarray(pol.act(obs), dtype=np.float64).reshape(2)
            acts.append(a.tolist())
            if not np.all(np.isfinite(a)):
                raise ValueError("non-finite action")
            st, r, d = W.step(st, torch.tensor(a[None], dtype=torch.float32, device=st["pos"].device))
            obs = W.observe(st)[0]
            n += 1
            if bool(d[0]):
                break
        tip = float(np.linalg.norm(obs["points"][0, :2] - obs["target"]))
        out.append(dict(success=bool(st["success"][0]), tip_dist=tip, n_steps=n,
                        final_p1=obs["points"][0, :2].tolist(),
                        final_speed=float((st["pos"] - st["prev_pos"]).norm(dim=-1).max()) / W.DT,
                        final_tool_speed=float((st["tool"] - st["prev_tool"]).norm(dim=-1).max()) / W.DT,
                        actions=acts if record_actions else None))
        if verbose:
            print(i, out[-1], flush=True)
    return out


def snapshot_obs(idx):
    d = np.load("runs/annex/observation/layouts.npz")
    return [dict(points=d["points"][i], tool_pos=d["tool_pos"][i], tool_vel=np.zeros(2), post=d["post"][i],
                 zone=d["zone"][i], target=d["target"][i], t=0) for i in idx]


if __name__ == "__main__":
    # verification: python experiments/annex/code_wm/screen.py <factory> '<json kwargs>' <first> <count>
    fac = sys.argv[1]
    kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    a, n = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 4)
    t0 = time.time()
    res = run_in_world(fac, kw, snapshot_obs(range(a, a + n)), verbose=True)
    print(json.dumps(dict(success=float(np.mean([r["success"] for r in res])),
                          tip=float(np.mean([r["tip_dist"] for r in res])), wall=time.time() - t0,
                          steps=int(sum(r["n_steps"] for r in res)))))
