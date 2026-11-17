"""Screen arbitrary Push-T/L control programs inside the code world (imagined episodes, no real cost).

run_in_world(policy_factory, policy_kwargs, obj, n_episodes, seed0, n_workers=1, batch=25) -> list of dicts
  policy_factory: "package.module:function" returning a policy with optional bind(task_card), set_seed(seed),
                  and reset(), act(obs) -> (2,) pusher target (the lab protocol of reallab.pusht.parallel).
  Each episode: the world (worlds/pusht/pusht_world.py, as evaluated, set_object(obj)) is stepped at 10 Hz for at most
  300 steps; the lab-format observation is built from world state each step:
      agent_pos = world pusher position, agent_vel = world pusher velocity (both exact world state),
      block_pose = world body pose with theta wrapped to [0, 2 pi), goal_pose = task-card goal,
      coverage = reallab.pusht.api.coverage_of(pose, object) (exact public geometry), t = step count.
  Success = the real rule (coverage_of > 0.95 terminates the episode). `success_world_estimate` = whether the world's
  own grid coverage estimate exceeded 0.95 at any step up to the end of the episode.
  Initial states (same sampler as eval_world.init_states): pusher uniform in [50, 462]^2; block position uniform in
  [lo, hi]^2 with angle uniform in [0, 2 pi), rejected if the pusher is within 25 u of the block outline, if the block
  outline crosses the walls, or if the initial coverage is >= 0.5. T: lo, hi = 100, 412 (as in the T tuning runs);
  L: lo, hi = 40, 472 (as in the L tuning runs, widened after L life 2 started 25 u from the top wall).
  Episode i uses seed seed0 + i for both the initial state and policy.set_seed.
"""
from __future__ import annotations

import importlib
import math
import multiprocessing as mp
import os

import numpy as np

RANGES = {"T": (100.0, 412.0), "L": (40.0, 472.0)}


def _load(spec):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def _init_state(rng, W, lo, hi):
    import torch
    while True:
        p = rng.uniform(50, 462, 2)
        q = np.array([*rng.uniform(lo, hi, 2), rng.uniform(0, 2 * math.pi)])
        qt = torch.tensor(q[None], dtype=torch.float32)
        cw = W.to_world(qt, torch.tensor(W.VERTS, dtype=torch.float32)[None])[0]
        if float(cw.min()) < W.WALL_LO or float(cw.max()) > W.WALL_HI:
            continue
        d, _ = W.t_sdf(W.to_body(qt, torch.tensor(p[None, None], dtype=torch.float32))[:, 0])
        if float(d[0]) > 25 and float(W.coverage(qt)[0]) < 0.5:
            return p, q


def _run_chunk(args):
    policy_factory, policy_kwargs, obj, seeds = args
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    import torch
    torch.set_num_threads(1)
    from reallab.pusht import api
    from worlds.pusht import pusht_world as W
    W.set_object(obj)
    card = api.task_card(object=obj) if obj != "T" else api.task_card()
    factory = _load(policy_factory)
    n = len(seeds)
    P, Q = zip(*[_init_state(np.random.default_rng(s), W, *RANGES[obj]) for s in seeds])
    f = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
    st = W.make_state(f(P), torch.zeros(n, 2), f(Q))
    pols = []
    for s in seeds:
        pol = factory(**(policy_kwargs or {}))
        if hasattr(pol, "bind"):
            pol.bind(card)
        if hasattr(pol, "set_seed"):
            pol.set_seed(int(s))
        pol.reset()
        pols.append(pol)
    goal = np.array(W.GOAL, float)
    done = np.zeros(n, bool); succ = np.zeros(n, bool); succ_w = np.zeros(n, bool)
    steps = np.zeros(n, int)
    pose = st["q"].double().numpy()
    cov = np.array([api.coverage_of(tuple(pose[i]), object=obj) for i in range(n)])
    maxcov = cov.copy()
    errors = [None] * n
    for t in range(300):
        p_np, pv_np = st["p"].double().numpy(), st["pv"].double().numpy()
        acts = np.zeros((n, 2))
        for i in range(n):
            if done[i]:
                acts[i] = p_np[i]
                continue
            obs = dict(agent_pos=p_np[i].copy(), agent_vel=pv_np[i].copy(),
                       block_pose=np.array([pose[i, 0], pose[i, 1], pose[i, 2] % (2 * math.pi)]),
                       goal_pose=goal.copy(), coverage=float(cov[i]), t=t)
            try:
                a = np.asarray(pols[i].act(obs), dtype=np.float64).reshape(2)
                if not np.all(np.isfinite(a)):
                    raise ValueError("non-finite action")
            except Exception as ex:          # a crashing program ends its episode as a failure
                errors[i] = f"{type(ex).__name__}: {ex}"
                done[i] = True
                acts[i] = p_np[i]
                continue
            acts[i] = np.clip(a, 0.0, 512.0)
        live = ~done
        if not live.any():
            break
        st = W.physics(st, f(acts))
        pose = st["q"].double().numpy()
        wcov = W.coverage(st["q"]).numpy()
        for i in np.where(live)[0]:
            cov[i] = api.coverage_of(tuple(pose[i]), object=obj)
            steps[i] += 1
            maxcov[i] = max(maxcov[i], cov[i])
            succ_w[i] |= wcov[i] > 0.95
            if cov[i] > 0.95:
                succ[i] = done[i] = True
    return [dict(seed=int(s), success=bool(succ[i]), success_world_estimate=bool(succ_w[i]),
                 max_coverage=float(maxcov[i]), n_steps=int(steps[i]), error=errors[i])
            for i, s in enumerate(seeds)]


def run_in_world(policy_factory: str, policy_kwargs: dict, obj: str, n_episodes: int, seed0: int,
                 n_workers: int = 1, batch: int = 25) -> list[dict]:
    seeds = list(range(seed0, seed0 + n_episodes))
    chunks = [(policy_factory, policy_kwargs, obj, seeds[i:i + batch]) for i in range(0, n_episodes, batch)]
    if n_workers <= 1:
        out = [_run_chunk(c) for c in chunks]
    else:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            out = pool.map(_run_chunk, chunks)
    return [e for c in out for e in c]


def _summ(eps):
    return dict(n=len(eps), success=float(np.mean([e["success"] for e in eps])),
                success_world_estimate=float(np.mean([e["success_world_estimate"] for e in eps])),
                max_coverage=float(np.mean([e["max_coverage"] for e in eps])),
                n_steps=float(np.mean([e["n_steps"] for e in eps])),
                n_errors=int(sum(e["error"] is not None for e in eps)),
                first_error=next((e["error"] for e in eps if e["error"]), None))


def main():
    """Screen the other pilot's v1-v15 (v1-v8 on the T, v9-v15 on the L): 200 episodes each, 16 workers."""
    import json
    import sys
    import time
    from pathlib import Path
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    t0 = time.time()
    jobs = [("T", v) for v in range(1, 9)] + [("L", v) for v in range(9, 16)]
    chunks, index = [], []
    for obj, v in jobs:
        for i in range(0, n, 25):
            chunks.append((f"pilots.aspire_pusht.policy_v{v}:make_policy", {}, obj, list(range(50000 + i, 50000 + min(n, i + 25)))))
            index.append((obj, v))
    with mp.get_context("spawn").Pool(16) as pool:
        res = pool.map(_run_chunk, chunks, chunksize=1)
    out = {"T": {}, "L": {}}
    per = {}
    for (obj, v), r in zip(index, res):
        per.setdefault((obj, v), []).extend(r)
    for (obj, v), eps in per.items():
        out[obj][f"v{v}"] = _summ(eps)
    out["wall_s"] = time.time() - t0
    out["protocol"] = dict(seeds="50000..50000+n-1 (same initial states for every version of an object)",
                           init_ranges=RANGES, world="worlds/pusht/pusht_world.py as evaluated",
                           success="reallab.pusht.api.coverage_of > 0.95 (terminates)",
                           success_world_estimate="world grid coverage > 0.95 at any step")
    Path("runs/probe").mkdir(parents=True, exist_ok=True)
    Path("runs/probe/screen_pusht.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
