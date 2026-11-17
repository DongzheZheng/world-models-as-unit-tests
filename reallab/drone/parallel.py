"""Parallel episode runner over the metered real-drone API (see reallab.drone.api for the manual)."""
from __future__ import annotations

import gzip
import importlib
import inspect
import json
import multiprocessing as mp
import os

import numpy as np

_BUILTIN = "reallab.drone._policies"
_POLICY_CACHE: dict = {}


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def _load_policy(factory: str, kwargs: dict):
    key = (factory, json.dumps(kwargs, sort_keys=True, default=str))
    if key not in _POLICY_CACHE:
        mod, _, fn = factory.partition(":")
        mod = mod or _BUILTIN
        _POLICY_CACHE[key] = getattr(importlib.import_module(mod), fn)(**kwargs)
    return _POLICY_CACHE[key]


def _worker_init():
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def run_episode(job: dict) -> dict:
    from ._lab import make_lab as make
    pol = _load_policy(job["policy_factory"], job["policy_kwargs"])
    seed = job["seed"]
    try:
        if "seed" in inspect.signature(pol.reset).parameters:
            pol.reset(seed=seed)
        else:
            pol.reset()
    except (TypeError, ValueError):
        pol.reset()
    env = make(job["task"], job["purpose"], job["tag"], **job["env_kwargs"])
    log = None
    try:
        obs, info0 = env.reset(seed=seed)
        rex = getattr(env, "render_extra", lambda: None)
        if job["log_dir"]:
            os.makedirs(job["log_dir"], exist_ok=True)
            log = gzip.open(os.path.join(job["log_dir"], f"{job['task']}_{job['purpose']}_{seed}.jsonl.gz"), "wt")
            log.write(json.dumps(_jsonable(dict(type="reset", seed=seed, task=job["task"], obs=obs, info=info0,
                                                 render_movers=rex()))) + "\n")
        ret, n, info = 0.0, 0, {}
        while True:
            act = np.asarray(pol.act(obs), dtype=np.float64)
            nobs, r, te, tr, info = env.step(act)
            ret += r
            n += 1
            if log:
                log.write(json.dumps(_jsonable(dict(type="step", action=act, reward=r, terminated=te, truncated=tr,
                                                    obs=nobs, render_movers=rex()))) + "\n")
            obs = nobs
            if te or tr:
                break
        if log:
            log.write(json.dumps(_jsonable(dict(type="end", info=info))) + "\n")
    finally:
        if log:
            log.close()
        env.close()
    out = dict(seed=seed, ret=float(ret), n_steps=n)
    out.update({k: v for k, v in info.items() if k != "course"})
    return out


def evaluate(policy_factory: str, policy_kwargs: dict, task: str, seeds: list[int], purpose: str = "eval",
             tag: str = "", n_workers: int = 32, log_dir: str | None = None, **env_kwargs) -> list[dict]:
    if task == "site" and purpose == "calibration":
        n_workers = 1            # lives are flown one at a time
        if env_kwargs.get("pilot") is None:
            raise ValueError("pilot=<name> is required for calibration at the site")
    jobs = [dict(policy_factory=policy_factory, policy_kwargs=dict(policy_kwargs or {}), task=task, seed=int(s),
                 purpose=purpose, tag=tag, log_dir=log_dir, env_kwargs=env_kwargs) for s in seeds]
    n_workers = max(1, min(n_workers, len(jobs)))
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        res = pool.map(run_episode, jobs, chunksize=1)
    return res
