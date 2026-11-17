"""Parallel episode runner over the metered real-world API (reallab.doom.api).

evaluate(policy_factory, policy_kwargs, scenario, seeds, ...) runs one episode per seed on a pool of spawned
worker processes. Each worker imports ``policy_factory`` ("package.module:function"), calls it once as
``function(**policy_kwargs)`` and reuses the returned policy for all its episodes.

Policy protocol:
    policy.reset() -> None            called before every episode
    policy.act(obs: dict) -> int      one discrete action id per agent step
Optional hooks (called if present): ``policy.bind(scenario, action_names)`` once after construction, and
``policy.set_seed(episode_seed)`` before ``reset()`` of each episode (lets stochastic policies be reproducible
per episode independent of which worker ran it).

Step logs (``log_dir``): one ``ep_<seed>.jsonl.gz`` per episode. Line 1: {"meta": {...}}. Line 2: {"reset":
{"obs": ..., "info": {"gt_objects": ...}}}. Then one line per agent step: {"action", "reward", "terminated",
"truncated", "obs" (observation *after* the step), "info"}. ``gt_objects`` only when purpose == "calibration".

Caveat: the calibration cap (api.CAL_CAP_STEPS) is enforced by the ledger Meter using the ledger total read when
each worker env makes its first step and its own count afterwards, so concurrent workers can together overshoot
the cap by up to (n_workers x one episode). Use few enough episodes to stay under it.
"""
from __future__ import annotations

import gzip
import importlib
import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- built-in policies
class _RandomPolicy:
    def __init__(self, seed: int = 0):
        self.base, self.n, self.rng = seed, None, np.random.default_rng(seed)

    def bind(self, scenario, action_names):
        self.n = len(action_names)

    def set_seed(self, s):
        self.rng = np.random.default_rng([self.base, int(s)])

    def reset(self):
        pass

    def act(self, obs):
        return int(self.rng.integers(self.n))


class _NoopPolicy:
    def reset(self):
        pass

    def act(self, obs):
        return 0


def random_policy(seed: int = 0):
    """Uniform random discrete actions; per-episode stream is a function of (seed, episode seed)."""
    return _RandomPolicy(seed)


def noop_policy():
    return _NoopPolicy()


# ---------------------------------------------------------------- worker
from reallab.doom.api import _jsonable  # noqa: E402


def _load(spec: str):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def _worker(wid, tasks, results, cfg):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    env = None
    try:
        from reallab.doom import api
        factory = _load(cfg["policy_factory"])
        policy = factory(**cfg["policy_kwargs"])
        if "torch" in sys.modules:
            sys.modules["torch"].set_num_threads(1)
        env = api.make(cfg["scenario"], cfg["purpose"], cfg["tag"], frame_skip=cfg["frame_skip"],
                       record_path="__pending__" if cfg["record_first"] else None,
                       record_format=cfg["record_format"], pilot=cfg["pilot"])
        env.record_path = None
        if hasattr(policy, "bind"):
            policy.bind(cfg["scenario"], env.action_names)
        while True:
            t = tasks.get()
            if t is None:
                break
            idx, seed = t
            try:
                results.put(("ok", idx, _episode(env, policy, cfg, idx, seed)))
            except Exception as ex:  # report and stop this worker
                results.put(("err", idx, (type(ex).__name__, str(ex), traceback.format_exc())))
                break
    except Exception as ex:
        results.put(("err", -1, (type(ex).__name__, str(ex), traceback.format_exc())))
    finally:
        if env is not None:
            env.close()


def _episode(env, policy, cfg, idx, seed):
    log = None
    if cfg["log_dir"]:
        Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
        log = gzip.open(Path(cfg["log_dir"]) / f"ep_{seed}.jsonl.gz", "wt")
        log.write(json.dumps({"meta": dict(scenario=cfg["scenario"], seed=seed, purpose=cfg["purpose"],
                                           tag=cfg["tag"], frame_skip=cfg["frame_skip"],
                                           action_names=env.action_names)}) + "\n")
    if idx < cfg["record_first"]:
        d = Path(cfg["video_dir"] or cfg["log_dir"] or "runs/doom/videos")
        env.record_path = str(d / f"ep_{seed}.{cfg['record_format']}")
        env.record_title = f"{cfg['scenario']} seed {seed}"
    if hasattr(policy, "set_seed"):
        policy.set_seed(seed)
    policy.reset()
    obs = env.reset(seed)
    if log:
        info0 = {"gt_objects": env.last_gt_objects} if cfg["purpose"] == "calibration" else {}
        log.write(json.dumps({"reset": {"obs": _jsonable(obs), "info": _jsonable(info0)}}) + "\n")
    ret, n = 0.0, 0
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            a = int(policy.act(obs))
            obs, r, terminated, truncated, info = env.step(a)
            ret += r
            n += 1
            if log:
                log.write(json.dumps(_jsonable(dict(action=a, reward=r, terminated=terminated,
                                                    truncated=truncated, obs=obs, info=info))) + "\n")
    finally:
        if log:
            log.close()
        env.stop_recording()
        env.record_path = None
    return dict(seed=seed, ret=ret, length_tics=obs["tic"], n_steps=n, terminated=bool(terminated),
                truncated=bool(truncated), player=_jsonable(obs["player"]))


# ---------------------------------------------------------------- driver
def evaluate(policy_factory: str, policy_kwargs: dict, scenario: str, seeds: list[int], purpose: str = "eval",
             tag: str = "", n_workers: int = 32, frame_skip: int = 4, log_dir: str | None = None,
             record_first: int = 0, record_format: str = "mp4", video_dir: str | None = None,
             pilot: str | None = None) -> list[dict]:
    """Run one episode per seed; returns one result dict per seed (same order as `seeds`):
    seed, ret, length_tics, n_steps, terminated, truncated, player (final player dict)."""
    from cawm.ledger import BudgetExceeded
    seeds = list(seeds)
    nw = max(1, min(1 if purpose == "calibration" else n_workers, len(seeds)))
    cfg = dict(policy_factory=policy_factory, policy_kwargs=policy_kwargs or {}, scenario=scenario,
               purpose=purpose, tag=tag, frame_skip=frame_skip, log_dir=log_dir, record_first=record_first,
               record_format=record_format, video_dir=video_dir, pilot=pilot)
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    for i, s in enumerate(seeds):
        tasks.put((i, s))
    for _ in range(nw):
        tasks.put(None)
    procs = [ctx.Process(target=_worker, args=(w, tasks, results, cfg), daemon=True) for w in range(nw)]
    for p in procs:
        p.start()
    out, err = {}, None
    try:
        while len(out) < len(seeds) and err is None:
            try:
                kind, idx, payload = results.get(timeout=5)
            except Exception:
                if not any(p.is_alive() for p in procs):
                    err = ("RuntimeError", "all workers died", "")
                continue
            if kind == "ok":
                out[idx] = payload
            else:
                err = payload
    finally:
        for p in procs:
            if err is not None and p.is_alive():
                p.terminate()
            p.join(timeout=30)
    if err is not None:
        if err[0] == "BudgetExceeded":
            raise BudgetExceeded(err[1])
        raise RuntimeError(f"worker failed: {err[0]}: {err[1]}\n{err[2]}")
    return [out[i] for i in range(len(seeds))]
