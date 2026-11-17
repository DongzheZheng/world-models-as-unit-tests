"""Parallel episode runner over the metered real-world API (reallab.pusht.api).

evaluate(policy_factory, policy_kwargs, seeds, ...) runs one episode per seed on a pool of spawned worker processes.
Each worker imports ``policy_factory`` ("package.module:function"), calls it once as ``function(**policy_kwargs)``
and reuses the returned policy for all its episodes.

Policy protocol:
    policy.reset() -> None            called before every episode
    policy.act(obs: dict) -> np.ndarray of shape (2,)     one pusher target per agent step
Optional hooks (called if present): ``policy.bind(task_card)`` once after construction, and
``policy.set_seed(episode_seed)`` before ``reset()`` of each episode (reproducible stochastic policies).
Built-in factories: "reallab.pusht.parallel:random_policy", ":noop_policy" (the empty module part
means this module, e.g. ":random_policy").

Step logs (``log_dir``): one ``ep_<seed>.npz`` per episode with the same arrays as the black-box log of api.py.
Recording: ``record_first=N`` writes GIFs for the first N seeds to ``video_dir`` (default ``log_dir`` or
runs/pusht/videos) as ep_<seed>.gif. Calibration forces a single worker.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- built-in policies
class _RandomPolicy:
    """Uniformly random pusher targets in the workspace; per-episode stream depends on (seed, episode seed)."""

    def __init__(self, seed: int = 0):
        self.base, self.rng = seed, np.random.default_rng(seed)

    def set_seed(self, s):
        self.rng = np.random.default_rng([self.base, int(s)])

    def reset(self):
        pass

    def act(self, obs):
        return self.rng.uniform(0.0, 512.0, 2)


class _NoopPolicy:
    """Holds the pusher where it is (target = current position)."""

    def reset(self):
        pass

    def act(self, obs):
        return np.array(obs["agent_pos"], dtype=np.float64)


def random_policy(seed: int = 0):
    return _RandomPolicy(seed)


def noop_policy():
    return _NoopPolicy()


# ---------------------------------------------------------------- worker
def _load(spec: str):
    mod, fn = spec.split(":")
    mod = mod or __name__
    return getattr(importlib.import_module(mod), fn)


def _worker(wid, tasks, results, cfg):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    env = None
    try:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
        from reallab.pusht import api
        factory = _load(cfg["policy_factory"])
        policy = factory(**cfg["policy_kwargs"])
        env = api.make(cfg["purpose"], pilot=cfg["pilot"], tag=cfg["tag"], object=cfg["object"])
        if hasattr(policy, "bind"):
            policy.bind(api.task_card(cfg["object"]) if cfg["object"] != "T" else api.task_card())
        if hasattr(policy, "bind_real"):         # lab-side privileged references only (reads env internals)
            policy.bind_real(env)
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
    log_dir = cfg["log_dir"]
    if idx < cfg["record_first"]:
        d = Path(cfg["video_dir"] or log_dir or "runs/pusht/videos")
        env.record_path = str(d / f"ep_{seed}.{cfg['record_format']}")
        env.record_title = f"{cfg['policy_factory'].split(':')[-1]} seed {seed}" + (" [L]" if cfg["object"] == "L" else "")
    else:
        env.record_path = None
    if hasattr(policy, "set_seed"):
        policy.set_seed(seed)
    policy.reset()
    obs = env.reset(seed)
    rows = dict(agent_pos=[obs["agent_pos"]], agent_vel=[obs["agent_vel"]], block_pose=[obs["block_pose"]],
                coverage=[obs["coverage"]], actions=[], rewards=[])
    ret, n = 0.0, 0
    terminated = truncated = False
    info = {}
    try:
        while not (terminated or truncated):
            a = np.asarray(policy.act(obs), dtype=np.float64)
            obs, r, terminated, truncated, info = env.step(a)
            ret += r
            n += 1
            if log_dir:
                rows["agent_pos"].append(obs["agent_pos"]); rows["agent_vel"].append(obs["agent_vel"])
                rows["block_pose"].append(obs["block_pose"]); rows["coverage"].append(obs["coverage"])
                rows["actions"].append(a); rows["rewards"].append(r)
    finally:
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(log_dir) / f"ep_{seed}.npz", seed=seed, goal_pose=obs["goal_pose"],
                                **{k: np.array(v) for k, v in rows.items()})
        env.stop_recording("SUCCESS" if info.get("success") else f"timeout - max coverage "
                                                                f"{info.get('max_coverage', 0.0):.2f}")
        env.record_path = None
    return dict(seed=seed, ret=float(ret), n_steps=n, success=bool(info["success"]),
                max_coverage=float(info["max_coverage"]), final_coverage=float(info["final_coverage"]))


def evaluate(policy_factory: str, policy_kwargs: dict | None, seeds: list[int], purpose: str = "eval",
             pilot: str = "", tag: str = "", n_workers: int = 16, log_dir: str | None = None,
             record_first: int = 0, record_format: str = "gif", video_dir: str | None = None, object: str = "T") -> list[dict]:
    """Run one episode per seed (object="T" or "L"); returns one dict per seed (same order as `seeds`):
    seed, ret, n_steps, success, max_coverage, final_coverage."""
    from cawm.ledger import BudgetExceeded
    if record_format != "gif":
        raise ValueError("record_format must be 'gif'")
    seeds = list(seeds)
    nw = max(1, min(1 if purpose == "calibration" else n_workers, len(seeds)))
    cfg = dict(policy_factory=policy_factory, policy_kwargs=policy_kwargs or {}, purpose=purpose, pilot=pilot,
               tag=tag, log_dir=log_dir, record_first=record_first, record_format=record_format,
               video_dir=video_dir, object=object)
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
