"""Parallel episode runner over the metered real-world API (reallab.station.api).

evaluate(policy_factory, policy_kwargs, seeds, ...) runs one episode per seed on a pool of spawned worker processes.
Each worker imports ``policy_factory`` ("package.module:function"), calls it once as ``function(**policy_kwargs)`` and
reuses the returned policy for all its episodes.

Policy protocol:
    policy.reset() -> None                               called before every episode
    policy.act(obs: dict) -> np.ndarray of shape (2,)    one tool command per agent step
Optional hooks (called if present): ``policy.bind(task_card)`` once after construction, and
``policy.set_seed(episode_seed)`` before ``reset()`` of each episode.
Built-in factories: "reallab.station.parallel:random_policy", ":noop_policy" (an empty module part means this module).

Step logs (``log_dir``): one ``ep_<seed>.npz`` per episode with the same arrays as the black-box log of api.py.
Calibration forces a single worker.  Per-episode result: seed, ret, n_steps, success, sides_ok, final_dist,
max_load, overload.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import traceback
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- built-in policies
class _RandomPolicy:
    """Uniformly random tool commands on the table; per-episode stream depends on (seed, episode seed)."""

    def __init__(self, seed: int = 0):
        self.base, self.rng = seed, np.random.default_rng(seed)

    def set_seed(self, s):
        self.rng = np.random.default_rng([self.base, int(s)])

    def reset(self):
        pass

    def act(self, obs):
        return self.rng.uniform([0.0, -0.4], [1.2, 0.4])


class _NoopPolicy:
    """Holds the tool where it is (command = current tool position)."""

    def reset(self):
        pass

    def act(self, obs):
        return np.array(obs["tool_pos"], dtype=np.float64)


def random_policy(seed: int = 0):
    return _RandomPolicy(seed)


def noop_policy():
    return _NoopPolicy()


# ---------------------------------------------------------------- worker
_KEYS = ("points", "tool_pos", "tool_vel", "posts", "post_sides", "sides_now", "target", "load", "t")


def _load(spec: str):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod or __name__), fn)


def _make_env(cfg):
    from reallab.station import api
    if cfg.get("lab_env"):                       # lab-side constructor (observation runs, recording)
        return _load(cfg["lab_env"])(cfg)
    return api.make(cfg["purpose"], pilot=cfg["pilot"], tag=cfg["tag"])


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
        from reallab.station import api
        policy = _load(cfg["policy_factory"])(**cfg["policy_kwargs"])
        env = _make_env(cfg)
        if hasattr(policy, "bind"):
            policy.bind(api.task_card())
        if hasattr(policy, "bind_real"):         # lab-side privileged references only (reads env internals)
            policy.bind_real(env)
        while True:
            t = tasks.get()
            if t is None:
                break
            idx, seed = t
            try:
                results.put(("ok", idx, _episode(env, policy, cfg, idx, seed)))
            except Exception as ex:
                results.put(("err", idx, (type(ex).__name__, str(ex), traceback.format_exc())))
                break
    except Exception as ex:
        results.put(("err", -1, (type(ex).__name__, str(ex), traceback.format_exc())))
    finally:
        if env is not None:
            env.close()


def _episode(env, policy, cfg, idx, seed):
    log_dir = cfg["log_dir"]
    hook = cfg.get("episode_hook")
    if hook:
        _load(hook)(env, cfg, idx, seed)
    if hasattr(policy, "set_seed"):
        policy.set_seed(seed)
    policy.reset()
    obs = env.reset(seed)
    rows = {k: [obs[k]] for k in _KEYS}
    rows.update(actions=[], rewards=[])
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
                for k in _KEYS:
                    rows[k].append(obs[k])
                rows["actions"].append(np.minimum(np.maximum(a, env.action_low), env.action_high))
                rows["rewards"].append(r)
    finally:
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(log_dir) / f"ep_{seed}.npz", seed=seed, **{k: np.array(v) for k, v in rows.items()})
    return dict(seed=seed, ret=float(ret), n_steps=n, success=bool(info["success"]), sides_ok=int(info["sides_ok"]),
                final_dist=float(info["final_dist"]), max_load=float(info["max_load"]), overload=bool(info["overload"]))


def _run(cfg: dict, seeds: list[int], n_workers: int) -> list[dict]:
    from cawm.ledger import BudgetExceeded
    seeds = list(seeds)
    nw = max(1, min(1 if cfg["purpose"] in ("calibration", "observation") else n_workers, len(seeds)))
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


def evaluate(policy_factory: str, policy_kwargs: dict | None, seeds: list[int], purpose: str = "eval",
             pilot: str = "", tag: str = "", n_workers: int = 16, log_dir: str | None = None) -> list[dict]:
    """Run one episode per seed; returns one dict per seed (same order as `seeds`):
    seed, ret, n_steps, success, sides_ok, final_dist, max_load, overload."""
    if purpose == "observation":
        raise ValueError("observation episodes are run by the lab only")
    cfg = dict(policy_factory=policy_factory, policy_kwargs=policy_kwargs or {}, purpose=purpose, pilot=pilot, tag=tag,
               log_dir=log_dir)
    return _run(cfg, seeds, n_workers)
