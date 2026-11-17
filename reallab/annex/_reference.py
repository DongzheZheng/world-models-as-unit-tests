"""Lab-side reference policies and lab-only runners.  Internal: pilots must not read this module.

ReplayPolicy(plans)    replays a precomputed command sequence per seed (plans made in the lab simulator by the
                       privileged / slow searches of _plan.py); uses nothing else.
GeoPolicy(consts)      the geometry-scheduled script: one fixed waypoint cast whose parameters are closed-form functions
                       of the post's bearing and distance (_sim.geo_params); uses only the observation.
lab_env(cfg)           lab constructor (optionally a speed-capped tool) for parallel._run(..., env_factory=lab_env).
"""
from __future__ import annotations

import numpy as np

from reallab.annex import _limits as S


class ReplayPolicy:
    def __init__(self, plans: dict):
        self.plans = {int(k): np.asarray(v, float) for k, v in plans.items()}

    def set_seed(self, s):
        self.cur = self.plans.get(int(s))

    def reset(self):
        pass

    def act(self, obs):
        if self.cur is None:
            return np.array(obs["tool_pos"])
        k = min(int(obs["t"]), len(self.cur) - 1)
        return self.cur[k]


class GeoPolicy:
    def __init__(self, consts):
        self.consts = np.asarray(consts, float)

    def reset(self):
        self.sched = None

    def act(self, obs):
        from reallab.annex._sim import Cast, geo_params
        if self.sched is None:
            post = np.asarray(obs["post"], float)
            dist = float(np.linalg.norm(post - np.array(S.ZONE_CENTRE)))
            self.sched = Cast(geo_params(self.consts, dist), post, 1, 3)
        return self.sched(int(obs["t"]))


def replay_policy(plans):
    return ReplayPolicy(plans)


def geo_policy(consts):
    return GeoPolicy(consts)


def lab_env(cfg):
    from reallab.annex import api
    q = {"v": cfg["vmax"]} if cfg.get("vmax") else None
    return api.RealAnnex(cfg["purpose"], cfg["pilot"], cfg["tag"], None, _q=q)


def lab_evaluate(policy_factory, policy_kwargs, seeds, purpose="eval", tag="", n_workers=12, log_dir=None, vmax=None):
    from reallab.annex.parallel import _run
    cfg = dict(policy_factory=policy_factory, policy_kwargs=policy_kwargs or {}, purpose=purpose, pilot="", tag=tag,
               log_dir=log_dir, vmax=vmax)
    return _run(cfg, seeds, n_workers, env_factory=lab_env)
