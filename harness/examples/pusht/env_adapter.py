"""Adapter: the repository's pushing environment (reallab/pusht) behind the environment contract of the harness.

Hand-written glue, not written by an agent.  It changes nothing about the environment:

    manual       the docstring of reallab/pusht/api.py, the manual the study's agents were given.  Its section on how
                 to open the environment (api.make, purposes) does not apply behind the harness, where lab.py is the
                 only door; the rest (units, frames, observation, action, success rule) applies unchanged.
    task_card()  reallab.pusht.api.task_card()
    reset(seed)  api.make(purpose).reset(seed), with purpose "eval" for seeds >= 1000 (the purpose of the recorded
                 runs of runs/probe/curves.json) and "smoke" below (uncapped; the harness keeps its own count)
    step(a)      api's (obs, reward, terminated, truncated, info) as (obs, terminated or truncated, info)
    max_steps    300, the environment's time limit

Every real episode is also counted by the repository's own budget counter (cawm/ledger.py), in runs/ledger.jsonl
unless `ledger_path` says otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # the repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PushT:
    max_steps = 300

    def __init__(self, object: str = "T", ledger_path: str | None = None, tag: str = "harness"):
        from reallab.pusht import api
        self._api, self.object, self._ledger_path, self._tag = api, object, ledger_path, tag
        self.manual = api.__doc__
        self._envs: dict = {}
        self._env = None

    def task_card(self) -> dict:
        return self._api.task_card(object="L") if self.object == "L" else self._api.task_card()

    def reset(self, seed: int) -> dict:
        purpose = "eval" if seed >= self._api.SEED_SPLIT else "smoke"
        if purpose not in self._envs:
            self._envs[purpose] = self._api.make(purpose, pilot="harness", tag=self._tag,
                                                 ledger_path=self._ledger_path, object=self.object)
        self._env = self._envs[purpose]
        return self._env.reset(seed=seed)

    def step(self, action):
        obs, _, terminated, truncated, info = self._env.step(np.asarray(action, dtype=np.float64))
        return obs, bool(terminated or truncated), info

    def close(self) -> None:
        for env in self._envs.values():
            env.close()


def make_env(**kwargs) -> PushT:
    return PushT(**kwargs)
