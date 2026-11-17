"""Adapter: the agent-written world model of the pushing task behind the `World` contract of the harness.

Hand-written glue, not written by an agent.  The world model is worlds/pusht/pusht_world.py, the file the
world-model agent wrote in the study, imported unchanged.  That file is a batched, functional torch module
(make_state, physics); the harness wants an object with reset(seed) and step(action) that speaks the environment's
formats.  This adapter supplies exactly what the study's own model-based evaluation supplied around the same file
(experiments/pusht/code_wm/screen.py), one episode at a time instead of 25 in a batch:

    reset(seed)      the initial state of that script's sampler, _init_state(numpy.random.default_rng(seed), ...),
                     imported from it unchanged
    observation      agent_pos and agent_vel from the model's pusher state, block_pose with the angle wrapped to
                     [0, 2 pi), goal_pose from the model, coverage from the public geometry
                     (reallab.pusht.api.coverage_of), t = steps taken
    step(action)     action as float64 (2,), clipped to [0, 512], one call of physics(); non-finite actions raise
    episode rules    success when coverage_of(pose) > 0.95 (ends the episode), time limit 300 steps
    reset_from(obs)  a state built from a real reset observation (block at rest)

The repository keeps reallab out of its world models (tests/test_no_leak.py).  This adapter imports the public
function coverage_of from it for the success rule, as the study's model-based evaluation did; the world model file
itself stays free of it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # the repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from experiments.pusht.code_wm.screen import RANGES, _init_state  # noqa: E402
from reallab.pusht import api  # noqa: E402
from worlds.pusht import pusht_world as W  # noqa: E402

OBJECT = "T"
MAX_STEPS = 300
SUCCESS_COVERAGE = 0.95


def _tensor(x) -> torch.Tensor:
    return torch.tensor(np.array(x), dtype=torch.float32)


class World:
    def __init__(self):
        W.set_object(OBJECT)
        self._goal = np.array(W.GOAL, float)
        self._state = None
        self._t = 0
        self._coverage = self._max_coverage = 0.0

    def _start(self, state) -> dict:
        self._state, self._t = state, 0
        self._coverage = self._max_coverage = api.coverage_of(tuple(self._pose()), object=OBJECT)
        return self._obs()

    def _pose(self) -> np.ndarray:
        return self._state["q"].double().numpy()[0]

    def _obs(self) -> dict:
        pose = self._pose()
        return dict(agent_pos=self._state["p"].double().numpy()[0].copy(),
                    agent_vel=self._state["pv"].double().numpy()[0].copy(),
                    block_pose=np.array([pose[0], pose[1], pose[2] % (2 * math.pi)]),
                    goal_pose=self._goal.copy(), coverage=float(self._coverage), t=self._t)

    def reset(self, seed: int) -> dict:
        p, q = _init_state(np.random.default_rng(seed), W, *RANGES[OBJECT])
        return self._start(W.make_state(_tensor([p]), torch.zeros(1, 2), _tensor([q])))

    def reset_from(self, obs: dict) -> dict:
        return self._start(W.make_state(_tensor([obs["agent_pos"]]), _tensor([obs["agent_vel"]]),
                                        _tensor([obs["block_pose"]])))

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite action")
        self._state = W.physics(self._state, _tensor([np.clip(a, 0.0, 512.0)]))
        self._t += 1
        self._coverage = api.coverage_of(tuple(self._pose()), object=OBJECT)
        self._max_coverage = max(self._max_coverage, self._coverage)
        success = self._coverage > SUCCESS_COVERAGE
        done = success or self._t >= MAX_STEPS
        info = dict(success=bool(success), max_coverage=float(self._max_coverage),
                    final_coverage=float(self._coverage), n_steps=self._t)
        return self._obs(), done, info
