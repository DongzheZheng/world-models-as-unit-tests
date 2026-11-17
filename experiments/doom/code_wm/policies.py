"""Real-world policies for the code_wm pilot (loaded by reallab.doom.parallel workers).

sticky_random: exploration policy for life 1 (holds a random action for a random 2..12 steps, so the log
               contains long constant-input segments from which acceleration / top speed / friction can be fitted).
trained:       a controller optimised inside worlds/fps/take_cover_world.py; real obs dicts are encoded with the
               SAME function the code world uses (worlds.fps.take_cover_world.encode_obs_dict).
"""
from __future__ import annotations

import numpy as np


class StickyRandom:
    def __init__(self, seed=0, min_hold=2, max_hold=12):
        self.base, self.lo, self.hi = seed, min_hold, max_hold
        self.rng = np.random.default_rng(seed)
        self.a, self.left = 0, 0

    def set_seed(self, s):
        self.rng = np.random.default_rng([self.base, int(s)])

    def reset(self):
        self.a, self.left = 0, 0

    def act(self, obs):
        if self.left <= 0:
            self.a = int(self.rng.integers(3))
            self.left = int(self.rng.integers(self.lo, self.hi + 1))
        self.left -= 1
        return self.a


def sticky_random(seed=0, min_hold=2, max_hold=12):
    return StickyRandom(seed, min_hold, max_hold)


class Trained:
    """Wraps an exported cawm.ppo.ActorCritic; keeps the encoder's memory (previous visible objects) itself."""

    def __init__(self, path, deterministic=True):
        import torch
        from cawm.ppo import ActorCritic
        from worlds.fps import take_cover_world as W
        self.W, self.torch = W, torch
        torch.set_num_threads(1)
        self.pi = ActorCritic.load(torch.load(path, map_location="cpu"), "cpu")
        self.det = deterministic
        self.mem = None
        self.rng = np.random.default_rng(0)

    def set_seed(self, s):
        self.rng = np.random.default_rng(int(s))
        self.torch.manual_seed(int(s))

    def reset(self):
        self.mem = None

    def act(self, obs):
        o, self.mem = self.W.encode_obs_dict(obs, self.mem)
        x = self.torch.as_tensor(o, dtype=self.torch.float32)[None]
        return int(self.pi.act(x, deterministic=self.det)[0])


def trained(path, deterministic=True):
    return Trained(path, deterministic)
