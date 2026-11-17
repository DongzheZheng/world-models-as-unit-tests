"""Run a policy file in world.py and print its success rate.

HAND-WRITTEN STAND-IN for an agent's own test script (see world.py).  Run it in a workspace:

    python evaluate_in_model.py policy.py 200 0        # policy file, number of episodes, first seed

The loop is the one that would run the policy in the environment, with World() in place of lab.Env().
"""
from __future__ import annotations

import importlib.util
import json
import sys

from world import World


def episode(policy, env, seed: int) -> bool:
    policy.reset()
    obs, done, info = env.reset(seed), False, {}
    while not done:
        obs, done, info = env.step(policy.act(obs))
    return bool(info["success"])


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "policy.py"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy, world = module.make_policy(), World()
    wins = sum(episode(policy, world, seed) for seed in range(first, first + episodes))
    print(json.dumps({"policy": path, "episodes": episodes, "first_seed": first, "successes": wins,
                      "success_rate": wins / episodes}))
