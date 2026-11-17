"""The two contracts of the harness: what an environment must look like, and what an agent must hand in.

THE ENVIRONMENT (what you implement for your own simulator or robot)
--------------------------------------------------------------------
The run config names a factory, "package.module:function".  The harness calls it (with the keyword arguments of the
config, in a process of its own) and expects an object with:

    manual       str    The text the agent is given as MANUAL.md: interface, units, frames, the exact success rule.
                        Leave out whatever the agent is supposed to find out by experiment.
    max_steps    int    The longest an episode can be.  The harness ends an episode that reaches it.
    task_card()  -> dict        Public facts of the task as JSON-serialisable data (numbers, strings, lists, dicts).
    reset(seed)  -> obs         Start an episode.  `seed` is an integer that names the initial situation; the harness
                                only ever passes seeds from the two ranges of the config.  `obs` is a dict whose values
                                are numbers or numpy arrays (the same keys and shapes on every step).
    step(action) -> (obs, done, info)
                                Apply one action.  `action` is whatever the manual says: a number, a list or a numpy
                                array (or a dict of those).  Raise ValueError for an action that is not acceptable;
                                the episode stays open.  `done` is True on the last step, and then info["success"]
                                (bool) must say whether the task was achieved.  `info` is a dict of numbers, strings
                                or arrays.
    close()                     Optional.  Called when the process stops.

The same seed with the same actions should give the same episode if the system allows it; say so in the manual.  One
instance serves one episode at a time.  Held-out evaluation creates one instance per worker process, so a system
that exists only once (a robot) needs `[verify] workers = 1`.

THE DELIVERABLES (what the agent leaves in its workspace; stated in the briefs, checked by `verify`)
----------------------------------------------------------------------------------------------------
    policy.py        make_policy() -> an object with reset() and act(obs) -> action.  Optional hooks, called if they
                     exist: bind(task_card) once after make_policy(), set_seed(seed) before reset() of each episode.
    world.py         class World, created as World(), with reset(seed) -> obs and step(action) -> (obs, done, info) in
                     exactly the environment's formats, so that a program written for the environment runs unchanged
                     in the model.  Optional: reset_from(obs) -> obs, which starts an episode from a real reset
                     observation.  Required for the roles that have a world model.
    prediction.json  {"predicted_success_rate": <number in [0, 1]>, ...}: the success rate the agent expects of
                     policy.py on held-out situations, written before it hands in.  Roles with a world model.
    notes/hypothesis.json, notes/diagnosis.json, notes/fidelity.json
                     Optional; `verify` reports them as present or missing.

Delivered files are untrusted code.  The harness never imports them into its own process or into the process that
holds the environment: they are loaded by harness/_host.py in subprocesses, with timeouts (see evaluate.py).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from harness import wire

POLICY_FILE = "policy.py"
WORLD_FILE = "world.py"
PREDICTION_FILE = "prediction.json"
NOTE_FILES = ("notes/hypothesis.json", "notes/diagnosis.json", "notes/fidelity.json")

# role -> which deliverables are required besides policy.py
ROLES = {
    "world_model": {"world": True, "prediction": True},
    "cap": {"world": False, "prediction": False},
    "cap_with_tests": {"world": True, "prediction": True},
}


class ContractError(Exception):
    """An environment that does not follow the contract above."""


def load_environment(spec: dict):
    """Create the environment named by the resolved `[environment]` section of a config.  Changes the working
    directory and the import path of the calling process, so call it only in a process made for the environment."""
    for p in reversed(spec.get("paths") or []):
        if p not in sys.path:
            sys.path.insert(0, p)
    if spec.get("cwd"):
        os.chdir(spec["cwd"])
    module_name, _, func_name = spec["factory"].partition(":")
    if not module_name or not func_name:
        raise ContractError(f"environment factory must look like 'package.module:function', got {spec['factory']!r}")
    factory = getattr(importlib.import_module(module_name), func_name)
    env = factory(**(spec.get("kwargs") or {}))
    check_environment(env)
    return env


def check_environment(env) -> None:
    problems = []
    if not isinstance(getattr(env, "manual", None), str) or not env.manual.strip():
        problems.append("`manual` must be a non-empty string")
    max_steps = getattr(env, "max_steps", None)
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps <= 0:
        problems.append("`max_steps` must be a positive integer")
    for name in ("task_card", "reset", "step"):
        if not callable(getattr(env, name, None)):
            problems.append(f"`{name}` must be a method")
    if problems:
        raise ContractError("the environment does not follow the contract (see harness/contract.py): " + "; ".join(problems))


def describe_environment(env) -> dict:
    """The parts of an environment that go into the workspace: manual, task card, step limit."""
    card = env.task_card()
    try:
        card = json.loads(json.dumps(wire.pack(card)))
    except TypeError as ex:
        raise ContractError(f"task_card() must return JSON-serialisable data: {ex}") from None
    if not isinstance(card, dict):
        raise ContractError("task_card() must return a dict")
    return {"manual": env.manual, "task_card": card, "max_steps": int(env.max_steps)}


def read_prediction(path: Path):
    """(value, problem): the predicted success rate in prediction.json, or None and what is wrong with the file."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as ex:
        return None, f"not readable as JSON ({type(ex).__name__})"
    value = data.get("predicted_success_rate") if isinstance(data, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, 'no number under the key "predicted_success_rate"'
    if not 0.0 <= value <= 1.0:
        return None, f"predicted_success_rate = {value} is not in [0, 1]"
    return float(value), None


def obs_signature(obs) -> dict | None:
    """Keys and shapes of an observation, for comparing a world model's format with the environment's."""
    if not isinstance(obs, dict):
        return None
    sig = {}
    for key, value in obs.items():
        try:
            arr = np.asarray(value)
            sig[str(key)] = {"shape": list(arr.shape), "kind": arr.dtype.kind}
        except Exception:
            sig[str(key)] = {"shape": None, "kind": "?"}
    return sig


def compare_signatures(env_sig: dict | None, world_sig: dict | None) -> list[str]:
    """Differences between the environment's observation format and a world model's, as short statements."""
    if env_sig is None or world_sig is None:
        return []
    out = []
    missing = sorted(set(env_sig) - set(world_sig))
    extra = sorted(set(world_sig) - set(env_sig))
    if missing:
        out.append("observation keys missing in the world model: " + ", ".join(missing))
    if extra:
        out.append("observation keys only in the world model: " + ", ".join(extra))
    for key in sorted(set(env_sig) & set(world_sig)):
        a, b = env_sig[key]["shape"], world_sig[key]["shape"]
        if a != b:
            out.append(f"shape of observation '{key}': environment {tuple(a) if a is not None else '?'}, "
                       f"world model {tuple(b) if b is not None else '?'}")
        elif (env_sig[key]["kind"] in "biuf") != (world_sig[key]["kind"] in "biuf"):
            out.append(f"observation '{key}' is numeric in one and not in the other")
    return out
