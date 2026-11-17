"""Helpers shared by the harness tests.  Everything runs on the toy environment (harness/examples/toy) except the
pusht tests; no test calls a model or spends tokens.

    python -m pytest harness/tests -q                   everything, including the slow pusht reproduction
    python -m pytest harness/tests -q -m "not slow"     the quick suite
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import itertools
import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
ROOT = HARNESS.parent
TOY = HARNESS / "examples" / "toy"
REFERENCE = TOY / "reference"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import config, evaluate, scaffold  # noqa: E402

_counter = itertools.count()

# a policy for the toy environment that needs no other file
PLAIN_POLICY = (REFERENCE / "policy_poor.py").read_text()


def toml_text(table: dict) -> str:
    """A nested dict of strings, numbers, booleans and lists as TOML."""
    lines: list[str] = []

    def value(v):
        return ("true" if v else "false") if isinstance(v, bool) else json.dumps(v)

    def emit(prefix: str, tab: dict):
        if prefix:
            lines.append(f"[{prefix}]")
        for k, v in tab.items():
            if not isinstance(v, dict):
                lines.append(f"{json.dumps(k)} = {value(v)}")
        for k, v in tab.items():
            if isinstance(v, dict):
                emit(f"{prefix}.{json.dumps(k)}" if prefix else json.dumps(k), v)

    emit("", table)
    return "\n".join(lines) + "\n"


def merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def base_config() -> dict:
    """A small run on the toy environment: 3 real trials, 4 held-out episodes, 6 model episodes."""
    return {
        "run": {"name": "test"},
        "environment": {"factory": "toy_env:make_env", "paths": [str(TOY)], "name": "puck table"},
        "budget": {"trials": 3, "seed_range": [0, 99], "held_out_seed_range": [1000, 1999], "held_out_episodes": 4},
        "agent": {"kind": "none", "role": "world_model"},
        "verify": {"model_episodes": 6, "workers": 2, "episode_timeout_s": 20, "load_timeout_s": 30},
    }


def new_run(base: Path, overrides=None, init=True, files=None) -> dict:
    """Write a config under `base`, load it and (by default) initialise its run directory.
    `files` maps a path in the workspace to its text, or to a Path whose text is copied."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    cfg_file = base / "run.toml"
    cfg_file.write_text(toml_text(merge(base_config(), overrides or {})))
    cfg = config.load(cfg_file, base / "run")
    if init:
        scaffold.init(cfg)
        put_files(cfg, files or {})
    return cfg


def put_files(cfg: dict, files: dict) -> None:
    ws = config.paths(cfg).workspace
    for name, content in files.items():
        target = ws / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.read_text() if isinstance(content, Path) else content)


def reference_files(*names) -> dict:
    names = names or ("world.py", "policy.py", "prediction.json")
    return {name: REFERENCE / name for name in names}


@contextlib.contextmanager
def workspace_server(cfg: dict):
    """The environment server of a run directory, as `run` and `serve` start it."""
    p = config.paths(cfg)
    server = evaluate.ServerProcess(p.config_json, "agent", p.logs / "server.log", 60.0)
    try:
        yield server
    finally:
        server.stop()


@contextlib.contextmanager
def lab_client(cfg: dict):
    """The workspace's own lab.py, imported the way a script in the workspace would see it."""
    path = config.paths(cfg).workspace / "lab.py"
    spec = importlib.util.spec_from_file_location(f"lab_under_test_{next(_counter)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        module.close()


def read_json(path):
    return json.loads(Path(path).read_text())
