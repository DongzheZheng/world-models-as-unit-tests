"""The run config (TOML) and the layout of a run directory.

All keys with their defaults.  Paths are relative to the config file unless they are absolute.

    [run]
    name = "<config file name>"       # shown in the brief and in reports
    dir = "runs/<name>"               # the run directory; `--run-dir` on the command line overrides it
    workspace = "<dir>/workspace"     # the agent's workspace; put it elsewhere to keep it away from the state files

    [environment]
    factory = "package.module:function"     # required; returns an object that follows harness/contract.py
    paths = []                        # extra import paths for the environment process only
    kwargs = {}                       # keyword arguments of the factory
    cwd = "."                         # working directory of the environment process
    name = "<run name>"               # what the brief calls the environment

    [budget]
    trials = 5                        # real trials the agent may spend
    seed_range = [0, 999]             # seeds the agent may use (inclusive)
    held_out_seed_range = [1000, 1999]      # seeds reserved for held-out evaluation; never served to the workspace
    held_out_episodes = 50            # `verify` uses the first this-many seeds of that range

    [agent]
    kind = "command"                  # "command": run `command`; "replay": follow `script`, no tokens; "none"
    role = "world_model"              # brief template: "world_model", "cap" or "cap_with_tests"
    brief = ""                        # path of your own brief template; replaces the one chosen by `role`
    command = []                      # argument list; placeholders {workspace} {brief_file} {brief_text} {python}
    stdin = "none"                    # "brief" feeds the brief text to the command's standard input
    timeout_s = 7200                  # the command is killed after this many seconds
    time_allowance_min = 90           # what the brief tells the agent (default: three quarters of the timeout)
    env = {}                          # extra environment variables for the command
    script = ""                       # kind = "replay": the JSON script (see harness/agents.py)
    python = "<the harness's interpreter>"  # the interpreter the brief tells the agent to use

    [workspace.files]                 # extra files copied into the workspace at init: "path in workspace" = "source"
                                      # (role "cap_with_tests" needs "world.py" here: the world model it is given)

    [verify]
    model_episodes = 200              # episodes of the delivered policy in the delivered world model
    model_seed_first = 100000         # they use seeds model_seed_first, model_seed_first + 1, ...
    paired = true                     # if World has reset_from(): also start the model from the held-out resets
    workers = 4                       # processes for episodes (environment instances for the held-out evaluation)
    load_timeout_s = 120              # importing a delivered file and constructing its object
    episode_timeout_s = 120           # one episode, wall clock

    [screen]                          # defaults of `python -m harness screen`
    world = ""                        # the world model file
    policies = []                     # policy files (or [[screen.policies]] tables: file, name)
    names = []                        # optional names, one per policy
    paired = false
    out = "<run dir>/screen"          # output directory

    [server]
    transport = "auto"                # how the workspace reaches the environment: "unix" (socket file in the
                                      # workspace), "tcp" (127.0.0.1) or "auto" (unix where available)
    start_timeout_s = 120             # how long the environment may take to start

Layout of a run directory (config.paths()):

    <dir>/workspace/     BRIEF.md, MANUAL.md, task_card.json, lab.py, .lab/, trials/, notes/ and the deliverables
    <dir>/state/         config.toml, config.json (resolved), environment.json, budget.json, trials/ (the harness's
                         own copies of the trial records), agent.json, logs/
    <dir>/verify/        per-episode results of `verify`
    <dir>/report.json, <dir>/report.md
"""
from __future__ import annotations

import copy
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

from harness.contract import ROLES

DEFAULTS = {
    "run": {"name": None, "dir": None, "workspace": None},
    "environment": {"factory": None, "paths": [], "kwargs": {}, "cwd": ".", "name": None},
    "budget": {"trials": 5, "seed_range": [0, 999], "held_out_seed_range": [1000, 1999], "held_out_episodes": 50},
    "agent": {"kind": "command", "role": "world_model", "brief": "", "command": [], "stdin": "none",
              "timeout_s": 7200, "time_allowance_min": None, "env": {}, "script": "", "python": None},
    "workspace": {"files": {}},
    "verify": {"model_episodes": 200, "model_seed_first": 100000, "paired": True, "workers": 4,
               "load_timeout_s": 120, "episode_timeout_s": 120},
    "screen": {"world": "", "policies": [], "names": [], "paired": False, "out": None},
    "server": {"transport": "auto", "start_timeout_s": 120},
}
AGENT_KINDS = ("command", "replay", "none")
BRIEF_DIR = Path(__file__).resolve().parent / "briefs"


class ConfigError(Exception):
    pass


def _range(value, what):
    ok = (isinstance(value, list) and len(value) == 2
          and all(isinstance(v, int) and not isinstance(v, bool) for v in value) and 0 <= value[0] <= value[1])
    if not ok:
        raise ConfigError(f"{what} must be [lowest, highest] with integers 0 <= lowest <= highest")
    return [int(value[0]), int(value[1])]


def _number(value, what, minimum, integer=False):
    kinds = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds) or value < minimum:
        raise ConfigError(f"{what} must be {'an integer' if integer else 'a number'} >= {minimum}")
    return value


def load(path, run_dir=None) -> dict:
    """Read a TOML config, fill in the defaults, check it and make every path absolute.  Returns a plain dict that
    can be stored as JSON (state/config.json) and read back with load_resolved()."""
    path = Path(path).resolve()
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"no config file {path}") from None
    except tomllib.TOMLDecodeError as ex:
        raise ConfigError(f"{path}: {ex}") from None
    base = path.parent
    cfg = copy.deepcopy(DEFAULTS)
    for section, values in raw.items():
        if section not in DEFAULTS:
            raise ConfigError(f"{path}: unknown section [{section}]")
        if not isinstance(values, dict):
            raise ConfigError(f"{path}: [{section}] must be a table")
        for key, value in values.items():
            if key not in DEFAULTS[section]:
                raise ConfigError(f"{path}: unknown key `{key}` in [{section}]")
            cfg[section][key] = value

    def absolute(p):
        p = Path(p).expanduser()
        return str(p if p.is_absolute() else (base / p).resolve())

    run, env, budget, agent, verify, screen, server = (cfg[k] for k in ("run", "environment", "budget", "agent",
                                                                        "verify", "screen", "server"))
    run["name"] = str(run["name"] or path.stem)
    run["dir"] = str(Path(run_dir).resolve()) if run_dir else absolute(run["dir"] or f"runs/{run['name']}")
    run["workspace"] = absolute(run["workspace"]) if run["workspace"] else str(Path(run["dir"]) / "workspace")

    if not isinstance(env["factory"], str) or ":" not in env["factory"]:
        raise ConfigError("[environment] factory = \"package.module:function\" is required")
    if not isinstance(env["paths"], list) or not isinstance(env["kwargs"], dict):
        raise ConfigError("[environment] paths must be a list and kwargs a table")
    env["paths"] = [absolute(p) for p in env["paths"]]
    env["cwd"] = absolute(env["cwd"] or ".")
    env["name"] = str(env["name"] or run["name"])

    budget["trials"] = _number(budget["trials"], "[budget] trials", 0, integer=True)
    budget["seed_range"] = _range(budget["seed_range"], "[budget] seed_range")
    budget["held_out_seed_range"] = _range(budget["held_out_seed_range"], "[budget] held_out_seed_range")
    (lo, hi), (hlo, hhi) = budget["seed_range"], budget["held_out_seed_range"]
    if not (hi < hlo or hhi < lo):
        raise ConfigError("[budget] seed_range and held_out_seed_range must not overlap")
    budget["held_out_episodes"] = _number(budget["held_out_episodes"], "[budget] held_out_episodes", 1, integer=True)
    if budget["held_out_episodes"] > hhi - hlo + 1:
        raise ConfigError("[budget] held_out_episodes is larger than held_out_seed_range")

    if agent["kind"] not in AGENT_KINDS:
        raise ConfigError(f"[agent] kind must be one of {AGENT_KINDS}")
    if agent["brief"]:
        agent["brief"] = absolute(agent["brief"])
        if not Path(agent["brief"]).is_file():
            raise ConfigError(f"[agent] brief: no file {agent['brief']}")
    if agent["role"] not in ROLES:
        raise ConfigError(f"[agent] role must be one of {tuple(ROLES)}")
    if not isinstance(agent["command"], list) or not all(isinstance(a, str) for a in agent["command"]):
        raise ConfigError("[agent] command must be a list of strings")
    if agent["kind"] == "command" and not agent["command"]:
        raise ConfigError("[agent] kind = \"command\" needs a non-empty command")
    if agent["stdin"] not in ("none", "brief"):
        raise ConfigError("[agent] stdin must be \"none\" or \"brief\"")
    agent["timeout_s"] = _number(agent["timeout_s"], "[agent] timeout_s", 1)
    if agent["time_allowance_min"] is None:
        agent["time_allowance_min"] = max(1, int(agent["timeout_s"] * 0.75 // 60))
    _number(agent["time_allowance_min"], "[agent] time_allowance_min", 1)
    if not isinstance(agent["env"], dict) or not all(isinstance(v, str) for v in agent["env"].values()):
        raise ConfigError("[agent] env must be a table of strings")
    if agent["kind"] == "replay":
        if not agent["script"]:
            raise ConfigError("[agent] kind = \"replay\" needs script = \"<file>.json\"")
        agent["script"] = absolute(agent["script"])
        if not Path(agent["script"]).is_file():
            raise ConfigError(f"[agent] script: no file {agent['script']}")
    agent["python"] = str(agent["python"] or sys.executable)

    files = cfg["workspace"]["files"]
    if not isinstance(files, dict):
        raise ConfigError("[workspace.files] must be a table: \"path in workspace\" = \"source file\"")
    resolved = {}
    for dest, source in files.items():
        if Path(dest).is_absolute() or ".." in Path(dest).parts:
            raise ConfigError(f"[workspace.files] {dest!r} must be a path inside the workspace")
        resolved[dest] = absolute(source)
        if not Path(resolved[dest]).is_file():
            raise ConfigError(f"[workspace.files] {dest}: no file {resolved[dest]}")
    cfg["workspace"]["files"] = resolved
    if agent["role"] == "cap_with_tests" and not agent["brief"] and "world.py" not in resolved:
        raise ConfigError("role \"cap_with_tests\" needs the given world model: [workspace.files] \"world.py\" = \"...\"")

    verify["model_episodes"] = _number(verify["model_episodes"], "[verify] model_episodes", 1, integer=True)
    verify["model_seed_first"] = _number(verify["model_seed_first"], "[verify] model_seed_first", 0, integer=True)
    verify["workers"] = _number(verify["workers"], "[verify] workers", 1, integer=True)
    _number(verify["load_timeout_s"], "[verify] load_timeout_s", 0.1)
    _number(verify["episode_timeout_s"], "[verify] episode_timeout_s", 0.1)
    if not isinstance(verify["paired"], bool) or not isinstance(screen["paired"], bool):
        raise ConfigError("`paired` must be true or false")

    policies, names = [], list(screen["names"] or [])
    for i, item in enumerate(screen["policies"] or []):
        if isinstance(item, dict):
            if "file" not in item:
                raise ConfigError("[[screen.policies]] needs `file`")
            policies.append(absolute(item["file"]))
            if "name" in item:
                names = names + [None] * (i + 1 - len(names))
                names[i] = str(item["name"])
        else:
            policies.append(absolute(item))
    if names and len(names) != len(policies):
        raise ConfigError("[screen] names must have one entry per policy")
    screen["policies"], screen["names"] = policies, names
    screen["world"] = absolute(screen["world"]) if screen["world"] else ""
    screen["out"] = absolute(screen["out"]) if screen["out"] else str(Path(run["dir"]) / "screen")

    if server["transport"] not in ("auto", "unix", "tcp"):
        raise ConfigError("[server] transport must be \"auto\", \"unix\" or \"tcp\"")
    _number(server["start_timeout_s"], "[server] start_timeout_s", 1)

    cfg["config_file"] = str(path)
    return cfg


def paths(cfg: dict) -> SimpleNamespace:
    """Where things are in a run directory."""
    run = Path(cfg["run"]["dir"])
    state = run / "state"
    return SimpleNamespace(
        run=run, workspace=Path(cfg["run"]["workspace"]), state=state, logs=state / "logs",
        config_json=state / "config.json", config_toml=state / "config.toml", environment=state / "environment.json",
        budget=state / "budget.json", trials=state / "trials", agent=state / "agent.json",
        hypothesis_snapshot=state / "hypothesis_at_first_trial.json", verify=run / "verify",
        report_json=run / "report.json", report_md=run / "report.md")


def save_resolved(cfg: dict) -> None:
    p = paths(cfg)
    p.state.mkdir(parents=True, exist_ok=True)
    p.config_json.write_text(json.dumps(cfg, indent=1))
    p.config_toml.write_text(Path(cfg["config_file"]).read_text())


def load_resolved(run_dir) -> dict:
    """The config a run directory was initialised with."""
    f = Path(run_dir).resolve() / "state" / "config.json"
    if not f.is_file():
        raise ConfigError(f"{Path(run_dir)} is not an initialised run directory (no state/config.json); "
                          f"run `python -m harness init` first")
    return json.loads(f.read_text())


def is_initialised(cfg: dict) -> bool:
    return paths(cfg).config_json.is_file()


def held_out_seeds(cfg: dict) -> list[int]:
    first = cfg["budget"]["held_out_seed_range"][0]
    return list(range(first, first + cfg["budget"]["held_out_episodes"]))


def model_seeds(cfg: dict, episodes: int | None = None, first: int | None = None) -> list[int]:
    first = cfg["verify"]["model_seed_first"] if first is None else first
    n = cfg["verify"]["model_episodes"] if episodes is None else episodes
    return list(range(first, first + n))


def brief_template(cfg: dict) -> Path:
    return Path(cfg["agent"]["brief"]) if cfg["agent"]["brief"] else BRIEF_DIR / f"{cfg['agent']['role']}.md"
