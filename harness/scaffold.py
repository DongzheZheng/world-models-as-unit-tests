"""`init`: create a run directory and the agent's workspace.

The workspace gets what the agent may see and nothing else:

    BRIEF.md          the instructions, rendered from a template in harness/briefs/ (or `[agent] brief`)
    MANUAL.md         the environment's `manual`
    task_card.json    the environment's task_card()
    lab.py            the client (a copy of harness/client/lab.py)
    trials/           records of the real trials appear here
    notes/            for hypothesis.json, diagnosis.json, fidelity.json
    .lab/             connection file and socket of the environment server, while it runs
    <files of [workspace.files]>

The environment's source, the config and the budget counter stay outside (state/).  The manual and the task card are
read from the environment in a short-lived process, so this process never imports the environment either.

Placeholders of a brief template, written {{name}}: environment_name, run_name, trial_budget, seed_lo, seed_hi,
max_steps, time_allowance_min, python, held_out_episodes, model_episodes, episode_timeout_s.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from harness import config, evaluate
from harness.budget import Budget

CLIENT_FILE = Path(__file__).resolve().parent / "client" / "lab.py"
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_]+)\s*\}\}")


class ScaffoldError(Exception):
    pass


def describe_environment(config_json: Path, out_file: Path, log_path: Path, timeout: float) -> dict:
    """Ask a short-lived environment process for the manual, the task card and the step limit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "harness.server", "--config", str(config_json), "--mode", "describe",
           "--out", str(out_file)]
    with open(log_path, "ab") as log:
        try:
            code = subprocess.run(cmd, env=evaluate.trusted_env(), stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            raise ScaffoldError(f"the environment did not start within {timeout:g} s (log: {log_path})") from None
    if code != 0 or not out_file.is_file():
        tail = "\n".join(log_path.read_text(errors="replace").strip().splitlines()[-6:])
        raise ScaffoldError(f"the environment could not be created (log: {log_path}):\n{tail}")
    return json.loads(out_file.read_text())


def brief_context(cfg: dict, info: dict) -> dict:
    return {
        "environment_name": cfg["environment"]["name"],
        "run_name": cfg["run"]["name"],
        "trial_budget": cfg["budget"]["trials"],
        "seed_lo": cfg["budget"]["seed_range"][0],
        "seed_hi": cfg["budget"]["seed_range"][1],
        "max_steps": info["max_steps"],
        "time_allowance_min": cfg["agent"]["time_allowance_min"],
        "python": cfg["agent"]["python"],
        "held_out_episodes": cfg["budget"]["held_out_episodes"],
        "model_episodes": cfg["verify"]["model_episodes"],
        "episode_timeout_s": f"{cfg['verify']['episode_timeout_s']:g}",
    }


def render_brief(template: str, context: dict) -> str:
    def fill(match):
        name = match.group(1)
        if name not in context:
            raise ScaffoldError(f"brief template: unknown placeholder {{{{{name}}}}}; known: {', '.join(sorted(context))}")
        return str(context[name])
    return _PLACEHOLDER.sub(fill, template)


def init(cfg: dict) -> None:
    """Create the run directory of a resolved config (harness/config.py: load)."""
    p = config.paths(cfg)
    if config.is_initialised(cfg):
        raise ScaffoldError(f"{p.run} is already initialised")
    if p.workspace.exists() and any(p.workspace.iterdir()):
        raise ScaffoldError(f"the workspace directory {p.workspace} is not empty")
    for d in (p.state, p.logs, p.trials):
        d.mkdir(parents=True, exist_ok=True)
    config.save_resolved(cfg)
    try:
        info = describe_environment(p.config_json, p.environment, p.logs / "describe.log",
                                    float(cfg["server"]["start_timeout_s"]))
        template = config.brief_template(cfg).read_text()
        brief = render_brief(template, brief_context(cfg, info))
    except Exception:
        p.config_json.unlink(missing_ok=True)           # leave the directory "not initialised"
        raise
    ws = p.workspace
    for d in (ws, ws / "trials", ws / "notes", ws / ".lab"):
        d.mkdir(parents=True, exist_ok=True)
    (ws / "BRIEF.md").write_text(brief)
    (ws / "MANUAL.md").write_text(info["manual"].rstrip() + "\n")
    (ws / "task_card.json").write_text(json.dumps(info["task_card"], indent=1) + "\n")
    shutil.copyfile(CLIENT_FILE, ws / "lab.py")
    for dest, source in cfg["workspace"]["files"].items():
        target = ws / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    Budget(p.budget, cfg["budget"]["trials"])


def environment_info(cfg: dict) -> dict:
    """Manual, task card and step limit as stored at init."""
    return json.loads(config.paths(cfg).environment.read_text())
