"""Agent adapters: how the harness starts an agent in its workspace.  The harness knows nothing about any model.

kind = "command"
    Runs `[agent] command` (an argument list, no shell) with the workspace as working directory and waits for it.
    Placeholders, replaced inside every argument:
        {workspace}    absolute path of the workspace
        {brief_file}   absolute path of BRIEF.md in the workspace
        {brief_text}   the text of BRIEF.md
        {python}       the interpreter the brief names (`[agent] python`)
        {state_dir}    absolute path of the harness's state directory, for flags that deny the agent access to it
    `[agent] stdin = "brief"` feeds the brief text to the command's standard input.  Standard output and error go to
    state/logs/agent_stdout.log and agent_stderr.log.  After `[agent] timeout_s` seconds the command and everything
    it started (its process group) are killed.  The outcome is written to state/agent.json.
    Examples for two agent command-line tools: harness/examples/agents/.

kind = "replay"
    Spends no tokens.  Runs harness/_replay_agent.py in the workspace, which follows `[agent] script`, a JSON file:
        {"steps": [{"copy":  {"<path in workspace>": "<source, relative to the script>", ...}},
                   {"trial": {"seed": 3, "actions": [[...], {"repeat": 10, "action": [...]}, ...]}},
                   {"trial": {"record": "<a trial_<k>.npz saved earlier>"}},
                   ...]}
    Steps run in order.  A trial is re-flown through lab.py, so it is counted and recorded like any other; its
    actions are listed one by one, as {"repeat": n, "action": a} runs, or taken from a saved trial record.

kind = "none"
    Starts nothing (`run` then only verifies what is in the workspace).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from harness import config

REPLAY_SCRIPT = Path(__file__).resolve().parent / "_replay_agent.py"


def build_command(cfg: dict) -> list[str]:
    p = config.paths(cfg)
    brief_file = p.workspace / "BRIEF.md"
    if cfg["agent"]["kind"] == "replay":
        return [sys.executable, "-P", str(REPLAY_SCRIPT), cfg["agent"]["script"]]
    values = {"{workspace}": str(p.workspace), "{brief_file}": str(brief_file), "{python}": cfg["agent"]["python"],
              "{state_dir}": str(p.state), "{brief_text}": brief_file.read_text()}
    out = []
    for arg in cfg["agent"]["command"]:
        for key, value in values.items():
            arg = arg.replace(key, value)
        out.append(arg)
    return out


def _kill_group(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
    try:
        os.killpg(proc.pid, signal.SIGKILL)         # whatever the command left behind in its group
    except (ProcessLookupError, PermissionError):
        pass


def run_agent(cfg: dict) -> dict:
    """Start the agent, wait until it exits or its time is up, and return (and store) the outcome."""
    p = config.paths(cfg)
    kind = cfg["agent"]["kind"]
    record = {"kind": kind, "role": cfg["agent"]["role"], "command": None, "started": time.time(), "wall_s": 0.0,
              "exit_code": None, "timed_out": False, "error": None, "stdout": None, "stderr": None}
    if kind != "none":
        argv = build_command(cfg)
        shown = cfg["agent"]["command"] if kind == "command" else argv      # without the expanded brief text
        record.update(command=shown, stdout=str(p.logs / "agent_stdout.log"), stderr=str(p.logs / "agent_stderr.log"))
        p.logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(cfg["agent"]["env"])
        feed = (p.workspace / "BRIEF.md").read_bytes() if cfg["agent"]["stdin"] == "brief" and kind == "command" else None
        t0 = time.monotonic()
        with open(record["stdout"], "ab") as out, open(record["stderr"], "ab") as err:
            try:
                proc = subprocess.Popen(argv, cwd=str(p.workspace), env=env, stdout=out, stderr=err,
                                        stdin=subprocess.PIPE if feed is not None else subprocess.DEVNULL,
                                        start_new_session=True)
            except OSError as ex:
                record["error"] = f"the command could not be started: {type(ex).__name__}: {ex}"
            else:
                try:
                    proc.communicate(input=feed, timeout=float(cfg["agent"]["timeout_s"]))
                except subprocess.TimeoutExpired:
                    record["timed_out"] = True
                finally:
                    _kill_group(proc)
                record["exit_code"] = proc.returncode
        record["wall_s"] = round(time.monotonic() - t0, 2)
    p.agent.parent.mkdir(parents=True, exist_ok=True)
    p.agent.write_text(json.dumps(record, indent=1))
    return record
