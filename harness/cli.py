"""Command line of the harness: python -m harness <command> ...   (run from the repository root)"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from harness import __version__, agents, budget, config, contract, evaluate, scaffold, screen, verify

DESCRIPTION = """\
Point a coding agent at your own environment, let it write the environment's world model as code,
and verify the program it hands in: in its world model (predicted) and on held-out seeds (real).

  init     create a run directory: the agent's workspace and, outside it, the harness's state
  serve    keep the environment server of a run directory up until Ctrl-C (for an agent you start yourself)
  run      init if needed, start the environment server, run the agent, stop the server, verify
  verify   score the delivered policy in the delivered world model and on the held-out seeds
  screen   score any policy files in a world model and in the environment; table and scatter plot
  status   real trials used and deliverables present in a run directory

The config file is described in harness/config.py, the environment and deliverable contracts in
harness/contract.py, the agent adapters in harness/agents.py.  Zero-token examples: harness/examples/.
"""


def _parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _run_config(args) -> dict:
    """The config of a run directory: as stored at init if it is initialised, else read from --config."""
    if args.config:
        cfg = config.load(args.config, args.run_dir)
        if config.is_initialised(cfg):
            cfg = config.load_resolved(cfg["run"]["dir"])
        return cfg
    if not args.run_dir:
        raise config.ConfigError("give --config or --run-dir")
    return config.load_resolved(args.run_dir)


def cmd_init(args) -> int:
    cfg = config.load(args.config, args.run_dir)
    scaffold.init(cfg)
    p = config.paths(cfg)
    print(f"run directory: {p.run}\nworkspace:     {p.workspace}\nbrief:         {p.workspace / 'BRIEF.md'}")
    return 0


def _server(cfg: dict) -> evaluate.ServerProcess:
    p = config.paths(cfg)
    return evaluate.ServerProcess(p.config_json, "agent", p.logs / "server.log", float(cfg["server"]["start_timeout_s"]))


def cmd_serve(args) -> int:
    cfg = _run_config(args)
    if not config.is_initialised(cfg):
        scaffold.init(cfg)
    p = config.paths(cfg)
    with _server(cfg) as server:
        print(f"environment server is up (log: {p.logs / 'server.log'})\nworkspace: {p.workspace}\n"
              f"Start the agent in the workspace with BRIEF.md as its instructions.  Ctrl-C stops the server.",
              flush=True)
        try:
            while server.proc.poll() is None:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print()
    counter = budget.read(p.budget) or {"trials": []}
    print(f"stopped; real trials used: {len(counter['trials'])} of {cfg['budget']['trials']}")
    return 0


def cmd_run(args) -> int:
    cfg = config.load(args.config, args.run_dir)
    if config.is_initialised(cfg):
        cfg = config.load_resolved(cfg["run"]["dir"])
        print(f"run directory {cfg['run']['dir']} exists; using the config it was initialised with")
    else:
        scaffold.init(cfg)
    p = config.paths(cfg)
    print(f"workspace: {p.workspace}", flush=True)
    with _server(cfg):
        print(f"environment server is up; agent: {cfg['agent']['kind']} ({cfg['agent']['role']})", flush=True)
        record = agents.run_agent(cfg)
    if record["kind"] != "none":
        print(f"agent finished: exit code {record['exit_code']}, {record['wall_s']} s"
              + (", stopped at the time limit" if record["timed_out"] else "")
              + (f", {record['error']}" if record["error"] else ""), flush=True)
    if args.skip_verify:
        return 0
    return _verify(cfg, args)


def _verify(cfg: dict, args) -> int:
    rep = verify.verify(cfg, workers=args.workers, model_episodes=getattr(args, "model_episodes", None),
                        skip_real=getattr(args, "skip_real", False), skip_model=getattr(args, "skip_model", False))
    p = config.paths(cfg)
    print(p.report_md.read_text())
    print(f"report: {p.report_json}")
    failed = rep["real"] is not None and any(e == "environment_error" for e in rep["real"]["flags"])
    return 1 if failed else 0


def cmd_verify(args) -> int:
    cfg = _run_config(args)
    if not config.is_initialised(cfg):
        raise config.ConfigError(f"{cfg['run']['dir']} is not initialised; run `python -m harness init` first")
    return _verify(cfg, args)


def cmd_screen(args) -> int:
    cfg = config.load(args.config, args.run_dir)
    out = screen.screen(cfg, world=args.world, policies=args.policy or None, names=args.name or None, out_dir=args.out,
                        model_episodes=args.model_episodes, model_seed_first=args.model_seed_first,
                        real_seeds=_parse_seeds(args.real_seeds) if args.real_seeds else None, workers=args.workers,
                        paired=True if args.paired else None, skip_real=args.skip_real, skip_model=args.skip_model)
    print((Path(out["out_dir"]) / "screen.md").read_text())
    print(f"outputs: {out['out_dir']}")
    return 0


def cmd_status(args) -> int:
    cfg = _run_config(args)
    if not config.is_initialised(cfg):
        raise config.ConfigError(f"{cfg['run']['dir']} is not initialised")
    p = config.paths(cfg)
    counter = budget.read(p.budget) or {"trials": [], "trial_budget": cfg["budget"]["trials"]}
    role = contract.ROLES[cfg["agent"]["role"]]
    required = {contract.POLICY_FILE: True, contract.WORLD_FILE: role["world"], contract.PREDICTION_FILE: role["prediction"]}
    files = {name: {"present": (p.workspace / name).is_file(), "required": required.get(name, False)}
             for name in (contract.POLICY_FILE, contract.WORLD_FILE, contract.PREDICTION_FILE, *contract.NOTE_FILES)}
    out = {"run": cfg["run"]["name"], "role": cfg["agent"]["role"], "workspace": str(p.workspace),
           "trials_used": len(counter["trials"]), "trial_budget": counter["trial_budget"],
           "trials": counter["trials"], "server_running": (p.workspace / ".lab" / "connection.json").exists(),
           "deliverables": files, "report": str(p.report_json) if p.report_json.is_file() else None}
    print(json.dumps(out, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m harness", description=DESCRIPTION,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")

    def common(sp, config_required):
        sp.add_argument("--config", required=config_required, help="run config (TOML); see harness/config.py")
        sp.add_argument("--run-dir", help="run directory (default: [run] dir of the config)")

    def workers(sp):
        sp.add_argument("--workers", type=int, help="processes for episodes (default: [verify] workers)")

    sp = sub.add_parser("init", help="create the run directory and the agent's workspace",
                        description="Create the run directory: workspace/ (BRIEF.md, MANUAL.md, task_card.json, lab.py, "
                                    "trials/, notes/) and state/ (config copy, budget counter, logs).")
    common(sp, True)
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("serve", help="keep the environment server up until Ctrl-C",
                        description="Start the environment server of a run directory (initialising it from --config if "
                                    "needed) and keep it up until Ctrl-C.  Use it when you start the agent yourself, "
                                    "for example interactively, in the workspace; then run `verify`.")
    common(sp, False)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("run", help="init if needed, serve, run the agent, stop, verify",
                        description="The whole procedure: create the run directory if it does not exist, start the "
                                    "environment server, run the agent of the config in the workspace, stop the server "
                                    "and verify what the agent left behind.")
    common(sp, True)
    workers(sp)
    sp.add_argument("--skip-verify", action="store_true", help="stop after the agent has finished")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("verify", help="score the delivered policy in the delivered world model and for real",
                        description="Check the deliverables in the workspace, run policy.py in world.py (predicted "
                                    "success) and on the held-out seeds in the environment (real success), and write "
                                    "report.json and report.md into the run directory.  Delivered code runs in "
                                    "subprocesses with time limits.")
    common(sp, False)
    workers(sp)
    sp.add_argument("--model-episodes", type=int, help="episodes in the world model (default: [verify] model_episodes)")
    sp.add_argument("--skip-real", action="store_true", help="do not run the held-out evaluation in the environment")
    sp.add_argument("--skip-model", action="store_true", help="do not run the policy in the world model")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("screen", help="score policy files in a world model and in the environment",
                        description="Score any list of policy files in a world model file (predicted success) and in "
                                    "the environment (real success); write screen.json, screen.md and screen.svg.  "
                                    "Arguments not given come from the [screen] section of the config.")
    common(sp, True)
    workers(sp)
    sp.add_argument("--world", help="world model file (default: [screen] world)")
    sp.add_argument("--policy", action="append", default=[], metavar="FILE",
                    help="policy file; repeat for several (default: [screen] policies)")
    sp.add_argument("--name", action="append", default=[], help="name of the policy with the same position; repeat")
    sp.add_argument("--out", help="output directory (default: [screen] out)")
    sp.add_argument("--model-episodes", type=int, help="episodes per program in the world model")
    sp.add_argument("--model-seed-first", type=int, help="first seed of the model episodes")
    sp.add_argument("--real-seeds", metavar="A-B", help="seeds of the real episodes, e.g. 6000-6049 (default: held-out seeds)")
    sp.add_argument("--paired", action="store_true",
                    help="also start the world model from the real reset observations (needs World.reset_from)")
    sp.add_argument("--skip-real", action="store_true", help="world model only")
    sp.add_argument("--skip-model", action="store_true", help="environment only")
    sp.set_defaults(func=cmd_screen)

    sp = sub.add_parser("status", help="real trials used and deliverables present",
                        description="Print the budget counter and which deliverables are in the workspace, as JSON.")
    common(sp, False)
    sp.set_defaults(func=cmd_status)
    return ap


def _terminate(*_):
    raise KeyboardInterrupt


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        signal.signal(signal.SIGTERM, _terminate)       # so that servers and agents are stopped on the way out
    except ValueError:
        pass                                            # not the main thread
    try:
        return args.func(args)
    except (config.ConfigError, scaffold.ScaffoldError, screen.ScreenError, evaluate.ServerError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
