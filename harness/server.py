"""The environment server: the one process that holds the environment, and the rules around it.

    python -m harness.server --config <resolved config .json> --mode agent|eval --control-fd N
    python -m harness.server --config <resolved config .json> --mode describe --out FILE

The harness starts it (harness/evaluate.py: ServerProcess); it is not meant to be started by hand.

Two kinds of callers reach the environment, and they are told apart by the channel they come in on:

* the workspace (mode "agent" only).  The server listens on a socket whose address and key it writes to
  <workspace>/.lab/connection.json; lab.py connects there.  Requests on this endpoint are bound by the rules:
    - a reset() needs an unused real trial.  The count is kept in state/budget.json (harness/budget.py); when the
      budget is used up the request fails with BudgetExceeded and the environment is not touched;
    - the seed must be an integer inside [budget] seed_range, else SeedNotAllowed (no trial is used).  Held-out
      seeds are outside that range, so the workspace cannot ask for them;
    - one episode runs at a time; a new reset() or a closed connection abandons the running one.  One server
      runs per run directory at a time (state/server.lock);
    - every trial is recorded: state/trials/trial_<k>.npz, copied to <workspace>/trials/;
    - when the environment raises, the workspace gets the exception's type and message, never a traceback (so no
      source lines or file names of the environment, unless its own messages contain them).
* the harness.  The control channel is a socket this process inherited from the process that started it; nothing on
  disk gives access to it.  It accepts `status`, `shutdown`, `reset`/`step` with any seed, and `evaluate`: run a
  delivered policy on a list of seeds (harness/evaluate.py: drive_policy).  Held-out episodes are requested this way.

Mode "eval" has no workspace endpoint: it is used for held-out evaluation after the agent has finished.
Mode "describe" writes the manual, the task card and the step limit of the environment to a file and exits.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import secrets
import shutil
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from harness import contract, wire
from harness.budget import Budget, BudgetExceeded

RESERVED_KEYS = ("actions", "seed", "trial", "n_steps", "finished", "success", "info_json")


def log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, file=sys.stderr, flush=True)


class Refused(Exception):
    """A request that breaks a rule; `kind` becomes the error name the client sees."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


# ------------------------------------------------------------------------------------------------------------------
# Trial records
# ------------------------------------------------------------------------------------------------------------------
def write_trial_record(directory: Path, k: int, seed: int, observations: list, actions: list, finished: bool,
                       success, info) -> str:
    """Save one real trial as trial_<k>.npz (or .json if the values do not stack into arrays); returns the name."""
    directory.mkdir(parents=True, exist_ok=True)
    try:
        info_json = json.dumps(wire.pack(info if isinstance(info, dict) else {}))
    except TypeError:
        info_json = "{}"
    try:
        out = {}
        for key in observations[0]:
            name = f"obs_{key}" if key in RESERVED_KEYS else str(key)
            out[name] = np.stack([np.asarray(o[key]) for o in observations])
            if out[name].dtype.kind not in "biuf":
                raise TypeError(f"observation '{key}' is not numeric")
        if actions and isinstance(actions[0], dict):
            for key in actions[0]:
                out[f"action_{key}"] = np.stack([np.asarray(a[key], dtype=float) for a in actions])
        else:
            acts = np.asarray(actions, dtype=float)
            out["actions"] = acts if len(actions) else np.zeros((0,))
        out.update(seed=int(seed), trial=int(k), n_steps=len(actions), finished=bool(finished),
                   success=bool(success) if success is not None else False, info_json=info_json)
        name = f"trial_{k}.npz"
        tmp = directory / f".trial_{k}.tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, directory / name)
    except Exception as ex:                       # ragged or non-numeric values: keep everything as JSON
        log(f"trial {k}: not stackable into arrays ({type(ex).__name__}: {ex}); writing JSON")
        name = f"trial_{k}.json"
        data = {"seed": int(seed), "trial": int(k), "n_steps": len(actions), "finished": bool(finished),
                "success": bool(success) if success is not None else False, "info_json": info_json,
                "observations": observations, "actions": actions}
        (directory / name).write_text(json.dumps(wire.pack(data)))
    return name


def _copy_value(v):
    return np.array(v) if isinstance(v, np.ndarray) else v


# ------------------------------------------------------------------------------------------------------------------
# The gate: the environment plus the rules
# ------------------------------------------------------------------------------------------------------------------
class Gate:
    def __init__(self, env, cfg: dict | None, with_workspace: bool):
        self.env = env
        self.lock = threading.RLock()
        self.info = contract.describe_environment(env)
        self.max_steps = self.info["max_steps"]
        self.with_workspace = with_workspace
        self.episode = None             # the running episode: dict(owner, role, seed, trial, observations, actions)
        self.budget = None
        if with_workspace:
            from harness import config as config_module
            self.paths = config_module.paths(cfg)
            self.seed_range = cfg["budget"]["seed_range"]
            self.budget = Budget(self.paths.budget, cfg["budget"]["trials"])

    # ---- requests -----------------------------------------------------------------------------------------------
    def handle(self, role: str, owner, request) -> dict:
        """One request -> one reply dict.  Never raises."""
        with self.lock:
            try:
                if not isinstance(request, dict):
                    raise Refused("LabError", "a request must be a JSON object")
                op = request.get("op")
                if op == "status":
                    return {"ok": True, **self.status()}
                if op == "task_card":
                    return {"ok": True, "task_card": self.info["task_card"]}
                if op == "reset":
                    return {"ok": True, **self.reset(role, owner, request.get("seed"))}
                if op == "step":
                    return {"ok": True, **self.step(role, owner, request.get("action"))}
                raise Refused("LabError", f"unknown request {op!r}")
            except Refused as ex:
                return {"ok": False, "error": ex.kind, "message": str(ex)}
            except BudgetExceeded as ex:
                return {"ok": False, "error": "BudgetExceeded", "message": str(ex)}
            except Exception:
                log("internal error:\n" + traceback.format_exc())
                return {"ok": False, "error": "LabError", "message": "internal error in the environment process"}

    def status(self) -> dict:
        ep = self.episode
        out = {"max_steps": self.max_steps, "episode_running": ep is not None,
               "steps_in_episode": len(ep["actions"]) if ep else 0}
        if self.budget is not None:
            out.update(trial_budget=self.budget.trial_budget, trials_used=self.budget.used,
                       trials_left=self.budget.left, seed_range=list(self.seed_range))
        return out

    def reset(self, role: str, owner, seed) -> dict:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise Refused("SeedNotAllowed", "the seed must be an integer")
        counted = role == "agent"
        if counted:
            lo, hi = self.seed_range
            if not lo <= seed <= hi:
                raise Refused("SeedNotAllowed", f"seed {seed} is outside the range you may use, {lo} to {hi}")
            self.budget.check()
        self.abandon()
        try:
            obs = self.env.reset(seed)
        except Exception as ex:
            log(f"environment reset({seed}) failed:\n" + traceback.format_exc())
            raise Refused("LabError", f"the environment could not be reset: {type(ex).__name__}: {ex}") from None
        if not isinstance(obs, dict):
            raise Refused("LabError", "the environment returned an observation that is not a dict")
        trial = None
        if counted:
            if self.budget.used == 0:
                self._snapshot_hypothesis()
            trial = self.budget.start(seed)
            log(f"trial {trial} started: seed {seed}")
        self.episode = {"owner": owner, "role": role, "seed": seed, "trial": trial,
                        "observations": [{k: _copy_value(v) for k, v in obs.items()}], "actions": []}
        reply = {"obs": obs}
        if counted:
            reply.update(trial=trial, trials_left=self.budget.left)
        return reply

    def step(self, role: str, owner, action) -> dict:
        ep = self.episode
        if ep is None or ep["role"] != role or (role == "agent" and ep["owner"] != owner):
            raise Refused("NoActiveTrial", "no episode is running on this connection; call reset(seed) first")
        try:
            result = self.env.step(action)
            obs, done, info = result
        except Exception as ex:
            log(f"environment step failed ({type(ex).__name__}: {ex})")
            raise Refused("LabError", f"the environment rejected the step: {type(ex).__name__}: {ex}") from None
        info = dict(info) if isinstance(info, dict) else {}
        done = bool(done)
        ep["observations"].append({k: _copy_value(v) for k, v in obs.items()})
        ep["actions"].append(_copy_value(action) if not isinstance(action, dict)
                             else {k: _copy_value(v) for k, v in action.items()})
        if not done and len(ep["actions"]) >= self.max_steps:
            done = True
            info.setdefault("success", False)
            info["ended_by_step_limit"] = True
        if done:
            info["success"] = bool(info.get("success", False))
            self._finish(True, info["success"], info)
        return {"obs": obs, "done": done, "info": info}

    # ---- episodes -----------------------------------------------------------------------------------------------
    def abandon(self, owner=None) -> None:
        """End the running episode without a result (if `owner` is given: only if it belongs to that connection)."""
        with self.lock:
            ep = self.episode
            if ep is not None and (owner is None or ep["owner"] == owner):
                self._finish(False, None, {})

    def _finish(self, finished: bool, success, info) -> None:
        ep, self.episode = self.episode, None
        if ep["trial"] is None:
            return
        k = ep["trial"]
        try:
            name = write_trial_record(self.paths.trials, k, ep["seed"], ep["observations"], ep["actions"], finished,
                                      success, info)
            dest = self.paths.workspace / "trials"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.paths.trials / name, dest / name)
        except Exception:
            name = None
            log(f"trial {k}: the record could not be written:\n" + traceback.format_exc())
        self.budget.end(k, len(ep["actions"]), finished, success, name)
        log(f"trial {k} {'finished' if finished else 'abandoned'}: {len(ep['actions'])} steps"
            + (f", success={bool(success)}" if finished else ""))

    def _snapshot_hypothesis(self) -> None:
        """Keep a copy of notes/hypothesis.json as it was when the first real trial started."""
        src = self.paths.workspace / "notes" / "hypothesis.json"
        try:
            if src.is_file():
                shutil.copyfile(src, self.paths.hypothesis_snapshot)
            self.budget.note_hypothesis(src.is_file())
        except OSError:
            log("could not copy notes/hypothesis.json:\n" + traceback.format_exc())

    def close(self) -> None:
        self.abandon()
        if callable(getattr(self.env, "close", None)):
            try:
                self.env.close()
            except Exception:
                log("environment close() failed:\n" + traceback.format_exc())


# ------------------------------------------------------------------------------------------------------------------
# The workspace endpoint
# ------------------------------------------------------------------------------------------------------------------
class AgentEndpoint:
    """Listens for lab.py.  Address and key go to <workspace>/.lab/connection.json, removed again on close()."""

    def __init__(self, gate: Gate, workspace: Path, transport: str):
        self.gate = gate
        self.dir = workspace / ".lab"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "connection.json"
        self.key = secrets.token_bytes(32)
        self.socket_path = None
        self.stop = threading.Event()
        self.sessions = itertools.count(1)
        if transport in ("auto", "unix") and hasattr(socket, "AF_UNIX"):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket_path = self.dir / "lab.sock"
            self.socket_path.unlink(missing_ok=True)
            if len(os.fsencode(str(self.socket_path))) <= wire.MAX_UNIX_PATH:
                self.sock.bind(str(self.socket_path))
            else:                                   # socket paths are limited to about 100 bytes: bind by name
                cwd = os.getcwd()
                os.chdir(self.dir)
                try:
                    self.sock.bind("lab.sock")
                finally:
                    os.chdir(cwd)
            os.chmod(self.socket_path, 0o600)
            info = {"family": "unix", "socket": "lab.sock"}
        elif transport == "unix":
            raise RuntimeError("[server] transport = \"unix\" is not available on this platform")
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind(("127.0.0.1", 0))
            info = {"family": "tcp", "host": "127.0.0.1", "port": self.sock.getsockname()[1]}
        self.sock.listen(16)
        self.sock.settimeout(0.25)
        self.public = dict(info)
        info["key"] = self.key.hex()
        tmp = self.dir / "connection.json.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(info, f)
        os.replace(tmp, self.file)
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def _accept(self) -> None:
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn) -> None:
        owner = next(self.sessions)
        try:
            wire.server_handshake(conn, self.key)
            while not self.stop.is_set():
                try:
                    request = wire.recv(conn)
                except wire.ProtocolError:
                    wire.send(conn, {"ok": False, "error": "LabError", "message": "malformed request"})
                    break
                reply = self.gate.handle("agent", owner, request)
                try:
                    data = wire.dumps(reply)
                except TypeError as ex:
                    log(f"reply not encodable: {ex}")
                    data = wire.dumps({"ok": False, "error": "LabError",
                                       "message": "the environment returned a value that cannot be sent"})
                wire.send_frame(conn, data)
        except (EOFError, OSError, PermissionError):
            pass
        finally:
            self.gate.abandon(owner)
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self.stop.set()
        self.file.unlink(missing_ok=True)
        try:
            self.sock.close()
        except OSError:
            pass
        if self.socket_path is not None:
            self.socket_path.unlink(missing_ok=True)
        self.thread.join(timeout=2)


# ------------------------------------------------------------------------------------------------------------------
# The control channel
# ------------------------------------------------------------------------------------------------------------------
def serve_control(control, gate: Gate) -> None:
    """Answer the harness until it says `shutdown` or goes away."""
    while True:
        try:
            request = wire.recv(control)
        except (EOFError, OSError, wire.ProtocolError):
            log("control channel closed; stopping")
            return
        op = request.get("op") if isinstance(request, dict) else None
        if op == "shutdown":
            wire.send(control, {"ok": True})
            return
        if op == "evaluate":
            try:
                from harness import evaluate
                with gate.lock:
                    gate.abandon()
                    result = evaluate.drive_policy(
                        gate.env, policy=request["policy"], cwd=request["cwd"], seeds=request["seeds"],
                        task_card=gate.info["task_card"], max_steps=gate.max_steps, limits=request["limits"],
                        log_path=request["log"], keep_reset_obs=bool(request.get("keep_reset_obs")),
                        sys_path=request.get("sys_path") or [])
                reply = {"ok": True, **result}
            except Exception as ex:
                log("evaluate failed:\n" + traceback.format_exc())
                reply = {"ok": False, "error": "LabError", "message": f"{type(ex).__name__}: {ex}"}
        else:
            reply = gate.handle("harness", "control", request)
        try:
            wire.send(control, reply)
        except TypeError as ex:
            wire.send(control, {"ok": False, "error": "LabError", "message": f"reply cannot be sent: {ex}"})


def lock_run_directory(cfg: dict):
    """Allow one environment server per run directory, so that one process owns the budget counter."""
    import fcntl
    from harness import config as config_module
    handle = open(config_module.paths(cfg).state / "server.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("an environment server is already running for this run directory") from None
    return handle


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.server", description="Environment server of the harness (started by `python -m harness`).")
    ap.add_argument("--config", required=True, help="resolved config (state/config.json of a run directory)")
    ap.add_argument("--mode", choices=("agent", "eval", "describe"), required=True)
    ap.add_argument("--control-fd", type=int, help="inherited socket to the harness (modes agent and eval)")
    ap.add_argument("--out", help="mode describe: file to write")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())

    if args.mode == "describe":
        env = contract.load_environment(cfg["environment"])
        info = contract.describe_environment(env)
        if callable(getattr(env, "close", None)):
            env.close()
        Path(args.out).write_text(json.dumps(info, indent=1))
        return 0

    if args.control_fd is None:
        ap.error("--control-fd is required for modes agent and eval")
    control = socket.socket(fileno=args.control_fd)
    signal.signal(signal.SIGINT, signal.SIG_IGN)            # the harness decides when to stop
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    gate = endpoint = lock = None
    try:
        try:
            if args.mode == "agent":
                lock = lock_run_directory(cfg)              # held until this process ends
            env = contract.load_environment(cfg["environment"])
            gate = Gate(env, cfg, with_workspace=args.mode == "agent")
            if args.mode == "agent":
                endpoint = AgentEndpoint(gate, Path(cfg["run"]["workspace"]), cfg["server"]["transport"])
        except Exception as ex:
            log("could not start:\n" + traceback.format_exc())
            wire.send(control, {"ok": False, "message": f"{type(ex).__name__}: {ex}"})
            return 1
        ready = {"ok": True, "pid": os.getpid(), "max_steps": gate.max_steps, "task_card": gate.info["task_card"],
                 "endpoint": endpoint.public if endpoint else None}
        wire.send(control, ready)
        log(f"ready ({args.mode}); environment {cfg['environment']['factory']}")
        serve_control(control, gate)
        return 0
    finally:
        if endpoint is not None:
            endpoint.close()
        if gate is not None:
            gate.close()
        log("stopped")


if __name__ == "__main__":
    sys.exit(main())
