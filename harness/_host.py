"""Host process for untrusted code: the only place where delivered policy and world-model files are imported.

Started by harness/evaluate.py as a script of its own,

    python -P _host.py --fd <inherited socket> --mode check|policy|model [--path DIR ...]

with the directory of the delivered files as working directory.  It talks to its parent over the inherited socket in
the wire format of client/lab.py and gets nothing else: no environment object, no key, no seed list.  Its standard
output and error go to a log file.  The parent enforces every time limit by killing this process.

    check    import one file and construct its object:  {"kind": "policy"|"world", "path", "task_card"}
             -> {"ok", "error", "hooks" | "has_reset_from"}
    policy   answer requests of an episode loop that runs elsewhere (the environment never enters this process):
             {"op": "load", "policy", "task_card"}, {"op": "begin", "seed"}, {"op": "act", "obs"} -> {"ok", ...}
    model    run a policy in a world model, both delivered files in this process:
             {"world", "policy", "task_card", "episodes": [{"seed", "obs"?}, ...], "max_steps"}
             -> {"event": "loaded", ...}, one {"event": "episode", ...} per episode, {"event": "end"}

An episode result is {"seed", "success", "n_steps", "flag", "error", "wall_s"}.  `flag` is None for an episode that
ran to its end, else one of: policy_error (the program raised), world_error (the world model raised), world_contract
(wrong return shape), no_end (the world model never ended the episode).  A flagged episode counts as a failure.

This is not a sandbox: the code runs with the rights of the user who started the harness.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys
import time
import traceback
from pathlib import Path


def _load_wire():
    spec = importlib.util.spec_from_file_location("_harness_wire", Path(__file__).resolve().parent / "client" / "lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Problem(Exception):
    """A delivered file that does not follow the contract."""


def _describe(ex: BaseException) -> str:
    """Exception type, message and where in the delivered code it was raised (file name and line only)."""
    text = f"{type(ex).__name__}: {ex}".strip().rstrip(":")
    if isinstance(ex, Problem):
        return str(ex)
    here = os.path.abspath(__file__)
    frames = [f for f in traceback.extract_tb(ex.__traceback__)
              if not f.filename.startswith("<") and os.path.abspath(f.filename) != here]
    if frames and not isinstance(ex, SyntaxError):          # a SyntaxError names its own file and line
        text += f" ({os.path.basename(frames[-1].filename)}, line {frames[-1].lineno})"
    return text[:500]


def _import_file(path: str):
    path = os.path.abspath(path)
    name = Path(path).stem
    if name in sys.modules and os.path.abspath(getattr(sys.modules[name], "__file__", "") or "") != path:
        name = f"_delivered_{name}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Problem(f"{os.path.basename(path)} cannot be imported as a Python file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _make_policy(path: str, task_card: dict):
    module = _import_file(path)
    if not callable(getattr(module, "make_policy", None)):
        raise Problem(f"{os.path.basename(path)} has no function make_policy()")
    policy = module.make_policy()
    for name in ("reset", "act"):
        if not callable(getattr(policy, name, None)):
            raise Problem(f"make_policy() returned an object without a method {name}()")
    if callable(getattr(policy, "bind", None)):
        policy.bind(task_card)
    return policy


def _make_world(path: str):
    module = _import_file(path)
    if not isinstance(getattr(module, "World", None), type):
        raise Problem(f"{os.path.basename(path)} has no class World")
    world = module.World()
    for name in ("reset", "step"):
        if not callable(getattr(world, name, None)):
            raise Problem(f"World has no method {name}()")
    return world


def _hooks(obj, names) -> list:
    return [n for n in names if callable(getattr(obj, n, None))]


def _signature(obs):
    import numpy as np
    sig = {}
    for key, value in obs.items():
        try:
            arr = np.asarray(value)
            sig[str(key)] = {"shape": list(arr.shape), "kind": arr.dtype.kind}
        except Exception:
            sig[str(key)] = {"shape": None, "kind": "?"}
    return sig


# ------------------------------------------------------------------------------------------------------------------
def run_check(sock, wire) -> None:
    msg = wire._recv(sock)
    try:
        if msg["kind"] == "policy":
            policy = _make_policy(msg["path"], msg.get("task_card") or {})
            reply = {"ok": True, "error": None, "hooks": _hooks(policy, ("bind", "set_seed"))}
        else:
            world = _make_world(msg["path"])
            reply = {"ok": True, "error": None, "has_reset_from": bool(_hooks(world, ("reset_from",)))}
    except (Exception, SystemExit) as ex:
        reply = {"ok": False, "error": _describe(ex)}
    wire._send(sock, reply)


def run_policy(sock, wire) -> None:
    policy = None
    while True:
        try:
            msg = wire._recv(sock)
        except EOFError:
            return
        op = msg.get("op")
        try:
            if op == "load":
                policy = _make_policy(msg["policy"], msg.get("task_card") or {})
                data = wire._dumps({"ok": True})
            elif op == "begin":
                if callable(getattr(policy, "set_seed", None)):
                    policy.set_seed(msg["seed"])
                policy.reset()
                data = wire._dumps({"ok": True})
            elif op == "act":
                action = policy.act(msg["obs"])
                try:
                    data = wire._dumps({"ok": True, "action": action})
                except TypeError as ex:
                    raise Problem(f"act() returned something that is not an action: {ex}") from None
            else:
                return
        except (Exception, SystemExit) as ex:
            data = wire._dumps({"ok": False, "error": _describe(ex)})
        wire._send_frame(sock, data)


def _model_episode(world, policy, start: dict, max_steps: int) -> dict:
    seed, t0 = start["seed"], time.time()
    state = {"n": 0, "sig": None}

    def out(success, flag=None, error=None):
        return {"seed": seed, "success": bool(success) and flag is None, "n_steps": state["n"], "flag": flag,
                "error": error, "wall_s": round(time.time() - t0, 4), "obs_signature": state["sig"]}

    try:
        if callable(getattr(policy, "set_seed", None)):
            policy.set_seed(seed)
        policy.reset()
    except (Exception, SystemExit) as ex:
        return out(False, "policy_error", _describe(ex))
    try:
        obs = world.reset_from(start["obs"]) if start.get("obs") is not None else world.reset(seed)
    except (Exception, SystemExit) as ex:
        return out(False, "world_error", _describe(ex))
    if not isinstance(obs, dict):
        return out(False, "world_contract", f"World.reset() must return a dict of observations, got {type(obs).__name__}")
    state["sig"] = _signature(obs)
    for _ in range(max_steps):
        try:
            action = policy.act(obs)
        except (Exception, SystemExit) as ex:
            return out(False, "policy_error", _describe(ex))
        try:
            result = world.step(action)
        except (Exception, SystemExit) as ex:
            return out(False, "world_error", _describe(ex))
        state["n"] += 1
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            got = f"{len(result)} values" if isinstance(result, (tuple, list)) else type(result).__name__
            return out(False, "world_contract", f"World.step() must return (obs, done, info), got {got}")
        obs, done, info = result
        if not isinstance(obs, dict):
            return out(False, "world_contract", f"World.step() must return a dict of observations first, got {type(obs).__name__}")
        try:
            done = bool(done)
        except Exception:
            return out(False, "world_contract", "World.step(): `done` is not a truth value")
        if done:
            if not isinstance(info, dict) or "success" not in info:
                return out(False, "world_contract", 'World.step() must set info["success"] on the last step')
            try:
                return out(bool(info["success"]))
            except Exception:
                return out(False, "world_contract", 'info["success"] is not a truth value')
    return out(False, "no_end", f"the world model did not end the episode within {max_steps} steps")


def run_model(sock, wire) -> None:
    msg = wire._recv(sock)
    card = msg.get("task_card") or {}
    try:
        what = "world"
        world = _make_world(msg["world"])
        what = "policy"
        policy = _make_policy(msg["policy"], card)
    except (Exception, SystemExit) as ex:
        wire._send(sock, {"event": "loaded", "ok": False, "what": what, "error": _describe(ex)})
        return
    wire._send(sock, {"event": "loaded", "ok": True, "has_reset_from": bool(_hooks(world, ("reset_from",)))})
    for index, start in enumerate(msg["episodes"]):
        wire._send(sock, {"event": "episode", "index": index, **_model_episode(world, policy, start, int(msg["max_steps"]))})
    wire._send(sock, {"event": "end"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Host process for delivered policy and world-model files (started by the harness).")
    ap.add_argument("--fd", type=int, required=True, help="inherited socket to the parent")
    ap.add_argument("--mode", choices=("check", "policy", "model"), required=True)
    ap.add_argument("--path", action="append", default=[], help="extra import path (the delivered files' directories)")
    args = ap.parse_args()
    wire = _load_wire()
    sock = socket.socket(fileno=args.fd)
    for p in [os.getcwd(), *args.path]:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        {"check": run_check, "policy": run_policy, "model": run_model}[args.mode](sock, wire)
    except (EOFError, OSError):
        pass                    # the parent went away
    finally:
        try:
            sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
