"""Episodes of delivered programs: in the environment (real) and in a delivered world model (predicted).

Programs and world models written by an agent are untrusted code.  They are never imported into the harness's own
process or into a process that holds the environment.  They run in subprocesses (harness/_host.py), with the
directory of the delivered files as working directory, and every wait on them has a time limit; on a timeout the
subprocess is killed.  This is isolation against accidents and against a program that reaches for the environment,
not a security sandbox: the subprocess runs with the rights of the user who started the harness.

    real_episodes(...)     Environment servers (harness/server.py, one environment instance each) run the episode
                           loop: they send each observation to a policy host and get an action back.  The delivered
                           program never holds the environment or the seed list, so it cannot retry an episode.
    model_episodes(...)    One host process per chunk loads world.py and policy.py together and runs the episodes.
    check_file(...)        Import one delivered file and construct its object, to tell "broken" from "present".

Every episode becomes a dict {"seed", "success", "n_steps", "flag", "error", "wall_s"}.  `flag` is None for an
episode that ran to its end.  A flagged episode counts as a failure:

    policy_error       the program raised (or its process stopped)
    action_rejected    the environment raised on the program's action
    world_error        the world model raised          world_contract   the world model returned the wrong shape
    no_end             the episode was not ended within the step limit
    timeout            the episode took longer than the time limit; after two timeouts in a row the rest of the
                       chunk is not run (flag not_run)
    process_died       the host process stopped during the episode
    load_error         the delivered file could not be imported or its object not constructed
    environment_error  the environment itself failed

Every process started here gets OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS = 1 unless they are
already set, so that `workers` processes use about `workers` cores.  Host processes also get
PYTHONDONTWRITEBYTECODE = 1, so that verifying leaves no __pycache__ next to the delivered files.
"""
from __future__ import annotations

import concurrent.futures
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from harness import wire

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_DIR.parent
HOST_SCRIPT = PACKAGE_DIR / "_host.py"
_MARK = "HARNESS_ADDED_PYTHONPATH"
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


class HostTimeout(Exception):
    pass


class HostDied(Exception):
    pass


class ServerError(Exception):
    """The environment process failed (it did not start, or it stopped during a request)."""


# ------------------------------------------------------------------------------------------------------------------
# Subprocess plumbing
# ------------------------------------------------------------------------------------------------------------------
def trusted_env() -> dict:
    """Environment variables for a harness process (the environment server): the package is importable."""
    env = dict(os.environ)
    if env.get(_MARK) != str(PACKAGE_PARENT):
        env["PYTHONPATH"] = str(PACKAGE_PARENT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env[_MARK] = str(PACKAGE_PARENT)
    for name in _THREAD_VARS:
        env.setdefault(name, "1")
    return env


def untrusted_env() -> dict:
    """Environment variables for a host of delivered code: without the import path the harness added for itself."""
    env = dict(os.environ)
    added = env.pop(_MARK, None)
    if added:
        parts = env.get("PYTHONPATH", "").split(os.pathsep)
        if added in parts:
            parts.remove(added)
        parts = [p for p in parts if p]
        if parts:
            env["PYTHONPATH"] = os.pathsep.join(parts)
        else:
            env.pop("PYTHONPATH", None)
    for name in _THREAD_VARS:
        env.setdefault(name, "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")      # leave no __pycache__ next to the delivered files
    return env


def kill_process(proc: subprocess.Popen, group: bool = True) -> None:
    """Kill a subprocess (and, if it leads a session of its own, everything it started)."""
    if proc.poll() is None:
        try:
            if group:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


class Host:
    """A subprocess running harness/_host.py, and the socket to it."""

    def __init__(self, mode: str, cwd, log_path, sys_path=()):
        parent, child = socket.socketpair()
        cmd = [sys.executable, "-P", str(HOST_SCRIPT), "--fd", str(child.fileno()), "--mode", mode]
        for p in sys_path:
            cmd += ["--path", str(p)]
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "ab")
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(cwd), env=untrusted_env(), stdin=subprocess.DEVNULL,
                                         stdout=self._log, stderr=subprocess.STDOUT, pass_fds=(child.fileno(),),
                                         start_new_session=True)
        except BaseException:
            parent.close()
            self._log.close()
            raise
        finally:
            child.close()
        self.sock = parent

    def _gone(self) -> str:
        try:
            return f"exit code {self.proc.wait(timeout=2)}"
        except subprocess.TimeoutExpired:
            return "connection lost"

    def send(self, msg, timeout: float = 60.0) -> None:
        try:
            self.sock.settimeout(timeout)               # a host that does not read must not block the harness
            wire.send(self.sock, msg)
        except TimeoutError:
            raise HostTimeout() from None
        except OSError:
            raise HostDied(self._gone()) from None

    def recv(self, timeout: float):
        try:
            return wire.recv(self.sock, max(float(timeout), 0.001))
        except TimeoutError:
            raise HostTimeout() from None
        except (EOFError, OSError, wire.ProtocolError):
            raise HostDied(self._gone()) from None

    def request(self, msg, timeout: float):
        self.send(msg)
        return self.recv(timeout)

    def close(self) -> None:
        kill_process(self.proc)
        for f in (self.sock, self._log):
            try:
                f.close()
            except OSError:
                pass


def _failed(seed, flag: str, error: str, n_steps: int = 0) -> dict:
    return {"seed": seed, "success": False, "n_steps": n_steps, "flag": flag, "error": error, "wall_s": 0.0}


def check_file(kind: str, path, cwd, task_card: dict, timeout: float, log_path, sys_path=()) -> dict:
    """Import a delivered file ("policy" or "world") and construct its object in a host process.
    Returns {"ok", "error"} plus "hooks" (policy) or "has_reset_from" (world)."""
    try:
        host = Host("check", cwd, log_path, sys_path)
    except OSError as ex:
        return {"ok": False, "error": f"the host process could not be started: {type(ex).__name__}: {ex}"}
    try:
        return host.request({"kind": kind, "path": str(path), "task_card": task_card}, timeout)
    except HostTimeout:
        return {"ok": False, "error": f"timeout: not loaded within {timeout:g} s"}
    except HostDied as ex:
        return {"ok": False, "error": f"the process stopped while loading ({ex})"}
    finally:
        host.close()


# ------------------------------------------------------------------------------------------------------------------
# Real episodes: this part runs inside the environment server (the `evaluate` request of harness/server.py)
# ------------------------------------------------------------------------------------------------------------------
def _env_episode(env, host: Host, seed: int, max_steps: int, time_limit: float, keep_reset_obs: bool) -> dict:
    t0 = time.monotonic()
    state = {"n": 0, "reset_obs": None}

    def out(success, flag=None, error=None, restart=False):
        ep = {"seed": seed, "success": bool(success) and flag is None, "n_steps": state["n"], "flag": flag,
              "error": error, "wall_s": round(time.monotonic() - t0, 4)}
        if keep_reset_obs:
            ep["reset_obs"] = state["reset_obs"]
        if restart:
            ep["_restart"] = True
        return ep

    def left():
        return time_limit - (time.monotonic() - t0)

    try:
        reply = host.request({"op": "begin", "seed": seed}, left())
        if not reply.get("ok"):
            return out(False, "policy_error", reply.get("error"))
        try:
            obs = env.reset(seed)
            if keep_reset_obs:
                state["reset_obs"] = wire.pack(obs)
        except Exception as ex:
            return out(False, "environment_error", f"reset({seed}) raised {type(ex).__name__}: {ex}")
        for _ in range(max_steps):
            if left() <= 0:
                raise HostTimeout()
            try:
                host.send({"op": "act", "obs": obs})
            except TypeError as ex:
                return out(False, "environment_error", f"the observation cannot be sent: {ex}")
            reply = host.recv(left())
            if not reply.get("ok"):
                return out(False, "policy_error", reply.get("error"))
            try:
                obs, done, info = env.step(reply.get("action"))
            except Exception as ex:
                return out(False, "action_rejected", f"{type(ex).__name__}: {ex}"[:500])
            state["n"] += 1
            if done:
                return out(bool(info.get("success", False)) if isinstance(info, dict) else False)
        return out(False, "no_end", f"the environment did not end the episode within {max_steps} steps")
    except HostTimeout:
        return out(False, "timeout", f"the episode took longer than {time_limit:g} s", restart=True)
    except HostDied as ex:
        return out(False, "policy_error", f"the policy process stopped ({ex})", restart=True)


def drive_policy(env, *, policy, cwd, seeds, task_card: dict, max_steps: int, limits: dict, log_path,
                 keep_reset_obs: bool = False, sys_path=()) -> dict:
    """Run one episode per seed in `env` with the policy file `policy`, which is loaded in a host process.
    Returns {"load_error": str | None, "episodes": [...]}."""
    episodes, host, load_error, timeouts = [], None, None, 0
    try:
        for seed in seeds:
            if load_error is not None:
                episodes.append(_failed(seed, "load_error", load_error))
                continue
            if timeouts >= 2:
                episodes.append(_failed(seed, "not_run", "not run: two episodes in a row timed out"))
                continue
            if host is None:
                host = Host("policy", cwd, log_path, sys_path)
                try:
                    reply = host.request({"op": "load", "policy": str(policy), "task_card": task_card},
                                         limits["load_timeout_s"])
                    if not reply.get("ok"):
                        load_error = reply.get("error") or "could not be loaded"
                except HostTimeout:
                    load_error = f"timeout: not loaded within {limits['load_timeout_s']:g} s"
                except HostDied as ex:
                    load_error = f"the process stopped while loading ({ex})"
                if load_error is not None:
                    host.close()
                    host = None
                    episodes.append(_failed(seed, "load_error", load_error))
                    continue
            ep = _env_episode(env, host, seed, max_steps, limits["episode_timeout_s"], keep_reset_obs)
            if ep.pop("_restart", False):
                host.close()
                host = None
            timeouts = timeouts + 1 if ep["flag"] == "timeout" else 0
            episodes.append(ep)
    finally:
        if host is not None:
            host.close()
    return {"load_error": load_error, "episodes": episodes}


# ------------------------------------------------------------------------------------------------------------------
# Environment servers as seen from the harness
# ------------------------------------------------------------------------------------------------------------------
class ServerProcess:
    """An environment server (python -m harness.server) in a subprocess, and the control channel to it.

    The control channel is a socket the server inherits from this process.  It is the only way to ask for held-out
    seeds; there is no key for it on disk.  mode "agent" also opens the workspace's endpoint, mode "eval" does not."""

    def __init__(self, config_json, mode: str, log_path, start_timeout: float = 120.0):
        parent, child = socket.socketpair()
        cmd = [sys.executable, "-m", "harness.server", "--config", str(config_json), "--mode", mode,
               "--control-fd", str(child.fileno())]
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path)
        self._log = open(log_path, "ab")
        try:
            self.proc = subprocess.Popen(cmd, env=trusted_env(), stdin=subprocess.DEVNULL, stdout=self._log,
                                         stderr=subprocess.STDOUT, pass_fds=(child.fileno(),))
        except BaseException:
            parent.close()
            self._log.close()
            raise
        finally:
            child.close()
        self.sock = parent
        try:
            self.info = wire.recv(self.sock, start_timeout)
        except TimeoutError:
            self.kill()
            raise ServerError(f"the environment did not start within {start_timeout:g} s (log: {self.log_path})") from None
        except (EOFError, OSError, wire.ProtocolError):
            self.kill()
            raise ServerError(f"the environment process stopped while starting (log: {self.log_path})") from None
        if not self.info.get("ok"):
            self.kill()
            raise ServerError(f"the environment could not be started: {self.info.get('message')} (log: {self.log_path})")

    def call(self, op: str, timeout: float | None = None, **fields) -> dict:
        try:
            wire.send(self.sock, {"op": op, **fields})
            reply = wire.recv(self.sock, timeout)
        except TimeoutError:
            raise ServerError(f"the environment process did not answer `{op}` in time") from None
        except (EOFError, OSError, wire.ProtocolError):
            raise ServerError(f"the environment process stopped during `{op}` (log: {self.log_path})") from None
        if not reply.get("ok"):
            raise ServerError(f"{reply.get('error')}: {reply.get('message')}")
        return reply

    def stop(self) -> None:
        """Ask the server to finish (it saves the record of a running trial) and wait for it."""
        if self.proc.poll() is None:
            try:
                self.call("shutdown", timeout=30.0)
                self.proc.wait(timeout=30)
            except (ServerError, subprocess.TimeoutExpired):
                pass
        self.kill()

    def kill(self) -> None:
        kill_process(self.proc, group=False)
        for f in (self.sock, self._log):
            try:
                f.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def _split(items: list, k: int) -> list[list]:
    """Up to k contiguous chunks of nearly equal size."""
    k = max(1, min(k, len(items)))
    size, extra = divmod(len(items), k)
    out, i = [], 0
    for j in range(k):
        n = size + (1 if j < extra else 0)
        out.append(items[i:i + n])
        i += n
    return out


def _run_env_jobs(config_json, jobs: list[dict], workers: int, log_dir, start_timeout: float) -> list[dict]:
    """Send each job (the fields of an `evaluate` request) to one of `workers` environment servers.  A reply is
    {"load_error", "episodes"}, or {"server_error": message} if the environment process failed."""
    results: list = [None] * len(jobs)
    todo: queue.Queue = queue.Queue()
    for item in enumerate(jobs):
        todo.put(item)
    fatal: list[str] = []

    def work(w: int) -> None:
        server = None
        try:
            while True:
                try:
                    i, job = todo.get_nowait()
                except queue.Empty:
                    return
                if fatal:
                    results[i] = {"server_error": fatal[0]}
                    continue
                for attempt in (1, 2):
                    try:
                        if server is None:
                            try:
                                server = ServerProcess(config_json, "eval", Path(log_dir) / f"environment_{w}.log",
                                                       start_timeout)
                            except ServerError as ex:
                                fatal.append(str(ex))
                                results[i] = {"server_error": str(ex)}
                                break
                        results[i] = server.call("evaluate", **job)
                        break
                    except ServerError as ex:
                        server.kill()
                        server = None
                        results[i] = {"server_error": str(ex)}
        except Exception as ex:                         # e.g. no process could be started at all
            fatal.append(f"{type(ex).__name__}: {ex}")
        finally:
            if server is not None:
                server.stop()

    threads = [threading.Thread(target=work, args=(w,), daemon=True) for w in range(max(1, min(workers, len(jobs))))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r if r is not None else {"server_error": fatal[0] if fatal else "the job was not run"} for r in results]


def limits_of(cfg: dict) -> dict:
    return {"load_timeout_s": float(cfg["verify"]["load_timeout_s"]),
            "episode_timeout_s": float(cfg["verify"]["episode_timeout_s"])}


def real_episodes(config_json, cfg: dict, policies: list[dict], seeds: list[int], *, workers: int, log_dir,
                  keep_reset_obs: bool = False) -> list[dict]:
    """Run every policy on every seed in the environment.

    policies: [{"file": path, "cwd": path, "name": str}, ...].  Returns one dict per policy:
    {"load_error", "error" (environment process failure or None), "episodes": [... in the order of `seeds`]}."""
    limits = limits_of(cfg)
    chunks = _split(list(seeds), workers)
    jobs, index = [], []
    for pi, pol in enumerate(policies):
        for chunk in chunks:
            jobs.append(dict(policy=str(pol["file"]), cwd=str(pol["cwd"]), seeds=chunk, limits=limits,
                             log=str(Path(log_dir) / f"policy_{pol['name']}_real.log"), keep_reset_obs=keep_reset_obs,
                             sys_path=[str(p) for p in pol.get("sys_path", [])]))
            index.append(pi)
    replies = _run_env_jobs(config_json, jobs, workers, log_dir, float(cfg["server"]["start_timeout_s"]))
    out = [{"load_error": None, "error": None, "episodes": []} for _ in policies]
    for pi, job, reply in zip(index, jobs, replies):
        res = out[pi]
        if "server_error" in reply:
            res["error"] = res["error"] or reply["server_error"]
            res["episodes"] += [_failed(s, "environment_error", reply["server_error"]) for s in job["seeds"]]
        else:
            res["load_error"] = res["load_error"] or reply.get("load_error")
            res["episodes"] += reply["episodes"]
    return out


# ------------------------------------------------------------------------------------------------------------------
# Model episodes
# ------------------------------------------------------------------------------------------------------------------
def model_chunk(*, world, policy, cwd, starts: list[dict], task_card: dict, max_steps: int, limits: dict, log_path,
                sys_path=()) -> dict:
    """Run a policy file in a world-model file, one episode per start ({"seed"} or {"seed", "obs"}), in host
    processes.  Returns {"load_error", "load_what", "episodes", "has_reset_from", "obs_signature"}."""
    n = len(starts)
    results: list = [None] * n
    pos, strikes = 0, 0
    load_error = load_what = has_reset_from = signature = None
    while pos < n and load_error is None:
        if strikes >= 2:
            for i in range(pos, n):
                results[i] = _failed(starts[i]["seed"], "not_run", "not run: two episodes in a row timed out or stopped the process")
            pos = n
            break
        host = Host("model", cwd, log_path, sys_path)
        loaded = False
        try:
            host.send({"world": str(world), "policy": str(policy), "task_card": task_card, "episodes": starts[pos:],
                       "max_steps": max_steps})
            msg = host.recv(limits["load_timeout_s"])
            if not msg.get("ok"):
                load_what = msg.get("what")
                load_error = msg.get("error") or "could not be loaded"
                continue
            loaded, has_reset_from = True, msg.get("has_reset_from")
            while pos < n:
                msg = host.recv(limits["episode_timeout_s"])
                if msg.get("event") != "episode":
                    raise HostDied("unexpected message")
                sig = msg.pop("obs_signature", None)
                signature = signature if signature is not None else sig
                msg.pop("event", None)
                msg.pop("index", None)
                results[pos] = msg
                pos += 1
                strikes = 0
        except HostTimeout:
            if not loaded:
                load_error = f"timeout: not loaded within {limits['load_timeout_s']:g} s"
            else:
                results[pos] = _failed(starts[pos]["seed"], "timeout",
                                       f"the episode took longer than {limits['episode_timeout_s']:g} s")
                pos += 1
                strikes += 1
        except HostDied as ex:
            if not loaded:
                load_error = f"the process stopped while loading ({ex})"
            else:
                results[pos] = _failed(starts[pos]["seed"], "process_died", f"the process stopped during the episode ({ex})")
                pos += 1
                strikes += 1
        finally:
            host.close()
    if load_error is not None:
        for i in range(pos, n):
            results[i] = _failed(starts[i]["seed"], "load_error", load_error)
    return {"load_error": load_error, "load_what": load_what, "episodes": results, "has_reset_from": has_reset_from,
            "obs_signature": signature}


def model_episodes(world, policies: list[dict], starts: list[dict], *, task_card: dict, max_steps: int, limits: dict,
                   workers: int, log_dir, tag: str = "model") -> list[dict]:
    """Run every policy in the world model `world` from every start.  Returns one dict per policy:
    {"load_error", "load_what", "has_reset_from", "obs_signature", "episodes": [... in the order of `starts`]}."""
    world = Path(world)
    chunks = _split(list(starts), workers)
    jobs = [(pi, chunk) for pi in range(len(policies)) for chunk in chunks]

    def run(job):
        pi, chunk = job
        pol = policies[pi]
        try:
            return model_chunk(world=world, policy=pol["file"], cwd=pol["cwd"], starts=chunk, task_card=task_card,
                               max_steps=max_steps, limits=limits,
                               log_path=Path(log_dir) / f"policy_{pol['name']}_{tag}.log",
                               sys_path=[str(world.parent), *[str(p) for p in pol.get("sys_path", [])]])
        except Exception as ex:                         # e.g. no process could be started at all
            error = f"the host process could not be run: {type(ex).__name__}: {ex}"
            return {"load_error": error, "load_what": None, "has_reset_from": None, "obs_signature": None,
                    "episodes": [_failed(s["seed"], "load_error", error) for s in chunk]}

    with concurrent.futures.ThreadPoolExecutor(max(1, min(workers, len(jobs)))) as pool:
        replies = list(pool.map(run, jobs))
    out = [{"load_error": None, "load_what": None, "has_reset_from": None, "obs_signature": None, "episodes": []}
           for _ in policies]
    for (pi, _), reply in zip(jobs, replies):
        res = out[pi]
        for key in ("load_error", "load_what", "has_reset_from", "obs_signature"):
            if res[key] is None:
                res[key] = reply[key]
        res["episodes"] += reply["episodes"]
    return out


def summarise(episodes: list[dict]) -> dict:
    """Counts of a list of episode dicts."""
    n = len(episodes)
    successes = sum(1 for e in episodes if e["success"])
    flags: dict = {}
    for e in episodes:
        if e.get("flag"):
            flags[e["flag"]] = flags.get(e["flag"], 0) + 1
    return {"episodes": n, "successes": successes, "success_rate": successes / n if n else None,
            "flagged": sum(flags.values()), "flags": flags,
            "first_error": next((e["error"] for e in episodes if e.get("flag")), None)}
