"""lab.py: the client of the environment behind this workspace.

The environment runs in another process.  This file is the only way to reach it; it needs the standard library and
numpy.  MANUAL.md describes the environment (interface, units, success rule); this docstring describes the client.

    import lab

    lab.status()                 # free: trials used and left, the seeds you may use, the step limit
    lab.task_card()              # free: the public facts of the task, as a dict (same as task_card.json)

    obs = lab.reset(seed)        # starts ONE REAL TRIAL and returns the first observation (a dict)
    obs, done, info = lab.step(action)
    ...                          # until done is True; info["success"] says how the trial ended

    result = lab.run_trial(policy, seed)     # the loop above for an object with reset() and act(obs) -> action
    log = lab.load_trial(k)                  # the saved record of your k-th real trial, as a dict of arrays

    env = lab.Env()              # the same two calls as methods, env.reset(seed) and env.step(action), so that code
                                 # written against `env` also runs against an object with the same two methods

Rules enforced by the environment process, not by this file:

* Every reset() uses up one real trial, also when the previous episode was not finished.  When the budget is used
  up, reset() raises lab.BudgetExceeded.
* Only seeds inside the range reported by status() are accepted; anything else raises lab.SeedNotAllowed and costs
  nothing.  The same seed with the same actions gives the same episode if the environment is deterministic (see
  MANUAL.md).
* step() without a running episode raises lab.NoActiveTrial.  An action the environment rejects raises lab.LabError
  with the environment's message; the episode stays open and the step is not taken.
* Every real trial is saved when it ends or is abandoned: trials/trial_<k>.npz.  For each observation key there is
  an array with one row per observation (row 0 is the reset observation, so T + 1 rows after T steps); `actions`
  has T rows; `seed`, `trial`, `n_steps`, `finished` (False if abandoned), `success` and `info_json` (the last info
  dict as JSON text) are scalars.  If the values cannot be stacked into arrays the record is trials/trial_<k>.json.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import socket
import struct
from pathlib import Path

import numpy as np

__all__ = ["status", "task_card", "reset", "step", "run_trial", "load_trial", "trial_files", "close", "Env",
           "LabError", "BudgetExceeded", "SeedNotAllowed", "NoActiveTrial", "NotConnected"]

_HERE = Path(__file__).resolve().parent
_CONNECTION_FILE = _HERE / ".lab" / "connection.json"
_MAX_MESSAGE = 64 * 2 ** 20      # bytes; larger messages are refused
_MAX_UNIX_PATH = 100             # longest socket path passed to connect() directly
_ND = "__ndarray__"


class LabError(RuntimeError):
    """Any failure reported by the environment process."""


class BudgetExceeded(LabError):
    """All real trials of this run are used up."""


class SeedNotAllowed(LabError):
    """The seed is not an integer inside the range reported by status()."""


class NoActiveTrial(LabError):
    """step() was called without a running episode; call reset(seed) first."""


class NotConnected(LabError):
    """The environment process cannot be reached (it is not running, or the connection was lost)."""


class ProtocolError(LabError):
    """A message that does not follow the wire format."""


_ERRORS = {"BudgetExceeded": BudgetExceeded, "SeedNotAllowed": SeedNotAllowed, "NoActiveTrial": NoActiveTrial}


# ------------------------------------------------------------------------------------------------------------------
# Wire format: length-prefixed JSON messages.  Numpy arrays travel as base64 of their bytes; nothing is ever
# unpickled.  The harness imports these functions from this file, so both ends share one implementation.
# ------------------------------------------------------------------------------------------------------------------
def _pack(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind not in "biuf":
            raise TypeError(f"cannot send an array of dtype {obj.dtype}")
        data = np.ascontiguousarray(obj)
        return {_ND: [data.dtype.str, list(data.shape), base64.b64encode(data.tobytes()).decode("ascii")]}
    if isinstance(obj, np.generic):
        return _pack(obj.item())
    if isinstance(obj, dict):
        return {str(k): _pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pack(v) for v in obj]
    raise TypeError(f"cannot send a value of type {type(obj).__name__}")


def _unpack(obj):
    if isinstance(obj, dict):
        if len(obj) == 1 and _ND in obj:
            dtype, shape, data = obj[_ND]
            dt = np.dtype(dtype)
            if dt.kind not in "biuf":
                raise ProtocolError(f"array of dtype {dt} refused")
            arr = np.frombuffer(base64.b64decode(data), dtype=dt)
            return arr.reshape([int(n) for n in shape]).copy()
        return {k: _unpack(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack(v) for v in obj]
    return obj


def _dumps(obj) -> bytes:
    return json.dumps(_pack(obj), separators=(",", ":")).encode("utf-8")


def _loads(data: bytes):
    try:
        return _unpack(json.loads(data.decode("utf-8")))
    except ProtocolError:
        raise
    except Exception as ex:
        raise ProtocolError(f"malformed message: {type(ex).__name__}") from None


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise EOFError("connection closed")
        buf += chunk
    return bytes(buf)


def _send_frame(sock, payload: bytes) -> None:
    if len(payload) > _MAX_MESSAGE:
        raise ProtocolError("message too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_frame(sock, timeout: float | None = None) -> bytes:
    """One message.  With a timeout, TimeoutError is raised if it does not arrive in time; the connection must then be
    closed, since part of a message may have been read."""
    sock.settimeout(timeout)
    (n,) = struct.unpack("!I", _recv_exact(sock, 4))
    if n > _MAX_MESSAGE:
        raise ProtocolError("message too large")
    return _recv_exact(sock, n)


def _send(sock, obj) -> None:
    _send_frame(sock, _dumps(obj))


def _recv(sock, timeout: float | None = None):
    return _loads(_recv_frame(sock, timeout))


def _mac(key: bytes, label: bytes, nonce: bytes) -> bytes:
    return hmac.new(key, label + nonce, hashlib.sha256).digest()


def _client_handshake(sock, key: bytes, timeout: float = 20.0) -> None:
    """Prove knowledge of the key to the server and check that the server knows it too."""
    server_nonce = _recv_frame(sock, timeout)
    my_nonce = os.urandom(32)
    _send_frame(sock, _mac(key, b"client:", server_nonce) + my_nonce)
    answer = _recv_frame(sock, timeout)
    if not hmac.compare_digest(answer, _mac(key, b"server:", my_nonce)):
        raise NotConnected("the environment process did not accept this workspace's key")


def _server_handshake(sock, key: bytes, timeout: float = 20.0) -> None:
    """The server's half of the handshake; raises PermissionError if the client does not know the key."""
    my_nonce = os.urandom(32)
    _send_frame(sock, my_nonce)
    answer = _recv_frame(sock, timeout)
    if len(answer) != 64 or not hmac.compare_digest(answer[:32], _mac(key, b"client:", my_nonce)):
        raise PermissionError("wrong key")
    _send_frame(sock, _mac(key, b"server:", answer[32:]))


# ------------------------------------------------------------------------------------------------------------------
# Connection
# ------------------------------------------------------------------------------------------------------------------
_sock = None


def _connect():
    try:
        info = json.loads(_CONNECTION_FILE.read_text())
    except FileNotFoundError:
        raise NotConnected("the environment process is not running (no .lab/connection.json in this workspace)") from None
    key = bytes.fromhex(info["key"])
    try:
        if info["family"] == "unix":
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            path = str(_CONNECTION_FILE.parent / info["socket"])
            if len(os.fsencode(path)) <= _MAX_UNIX_PATH:
                sock.connect(path)
            else:                       # socket paths are limited to about 100 bytes: connect by its relative name
                cwd = os.getcwd()
                os.chdir(_CONNECTION_FILE.parent)
                try:
                    sock.connect(info["socket"])
                finally:
                    os.chdir(cwd)
        else:
            sock = socket.create_connection((info["host"], int(info["port"])), timeout=20.0)
        _client_handshake(sock, key)
        sock.settimeout(None)
    except NotConnected:
        raise
    except (OSError, EOFError) as ex:
        raise NotConnected(f"cannot reach the environment process: {type(ex).__name__}: {ex}") from None
    return sock


def close() -> None:
    """Close the connection.  A running episode is abandoned (its record is saved)."""
    global _sock
    if _sock is not None:
        try:
            _sock.close()
        finally:
            _sock = None


def _call(op: str, **fields) -> dict:
    global _sock
    if _sock is None:
        _sock = _connect()
    try:
        _send(_sock, {"op": op, **fields})
        reply = _recv(_sock)
    except (OSError, EOFError) as ex:
        close()
        raise NotConnected(f"connection to the environment process lost: {type(ex).__name__}: {ex}") from None
    if not reply.get("ok"):
        raise _ERRORS.get(reply.get("error"), LabError)(reply.get("message", "request refused"))
    return reply


# ------------------------------------------------------------------------------------------------------------------
# The interface
# ------------------------------------------------------------------------------------------------------------------
def status() -> dict:
    """Free.  Keys: trial_budget, trials_used, trials_left, seed_range ([lowest, highest], inclusive), max_steps,
    episode_running (bool), steps_in_episode."""
    reply = _call("status")
    reply.pop("ok", None)
    return reply


def task_card() -> dict:
    """Free.  The public facts of the task (the same dict as task_card.json)."""
    return _call("task_card")["task_card"]


def reset(seed: int) -> dict:
    """Start one real trial with this seed and return the first observation.  Uses up one trial."""
    return _call("reset", seed=seed)["obs"]


def step(action):
    """Apply one action in the running episode.  Returns (obs, done, info); on the last step info["success"] is set."""
    reply = _call("step", action=action)
    return reply["obs"], bool(reply["done"]), reply["info"]


def run_trial(policy, seed: int) -> dict:
    """One real trial with `policy` (an object with reset() and act(obs) -> action; set_seed(seed) is called first if
    it exists).  Returns {"seed", "success", "n_steps", "info"}.  Uses up one trial."""
    if hasattr(policy, "set_seed"):
        policy.set_seed(seed)
    policy.reset()
    obs = reset(seed)
    done, info, n = False, {}, 0
    while not done:
        obs, done, info = step(policy.act(obs))
        n += 1
    return {"seed": seed, "success": bool(info.get("success", False)), "n_steps": n, "info": info}


def trial_files() -> list:
    """Paths of the saved records of your real trials, in order."""
    files = [p for p in (_HERE / "trials").glob("trial_*.*") if p.suffix in (".npz", ".json")]
    return sorted(files, key=lambda p: int(p.stem.split("_")[1]))


def load_trial(k: int) -> dict:
    """The saved record of real trial k as a dict (see the module docstring for the keys)."""
    npz = _HERE / "trials" / f"trial_{int(k)}.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    return _unpack(json.loads((_HERE / "trials" / f"trial_{int(k)}.json").read_text()))


class Env:
    """The environment as an object: reset(seed) -> obs, step(action) -> (obs, done, info), task_card(), max_steps."""

    def __init__(self):
        self.max_steps = int(status()["max_steps"])

    def task_card(self) -> dict:
        return task_card()

    def reset(self, seed: int) -> dict:
        return reset(seed)

    def step(self, action):
        return step(action)
