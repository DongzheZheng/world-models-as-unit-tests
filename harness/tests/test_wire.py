"""The wire format: JSON messages with numpy arrays, a size limit, and the key handshake."""
from __future__ import annotations

import json
import socket
import struct
import threading

import numpy as np
import pytest

import harness_support as hs  # noqa: F401  (puts the repository root on the import path)
from harness import wire


def test_round_trip_keeps_values_shapes_and_dtypes():
    msg = {"obs": {"pos": np.array([1.5, -2.25]), "grid": np.arange(6, dtype=np.int32).reshape(2, 3),
                   "flag": np.array([True, False]), "t": 7, "x": float("nan")},
           "done": False, "info": {"name": "a", "none": None, "list": [1, 2.5, np.float32(0.5), np.int64(3)]}}
    out = wire.loads(wire.dumps(msg))
    assert out["obs"]["pos"].dtype == np.float64 and np.array_equal(out["obs"]["pos"], msg["obs"]["pos"])
    assert out["obs"]["grid"].dtype == np.int32 and out["obs"]["grid"].shape == (2, 3)
    assert out["obs"]["flag"].dtype == np.bool_ and out["obs"]["flag"].tolist() == [True, False]
    assert out["obs"]["t"] == 7 and np.isnan(out["obs"]["x"])
    assert out["done"] is False and out["info"] == {"name": "a", "none": None, "list": [1, 2.5, 0.5, 3]}
    out["obs"]["pos"][0] = 9.0                      # decoded arrays are writable copies


def test_tuples_arrive_as_lists_and_floats_are_exact():
    x = 0.1 + 0.2
    assert wire.loads(wire.dumps((x, [x]))) == [x, [x]]


@pytest.mark.parametrize("value", [object(), {1, 2}, np.array(["a", "b"]), np.array([object()]), b"bytes"])
def test_values_that_are_not_data_are_refused_when_sending(value):
    with pytest.raises(TypeError):
        wire.dumps({"x": value})


def test_arrays_of_objects_are_refused_when_receiving():
    forged = json.dumps({"__ndarray__": ["|O", [1], "AAAA"]}).encode()
    with pytest.raises(wire.ProtocolError):
        wire.loads(forged)
    with pytest.raises(wire.ProtocolError):
        wire.loads(b"\x80\x04not json")


def test_send_and_receive_over_a_socket_with_timeout():
    a, b = socket.socketpair()
    try:
        wire.send(a, {"n": np.zeros(3)})
        assert wire.recv(b, 5.0)["n"].shape == (3,)
        with pytest.raises(TimeoutError):
            wire.recv(b, 0.05)
        a.close()
        with pytest.raises(EOFError):
            wire.recv(b, 5.0)
    finally:
        b.close()


def test_oversized_message_is_refused():
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack("!I", 2 ** 31))
        with pytest.raises(wire.ProtocolError):
            wire.recv(b, 5.0)
    finally:
        a.close()
        b.close()


def _handshake(server_key: bytes, client_key: bytes):
    a, b = socket.socketpair()
    result = {}

    def serve():
        try:
            wire.server_handshake(a, server_key, timeout=5.0)
            result["server"] = "ok"
        except Exception as ex:
            result["server"] = type(ex).__name__
        finally:
            a.close()

    t = threading.Thread(target=serve)
    t.start()
    try:
        wire.client_handshake(b, client_key, timeout=5.0)
        result["client"] = "ok"
    except Exception as ex:
        result["client"] = type(ex).__name__
    finally:
        b.close()
        t.join()
    return result


def test_handshake_accepts_the_shared_key_and_refuses_another():
    assert _handshake(b"k" * 32, b"k" * 32) == {"server": "ok", "client": "ok"}
    bad = _handshake(b"k" * 32, b"x" * 32)
    assert bad["server"] == "PermissionError" and bad["client"] != "ok"
