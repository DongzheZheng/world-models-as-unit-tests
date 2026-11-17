"""Wire format of the harness: length-prefixed JSON messages over a socket, numpy arrays as base64 of their bytes.

Nothing is pickled, so a message from untrusted code (the agent's client, a delivered policy) cannot run code in the
receiving process.  The implementation is in client/lab.py, the file copied into the agent's workspace, and is
re-exported here so that both ends of every connection use the same functions.

    send(sock, obj) / recv(sock, timeout=None)      one message; recv raises TimeoutError, EOFError or ProtocolError
    dumps(obj) / loads(data)                        the encoding alone
    client_handshake(sock, key) / server_handshake(sock, key)       mutual proof of a shared key (HMAC-SHA256)
"""
from harness.client.lab import ProtocolError
from harness.client.lab import _MAX_UNIX_PATH as MAX_UNIX_PATH
from harness.client.lab import _client_handshake as client_handshake
from harness.client.lab import _dumps as dumps
from harness.client.lab import _loads as loads
from harness.client.lab import _pack as pack
from harness.client.lab import _recv as recv
from harness.client.lab import _recv_frame as recv_frame
from harness.client.lab import _send as send
from harness.client.lab import _send_frame as send_frame
from harness.client.lab import _server_handshake as server_handshake
from harness.client.lab import _unpack as unpack

__all__ = ["ProtocolError", "MAX_UNIX_PATH", "client_handshake", "server_handshake", "dumps", "loads", "pack",
           "unpack", "send", "recv", "send_frame", "recv_frame"]
