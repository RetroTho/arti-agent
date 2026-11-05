"""The connection to Chrome's DevTools endpoint, for the browser plugin.

Chrome is driven with the DevTools protocol (CDP): JSON messages over a WebSocket. Arti sends a
command, `{"id", "method", "params"}`, and Chrome answers with a message carrying the same id. In
between, Chrome sends events, `{"method", "params"}`, such as "a request is waiting for you". A
message may also carry a `sessionId`, naming the tab, frame or worker it is about; without one it
is about the browser itself.

This file is only that plumbing: a minimal WebSocket client (just enough for Chrome on 127.0.0.1)
and `DevTools`, which sends commands, waits for their answers and hands every other message to
`handle_event`. What the events mean, and the safety rules, are in the plugin's __init__.py.
"""

import base64
import hashlib
import json
import os
import re
import socket
import struct
import threading
import time

from arti.tools import ToolError

COMMAND_SECONDS = 30
POLL = 0.2  # how often a wait checks for /stop
MAX_MESSAGE_BYTES = 20_000_000  # one message; a larger one ends the connection


class BrowserError(Exception):
    """The connection to Chrome broke, or Chrome did something it must not; start over."""


class Cancelled(Exception):
    """/stop arrived while the browser was busy."""


# --- WebSocket (RFC 6455) -------------------------------------------------------------------


def encode_frame(opcode: int, payload: bytes, mask: bytes | None = None) -> bytes:
    """One final frame. A client must mask what it sends (mask=None picks a random one)."""
    mask = os.urandom(4) if mask is None else mask
    size = len(payload)
    if size < 126:
        header = struct.pack("!BB", 0x80 | opcode, 0x80 | size)
    elif size < 1 << 16:
        header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, size)
    else:
        header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, size)
    return header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def decode_frame(data: bytes):
    """(final, opcode, payload, frame size) of the frame at the start of data, or None if
    data does not hold a whole frame yet."""
    if len(data) < 2:
        return None
    final, opcode, masked, size = data[0] & 0x80, data[0] & 0x0F, data[1] & 0x80, data[1] & 0x7F
    start = 2
    if size >= 126:
        width = 2 if size == 126 else 8
        if len(data) < 2 + width:
            return None
        size, start = int.from_bytes(data[2 : 2 + width], "big"), 2 + width
    if size > MAX_MESSAGE_BYTES:  # refused from the header, before it is read
        raise BrowserError("the browser sent a message over 20 MB")
    mask = data[start : start + 4] if masked else b""
    start += len(mask)
    if len(data) < start + size:
        return None
    payload = data[start : start + size]
    if mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return bool(final), opcode, payload, start + size


class WebSocket:
    """A text-message WebSocket client for ws://127.0.0.1 only (DevTools listens there)."""

    def __init__(self, url: str, timeout: float = 10):
        match = re.fullmatch(r"ws://127\.0\.0\.1:(\d+)(/\S*)", url)
        if not match:
            raise BrowserError("DevTools must listen on ws://127.0.0.1")
        port, path = match[1], match[2]
        self.sock = socket.create_connection(("127.0.0.1", int(port)), timeout)
        self.buffer = bytearray()  # appended in place: no quadratic copying of a large message
        self.parts: list[bytes] = []  # a fragmented message so far (kept across timeouts)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode())
        while b"\r\n\r\n" not in self.buffer:
            self._fill(time.monotonic() + timeout)
        head, _, rest = self.buffer.partition(b"\r\n\r\n")
        self.buffer = bytearray(rest)
        accept = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        )
        if b" 101 " not in head.split(b"\r\n")[0] or accept not in head:
            raise BrowserError("DevTools refused the WebSocket upgrade")

    def _fill(self, deadline: float) -> None:
        self.sock.settimeout(max(deadline - time.monotonic(), 0.001))
        chunk = self.sock.recv(1 << 20)  # socket.timeout is a TimeoutError
        if not chunk:
            raise BrowserError("the browser closed the connection")
        self.buffer += chunk

    def send(self, text: str) -> None:
        self.sock.sendall(encode_frame(0x1, text.encode()))

    def recv(self, timeout: float) -> str:
        """The next text message; TimeoutError when none arrives in time."""
        deadline = time.monotonic() + timeout
        while True:
            frame = decode_frame(self.buffer)
            if frame is None:
                self._fill(deadline)
                continue
            final, opcode, payload, used = frame
            del self.buffer[:used]
            if opcode == 0x9:  # ping
                self.sock.sendall(encode_frame(0xA, bytes(payload)))
            elif opcode == 0x8:  # close
                raise BrowserError("the browser closed the connection")
            elif opcode in (0x0, 0x1):  # a continuation or a text frame
                self.parts.append(bytes(payload))
                if sum(map(len, self.parts)) > MAX_MESSAGE_BYTES:
                    raise BrowserError("the browser sent a message over 20 MB")
                if final:
                    message, self.parts = b"".join(self.parts), []
                    return message.decode("utf-8", "replace")

    def close(self) -> None:
        try:
            self.sock.sendall(encode_frame(0x8, b"\x03\xe8"))  # close, "normal closure"
        except OSError:
            pass
        self.sock.close()


# --- DevTools commands and events -----------------------------------------------------------


class DevTools:
    """Commands and events over one WebSocket. cmd() sends to `session` (None: the browser
    itself); setting `cancel` (the current call's /stop) ends any wait with Cancelled."""

    def __init__(self, url: str):
        self.ws = WebSocket(url)
        self.session: str | None = None
        self.cancel = threading.Event()
        self._next_id = 0

    def handle_event(self, message: dict) -> None:
        """Called with every message that is not the answer being waited for."""

    def send(self, method: str, params: dict, session: str | None) -> int:
        """Send one command without waiting; returns its id."""
        self._next_id += 1
        message = {"id": self._next_id, "method": method, "params": params}
        if session:
            message["sessionId"] = session
        self.ws.send(json.dumps(message))
        return self._next_id

    def _receive(self, deadline: float) -> dict | None:
        """The next message from Chrome, or None once deadline has passed; checks /stop."""
        while (remaining := deadline - time.monotonic()) > 0:
            if self.cancel.is_set():
                raise Cancelled()
            try:
                return json.loads(self.ws.recv(min(remaining, POLL)))
            except TimeoutError:
                pass
        return None

    def cmd(self, method: str, timeout: float = COMMAND_SECONDS, **params) -> dict:
        """Send one command and wait for its answer, handling events on the way."""
        wanted = self.send(method, params, self.session)
        deadline = time.monotonic() + timeout
        while (message := self._receive(deadline)) is not None:
            if message.get("id") == wanted:
                if "error" in message:  # the command was refused; the connection is fine
                    raise ToolError(f"{method}: {message['error'].get('message')}")
                return message.get("result", {})
            self.handle_event(message)
        raise TimeoutError(f"{method} did not answer in {timeout:.0f} s")

    def pump(self, seconds: float) -> None:
        """Only handle events for a while (a paused request keeps its page waiting meanwhile)."""
        deadline = time.monotonic() + seconds
        while (message := self._receive(deadline)) is not None:
            self.handle_event(message)
