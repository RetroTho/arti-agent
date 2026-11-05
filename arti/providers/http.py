"""The one HTTP client of the providers: HTTP/1.1 over asyncio streams, plus server-sent events.

One connection per request (`Connection: close`), TLS verified against the system store, no
proxy and no redirects, so credentials never follow a hop to another host. Plain HTTP only to
loopback (tests, local servers). Cancelling the task closes the socket at once; a stream that
stays silent for IDLE seconds, or ends mid-body, is a retryable "dropped" error; TCP keepalive
notices a dead connection sooner.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
import ssl
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from arti.providers import ProviderError, bad_response, classify, dropped, retry_info

CONNECT_TIMEOUT = 15
IDLE = 180  # seconds of upstream silence before a stream counts as dropped
# TCP keepalive: probe after 15 s of quiet, every 10 s, 3 times, so a dead path (sleep, a
# changed network) fails in about 45 s, while a live but quiet stream is left alone.
KEEPALIVE = (("TCP_KEEPIDLE", 15), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 3))
HEAD_LIMIT = 64 * 1024  # status line + headers
EVENT_LIMIT = 16 * 1024 * 1024  # one server-sent event
ERROR_LIMIT = 64 * 1024  # an error body, read only to classify it
USER_AGENT = "arti/1"
_FIXED = ("host", "content-length", "connection", "transfer-encoding", "accept-encoding")
_BLANK_LINE = re.compile(rb"\r?\n\r?\n")


def is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def parse_head(head: bytes) -> tuple[str, list[tuple[str, str]]]:
    """(the first line, the (name, value) header pairs) of an HTTP message head."""
    lines = head.decode("latin-1").split("\r\n")
    pairs = [
        tuple(part.strip() for part in line.split(":", 1)) for line in lines[1:] if ":" in line
    ]
    return lines[0], pairs


async def _guard(coro):
    """Await one read of a response: silence for IDLE seconds or a broken connection is a
    retryable error (nothing of the reply was used)."""
    try:
        async with asyncio.timeout(IDLE):
            return await coro
    except TimeoutError:
        raise ProviderError(
            f"The model stream went silent for {IDLE:.0f} s", "server", retryable=True
        ) from None
    except (asyncio.IncompleteReadError, ConnectionError, ssl.SSLError, ValueError):
        raise dropped() from None


def _keepalive(sock) -> None:
    """Turn TCP keepalive on with KEEPALIVE's timing, as far as the platform has the options."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in KEEPALIVE:
            if (option := getattr(socket, name, None)) is not None:
                sock.setsockopt(socket.IPPROTO_TCP, option, value)
    except OSError:
        pass  # a missing option only means a dead path is noticed after IDLE instead


class Response:
    """The status and headers of a response; the body is read with `chunks` or `read`."""

    def __init__(self, reader, writer, head: bytes):
        self.reader, self.writer = reader, writer
        status_line, self.raw_headers = parse_head(head)
        try:
            version, status = status_line.split(" ", 2)[:2]
            if not version.startswith("HTTP/1."):
                raise ValueError
            self.status = int(status)
        except ValueError:
            raise bad_response("a bad HTTP status line") from None
        self.headers = {name.lower(): value for name, value in self.raw_headers}

    async def chunks(self):
        """Yield the body as it arrives: chunked, Content-Length or close-delimited."""
        if "chunked" in self.headers.get("transfer-encoding", "").lower():
            while True:
                line = await _guard(self.reader.readline())
                try:
                    size = int(line.split(b";")[0].strip(), 16)
                except ValueError:
                    raise dropped() from None
                if size == 0:
                    return  # trailers are never used; the connection closes anyway
                async for data in self._exactly(size):
                    yield data
                await _guard(self.reader.readexactly(2))  # the chunk's CRLF
        elif "content-length" in self.headers:
            try:
                length = int(self.headers["content-length"])
            except ValueError:
                raise bad_response("a bad Content-Length") from None
            async for data in self._exactly(length):
                yield data
        else:
            while data := await _guard(self.reader.read(65536)):
                yield data

    async def _exactly(self, size: int):
        while size > 0:
            data = await _guard(self.reader.read(min(size, 65536)))
            if not data:
                raise dropped()
            size -= len(data)
            yield data

    async def read(self, limit: int) -> bytes:
        """The body, or its first `limit` bytes when it is longer."""
        body = bytearray()
        async for data in self.chunks():
            body += data
            if len(body) >= limit:
                break
        return bytes(body[:limit])

    def close(self) -> None:
        self.writer.transport.abort()  # immediate: the exchange is over or cancelled


async def request(url: str, *, headers: dict, body: bytes = b"") -> Response:
    """POST one request and read the response head. The caller must `close()` the response.
    Only HTTPS, or plain HTTP to loopback, is ever spoken."""
    parts = urlsplit(url)
    tls, host = parts.scheme == "https", parts.hostname
    if not host or not (tls or (parts.scheme == "http" and is_loopback(host))):
        raise ProviderError(
            "Refusing a URL that is not https:// (or http:// on loopback)", "invalid"
        )
    try:
        async with asyncio.timeout(CONNECT_TIMEOUT):
            reader, writer = await asyncio.open_connection(
                host,
                parts.port or (443 if tls else 80),
                ssl=ssl.create_default_context() if tls else None,
                limit=HEAD_LIMIT,
            )
    except TimeoutError:
        raise ProviderError(
            f"Could not connect to {host} in time", "timeout", retryable=True
        ) from None
    except ssl.SSLCertVerificationError:
        raise ProviderError(f"TLS certificate check failed for {host}", "invalid") from None
    except OSError as exc:
        raise ProviderError(
            f"Could not connect to {host}: {exc.strerror or exc}", "server", retryable=True
        ) from None
    try:
        if (sock := writer.get_extra_info("socket")) is not None:
            _keepalive(sock)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        lines = [f"POST {path} HTTP/1.1", "Host: " + parts.netloc.rpartition("@")[2]]
        if "user-agent" not in {name.lower() for name in headers}:
            lines.append(f"User-Agent: {USER_AGENT}")
        for name, value in headers.items():
            if name.lower() in _FIXED:
                continue
            if any(c in f"{name}{value}" for c in "\r\n"):
                raise ProviderError("A header contains a line break", "invalid")
            lines.append(f"{name}: {value}")
        lines += [f"Content-Length: {len(body)}", "Accept-Encoding: identity", "Connection: close"]
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
        return Response(reader, writer, await _guard(reader.readuntil(b"\r\n\r\n")))
    except asyncio.LimitOverrunError:
        writer.transport.abort()
        raise bad_response("oversized response headers") from None
    except BaseException:
        writer.transport.abort()
        raise


async def sse(chunks):
    """Yield the JSON events of a server-sent event stream (`[DONE]` and comments skipped).

    An event's `type` comes from its `event:` line when the JSON has none.
    """
    buffer = bytearray()
    async for chunk in chunks:
        start = max(0, len(buffer) - 3)  # a blank line may straddle two chunks
        buffer += chunk
        while match := _BLANK_LINE.search(buffer, start):
            event = _parse_event(bytes(buffer[: match.start()]))
            del buffer[: match.end()]
            start = 0
            if event is not None:
                yield event
        if len(buffer) > EVENT_LIMIT:
            raise bad_response("an oversized stream event")


def _parse_event(block: bytes) -> dict | None:
    name, data = "", []
    for line in block.decode("utf-8", "replace").splitlines():
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "data":
            data.append(value)
        elif field == "event":
            name = value
    text = "\n".join(data)
    if not data or text == "[DONE]":
        return None
    try:
        event = json.loads(text)
    except ValueError:
        raise bad_response("an event that is not JSON") from None
    if not isinstance(event, dict):
        raise bad_response("an event that is not an object")
    if name and "type" not in event:
        event["type"] = name
    return event


@asynccontextmanager
async def stream(url: str, headers: dict, payload: dict, rate=None):
    """POST `payload` as JSON and yield (events, rate_limit) of the 200 event stream.

    `rate(headers)` reads the provider's own quota headers (lower-cased) into the rate_limit
    dict, over the standard retry ones (`retry_info`). Any other status raises its classified
    error.
    """
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    headers = {**headers, "Content-Type": "application/json", "Accept": "text/event-stream"}
    response = await request(url, headers=headers, body=body)
    try:
        rate_limit = retry_info(response.headers) | (rate(response.headers) if rate else {})
        if response.status != 200:
            raise classify(response.status, await response.read(ERROR_LIMIT), rate_limit)
        if not response.headers.get("content-type", "").startswith("text/event-stream"):
            raise bad_response("a response that is not an event stream")
        yield sse(response.chunks()), rate_limit
    finally:
        response.close()


async def post(
    url: str, *, headers: dict, data: bytes, timeout: float, limit=1 << 20, read_errors=True
) -> tuple[int, dict, bytes]:
    """One short POST: (status, headers, body). A body over `limit` is an error. With
    `read_errors=False` the body of a non-200 response is never read (returned empty)."""
    async with asyncio.timeout(timeout):
        response = await request(url, headers=headers, body=data)
        try:
            if response.status != 200 and not read_errors:
                return response.status, response.headers, b""
            body = await response.read(limit + 1)
            if len(body) > limit:
                raise bad_response("an oversized response")
            return response.status, response.headers, body
        finally:
            response.close()
