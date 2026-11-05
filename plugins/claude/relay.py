"""The loopback relay between the Claude CLI and the Messages API: one request, then no more.

claude.py points the CLI's ANTHROPIC_BASE_URL here. The relay listens on 127.0.0.1 behind a
random 32-byte path and admits the first POST to <path>/v1/messages that has no Origin header
(so no web page can use it). It rewrites that request for caching (rewrite.py),
forwards it upstream with the CLI's own headers (its login; Arti never reads it), streams the
answer back to the CLI and captures it with an Assembler. The capture is Arti's reply.

Every later request gets a local 400, so the CLI can never spend quota on a second
generation. Nothing of a request or its headers is logged.
"""

from __future__ import annotations

import asyncio
import secrets
import time

from arti.providers import ProviderError, Reply, bad_response, dropped, http
from plugins.anthropic.messages import Assembler

from .rewrite import rewrite

HEAD_TIMEOUT = 10  # for the CLI to send its request head
BODY_TIMEOUT = 60
REQUEST_LIMIT = 64 * 1024 * 1024
# What a second request gets: shaped as an API error, so the CLI gives up instead of retrying.
CONSUMED = (
    b'{"type":"error","error":{"type":"invalid_request_error",'
    b'"message":"ARTI_MODEL_ADMISSION_CONSUMED"}}'
)
# Headers of the CLI's hop to the relay, not of the relay's hop upstream (http.request drops
# host, content-length, connection, transfer-encoding and accept-encoding itself).
_HOP_HEADERS = ("proxy-authorization", "proxy-connection", "keep-alive", "te", "upgrade")
_HOP_RESPONSE = ("connection", "transfer-encoding", "server", "date", "keep-alive")


class Relay:
    """One run's relay. `queried` is the newest user turn as Arti sent it, for the rewrite.

    After the run, claude.py reads what happened: `admitted`, `refused`, `denied`, the
    upstream `status`, `headers` and `error_body`, and the captured `reply` or `failure`.
    """

    def __init__(self, queried: list[dict], upstream: str):
        self.queried = queried
        self.upstream = upstream.rstrip("/")
        self.prefix = "/admit/" + secrets.token_urlsafe(32)
        self.url = ""  # set by start()
        self.admitted = False
        self.active = time.monotonic()  # the last sign of life, for claude.py's watchdog
        self.refused: ProviderError | None = None  # why the CLI's request was not admitted
        self.denied = 0  # requests answered with CONSUMED
        self.status: int | None = None
        self.headers: dict = {}
        self.request_id = ""
        self.restored = ""  # what the rewrite did to the tool results, for the logs
        self.assembler = Assembler()
        self.reply: Reply | None = None
        self.error_body = b""
        self.failure: ProviderError | None = None
        self._server = None
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._serve, "127.0.0.1", 0, limit=http.HEAD_LIMIT
        )
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}{self.prefix}"

    def touch(self) -> None:
        self.active = time.monotonic()

    async def close(self) -> None:
        self._server.close()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _serve(self, reader, writer) -> None:
        self._tasks.add(asyncio.current_task())
        try:
            await self._handle(reader, writer)
        except ProviderError as exc:
            self.failure = self.failure or exc
        except (OSError, ValueError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            if self.admitted and self.reply is None:
                self.failure = self.failure or dropped()
        finally:
            writer.close()
            self._tasks.discard(asyncio.current_task())

    async def _handle(self, reader, writer) -> None:
        async with asyncio.timeout(HEAD_TIMEOUT):
            head = await reader.readuntil(b"\r\n\r\n")
        request_line, headers = http.parse_head(head)
        method, target = (request_line.split(" ") + ["", ""])[:2]
        names = {name.lower(): value for name, value in headers}
        path, _, query = target.partition("?")
        if method != "POST" or path != self.prefix + "/v1/messages" or "origin" in names:
            return await _answer(writer, 404, b"")
        if self.admitted:
            self.denied += 1
            return await _answer(writer, 400, CONSUMED)
        length = int(names.get("content-length", "-1"))
        if not 0 <= length <= REQUEST_LIMIT or "transfer-encoding" in names:
            self.refused = bad_response("an unreadable request from the Claude CLI")
            return await _answer(writer, 400, b"")
        async with asyncio.timeout(BODY_TIMEOUT):
            body = await reader.readexactly(length)
        if self.admitted:  # another request was admitted while this one's body was read
            self.denied += 1
            return await _answer(writer, 400, CONSUMED)
        self.admitted = True
        self.touch()
        body, self.restored = await asyncio.to_thread(rewrite, body, self.queried)
        forward = {name: value for name, value in headers if name.lower() not in _HOP_HEADERS}
        url = self.upstream + "/v1/messages" + (f"?{query}" if query else "")
        response = await http.request(url, headers=forward, body=body)
        try:
            await self._pass_on(response, writer)
        finally:
            response.close()

    async def _pass_on(self, response, writer) -> None:
        """Send the upstream response to the CLI as it arrives, and capture it: the reply of
        a 200 event stream, or the start of an error body (to classify it)."""
        self.status, self.headers = response.status, response.headers
        self.request_id = response.headers.get("request-id", "")[:100]
        head = [f"HTTP/1.1 {response.status} Relayed"]
        head += [f"{k}: {v}" for k, v in response.raw_headers if k.lower() not in _HOP_RESPONSE]
        writer.write(("\r\n".join([*head, "Connection: close"]) + "\r\n\r\n").encode("latin-1"))
        body = self._forward(response, writer)
        if response.status != 200:
            async for chunk in body:
                self.error_body = (self.error_body + chunk)[: http.ERROR_LIMIT]
            return
        async for event in http.sse(body):
            if self.reply is None and self.failure is None:
                try:
                    self.reply = self.assembler.feed(event)
                except ProviderError as exc:  # an upstream `error` event, a malformed event
                    self.failure = exc

    async def _forward(self, response, writer):
        """Yield the response body while writing it to the CLI; each chunk is a sign of life."""
        async for chunk in response.chunks():
            self.touch()
            writer.write(chunk)
            await writer.drain()
            yield chunk


async def _answer(writer, status: int, body: bytes) -> None:
    """A local answer to the CLI (the relay's own refusals)."""
    head = f"HTTP/1.1 {status} Relay\r\nContent-Type: application/json\r\n"
    head += f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    writer.write(head.encode() + body)
    await writer.drain()
