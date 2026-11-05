"""Loopback fakes for the provider tests: a tiny asyncio HTTP server and SSE builders."""

import asyncio
import json


def sse(*events: dict) -> bytes:
    return b"".join(
        f"event: {e.get('type', '')}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    )


def response(status: int, body: bytes = b"", headers: dict | None = None) -> bytes:
    headers = {"Content-Length": str(len(body)), **(headers or {})}
    head = f"HTTP/1.1 {status} X\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    return head.encode() + b"\r\n" + body


def stream_response(body: bytes, status: int = 200) -> bytes:
    return response(status, body, {"Content-Type": "text/event-stream"})


def json_response(status: int, value) -> bytes:
    return response(status, json.dumps(value).encode(), {"Content-Type": "application/json"})


def messages_stream(blocks: list[dict], stop="end_turn", usage=None, stop_details=None) -> bytes:
    """A complete Anthropic Messages event stream for `blocks` (text/tool_use/thinking)."""
    events = [
        {
            "type": "message_start",
            "message": {"role": "assistant", "content": [], "usage": {"input_tokens": 10}},
        }
    ]
    for i, block in enumerate(blocks):
        if block["type"] == "tool_use":
            start = {**block, "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        elif block["type"] == "thinking":
            start = {"type": "thinking", "thinking": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        else:
            start = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"]}]
        events.append({"type": "content_block_start", "index": i, "content_block": start})
        events += [{"type": "content_block_delta", "index": i, "delta": d} for d in deltas]
        events.append({"type": "content_block_stop", "index": i})
    delta = {"stop_reason": stop}
    if stop_details:
        delta["stop_details"] = stop_details
    events.append(
        {
            "type": "message_delta",
            "delta": delta,
            "usage": usage or {"output_tokens": 5, "cache_read_input_tokens": 3},
        }
    )
    events.append({"type": "message_stop"})
    return sse(*events)


class FakeServer:
    """Serve scripted answers on 127.0.0.1. `answers` is a list (one per request) of bytes,
    or of async callables `(request, writer)` for streaming/stalling. Requests are recorded
    as dicts {method, path, headers (lower-cased), body}."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []
        self.closed = asyncio.Event()

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *exc):
        self.server.close()

    async def _serve(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path = lines[0].split(" ")[:2]
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    name, value = line.split(":", 1)
                    headers[name.strip().lower()] = value.strip()
            body = await reader.readexactly(int(headers.get("content-length", 0)))
            request = {"method": method, "path": path, "headers": headers, "body": body}
            self.requests.append(request)
            answer = self.answers.pop(0) if self.answers else response(500)
            if callable(answer):
                await answer(request, writer)
            else:
                writer.write(answer)
                await writer.drain()
            # wait for the client to hang up, so a cancelled client is observable
            await reader.read()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self.closed.set()
            writer.close()
