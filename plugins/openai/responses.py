"""The OpenAI Responses API: the `openai` route (API key) and `openai-codex` (ChatGPT login).

Both send the same payload: the history as Responses input items, `store: false`, and
encrypted reasoning included so it can be replayed (through the carrier) next time.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from arti.providers import (
    Provider,
    ProviderError,
    Reply,
    api_key,
    bad_response,
    classify,
    deadline,
    finish,
    http,
    native_items,
    number,
    read_reply,
)

from . import codex

API = "https://api.openai.com/v1"


def rate_info(headers: dict, now: float | None = None) -> dict:
    """Quota state for /status from the lower-cased headers: the API's requests and tokens
    (their resets are not worth parsing), and the ChatGPT backend's primary and secondary
    windows."""
    now = time.time() if now is None else now
    windows = {}
    for name in ("requests", "tokens"):
        limit = number(headers.get(f"x-ratelimit-limit-{name}"))
        left = number(headers.get(f"x-ratelimit-remaining-{name}"))
        if limit and left is not None:
            windows[name] = {"used": 1 - left / limit}
    for name in ("primary", "secondary"):
        used = number(headers.get(f"x-codex-{name}-used-percent"))
        after = number(headers.get(f"x-codex-{name}-reset-after-seconds"))
        if used is not None:
            windows[name] = {"used": used / 100} | ({"resets_at": now + after} if after else {})
    return {"windows": windows} if windows else {}


def _parts(blocks: list[dict]) -> list[dict]:
    """Text and image blocks as Responses input parts."""
    parts = []
    for block in blocks:
        if block["type"] == "text":
            parts.append({"type": "input_text", "text": block["text"]})
        elif block["type"] == "image":
            url = f"data:{block['media_type']};base64,{block['data']}"
            parts.append({"type": "input_image", "image_url": url})
        else:
            raise ProviderError(f"Cannot send a {block['type']!r} block here", "invalid")
    return parts


def _add(items: list[dict], role: str, parts: list[dict], phase=None) -> None:
    """Append parts to the trailing message item of the same role (and phase), or start one."""
    last = items[-1] if items else {}
    if last.get("type") == "message" and last["role"] == role and last.get("phase") == phase:
        last["content"].extend(parts)
        return
    item = {"type": "message", "role": role, "content": parts}
    if phase:
        item["phase"] = phase
    items.append(item)


def wire_input(messages: list[dict], provider: str, model: str) -> list[dict]:
    """Canonical history as Responses input items; a matching carrier is replayed as is."""
    items: list[dict] = []
    for message in messages:
        role = message["role"]
        native = native_items(message, provider, model) if role == "assistant" else None
        if native is not None:
            items.extend(native)
            continue
        for block in message["content"]:
            kind = block["type"]
            if kind == "tool_result":
                output = _parts(block["content"]) or [{"type": "input_text", "text": "(no output)"}]
                if block.get("is_error"):
                    output.insert(0, {"type": "input_text", "text": "Tool error:"})
                call_id = block["tool_use_id"]
                items.append(dict(type="function_call_output", call_id=call_id, output=output))
            elif kind == "tool_use":
                arguments = json.dumps(block["input"], ensure_ascii=False)
                call = dict(call_id=block["id"], name=block["name"], arguments=arguments)
                items.append({"type": "function_call", **call})
            elif role == "assistant" and kind == "text":
                part = {"type": "output_text", "text": block["text"], "annotations": []}
                _add(items, role, [part], block.get("phase"))
            else:
                _add(items, role, _parts([block]))
    return items


def _replayable(items: list[dict]) -> list[dict]:
    """The output items that can go back with `store: false`: without ids (nothing is stored
    server-side to point at), and reasoning only with its encrypted content."""
    kept = []
    for item in items:
        if item.get("type") == "reasoning" and (
            not item.get("encrypted_content") or str(item.get("id")).startswith("rs_tmp_")
        ):
            continue
        kept.append({k: v for k, v in item.items() if k != "id"})
    return kept


class ResponsesStream:
    """Read one Responses event stream; `feed` returns the Reply at the terminal event."""

    def __init__(self):
        self.done: dict[int, dict] = {}  # output_index -> completed item
        self.items: list[dict] = []  # the replayable output, for the carrier

    def feed(self, event: dict) -> Reply | None:
        try:
            kind = event["type"]
            if kind == "error":
                raise classify(None, event)
            if kind == "response.failed":
                raise classify(None, (event.get("response") or {}).get("error"))
            if kind == "response.output_item.done":
                self.done[event["output_index"]] = event["item"]
            elif kind in ("response.completed", "response.incomplete"):
                return self._reply(event["response"])
            return None
        except (KeyError, TypeError, AttributeError, ValueError):
            raise bad_response("a malformed stream event") from None

    def _reply(self, response: dict) -> Reply:
        # The ChatGPT backend may end with an empty `output` after streaming every item.
        items = response.get("output") or [self.done[i] for i in sorted(self.done)]
        stop = "stop"
        if response.get("status") == "incomplete":
            reason = (response.get("incomplete_details") or {}).get("reason")
            stop = "refusal" if reason == "content_filter" else "length"
        content = []
        for item in items:
            if item["type"] == "message":
                for part in item["content"]:
                    if part["type"] == "refusal":
                        stop = "refusal"
                    text = part.get("text") or part.get("refusal") or ""
                    block = {"type": "text", "text": text}
                    if item.get("phase"):
                        block["phase"] = item["phase"]
                    content.append(block)
            elif item["type"] == "function_call":
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except ValueError:
                    arguments = None  # fine on a cut-off reply, a bad_response otherwise
                block = {"id": item.get("call_id"), "name": item.get("name"), "input": arguments}
                content.append({"type": "tool_use", **block})
        if stop == "stop" and any(b["type"] == "tool_use" for b in content):
            stop = "tool_use"
        usage = response.get("usage") or {}
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
        self.items = _replayable(items)
        usage = {k: usage.get(k) or 0 for k in ("input_tokens", "output_tokens")}
        return Reply(content=content, stop_reason=stop, usage={**usage, "cached_tokens": cached})


class OpenAIProvider(Provider):
    """The `openai` route: OPENAI_API_KEY (or `api_key_env`) and the Responses API."""

    name = "openai"
    default_url = API

    def __init__(self, options: dict):
        self.opts = options
        self.url = (self.opts["base_url"] or self.default_url).rstrip("/") + "/responses"

    async def complete(self, messages, *, system, tools, model, effort, session_id) -> Reply:
        async with deadline(self.opts["timeout"]):
            items = await asyncio.to_thread(wire_input, messages, self.name, model)
            payload = {
                "model": model,
                "instructions": system,
                "input": items,
                "store": False,
                "stream": True,
                "include": ["reasoning.encrypted_content"],
                "prompt_cache_key": hashlib.sha256(session_id.encode()).hexdigest(),
                "reasoning": {"effort": effort, "summary": "auto"},
            }
            if tools:  # omitted when empty, as the ChatGPT backend expects
                payload["tool_choice"], payload["parallel_tool_calls"] = "auto", True
                payload["tools"] = [
                    {
                        "type": "function",
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                        "strict": False,
                    }
                    for tool in tools
                ]
            return await self._send(payload, model, session_id)

    async def _send(self, payload, model, session_id) -> Reply:
        if limit := self.opts["max_output_tokens"]:
            payload["max_output_tokens"] = limit
        headers = {"Authorization": "Bearer " + api_key(self.opts, "OPENAI_API_KEY")}
        return await self._exchange(headers, payload, model)

    async def _exchange(self, headers, payload, model) -> Reply:
        async with http.stream(self.url, headers, payload, rate_info) as (events, rate_limit):
            stream = ResponsesStream()
            reply = await read_reply(events, stream)
        reply.rate_limit = rate_limit
        return finish(reply, self.name, model, stream.items)


class CodexProvider(OpenAIProvider):
    """The `openai-codex` route: the same API on the ChatGPT backend, with Arti's own login."""

    name = "openai-codex"
    default_url = codex.BASE_URL

    def __init__(self, options: dict, auth: codex.CodexAuth):
        super().__init__(options)
        self.auth = auth

    async def _send(self, payload, model, session_id) -> Reply:
        """Send once; if the token is refused (401), refresh exactly that token and send once
        more. No max_output_tokens: the ChatGPT backend rejects that parameter."""
        extra = {"session_id": hashlib.sha256(session_id.encode()).hexdigest()}
        token = await self.auth.token()
        try:
            return await self._exchange({**codex.headers(token), **extra}, payload, model)
        except ProviderError as exc:
            if exc.status != 401:  # a 403 (or a mid-stream auth error) is not about the token
                raise
        token = await self.auth.token(rejected=token)
        return await self._exchange({**codex.headers(token), **extra}, payload, model)
