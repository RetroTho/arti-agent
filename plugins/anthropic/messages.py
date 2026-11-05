"""The Anthropic Messages API: history encoding and stream assembly, plus the API-key route.

The claude-subscription route speaks the same protocol through the Claude CLI, so it reuses
`wire_history` (to replay the history) and `Assembler` (to read the relay's capture).
"""

from __future__ import annotations

import asyncio
import json
import re

from arti.providers import (
    CUT_OFF,
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

API = "https://api.anthropic.com/v1"
STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "tool_use": "tool_use",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "refusal",
}
THINKING = ("thinking", "redacted_thinking")
BREAKPOINT = {"type": "ephemeral"}


def rate_info(headers: dict) -> dict:
    """Quota state for /status from the lower-cased Anthropic headers: the subscription's
    unified 5h and 7d windows (with their resets) and their status, and the API-key routes'
    requests and tokens (their resets are not worth parsing)."""
    found = {}  # window name -> (fraction used, reset time)
    for name in ("5h", "7d"):
        prefix = f"anthropic-ratelimit-unified-{name}-"
        used, reset = headers.get(prefix + "utilization"), headers.get(prefix + "reset")
        found[name] = number(used), number(reset)
    for name in ("requests", "tokens"):
        limit = number(headers.get(f"anthropic-ratelimit-{name}-limit"))
        left = number(headers.get(f"anthropic-ratelimit-{name}-remaining"))
        if limit and left is not None:
            found[name] = 1 - left / limit, None
    info = {}
    if windows := {
        name: {"used": used} | ({"resets_at": reset} if reset else {})
        for name, (used, reset) in found.items()
        if used is not None
    }:
        info["windows"] = windows
    status = headers.get("anthropic-ratelimit-unified-status", "")
    if re.fullmatch(r"[a-z_]{1,32}", status):
        info["status"] = status
    return info


def _blocks(blocks: list[dict], role: str, in_result: bool = False) -> list[dict]:
    """Canonical blocks as API blocks: only the API's own fields, blank text dropped."""
    out = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            if block["text"].strip():
                out.append({"type": "text", "text": block["text"]})
        elif kind == "image" and role == "user":
            source = {"type": "base64", "media_type": block["media_type"], "data": block["data"]}
            out.append({"type": "image", "source": source})
        elif kind == "tool_use" and role == "assistant":
            out.append({k: block[k] for k in ("type", "id", "name", "input")})
        elif kind == "tool_result" and role == "user" and not in_result:
            content = _blocks(block["content"], "user", True)
            out.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["tool_use_id"],
                    "is_error": bool(block.get("is_error")),
                    "content": content or [{"type": "text", "text": "(no output)"}],
                }
            )
        else:
            raise ProviderError(f"Cannot send a {kind!r} block in a {role} message", "invalid")
    return out


def wire_history(messages: list[dict], provider: str, model: str) -> list[dict]:
    """Canonical history as Messages turns.

    An assistant message whose carrier matches is replayed as its native blocks (signed
    thinking included). Adjacent turns of one role merge and tool results go first in a user
    turn, as the API requires.
    """
    turns = []
    for message in messages:
        role = message["role"]
        native = native_items(message, provider, model) if role == "assistant" else None
        if native is not None:
            blocks = [dict(b) for b in native if b["type"] != "text" or b["text"].strip()]
        else:
            blocks = _blocks(message["content"], role)
        if not blocks:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})
    for turn in turns:
        if turn["role"] == "user":
            turn["content"].sort(key=lambda b: b["type"] != "tool_result")
    return turns


class Assembler:
    """Rebuild one streamed Messages response; `feed` returns the Reply at `message_stop`.

    `blocks` keeps the native content (thinking included, tool inputs parsed) for the
    carrier, and `usage` the raw counters. An upstream `error` event raises its class.
    """

    def __init__(self):
        self.started = False
        self.blocks: list[dict] = []
        self.usage: dict = {}
        self.stop_reason = None
        self.stop_details = None
        self._json: dict[int, str] = {}

    def feed(self, event: dict) -> Reply | None:
        try:
            return self._feed(event)
        except (KeyError, TypeError, AttributeError, IndexError):
            raise bad_response("a malformed stream event") from None

    def _feed(self, event: dict) -> Reply | None:
        kind = event["type"]
        if kind == "error":
            raise classify(None, event)
        if kind == "message_start":
            self.started = True
            self.usage.update(event["message"].get("usage") or {})
        elif kind == "content_block_start":
            if event["index"] != len(self.blocks):
                raise bad_response("blocks out of order")
            self.blocks.append(dict(event["content_block"]))
        elif kind == "content_block_delta":
            self._delta(event["index"], event["delta"])
        elif kind == "message_delta":
            delta = event["delta"]
            self.stop_reason = delta.get("stop_reason") or self.stop_reason
            self.stop_details = delta.get("stop_details") or self.stop_details
            self.usage.update(
                {k: v for k, v in (event.get("usage") or {}).items() if v is not None}
            )
        elif kind == "message_stop":
            return self._reply()
        return None  # ping, content_block_stop and anything new

    def _delta(self, index: int, delta: dict) -> None:
        block, kind = self.blocks[index], delta["type"]
        if kind == "input_json_delta":
            self._json[index] = self._json.get(index, "") + delta["partial_json"]
        elif kind == "text_delta":
            block["text"] = block.get("text", "") + delta["text"]
        elif kind == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta["thinking"]
        elif kind == "signature_delta":
            block["signature"] = block.get("signature", "") + delta["signature"]
        elif kind == "citations_delta":
            block.setdefault("citations", []).append(delta["citation"])

    def _reply(self) -> Reply:
        if not self.started or self.stop_reason not in STOP_REASONS:
            raise bad_response(f"stop reason {str(self.stop_reason)[:40]!r}")
        stop = STOP_REASONS[self.stop_reason]
        content = []
        for index, block in enumerate(self.blocks):
            if block["type"] == "text":
                content.append({"type": "text", "text": block.get("text", "")})
            elif block["type"] == "tool_use" and stop not in CUT_OFF:
                raw = self._json.get(index, "")
                try:
                    block["input"] = json.loads(raw) if raw else block.get("input") or {}
                except ValueError:
                    raise bad_response("tool arguments that are not JSON") from None
                content.append({k: block[k] for k in ("type", "id", "name", "input")})
        if stop == "stop" and any(b["type"] == "tool_use" for b in content):
            stop = "tool_use"
        if stop == "refusal" and not any(
            b["type"] == "text" and b["text"].strip() for b in content
        ):
            details = self.stop_details if isinstance(self.stop_details, dict) else {}
            explanation, category = details.get("explanation"), details.get("category")
            if isinstance(category, str) and category.strip():
                category = f"The model declined this request (category: {category[:64]})."
            for text in (explanation, category):
                if isinstance(text, str) and text.strip():
                    content.append({"type": "text", "text": text})
                    break
        read = self.usage.get("cache_read_input_tokens") or 0
        written = self.usage.get("cache_creation_input_tokens") or 0
        usage = {
            "input_tokens": (self.usage.get("input_tokens") or 0) + read + written,
            "output_tokens": self.usage.get("output_tokens") or 0,
            "cached_tokens": read,
        }
        return Reply(content=content, stop_reason=stop, usage=usage)


def mark_breakpoint(turn: dict) -> None:
    """Put a cache breakpoint on the last block of `turn` that may carry one."""
    for block in reversed(turn["content"]):
        if block["type"] not in THINKING:
            block["cache_control"] = dict(BREAKPOINT)
            return


class AnthropicProvider(Provider):
    """The `anthropic` route: an API key from the environment, streamed Messages."""

    def __init__(self, options: dict):
        self.opts = options
        self.url = (self.opts["base_url"] or API).rstrip("/") + "/messages"

    async def complete(self, messages, *, system, tools, model, effort, session_id) -> Reply:
        async with deadline(self.opts["timeout"]):
            turns = await asyncio.to_thread(wire_history, messages, "anthropic", model)
            # Breakpoints: system, the end of the history, and the previous user turn, so a
            # large parallel tool round still finds the cached prefix within the lookback.
            users = [turn for turn in turns if turn["role"] == "user"]
            for turn in [*users[-2:-1], *turns[-1:]]:
                mark_breakpoint(turn)
            payload = {
                "model": model,
                "max_tokens": self.opts["max_output_tokens"] or 64000,
                "system": [{"type": "text", "text": system, "cache_control": BREAKPOINT}],
                "messages": turns,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
                "stream": True,
            }
            if not system:
                del payload["system"]  # the API refuses an empty text block
            if tools:
                payload["tools"] = [
                    {k: tool[k] for k in ("name", "description", "input_schema")} for tool in tools
                ]
            headers = {
                "x-api-key": api_key(self.opts, "ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
            }
            async with http.stream(self.url, headers, payload, rate_info) as (events, rate_limit):
                assembler = Assembler()
                reply = await read_reply(events, assembler)
            reply.rate_limit = rate_limit
            return finish(reply, "anthropic", model, assembler.blocks)
