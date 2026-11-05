"""The provider interface: the whole history in, one assistant reply out. Never a tool runner.

A provider is a plugin's (api.provider, plugin.py): the App builds it by name and calls
`Provider.complete`. This module and http.py (the one streaming HTTP client) are what every
provider shares: `Reply`, `ProviderError` and its classification, the API key and deadline
helpers, the standard retry headers, and the replay carrier that lets a provider see its own
signed or encrypted reasoning again. The providers themselves are all plugins'.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from arti import secret

CUT_OFF = ("length", "refusal")  # stop reasons whose tool calls are never run or replayed


@dataclass
class Reply:
    content: list[dict]  # text {text} and tool_use {id, name, input} blocks
    stop_reason: str = "stop"  # stop | tool_use | length | refusal
    usage: dict = field(default_factory=dict)  # input_tokens (cache incl.), output, cached
    native: dict = field(default_factory=dict)  # opaque replay carrier, see carrier()
    rate_limit: dict = field(default_factory=dict)  # retry_after?, status?, windows?

    @property
    def text(self) -> str:
        return "\n\n".join(b["text"] for b in self.content if b["type"] == "text")

    @property
    def tool_calls(self) -> list[dict]:
        return [b for b in self.content if b["type"] == "tool_use"]


class ProviderError(Exception):
    """A classified failure. `code` is one of: auth, rate_limit, context_length, server,
    timeout, bad_response, invalid. Core retries `retryable` ones once, compacts on
    context_length, and shows `message` to the owner otherwise."""

    def __init__(
        self, message, code, *, retryable=False, retry_after=None, rate_limit=None, status=None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after
        self.rate_limit = rate_limit
        self.status = status  # the HTTP status it came with; None for a mid-stream error


class Provider:
    """The interface every route implements."""

    async def complete(self, messages, *, system, tools, model, effort, session_id) -> Reply:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


def api_key(opts: dict, default_env: str) -> str:
    """The key from the environment variable named by `api_key_env` (never from config)."""
    name = opts.get("api_key_env") or default_env
    key = secret.get(name).strip()
    if not key:
        raise ProviderError(f"No API key: set {name} in ~/.arti/.env", "auth")
    return key


@asynccontextmanager
async def deadline(seconds):
    """`providers.<id>.timeout` for one request (None = only the caller's own deadline)."""
    try:
        async with asyncio.timeout(seconds or None):
            yield
    except TimeoutError:
        raise ProviderError("The model request timed out", "timeout", retryable=True) from None


# --- errors -----------------------------------------------------------------------------

_TEXT = {
    "auth": "The model API rejected the credentials",
    "context_length": "The conversation is too long for the model",
    "rate_limit": "The model's usage or rate limit was reached",
    "timeout": "The model request timed out",
    "server": "The model service is unavailable",
    "invalid": "The model API rejected the request",
}


def error_fields(body) -> tuple[str, str]:
    """(type and code, message) of an API error object; only these fields are ever read,
    because the rest of an error body can echo the request."""
    if isinstance(body, (bytes, str)):
        try:
            body = json.loads(body)
        except ValueError:
            return "", ""
    error = body.get("error", body) if isinstance(body, dict) else None
    if isinstance(error, str):
        return error, ""
    if not isinstance(error, dict):
        return "", ""
    kind = " ".join(str(error[k]) for k in ("type", "code") if isinstance(error.get(k), str))
    message = error.get("message")
    return kind.lower(), message if isinstance(message, str) else ""


def classify(status: int | None, body=None, rate_limit: dict | None = None) -> ProviderError:
    """Turn an HTTP status and a structured error into a ProviderError."""
    kind, message = error_fields(body)
    words = f"{kind} {message.lower()}"
    retryable = False
    if status in (401, 403) or "authentication" in kind or "permission" in kind:
        code = "auth"
    elif status == 413 or "context_length" in kind or "prompt is too long" in words:
        code = "context_length"
    elif status == 402 or "billing" in kind or "insufficient_quota" in kind:
        code = "rate_limit"
    elif status == 429 or "rate_limit" in kind or "usage_limit" in kind:
        code = "rate_limit"
        retryable = (rate_limit or {}).get("status") != "rejected"
    elif status in (408, 504) or "timeout" in kind:
        code, retryable = "timeout", True
    elif (
        status == 409
        or (status or 0) >= 500
        or any(word in kind for word in ("overloaded", "api_error", "server_error", "internal"))
    ):  # "server_error" and "internal…" are codes some APIs give mid-stream failures
        code, retryable = "server", True
    else:
        code = "invalid"
    text = _TEXT[code] + (f" (HTTP {status})" if status else "")
    if message:
        text += ": " + message[:300]
    retry_after = (rate_limit or {}).get("retry_after")
    return ProviderError(
        text,
        code,
        retryable=retryable,
        retry_after=retry_after,
        rate_limit=rate_limit,
        status=status,
    )


def dropped() -> ProviderError:
    """The stream ended before the reply was complete: nothing was used, a retry is safe."""
    return ProviderError("The model connection dropped mid-reply", "server", retryable=True)


def bad_response(what: str) -> ProviderError:
    return ProviderError(f"The model sent a malformed reply ({what})", "bad_response")


async def read_reply(events, parser) -> Reply:
    """Feed stream events to `parser` (whose `feed` returns the Reply once it is complete)
    until the reply is complete; a stream that ends first is dropped."""
    async for event in events:
        if (reply := parser.feed(event)) is not None:
            return reply
    raise dropped()


# --- rate limits ------------------------------------------------------------------------


def number(value) -> float | None:
    """A header value as a number >= 0, or None (missing, malformed, negative or NaN)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None  # NaN fails the comparison too


def retry_info(headers: dict) -> dict:
    """The standard part of the quota state, from lower-cased headers: {"retry_after"?} in
    seconds (`retry-after`, or `retry-after-ms`). A provider adds its own quota headers (http.
    stream's `rate`) in the shape /status reads: {"status"?, "windows"?: {name: {"used": 0..1,
    "resets_at"?: unix}}}."""
    retry = number(headers.get("retry-after"))
    if retry is None and (millis := number(headers.get("retry-after-ms"))) is not None:
        retry = millis / 1000
    return {} if retry is None else {"retry_after": retry}


# --- replay carriers and the reply's last checks ----------------------------------------


def _sha(content: list[dict]) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def carrier(provider: str, model: str, content: list[dict], items: list) -> dict:
    """The native output of a reply, bound to its provider, model and visible content."""
    return {"provider": provider, "model": model, "sha": _sha(content), "items": items}


def native_items(message: dict, provider: str, model: str) -> list | None:
    """An assistant message's native items, if they may be replayed to this provider and
    model: only while the visible content is exactly what the provider sent."""
    native = message.get("native") or {}
    if (
        native.get("provider") == provider
        and native.get("model") == model
        and isinstance(native.get("items"), list)
        and native.get("sha") == _sha(message["content"])
    ):
        return native["items"]
    return None


def finish(reply: Reply, provider: str, model: str, items: list) -> Reply:
    """Every route's last step: check the tool calls, hide those of a cut-off reply, give a
    textless refusal or cut-off a notice, and attach the replay carrier. A cut-off reply gets
    no carrier: its native output may hold a call that never ran, or half an answer."""
    cut_off = reply.stop_reason in CUT_OFF
    seen = set()
    content = []
    for block in reply.content:
        if block["type"] == "tool_use":
            if cut_off:
                continue
            if not block.get("id") or block["id"] in seen:
                raise bad_response("a missing or repeated tool call id")
            if not isinstance(block.get("input"), dict):
                raise bad_response("tool arguments that are not an object")
            seen.add(block["id"])
        elif not block.get("text", "").strip():
            continue  # a blank text block would make the history unsendable
        content.append(block)
    if cut_off and not any(b["type"] == "text" for b in content):
        notice = {
            "refusal": "The model declined this request without giving a reason.",
            "length": "The model reached its output limit before it finished answering.",
        }[reply.stop_reason]
        content.append({"type": "text", "text": notice})
    reply.content = content
    if not cut_off:
        reply.native = carrier(provider, model, content, items)
    return reply
