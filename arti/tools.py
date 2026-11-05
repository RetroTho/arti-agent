"""Tools: what a tool handler sees, and running one tool call.

The turn loop (turn.py) asks `tools_for` which tools to offer the model, and hands each tool call
the model makes to `run_tool`, which always returns a tool_result block for history.

Plugins register the tools (plugin.py). A tool call is checked in this order, and nothing has an
effect before the checks pass: the tool was offered, the arguments match its schema, the gate (a
plugin's, api.gate; with none, every call runs unreviewed) lets it run. The handler then runs in
its own thread with a deadline, which starts only once the gate has passed (the gate bounds its
own wait, for the owner's answer say); its result is redacted, capped and stored. A hung handler
thread is abandoned, never waited on forever.
"""

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from arti import media, schema, secret
from arti.fsutil import private_dir

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 20_000  # per text block in history; the full text is spilled to a file
MAX_ERROR_CHARS = 2_000
CANCEL_GRACE = 3  # seconds a timed-out or stopped handler gets to notice its cancel event
POLL = 0.2


class ToolError(Exception):
    """The call was refused or failed and nothing happened; the model may retry."""


class UnknownOutcome(Exception):
    """The call may have had an effect; the model is told verbatim and must check first."""


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    handler: Callable[[dict, "ToolContext"], Any]
    review: str | None | Callable[[dict], str | None]  # review kind for the gate, or fn(args)
    timeout: float = 300  # capped by agent.tool_timeout
    interactive: bool = False  # needs a chat: hidden in headless runs
    guest: bool = False  # also offered to anyone else's conversation (trusted False)
    available: Callable[[Any], bool] | None = None  # hidden when this returns False


@dataclass
class PromptInfo:
    """What a plugin's prompt section sees."""

    app: Any
    scope: str
    context_dir: Path
    key: str  # the conversation key (the channel's); "" in a headless run
    headless: bool
    trusted: bool  # the owner's conversation; False: someone else's (sees no private context)


def tools_for(conv) -> list[Tool]:
    """The tools this conversation may use right now."""

    def allowed(tool: Tool) -> bool:
        if conv.allowed is not None and tool.name not in conv.allowed:
            return False
        if tool.name in conv.excluded or (tool.interactive and conv.headless):
            return False
        if not (tool.guest or conv.trusted):
            return False
        try:
            return tool.available is None or bool(tool.available(conv.app))
        except Exception:
            log.exception("availability check of %s failed; hidden", tool.name)
            return False

    return [tool for tool in conv.app.tools.values() if allowed(tool)]


def specs(offered: list[Tool]) -> list[dict]:
    """The tool list as providers take it."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.schema} for t in offered
    ]


# --- the handler's view ------------------------------------------------------------------------


class ToolContext:
    """What a handler sees. It runs in its own thread; every effect checks `cancel` first."""

    def __init__(self, conv, call_id: str, cancel: threading.Event):
        self.conv, self.app, self.call_id, self.cancel = conv, conv.app, call_id, cancel
        self.key, self.scope, self.context_dir = conv.key, conv.scope, conv.context_dir
        self.state, self.headless = conv.state, conv.headless
        self.trusted = conv.trusted  # the owner's own conversation (Conversation.trusted)
        self._inputs_at_start = conv.person_inputs

    @property
    def unattended(self) -> bool:
        """Nobody can be asked: a run with no chat (a plugin's background run, --once).

        Exactly the headless runs: a Headless conversation always has the key "", and a Chat
        always has its chat's (app.conversation is only called with a real one)."""
        return self.headless

    def cache_dir(self, name: str) -> Path:
        """A private cache folder of this context: the home one is <data>/cache/<name>, a
        group's is contexts/<scope>/cache/<name>, so a group never sees another's files."""
        return private_dir(self.context_dir / "cache" / name)

    def check(self) -> None:
        if self.cancel.is_set():
            raise ToolError("cancelled")

    def wait(
        self,
        timeout: float,
        until: threading.Event | None = None,
        on_message: bool = True,
        watch=None,
    ) -> str:
        """Block until `until` is set, something new arrives from a person (`on_message`), the
        call is cancelled, or timeout.

        "Something new" is input to this conversation since the call began, or, given another
        conversation to `watch`, a message (not a reaction: a 👍 on the question is no answer) to
        that one since this wait began (an approval question watches the owner's home chat for a
        reply in words).

        Returns "done" | "message" | "cancelled" | "timeout".
        """

        def count() -> int:
            return self.conv.person_inputs if watch is None else watch.messages_in

        start = self._inputs_at_start if watch is None else count()
        deadline = time.monotonic() + timeout
        while True:
            if self.cancel.is_set():
                return "cancelled"
            if until is not None and until.is_set():
                return "done"
            if on_message and count() != start:
                return "message"
            if time.monotonic() >= deadline:
                return "timeout"
            self.cancel.wait(min(deadline - time.monotonic(), POLL))

    def call(self, coro, timeout: float = 60):
        """Run a coroutine on the App's loop and wait for it; a cancel stops waiting."""
        if self.cancel.is_set():
            coro.close()
            raise ToolError("cancelled")
        future = asyncio.run_coroutine_threadsafe(coro, self.app.loop)
        deadline = time.monotonic() + timeout
        while True:
            try:
                return future.result(POLL)
            except TimeoutError:
                if self.cancel.is_set() or time.monotonic() > deadline:
                    future.cancel()
                    raise ToolError(
                        "cancelled" if self.cancel.is_set() else f"timed out after {timeout:.0f}s"
                    ) from None

    def notify(self, text: str, at: float | None = None) -> None:
        """Queue a notice into this conversation (a background job finished), or deliver it
        `at` a later unix time (a wait)."""
        if self.headless:
            raise ToolError("there is no chat to notify in a background run")
        # Not `self.call`: a background job notifies after its call has returned (and maybe been
        # cancelled), and a timeout must not read as "nothing happened" (ToolError).
        coro = self.app.notify(self.key, text, at=at)
        future = asyncio.run_coroutine_threadsafe(coro, self.app.loop)
        try:
            future.result(30)
        except TimeoutError:
            future.cancel()
            raise

    def mark_answered(self) -> None:
        """A reaction counts as answering the latest message."""
        self.conv.mark_answered()

    def run_headless(self, prompt: str, **options) -> str:
        """Run a child agent in this context; its tools are a subset of this conversation's."""
        mine = set(self.conv.offered)
        wanted = options.pop("tools", None)
        options["tools"] = sorted(mine & set(wanted) if wanted is not None else mine)
        options.setdefault("scope", self.scope)
        options.setdefault("trusted", self.trusted)
        timeout = options.get("timeout") or self.app.config["agent"]["tool_timeout"]
        coro = self.app.run_headless(prompt, **options)
        return self.call(coro, timeout=timeout + 30)


# --- running a call ------------------------------------------------------------------------------


def _result(call_id: str, content: list[dict], is_error: bool = False) -> dict:
    return {"type": "tool_result", "tool_use_id": call_id, "is_error": is_error, "content": content}


def error_result(call_id: str, text: str) -> dict:
    # Redacted before it is shortened, so a cut can never leave half a secret behind.
    text = secret.redact(text)[:MAX_ERROR_CHARS]
    return _result(call_id, [{"type": "text", "text": text}], is_error=True)


def unknown_result(call_id: str, text: str) -> dict:
    """A result whose effect may or may not have happened; kept verbatim through compaction."""
    result = error_result(call_id, text)
    result["content"][0]["unknown"] = True
    return result


async def run_tool(conv, call: dict) -> dict:
    """Run one tool call and return its tool_result block (never raises, except on cancel).

    The gate and the handler run in a worker thread, which settles two futures on the loop, so
    the tool's deadline starts only once the gate has passed (the owner may take minutes):

        worker thread                      here, on the loop
        ─────────────                      ─────────────────
        app.gate (if any)                  wait for `approved` or `future` (no deadline)
          ├─ refused ─▶ `future` = refusal   ─▶ return it: nothing ran
          └─ passed ──▶ `approved` set       ─▶ wait for `future` (the tool's deadline)
             handler
             └────────▶ `future` = result    ─▶ return it
                                                at the deadline: cancel, a grace, "unknown"
    """
    app, call_id = conv.app, call["id"]
    tool = conv.offered.get(call["name"])
    if tool is None:
        return error_result(call_id, f"Error: unknown tool {call['name']!r}")
    args = call.get("input")
    try:
        schema.validate(args, tool.schema)
    except ValueError as error:
        return error_result(call_id, f"Error: invalid arguments: {error}")

    ctx = ToolContext(conv, call_id, threading.Event())
    approved = app.loop.create_future()  # set when the gate let the call through
    future = app.loop.create_future()  # the tool_result

    # The worker thread cannot touch asyncio futures directly: it hands results to the loop.
    def settle(target, value) -> None:
        try:
            app.loop.call_soon_threadsafe(lambda: target.done() or target.set_result(value))
        except RuntimeError:  # the loop is closed: Arti has shut down
            pass

    def worker() -> None:
        settle(future, _call_handler(ctx, tool, args, lambda: settle(approved, True)))

    timeout = min(tool.timeout, app.config["agent"]["tool_timeout"])
    threading.Thread(target=worker, name=f"tool-{tool.name}", daemon=True).start()
    try:
        # The gate's wait has no tool deadline: the gate's own timeout and the cancel bound it.
        await asyncio.wait({approved, future}, return_when=asyncio.FIRST_COMPLETED)
        return await asyncio.wait_for(asyncio.shield(future), timeout)
    except TimeoutError:
        ctx.cancel.set()
        await asyncio.wait({future}, timeout=CANCEL_GRACE)
        if future.done():  # it finished while it was being stopped: its real result
            return future.result()
        log.warning("tool %s timed out after %gs", tool.name, timeout)
        return unknown_result(call_id, f"[Arti: timed out after {timeout:g}s; outcome unknown]")
    except asyncio.CancelledError:
        ctx.cancel.set()
        raise
    finally:
        conv.save_state()


def _call_handler(ctx: ToolContext, tool: Tool, args: dict, on_approved: Callable) -> dict:
    """In the worker thread: the gate, the handler, then the result conversion."""
    gate = ctx.app.gate
    try:
        refusal = gate(ctx, tool, args) if gate else None
    except BaseException as error:  # a bug in the gate: nothing has run
        log.exception("the gate's review of %s raised", tool.name)
        return error_result(ctx.call_id, f"Error: {type(error).__name__}: {error}")
    if refusal:
        return error_result(ctx.call_id, refusal)
    on_approved()
    try:
        return _result(ctx.call_id, _blocks(ctx, tool.handler(args, ctx)))
    except ToolError as error:
        return error_result(ctx.call_id, f"Error: {error}")
    except UnknownOutcome as error:
        return unknown_result(ctx.call_id, f"[Arti: outcome unknown] {error}")
    except BaseException as error:  # a bug (SystemExit included): never crash the host
        # The handler may have done part of its work before it raised.
        log.exception("tool %s raised", tool.name)
        return unknown_result(
            ctx.call_id,
            f"[Arti: outcome unknown] failed with {type(error).__name__}: {error}; "
            "it may have partly run, so check before retrying",
        )


def _blocks(ctx: ToolContext, value) -> list[dict]:
    """A handler's return value as redacted, capped content blocks: text, or images by path."""
    if value is None:
        value = "(done)"
    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, indent=1)
    blocks = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, dict) and item.get("type") == "image":
            # Only a file path: history never holds image bytes. The handler has run by now,
            # so a failure here does not mean nothing happened.
            try:
                blocks.append(media.image_block(Path(item["path"]), ctx.context_dir))
            except (KeyError, OSError, ValueError) as error:
                raise UnknownOutcome(
                    f"the tool ran, but its result image could not be loaded: {error}"
                ) from None
            continue
        if isinstance(item, dict) and item.get("type") == "text":
            item = item["text"]
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        blocks.append({"type": "text", "text": _cap(ctx, secret.redact(text), len(blocks))})
    return blocks or [{"type": "text", "text": "(no output)"}]


def _cap(ctx: ToolContext, text: str, index: int) -> str:
    """Keep the head and tail of a huge text; the whole (redacted) text goes to a file."""
    if len(text) <= MAX_RESULT_CHARS:
        return text
    # The call id comes from the model: only safe characters of it go into a file name.
    name = re.sub(r"[^A-Za-z0-9_-]", "", ctx.call_id)[:64] or "call"
    path = ctx.cache_dir("tool-output") / f"{name}-{index}.txt"
    path.write_text(text, encoding="utf-8")
    half = MAX_RESULT_CHARS // 2
    return (
        f"{text[:half]}\n\n[Arti: output cut to {MAX_RESULT_CHARS} of {len(text)} characters; "
        f"the full text is in {path} (use read_file)]\n\n{text[-half:]}"
    )
