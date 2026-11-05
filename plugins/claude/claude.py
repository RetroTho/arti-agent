"""The `claude-subscription` route: the official `claude` CLI as one raw model call.

Each request runs a fresh CLI process on the owner's own Claude login (Arti never reads,
copies or refreshes Claude credentials). The CLI gets no built-in tools, no settings, no
session and one turn; an inert MCP stub (mcp_stub.py) only lists Arti's tools. The history
is replayed over stdin (historical turns with `shouldQuery: false`), and the CLI is pointed
at a loopback relay (relay.py) that admits exactly ONE upstream request, rewrites it
for caching (rewrite.py) and captures the streamed reply. That capture is the reply;
the CLI's own output only decides whether the run counts.

One request, and its reply:

  Arti (complete) ──stdin: replayed history──▶ claude CLI (+ mcp_stub.py: tool list only)
                                                  │ POST /v1/messages, the CLI's own login
                                                  ▼
                                 relay.py (127.0.0.1, admits one request)
                                                  │ rewrite.rewrite: Arti's frame and
                                                  │ cache breakpoints put back
                                                  ▼
                                         api.anthropic.com
  Reply: the API's stream ──▶ relay ──▶ CLI (which then exits); the relay's Assembler keeps a
  copy of that stream, and that copy is what `complete` returns. The CLI's stdout result and
  exit status only say whether the run succeeded.

This file builds the CLI's command and environment, drives the CLI and judges its run.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from arti import fsutil
from arti.providers import (
    Provider,
    ProviderError,
    Reply,
    bad_response,
    classify,
    deadline,
    dropped,
    finish,
    retry_info,
)
from plugins.anthropic.messages import rate_info, wire_history
from plugins.anthropic.models import claude_route
from plugins.safety import proc

from .relay import Relay

log = logging.getLogger(__name__)


PROVIDER = "claude-subscription"
PREFIX = "mcp__arti__"  # the CLI names MCP tools mcp__<server>__<tool>; the API allows 64 chars
UPSTREAM = "https://api.anthropic.com"
# The read-idle watchdog: a run that shows no sign of life (a CLI
# output line, the relay's admission, bytes from upstream) for this long is a retryable
# failure. Every sign restarts the count, so a long history replay is never cut short.
START_IDLE = 60  # until the CLI's request reaches the relay: start-up, each replayed turn
IDLE = 180  # after it; the same as http.IDLE
EXIT_GRACE = 5  # for the CLI to exit after its result
LINE_LIMIT = 64 * 1024 * 1024  # one line of the CLI's output
LOGIN_HINT = "The Claude login is missing or expired: run `claude auth login`."
# The CLI's own error codes (on an `assistant` event) -> (code, retryable, text).
CLI_ERRORS = {
    "authentication_failed": ("auth", False, LOGIN_HINT),
    # its own refusal before any request: a retry would be refused the same way
    "rate_limit": ("rate_limit", False, "The Claude subscription's usage limit was reached"),
    "billing_error": ("rate_limit", False, "The Claude subscription has a billing problem"),
    "overloaded": ("server", True, "Claude is overloaded"),
    "server_error": ("server", True, "The Claude service is unavailable"),
}
_ENV_KEEP = ("HOME", "PATH", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "TZ")
_CLI_WINDOWS = {"five_hour": "5h", "seven_day": "7d"}  # renamed as rate_info names them


class ClaudeProvider(Provider):
    def __init__(self, options: dict, data_dir: Path, upstream: str = UPSTREAM):
        self.opts = options
        self.data_dir = Path(data_dir)
        self.upstream = upstream  # injectable for tests; never configurable
        # A crash leaves request folders behind, with the system prompt (private memory) in
        # them. Only one Arti runs on a data dir (arti.lock), so none of them is in use now.
        for stale in (self.data_dir / "claude").glob("tmp*"):
            shutil.rmtree(stale, ignore_errors=True)

    async def complete(self, messages, *, system, tools, model, effort, session_id) -> Reply:
        async with deadline(self.opts["timeout"]):
            manifest, api_tools = inventory(tools)
            offered = {tool["name"] for tool in tools}
            frames = await asyncio.to_thread(wire_history, messages, PROVIDER, model)
            if not frames or frames[-1]["role"] != "user":
                raise ProviderError("The history must end with a user turn", "invalid")
            for frame in frames:  # an un-offered name replays as the model called it
                _prefix_offered(frame["content"], offered)
            binary = find_claude(self.opts["bin"])
            root = fsutil.private_dir(self.data_dir / "claude")
            cwd = fsutil.private_dir(root / "cwd")  # stable: the CLI writes its cwd into the prompt
            with tempfile.TemporaryDirectory(prefix="tmp", dir=root) as tmp:
                tmp = Path(tmp)
                write_request(tmp, system, manifest, api_tools, effort)
                command = cli_command(binary, tmp, model, effort)
                relay = Relay(frames[-1]["content"], self.upstream)
                await relay.start()
                try:
                    reply = await self._run(command, cwd, relay, frames)
                except BaseException as exc:  # cancellation and the deadline too
                    what = getattr(exc, "code", type(exc).__name__)
                    log.warning(
                        "claude failed model=%s error=%s %s", model, what, _diagnosis(relay)
                    )
                    raise
                finally:
                    await relay.close()
            return self._finish(reply, relay, model, offered)

    def _finish(self, reply: Reply, relay: Relay, model: str, offered: set) -> Reply:
        """Arti's tool names in the reply; the native blocks, prefixed, for the carrier."""
        items = copy.deepcopy(relay.assembler.blocks)
        _prefix_offered(items, offered)  # the model sometimes drops the prefix; replay teaches it
        for block in reply.tool_calls:
            block["name"] = block["name"].removeprefix(PREFIX)
        written = relay.assembler.usage.get("cache_creation_input_tokens", 0)
        log.info(
            "claude reply model=%s usage=%s cache_write=%s %s",
            *(model, reply.usage, written, _diagnosis(relay)),
        )
        return finish(reply, PROVIDER, model, items)

    async def _run(self, command: list[str], cwd: Path, relay: Relay, frames: list[dict]) -> Reply:
        """Start the CLI, replay `frames` to it under the watchdog, and kill it afterwards."""
        try:
            cli = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=cli_env(relay.url, self.opts["max_output_tokens"]),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,  # its own process group, with the MCP stub in it
                limit=LINE_LIMIT,
            )
        except OSError as exc:
            raise ProviderError(f"The Claude CLI could not start: {exc}", "invalid") from None
        proc.track(cli.pid)
        talk = asyncio.ensure_future(converse(cli, frames, relay.touch))
        try:
            return outcome(relay, await watch(relay, talk))
        finally:
            talk.cancel()
            try:
                os.killpg(cli.pid, signal.SIGKILL)  # on success too: nothing of it may linger
            except (ProcessLookupError, PermissionError):
                pass  # gone already (macOS answers EPERM when only zombies are left)
            try:
                async with asyncio.timeout(EXIT_GRACE):
                    await cli.wait()
            except TimeoutError:
                log.warning("claude CLI %d did not exit after SIGKILL", cli.pid)
            proc.untrack(cli.pid)


def _prefix_offered(blocks: list[dict], offered: set) -> None:
    """Give tool_use blocks that name an offered tool the CLI's MCP prefix."""
    for block in blocks:
        if block["type"] == "tool_use" and block.get("name") in offered:
            block["name"] = PREFIX + block["name"]


def _diagnosis(relay: Relay) -> str:
    """What the logs need to trace a request, on success and failure alike."""
    return (
        f"status={relay.status} denied={relay.denied} restored={relay.restored or '-'}"
        f" request_id={relay.request_id or '-'}"
    )


# --- the request the CLI is given -------------------------------------------------------


def find_claude(configured: str) -> str:
    """The CLI: `providers.claude-subscription.bin`, else ~/.local/bin/claude, else PATH."""
    for candidate in (configured, "~/.local/bin/claude"):
        if candidate and os.access(os.path.expanduser(candidate), os.X_OK):
            return os.path.expanduser(candidate)
    found = shutil.which("claude")
    if not found:
        raise ProviderError(
            "The Claude CLI was not found: install Claude Code or set "
            "providers.claude-subscription.bin",
            "invalid",
        )
    return found


def inventory(tools: list[dict]) -> tuple[list, list]:
    """(the MCP stub's manifest, the API tool definitions with prefixed names)."""
    manifest, api_tools, seen = [], [], set()
    for tool in tools:
        name = tool["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,53}", name) or name in seen:  # 53 + 11 = 64
            raise ProviderError(f"Tool name {name[:80]!r} is invalid or repeated", "invalid")
        seen.add(name)
        schema = tool["input_schema"]
        manifest.append({"name": name, "description": tool["description"], "inputSchema": schema})
        api_tools.append(
            {"name": PREFIX + name, "description": tool["description"], "input_schema": schema}
        )
    return manifest, api_tools


def write_request(tmp: Path, system: str, manifest: list, api_tools: list, effort: str) -> None:
    """The files cli_command points at. The CLI sends CLAUDE_CODE_EXTRA_BODY's fields with its
    request, which is how the model gets Arti's tools under their real API definitions."""
    extra = {"tools": api_tools, "output_config": {"effort": effort}}
    (tmp / "tools.json").write_text(json.dumps(manifest))
    (tmp / "system.md").write_text(system)
    settings = {"env": {"CLAUDE_CODE_EXTRA_BODY": json.dumps(extra)}}
    (tmp / "settings.json").write_text(json.dumps(settings))


def cli_command(binary: str, tmp: Path, model: str, effort: str) -> list[str]:
    """One raw model turn: no built-in tools, settings, session, slash commands or loop."""
    stub = [str(Path(__file__).with_name("mcp_stub.py")), str(tmp / "tools.json")]
    mcp = {"mcpServers": {"arti": {"command": sys.executable, "args": ["-I", *stub]}}}
    # One flag per line, its value beside it: kept out of the formatter, which would split them.
    return [
        binary, "-p",
        "--model", claude_route(model),
        "--effort", effort,
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",  # reveals a wrongly queried historical turn at once
        "--tools", "",
        "--system-prompt-file", str(tmp / "system.md"),
        "--settings", str(tmp / "settings.json"),
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--max-turns", "1",
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
        "--mcp-config", json.dumps(mcp),
    ]  # fmt: skip


def cli_env(relay_url: str, max_tokens: int | None = None) -> dict:
    """A whitelist of OS essentials plus fixed switches. Nothing else is inherited, so an
    ANTHROPIC_API_KEY, base URL, Bedrock/Vertex switch or proxy cannot change the backend."""
    env = {k: v for k, v in os.environ.items() if k in _ENV_KEEP or k.startswith("LC_")}
    env.update(
        ANTHROPIC_BASE_URL=relay_url,
        DISABLE_AUTOUPDATER="1",
        DISABLE_FEEDBACK_COMMAND="1",
        DISABLE_AUTO_COMPACT="1",
        DISABLE_COMPACT="1",
        ENABLE_TOOL_SEARCH="false",
        CLAUDE_CODE_MAX_RETRIES="0",  # the relay admits one request; retries would be denied
        CLAUDE_CODE_TOTAL_TOKENS_REMINDER="off",  # its replayed reminder would break caching
        DISABLE_TELEMETRY="1",
        DISABLE_ERROR_REPORTING="1",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        CLAUDE_CODE_DISABLE_LEGACY_MODEL_REMAP="1",  # the model asked for is the model used
        NO_PROXY="127.0.0.1,localhost",
    )
    if max_tokens:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
    return env


# --- driving the CLI ---------------------------------------------------------------------


@dataclass
class Output:
    """What the CLI printed about the queried turn."""

    results: list[dict] = field(default_factory=list)
    rate_limit: dict = field(default_factory=dict)
    error: str = ""  # the CLI's error code from an `assistant` event
    returncode: int | None = None  # None: still running EXIT_GRACE s after its result


async def converse(cli, frames: list[dict], touch) -> Output:
    """Replay the history over stdin and collect what the CLI says about the last turn."""
    out = Output()
    try:
        for index, frame in enumerate(frames):
            historical = frame["role"] == "user" and index < len(frames) - 1
            line = {"type": frame["role"], "message": frame}
            if historical:
                line["shouldQuery"] = False
            cli.stdin.write(json.dumps(line, ensure_ascii=False).encode() + b"\n")
            await cli.stdin.drain()
            if historical and not await _acknowledged(cli, out, touch):
                break
        cli.stdin.close()
    except (BrokenPipeError, ConnectionResetError):
        pass  # the CLI exited; what it printed says why
    wait = None  # no limit of its own until the result: the watchdog covers that time
    while (event := await _event(cli, touch, wait)) is not None:
        kind = event.get("type")
        if kind == "result":
            out.results.append(event)
            wait = EXIT_GRACE
        elif kind == "rate_limit_event":
            out.rate_limit = _cli_rate(event.get("rate_limit_info"))
        elif kind == "assistant" and event.get("error"):
            out.error = str(event["error"])[:64]
    try:
        async with asyncio.timeout(EXIT_GRACE):
            out.returncode = await cli.wait()
    except TimeoutError:
        pass
    return out


async def _acknowledged(cli, out: Output, touch) -> bool:
    """Wait for the CLI to accept a historical turn WITHOUT a model call (num_turns 0)."""
    while (event := await _event(cli, touch)) is not None:
        kind = event.get("type")
        if kind == "assistant" and event.get("error"):
            out.error = str(event["error"])[:64]  # e.g. logged out: it failed before any request
            return False
        if kind in ("assistant", "stream_event"):
            raise bad_response("the Claude CLI queried a historical turn")
        if kind == "result":
            if event.get("is_error"):
                out.results.append(event)
                return False
            if event.get("num_turns") != 0:
                raise bad_response("the Claude CLI queried a historical turn")
            return True
    return False


async def _event(cli, touch, timeout=None) -> dict | None:
    """The CLI's next stream-json event; None at EOF (or when `timeout` passes). Each line is
    a sign of life for the watchdog (`touch`)."""
    try:
        async with asyncio.timeout(timeout):
            line = await cli.stdout.readline()
    except TimeoutError:
        return None
    except ValueError:
        raise bad_response("an oversized CLI output line") from None
    if not line:
        return None
    touch()
    try:
        event = json.loads(line)
    except ValueError:
        raise bad_response(f"CLI output that is not JSON: {line[:200]!r}") from None
    if not isinstance(event, dict):
        raise bad_response("CLI output that is not an object")
    return event


async def watch(relay: Relay, talk: asyncio.Future) -> Output:
    """The conversation's Output, unless the run shows no sign of life for START_IDLE
    seconds (before admission) or IDLE seconds (after it): then a retryable failure."""
    while not talk.done():
        limit = IDLE if relay.admitted else START_IDLE
        left = relay.active + limit - time.monotonic()
        if left <= 0:
            raise ProviderError(
                f"The Claude CLI made no progress for {limit} seconds", "timeout", retryable=True
            )
        await asyncio.wait({talk}, timeout=min(left, 1))  # look again each second
    return talk.result()


# --- judging the run ---------------------------------------------------------------------


def outcome(relay: Relay, out: Output) -> Reply:
    """The captured reply, or why this run does not count as one."""
    if not relay.admitted:
        if relay.refused:
            raise relay.refused
        if out.error or any(r.get("is_error") for r in out.results):
            raise _cli_failure(out)
        raise bad_response("the Claude CLI made no model request")
    rate_limit = merge_rates(retry_info(relay.headers) | rate_info(relay.headers), out.rate_limit)
    if relay.status is not None and relay.status != 200:
        error = classify(relay.status, relay.error_body, rate_limit)
        if error.code == "auth":
            error = ProviderError(LOGIN_HINT, "auth", rate_limit=rate_limit)
        raise error
    if relay.reply is None:
        raise relay.failure or dropped()  # a truncated stream is never a reply
    if len(out.results) != 1:
        raise bad_response(f"{len(out.results)} CLI results instead of one")
    reply, final = relay.reply, out.results[0]
    # The reply is good even though the CLI reports an error in three cases: --max-turns 1
    # makes it exit 1 with error_max_turns when the model calls tools, and it also reports
    # one for a refusal and for a retry the relay denied.
    hit_max_turns = (
        final.get("subtype") == "error_max_turns"
        and reply.tool_calls
        and out.returncode in (1, None)
    )
    if not (hit_max_turns or relay.denied or reply.stop_reason == "refusal"):
        failed = out.returncode not in (0, None) or final.get("is_error") or out.error
        if failed or final.get("subtype") != "success":
            raise _cli_failure(out)
    reply.rate_limit = rate_limit
    return reply


def _cli_failure(out: Output) -> ProviderError:
    if out.error in CLI_ERRORS:
        code, retryable, text = CLI_ERRORS[out.error]
        return ProviderError(text, code, retryable=retryable, rate_limit=out.rate_limit)
    errors = [e for r in out.results for e in (r.get("errors") or []) if isinstance(e, str)]
    detail = "; ".join(errors)[:300] or (out.results[0].get("subtype", "") if out.results else "")
    return ProviderError(f"The Claude CLI failed: {detail or 'no result'}", "bad_response")


def _cli_rate(info) -> dict:
    """Quota state from a `rate_limit_event`'s `rate_limit_info` (its `unifiedWindows`), in
    rate_info's shape."""
    info = info if isinstance(info, dict) else {}
    result, windows = {}, {}
    if re.fullmatch(r"[a-z_]{1,32}", str(info.get("status"))):
        result["status"] = info["status"]
    unified = info.get("unifiedWindows")
    for key, window in (unified if isinstance(unified, dict) else {}).items():
        if isinstance(window, dict) and key in _CLI_WINDOWS:
            fields = (("utilization", "used"), ("resetsAt", "resets_at"))
            windows[_CLI_WINDOWS[key]] = {
                name: window[k] for k, name in fields if isinstance(window.get(k), (int, float))
            }
    if windows:
        result["windows"] = windows
    return result


def merge_rates(headers: dict, cli: dict) -> dict:
    """The relay's response headers and the CLI's event together; the headers win per window."""
    merged = {**cli, **headers}
    windows = {**cli.get("windows", {}), **headers.get("windows", {})}
    return {**merged, "windows": windows} if windows else merged
