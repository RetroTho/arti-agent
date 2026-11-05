"""Slash commands. They run at once, even in the middle of a turn, and always reply.

/stop /new /status /help work in any conversation, for that conversation only (where several
people talk to Arti, the sender's own). Everything else is the owner's and works only in their
one-on-one chat: /model and /effort (for that chat) and /restart. Plugins add their own
(api.chat_command, plugin.py), the owner's alone unless they say otherwise. An unknown /word is
an ordinary message.
"""

import asyncio
import inspect
import logging
import time
from datetime import datetime

from arti.config import EFFORTS

log = logging.getLogger(__name__)

NEW_SESSION_TIMEOUT = 15  # seconds for all on_new_session hooks of one /new

HELP = """\
/stop - stop the current task and drop queued messages
/new - start a fresh conversation
/status - model, effort, what I'm doing
/model [provider/]model - show or set the model
/effort level - show or set the reasoning effort (also /reasoning)
/restart - restart Arti"""

HOME_ONLY = {"restart", "model", "effort"}
CORE = HOME_ONLY | {"stop", "new", "status", "help", "reasoning"}  # no plugin may take these


def parse(app, text: str) -> tuple[str, str] | None:
    """("name", "argument") for a known command (the core's or a plugin's), else None."""
    word, _, arg = text.strip().partition(" ")
    name = {"/reasoning": "/effort"}.get(word.lower(), word.lower())
    known = name[1:] in CORE or name[1:] in app.chat_commands
    return (name[1:], arg.strip()) if name[:1] == "/" and known else None


async def run(app, inp, name: str, arg: str) -> str:
    """Run a command from `inp` and return the reply text."""
    command = app.chat_commands.get(name)
    home_only = command.home_only if command else name in HOME_ONLY
    if home_only and app.scope_for(inp.key) != "home":
        return f"/{name} works only in {app.owner_name}'s own chat with {app.agent_name}."
    if command:
        reply = command.func(app, inp, arg)
        return await reply if inspect.isawaitable(reply) else reply
    if name == "restart":
        app.request_restart()
        return "Restarting."
    conv = app.conversation(inp.key)
    if name == "stop":
        await conv.stop()  # app.admit already halted it (see App._dispatch)
        return "Stopped."
    if name == "new":
        await conv.stop(fresh=True)
        if not await new_session_hooks(app, inp.key):
            return "Started a new conversation (some background work is still stopping)."
        return "Started a new conversation."
    if name == "status":
        return status(app, conv)
    provider, model, effort = conv.route
    if name == "model":
        if not arg:
            return f"Model: {provider}/{model}"
        prefix, _, rest = arg.partition("/")
        if rest and prefix in app.provider_specs:
            provider, model = prefix, rest
        elif arg in app.provider_specs:
            provider, model = arg, app.provider_spec(arg).default_model
        else:
            model = arg
        conv.set_route((provider, model, effort))
        return f"Model set to {provider}/{model}."
    if name == "effort":
        if not arg:
            return f"Effort: {effort}"
        if arg not in EFFORTS:
            return "Effort must be one of: " + ", ".join(EFFORTS)
        conv.set_route((provider, model, arg))
        return f"Effort set to {arg}."
    # /help: the core's commands, then the plugins'.
    return "\n".join([HELP, *(f"/{c.name} {c.help}" for c in app.chat_commands.values())])


async def new_session_hooks(app, key: str) -> bool:
    """Run the plugins' on_new_session hooks for `key`, each in a thread; False when they did
    not all finish within NEW_SESSION_TIMEOUT (they keep running; a failure is logged)."""

    async def one(hook):
        try:
            await asyncio.to_thread(hook, app, key)
        except Exception:
            log.exception("on_new_session hook %r failed", hook)

    hooks = [one(hook) for hook in app.new_session_hooks]
    try:
        await asyncio.wait_for(asyncio.gather(*hooks), NEW_SESSION_TIMEOUT)
    except TimeoutError:
        return False
    return True


def status(app, conv) -> str:
    """/status: the model, what the conversation is doing, uptime, errors and quota."""
    provider, model, effort = conv.route
    uptime = int(time.time() - app.started)
    doing = f"Busy: running {conv.running_tool}" if conv.running_tool else "Busy"
    lines = [
        f"Model: {provider}/{model}, effort {effort}",
        doing if conv.busy else "Idle",
        f"Up {uptime // 3600}h {uptime % 3600 // 60}m",
    ]
    if conv.last_error:
        lines.append(f"Last error: {conv.last_error}")
    windows = conv.rate_limit.get("windows", {})
    for name, window in sorted(windows.items()):
        if "used" in window:
            line = f"Usage ({name} window): {window['used']:.0%} used"
            if window.get("resets_at"):
                when = datetime.fromtimestamp(window["resets_at"], app.tz)
                line += f", resets {when:%a %-I:%M %p}"
            lines.append(line)
    if conv.rate_limit.get("status") not in (None, "allowed"):
        lines.append(f"Usage status: {conv.rate_limit['status']}")
    return "\n".join(lines)
