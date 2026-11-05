"""Test helpers: a scripted model, a channel that records sends, and a real App on a temp dir.

app = await make_app(tmp, plugins=[register], replies=[text_reply("hi")])
sent = await run_message(app, "hello")

The App gets the four provider names (`fake_providers`, with their real model names), each
answered by the one FakeProvider, and a RecordingChannel whose home is HOME and owner OWNER. It
has no gate and no redactor unless the safety plugin is among `plugins` (`safety.register`).
"""

import asyncio
import copy
import hashlib
import itertools
import logging
import time
from pathlib import Path
from unittest import mock

from arti import config, lock
from arti.app import App
from arti.channel import Channel
from arti.providers import ProviderError, Reply
from arti.store import Input
from plugins.safety import proc

HOME = "home"
OWNER = "U1"
GROUP = "chat0002"  # a group with the owner and U2
GUEST = "U2"


def lane(person: str, chat: str = GROUP) -> str:
    """The key of `person`'s conversation in a group chat, as RecordingChannel spells it (like
    the link's keys, arti/link.py)."""
    return f"{chat}|{person}"


def scope(chat: str = GROUP) -> str:
    """The place of a group chat's conversations, as RecordingChannel names it."""
    return "chat-" + hashlib.sha256(chat.encode()).hexdigest()[:16]


_ids = itertools.count(1)
# Expected failures in tests log tracebacks; keep the test output readable.
for _name in ("arti", "plugins"):
    logging.getLogger(_name).addHandler(logging.NullHandler())


def text_reply(text: str) -> Reply:
    return Reply(content=[{"type": "text", "text": text}], stop_reason="stop")


def tool_reply(name: str, args: dict | None = None, text: str = "", call_id: str = "") -> Reply:
    """A reply that calls one tool (with optional interim text)."""
    content = [{"type": "text", "text": text}] if text else []
    content.append(
        {
            "type": "tool_use",
            "id": call_id or f"call{next(_ids)}",
            "name": name,
            "input": args or {},
        }
    )
    return Reply(content=content, stop_reason="tool_use")


class FakeProvider:
    """Returns scripted replies in order. A script item may be a Reply, an exception to raise,
    or a function(messages) -> Reply. Every request is recorded in `requests`."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.requests: list[dict] = []

    async def complete(self, messages, *, system, tools, model, effort, session_id):
        self.requests.append(
            {
                "messages": copy.deepcopy(messages),
                "system": system,
                "tools": [t["name"] for t in tools],
                "model": model,
                "effort": effort,
            }
        )
        if not self.replies:
            raise ProviderError("no scripted reply left", "invalid")
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item(messages) if callable(item) else item

    async def aclose(self):
        pass


def fake_providers(api):
    """The provider names the real plugins register, every one answered by `app.fake`."""
    claude = dict(
        default_model="claude-opus-5-5",
        judge_model="claude-haiku-5-5",
        summary_model="claude-sonnet-5-5",
        context_window=lambda model: 200_000,
    )
    gpt = dict(
        default_model="gpt-5.4",
        judge_model="gpt-5.4-mini",
        summary_model="gpt-5.4",
        context_window=lambda model: 128_000,
    )
    for name, models in (
        ("claude-subscription", claude),
        ("anthropic", claude),
        ("openai", gpt),
        ("openai-codex", gpt),
    ):
        api.provider(name, lambda app, options: app.fake, **models)


class RecordingChannel(Channel):
    """Records sends; `statuses` (if set) gives the status of each next send.

    Its keys are HOME, or `lane(person, chat)`: one person's conversation in a group chat, whose
    place is `scope(chat)` and is named by `places`."""

    def __init__(self, home: str = HOME, owner: str = OWNER):
        self.home, self.owner = home, owner
        self.sent: list[tuple[str, str]] = []
        self.files: list[tuple[str, str]] = []
        self.parts: list[tuple[str, list]] = []  # every send as the channel got it
        self.statuses: list[str] = []
        self.places: dict[str, str] = {}  # scope -> name

    def scope_of(self, key):
        chat = key.partition("|")[0]
        return "home" if chat and chat == self.home else scope(chat)

    def person_of(self, key):
        if key and key == self.home:
            return ""
        return key.partition("|")[2] or None

    def place_name(self, scope):
        return "" if scope == "home" else self.places.get(scope, "unknown group")

    async def send(self, key, parts):
        """Text parts are recorded together in `sent`, file parts each in `files`."""
        self.parts.append((key, list(parts)))
        if text := "\n\n".join(part.value for part in parts if part.kind == "text"):
            self.sent.append((key, text))
        self.files += [(key, part.value) for part in parts if part.kind == "file"]
        return self.statuses.pop(0) if self.statuses else "sent"

    def prompt_notes(self, key):
        return "# platform test notes"

    def texts(self, key: str = HOME) -> list[str]:
        return [text for k, text in self.sent if k == key]


def make_config(**agent) -> dict:
    cfg = copy.deepcopy(config.DEFAULTS)
    cfg["providers"] = {}
    # No hang watchdog unless a test asks for one (hang_timeout=...).
    defaults = dict(provider="anthropic", model="fake-model", user_name="Owner")
    cfg["agent"].update(defaults, context_window=200_000, hang_timeout=0)
    cfg["agent"].update(agent)
    return cfg


async def make_app(tmpdir, plugins=(), replies=(), channel=None, cfg=None, start=True) -> App:
    """A real App on `tmpdir` with a FakeProvider for every provider name."""
    # Debug-mode loops report every step over 0.1 s ("Executing <Task ...> took 0.1 seconds");
    # slow CI steps are not bugs, so keep the test output quiet.
    asyncio.get_running_loop().slow_callback_duration = 10
    cfg = cfg or make_config()
    plugins = [fake_providers, *plugins]
    app = App(cfg, Path(tmpdir), channel or RecordingChannel(), plugins=plugins)
    app.fake = FakeProvider(replies)
    app._providers[cfg["agent"]["provider"]] = app.fake  # made at once: shutdown closes it
    if start:
        # The instance lock and the safety plugin's records of child processes are process-wide
        # state; tests must not move them to their temp dirs.
        with mock.patch.object(lock, "acquire"), mock.patch.object(proc, "keep_records"):
            await app.start()
    return app


def message(
    text: str, key: str = HOME, sender: str = OWNER, kind: str = "message", **extra
) -> Input:
    return Input(
        event_id=f"test:{next(_ids)}",
        key=key,
        kind=kind,
        text=text,
        sender=sender,
        ts=time.time(),
        **extra,
    )


async def settle(app, timeout: float = 5) -> None:
    """Wait until no conversation is running a turn and background tasks are done."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        busy = [c for c in app.conversations.values() if c.busy]
        # The daily housekeeping and the heartbeat never finish; they are named in App.start.
        forever = {"housekeeping", "heartbeat"}
        others = [t for t in app._tasks if not t.done() and t.get_name() not in forever]
        if not busy and not others:
            return
    raise TimeoutError("the app did not settle")


async def run_message(app, text: str, key: str = HOME, sender: str = OWNER) -> list[str]:
    """Admit one message, wait for the turn to finish, and return the texts sent to `key`."""
    before = len(app.channel.texts(key))
    await app.admit([message(text, key=key, sender=sender)])
    await settle(app)
    return app.channel.texts(key)[before:]
