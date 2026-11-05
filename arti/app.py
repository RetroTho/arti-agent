"""The App: the one object that wires the store, the channel, providers and plugins together.

The App is one user's agent, running in that user's sandbox; its channel is the link to the host
(link.py), which owns the chat platform. Everything optional is a plugin (plugin.py): the tools,
the model providers and their settings. The App only knows them by name, from what the plugins
registered: `provider_specs`, `commands`, `tools`, `services` and the hooks.

How a message flows through it:

  channel ──▶ `admit` ──▶ inbox (SQLite) ──▶ its `Chat` (chat.py) ──▶ `turn.run_turn` ──▶ `send`

`admit` records inputs in the inbox before anything acts on them, then runs a slash command at
once or hands the input to its conversation's `Chat` (the owner's one-on-one, or one person's
conversation in another place the channel knows). `send` writes an outbox row before the channel
is called, so a send is never repeated after a crash.

The App also runs headless turns (plugins' background runs, --once), recovers after a crash,
does daily housekeeping, keeps the hang watchdog fed (watchdog.py) and shuts everything down
within bounded time.
"""

import asyncio
import inspect
import logging
import os
import time
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo

from arti import commands, config, lock, media, plugin, reply, secret, tools, watchdog
from arti.channel import Channel, Part
from arti.chat import Chat, Headless
from arti.config import code_dir
from arti.fsutil import delete_older_than, private_dir
from arti.link import Link
from arti.store import Input, Store
from arti.turn import open_calls, run_turn

log = logging.getLogger(__name__)

SEND_TIMEOUT = 300  # seconds for the channel (the host, through the link) to take one send
MAX_HEADLESS = 4  # headless runs at once; more wait their turn
HOUSEKEEPING_EVERY = 24 * 3600  # seconds of wall-clock time between housekeeping runs
WAKE_EVERY = 3600  # seconds between checks whether housekeeping is due
INBOX_DAYS = 7
IDLE_UNLOAD = 3600  # seconds a conversation must be quiet (no input, no turn) to be unloaded
RESTARTED = "[Arti: Arti restarted while this ran; its outcome is unknown]"
RESTART_NOTICE = (
    "[Arti: Arti restarted in the middle of your last task. Check what finished before "
    "continuing; outcomes above marked unknown may or may not have happened.]"
)


class App:
    def __init__(
        self, config: dict, data_dir: Path, channel: Channel | None = None, *, plugins=None
    ):
        """`channel`: None makes the link to the host at start (tests and --once pass their own).
        `plugins`: None loads <repo>/plugins; tests pass register(api) functions."""
        self.config, self.data_dir, self.channel = config, Path(data_dir), channel
        self.plugins = plugins
        self.code_dir = code_dir()
        self.tz = ZoneInfo(config["agent"]["timezone"])
        self.owner_name = config["agent"]["user_name"]
        self.agent_name = config["agent"]["name"]
        self.store: Store | None = None  # opened by start()
        self.watchdog: watchdog.Watchdog | None = None  # started by start() (serve)
        self.loop: asyncio.AbstractEventLoop | None = None  # likewise
        # What plugins register (plugin.add_plugin fills these; load_plugins runs them).
        self.plugins_loaded = False
        self.tools: dict[str, tools.Tool] = {}
        self.services: dict = {}
        self.prompt_sections, self.new_session_hooks, self.compact_hooks = [], [], []
        self.start_hooks: list[tuple] = []  # (fn(app), also for --once), in load order
        self.stop_hooks: list = []  # (fn(app), timeout), in load order (run in reverse)
        self.gate = None  # fn(ctx, tool, args) -> refusal | None (api.gate); None: no review
        self.redactor = None  # (text fn, obj fn) (api.redactor); secret.py redacts through it
        self.file_checks: list = []  # fn(path) -> why it must not be sent | None (api.file_check)
        self.chat_commands: dict[str, plugin.ChatCommand] = {}
        self.provider_specs: dict[str, plugin.ProviderSpec] = {}
        self.commands: dict[str, plugin.Command] = {}
        self.plugin_sections: dict[str, str] = {}  # config section -> the plugin that declared it
        self.provider_keys: list[str] = []  # "<section>.<key>" settings naming a provider
        self.state: dict = {}  # plugins' per-App objects
        self.conversations: dict[str, Chat] = {}  # key -> its loaded Chat
        self.started = time.time()
        self.stopping = self.restart = False
        self.exit_requested = asyncio.Event()
        self.headless_slots = asyncio.Semaphore(MAX_HEADLESS)
        self._providers: dict = {}
        self._tasks: set[asyncio.Task] = set()  # background tasks, cancelled at shutdown
        self._headless: set[asyncio.Task] = set()  # running headless turns, likewise

    @property
    def home(self) -> str:
        """The owner's one-on-one conversation key (the channel's); "" when there is none."""
        return self.channel.home if self.channel else ""

    @property
    def owner(self) -> str:
        """The owner's normalised handle (the channel's); "" when unknown."""
        return self.channel.owner if self.channel else ""

    def load_plugins(self) -> None:
        """Load the plugins, then check the configuration against what they declared (their
        sections, the providers). Once only; start() calls it, the command line earlier."""
        if self.plugins_loaded:
            return
        self.plugins_loaded = True
        skipped = set()
        if self.plugins is None:
            disabled = set(self.config["plugins"]["disabled"])
            skipped = plugin.load_plugins(self, self.code_dir / "plugins", disabled)
        for register in self.plugins or ():
            if not plugin.add_plugin(self, register.__module__, register):
                skipped.add(register.__module__)
        secret.set_redactor(self.redactor)  # the log and errors are redacted from here on too
        config.check_plugins(
            self.config, set(self.plugin_sections), skipped, self.provider_specs, self.provider_keys
        )
        if self.gate is None:
            log.warning("no plugin registered a gate: tool calls run without review")

    def make_channel(self) -> Channel:
        """The link to the host (not started)."""
        return Link(self.data_dir)

    async def start(self, serve: bool = True) -> None:
        """Load plugins, make the channel (unless one was given), take the data dir and open the
        store; with `serve` (not for --once) also recover, run plugin start hooks, start the
        channel and the queued conversations. --once runs only the start hooks marked `once`."""
        self.loop = asyncio.get_running_loop()
        self.load_plugins()
        if self.channel is None:
            self.channel = self.make_channel()
        private_dir(self.data_dir)
        lock.acquire(self.data_dir)
        self.store = Store(self.data_dir / "state.db")
        if not serve:
            # --once must not answer queued messages or run start hooks into a printing channel.
            await self._run_start_hooks(once=True)
            return
        self.recover()
        await self._run_start_hooks()
        await self.channel.start(self)
        for inp in self.store.pending_inputs():
            self._dispatch(inp, run_commands=False)
        self.spawn(self._housekeeping()).set_name("housekeeping")
        self._watch_for_hangs()
        log.info("Arti started (%s)", "/".join(self.default_route()))

    async def _run_start_hooks(self, once: bool = False) -> None:
        """The plugins' on_start hooks in load order (with `once`, only those for --once too)."""
        for hook, for_once in self.start_hooks:
            if once and not for_once:
                continue
            try:
                if inspect.isawaitable(result := hook(self)):
                    await result
            except Exception:
                log.exception("plugin on_start hook %r failed", hook)

    def _watch_for_hangs(self) -> None:
        """Report a hang the last run ended with, then start the watchdog and its heartbeat."""
        logs = private_dir(self.data_dir / "logs")
        hang = logs / "hang.log"
        if hang.exists():
            hang.replace(logs / "hang.log.1")
            log.warning("the last run stopped making progress and exited; see logs/hang.log.1")
            if self.home:
                text = (
                    f"[{self.agent_name}: I got stuck and restarted myself. Where it was stuck is "
                    "in logs/hang.log.1 in my folder.]"
                )
                self.spawn(self.send(self.home, text))
        if after := self.config["agent"]["hang_timeout"]:
            self.watchdog = watchdog.Watchdog(hang, after)
            self.watchdog.start()
            self.spawn(self._heartbeat()).set_name("heartbeat")
            log.info("watchdog on: Arti restarts itself after %.0f s without progress", after)

    async def _heartbeat(self) -> None:
        """Feed the watchdog while the loop runs and the channel still reads messages."""
        while True:
            if self.channel.alive():
                self.watchdog.beat()
            await asyncio.sleep(watchdog.BEAT)

    # --- inputs -------------------------------------------------------------------------------

    async def admit(self, inputs: list[Input]) -> None:
        """Record inputs in one transaction, then act on the new ones.

        An input seen before is dropped by the store (same event id). Raises while shutting
        down, so the channel keeps the inputs and hands them over again after the restart.
        """
        if self.stopping:
            raise RuntimeError("Arti is shutting down")
        for inp in inputs:
            inp.text = secret.redact(inp.text)
        for inp in self.store.admit(inputs):
            self._dispatch(inp, run_commands=True)

    def _dispatch(self, inp: Input, run_commands: bool) -> None:
        """Hand an input to its Chat, or run it as a slash command."""
        command = commands.parse(self, inp.text) if inp.kind == "message" else None
        if command is None:
            self.conversation(inp.key).add(inp)
            return
        # Marked done before it runs, so a crash never runs a command twice; one still pending
        # at startup (`run_commands` False) is dropped instead of run late.
        self.store.mark_inputs([inp.seq], "done")
        if not run_commands:
            return
        name, arg = command
        if name in ("stop", "new"):  # halted now, so input after it in this batch is kept
            conv = self.conversation(inp.key)
            if name == "stop" and conv.busy:
                # The cancelled turn writes this into history, so the next turn knows the task
                # was stopped on purpose instead of seeing it unfinished and resuming it.
                conv.note(
                    f"[Arti: {conv.person_name} sent /stop here, so the task above was "
                    "stopped. Do not pick it up again unless asked.]"
                )
            conv.halt()
        self.spawn(self._run_command(inp, name, arg))

    async def _run_command(self, inp: Input, name: str, arg: str) -> None:
        try:
            reply = await commands.run(self, inp, name, arg)
        except Exception as error:
            log.exception("command /%s failed", name)
            reply = f"/{name} failed: {error}"
        if reply:
            await self.send(inp.key, reply)

    async def notify(self, key: str, text: str, at: float | None = None) -> None:
        """Queue a notice: it starts a turn but needs no answer (a background job finished).
        With `at` (unix time) it is delivered then, also after a restart; /stop cancels it."""
        event_id = f"notice:{uuid.uuid4().hex}"
        inp = Input(event_id, key, "notice", text=text, ts=at or time.time(), due=at or 0.0)
        await self.admit([inp])

    async def deliver(self, key: str, text: str) -> str:
        """Send `text` to a conversation and record it there as a note (a scheduled result)."""
        status = await self.send(key, text)
        heard = "was sent" if status == "sent" else f"could not be confirmed sent ({status})"
        note = f"[Arti: this message from a scheduled task {heard} here:]\n{text}"
        await self.admit([Input(f"deliver:{uuid.uuid4().hex}", key, "note", text=note)])
        return status

    # --- sends --------------------------------------------------------------------------------

    async def send(self, key: str, text: str) -> str:
        """Send a reply to a conversation: "sent" | "failed" | "unknown". Its MEDIA lines are
        files (reply.py); one that may not go out is sent as a note saying why instead."""
        text = secret.redact(text)
        parts = []
        for part in reply.parts(text):
            if part.kind == "file" and (problem := await self._file_problem(key, part.value)):
                part = Part("text", f"[file not sent: {problem}]")
            parts.append(part)
        return await self._send(key, "text", text, self.channel.send(key, parts))

    async def _file_problem(self, key: str, path: str) -> str | None:
        if problem := await asyncio.to_thread(self.file_problem, key, path):
            log.warning("file not sent: %s", problem)
        return problem

    def file_problem(self, key: str, path: str | Path) -> str | None:
        """Why a file must not or cannot be sent to conversation `key`, or None. A place other
        than home gets only files inside its own context folder (everything made or received
        there lives there), so sending is no way around the approvals that guard every other
        path; raw tool output may hold secrets; and every plugin's file check must pass
        (api.file_check: one that refuses secret files, say). A channel that copies the file
        before sending checks the copy again, so a symlink swapped in between is caught too."""
        name = Path(path).name
        if not (os.path.isfile(path) and os.access(path, os.R_OK)):
            return f"{name} is not a readable file"
        real = Path(path).resolve()
        scope = self.scope_for(key)
        own = self.context_dir(scope).resolve()
        if real.is_relative_to(own / "cache" / "tool-output"):
            return f"{name} is raw tool output, which may hold secrets (send its text instead)"
        if scope != "home" and not real.is_relative_to(own):
            return f"{name} is outside this chat's files"
        for check in self.file_checks:
            if problem := check(str(path)):
                return problem
        return None

    async def _send(self, key: str, kind: str, body: str, sending) -> str:
        # The outbox row is committed before the channel is called: after a crash a row still
        # "sending" is reported as unknown, never sent again.
        loaded = self.conversations.get(key)
        row = self.store.outbox_begin(key, loaded.session_id if loaded else "", kind, body)
        status = "unknown"
        try:
            status = await asyncio.wait_for(sending, SEND_TIMEOUT)
        except asyncio.CancelledError:
            # Cut short by shutdown: left "sending", so the restart tells the model about it.
            if not self.stopping:
                self.store.outbox_end(row, "unknown")
            raise
        except TimeoutError:
            log.warning("send to %s timed out; outcome unknown", key)
        except Exception:
            log.exception("send to %s raised; outcome unknown", key)
        status = status if status in ("sent", "failed") else "unknown"
        self.store.outbox_end(row, status)
        return status

    # --- model calls without a chat -----------------------------------------------------------

    async def run_headless(
        self,
        prompt: str,
        *,
        scope: str = "home",
        instructions: str = "",
        tools: list[str] | None = None,
        exclude=(),
        provider: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        images=(),
        trusted: bool = True,
        trusted_prompt_writes: bool = False,
    ) -> str:
        """Run the turn loop with no chat and in-memory history; return the final reply text.
        `instructions` go at the end of its system prompt (the caller's own, e.g. a job's
        preloaded procedures). `trusted` False runs it as someone else's conversation (guest
        tools, no private context); `trusted_prompt_writes` lets its `prompt` calls run
        unreviewed (a plugin's own background run that writes what later prompts read)."""
        conv = Headless(
            self,
            scope,
            self.default_route(provider, model),
            tools=tools,
            exclude=exclude,
            instructions=instructions,
            trusted=trusted,
            trusted_prompt_writes=trusted_prompt_writes,
        )
        folder = self.context_dir(scope)
        blocks = [await asyncio.to_thread(media.image_block, Path(p), folder) for p in images]
        conv.append({"role": "user", "content": blocks + [{"type": "text", "text": prompt}]})
        # Tracked so shutdown cancels it, also when a tool's worker thread started it.
        task = asyncio.current_task()
        self._headless.add(task)
        try:
            async with self.headless_slots:
                async with asyncio.timeout(timeout or self.config["agent"]["tool_timeout"]):
                    return await run_turn(conv)
        finally:
            self._headless.discard(task)

    async def complete(
        self,
        system: str,
        text: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        effort: str = "low",
        timeout: float = 120,
    ) -> str:
        """One model call without tools or history (a plugin's quick check, a summary)."""
        name, model, _ = self.default_route(provider, model)
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        async with asyncio.timeout(timeout):
            reply = await self.provider(name).complete(
                messages,
                system=system,
                tools=[],
                model=model,
                effort=effort,
                session_id=uuid.uuid4().hex,
            )
        return reply.text.strip()

    # --- lookups ------------------------------------------------------------------------------

    def conversation(self, key: str) -> Chat:
        """The Chat of conversation `key`, loaded from the store on first use."""
        if key not in self.conversations:
            self.conversations[key] = Chat(self, key)
        return self.conversations[key]

    def scope_for(self, key: str) -> str:
        """The place conversation `key` lives in: "home" or another (Channel.scope_of)."""
        return self.channel.scope_of(key)

    def person_of(self, key: str) -> str | None:
        """Whose conversation `key` is (Channel.person_of): "" for the owner's one-on-one."""
        return self.channel.person_of(key)

    def owns(self, key: str) -> bool:
        """Conversation `key` is the owner's own (their one-on-one, or theirs in another
        place): trusted, with every tool and the owner's private context."""
        person = self.person_of(key)
        return person == "" or self.is_owner(person)

    def place_name(self, scope: str) -> str:
        """The name of a place for tags and questions (Channel.place_name); "" for home."""
        return self.channel.place_name(scope)

    def context_dir(self, scope: str) -> Path:
        """The home data dir, or contexts/<scope> for another place's files (private)."""
        return self.data_dir if scope == "home" else private_dir(self.data_dir / "contexts" / scope)

    def cache_dir(self, name: str) -> Path:
        return private_dir(self.data_dir / "cache" / name)

    def is_owner(self, handle: str | None) -> bool:
        """`handle` (normalised, as the channel gives senders) is the owner's."""
        return bool(handle) and handle == self.owner

    def default_route(self, provider: str | None = None, model: str | None = None) -> tuple:
        """(provider, model, effort) from the config; another provider gets its default model."""
        agent = self.config["agent"]
        route = (agent["provider"], agent["model"])
        if provider and provider != route[0]:
            route = (provider, self.provider_spec(provider).default_model)
        return route[0], model or route[1], agent["effort"]

    def provider_spec(self, name: str) -> plugin.ProviderSpec:
        """What a plugin registered for provider `name` (its models, window, factory)."""
        if name not in self.provider_specs:
            raise ValueError(f"unknown provider: {name}")
        return self.provider_specs[name]

    def provider(self, name: str):
        """One provider client per name, created on first use."""
        if name not in self._providers:
            spec = self.provider_spec(name)
            self._providers[name] = spec.factory(self, self.config["providers"][name])
        return self._providers[name]

    def spawn(self, coro) -> asyncio.Task:
        """Run `coro` in the background; the App keeps it alive and cancels it at shutdown. A
        task that fails is logged: a background failure is never invisible."""
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            log.error("background task %s failed", task.get_name(), exc_info=error)

    # --- recovery, housekeeping, shutdown -----------------------------------------------------

    def recover(self) -> None:
        """After a crash: answer tool calls left open, mark in-flight sends unknown, and queue
        one notice per affected conversation so the owner is not left hanging. The notices are
        admitted first (their event ids make that idempotent), so a crash in here cannot lose
        one."""
        notices: dict[str, Input] = {}

        def notice(key: str, event_id: str) -> Input:
            if key not in notices:
                notices[key] = Input(event_id, key, "notice", text=RESTART_NOTICE, ts=time.time())
                log.warning("recovering an interrupted turn in %s", key)
            return notices[key]

        sessions = self.store.open_sessions()
        histories = {session["id"]: self.store.history(session["id"]) for session in sessions}
        for session in sessions:
            # Answers an earlier, interrupted recover() added do not change the event id.
            ids = [m["id"] for m in histories[session["id"]] if not _recovered(m)]
            notice(session["key"], f"restart:{session['id']}:{ids[-1] if ids else 0}")
        sends = self.store.outbox_inflight()
        for row in sends:
            inp = notice(row["key"], f"outbox:{row['id']}")
            inp.text += (
                f"\n[Arti: this send may or may not have been delivered: {row['body'][:200]!r}]"
            )
        self.store.admit(list(notices.values()))

        for session in sessions:
            calls = open_calls(histories[session["id"]])
            results = [tools.unknown_result(call["id"], RESTARTED) for call in calls]
            if results:
                message = {"role": "user", "content": results}
                self.store.append(session["id"], session["key"], message)
            self.store.update_session(session["id"], turn_open=False)
        for row in sends:
            self.store.outbox_end(row["id"], "unknown")

    async def _housekeeping(self) -> None:
        """Run `housekeep` at start and then daily. Due by the wall clock, checked every
        WAKE_EVERY: the event loop's clock stops while the Mac sleeps, so a plain 24-hour sleep
        would drift later by every night the Mac slept."""
        due = 0.0
        while True:
            if time.time() >= due:
                due = time.time() + HOUSEKEEPING_EVERY
                try:
                    await self.housekeep()
                except Exception:
                    log.exception("housekeeping failed")
            await asyncio.sleep(WAKE_EVERY)

    async def housekeep(self) -> None:
        """Forget old inbox/outbox rows, delete old media and cache files, and unload idle
        conversations other than home (each holds its history in memory; app.conversation loads
        it again).

        Only a conversation quiet for IDLE_UNLOAD is unloaded: a timed-out tool thread may still be
        changing the old Chat's state, and a fresh Chat loaded for the key would not see that."""
        now = time.time()
        for key, conv in list(self.conversations.items()):
            active = conv.busy or conv.pending or conv.waits or conv.halting or conv.running_call
            if key != self.home and not active and now - conv.last_used > IDLE_UNLOAD:
                del self.conversations[key]
        self.store.prune(INBOX_DAYS)
        days = self.config["storage"]["cache_max_age_days"]
        await asyncio.to_thread(prune_files, self.prune_roots(), days)

    def prune_roots(self) -> list[Path]:
        """The disposable folders: media/ and cache/ of home and of every other context."""
        contexts = sorted(self.data_dir.glob("contexts/*"))
        return [base / sub for base in [self.data_dir, *contexts] for sub in ("media", "cache")]

    def request_restart(self) -> None:
        """/restart: shut down cleanly, then the daemon re-executes itself."""
        self.restart = True
        self.exit_requested.set()

    async def shutdown(self) -> None:
        """Stop inbound, stop every conversation (queued input stays pending), run the on_stop
        hooks (in reverse load order: a plugin before the plugins it requires), stop the channel
        and close the store. Every step is bounded."""
        if self.stopping:
            return
        self.stopping = True
        if self.store:  # not for a run that never started (another Arti held the lock, say)
            log.info("Arti shutting down")
        # All at once: each may wait up to STOP_WAIT for its turn to wind down.
        stops = [conv.stop(drop_pending=False) for conv in self.conversations.values()]
        for error in await asyncio.gather(*stops, return_exceptions=True):
            if isinstance(error, Exception):
                log.error("stopping a conversation failed: %r", error)
        for task in (self._tasks | self._headless) - {asyncio.current_task()}:
            task.cancel()
        steps = []
        for hook, timeout in reversed(self.stop_hooks):  # a plain one runs in an abandonable thread
            run = hook(self) if inspect.iscoroutinefunction(hook) else asyncio.to_thread(hook, self)
            steps.append((run, timeout))
        if self.channel:  # None when start failed before making it
            steps.append((self.channel.stop(), 10))
        steps += [(provider.aclose(), 5) for provider in self._providers.values()]
        for step, timeout in steps:
            try:
                await asyncio.wait_for(step, timeout)
            except Exception:
                log.exception("shutdown step %r failed", step)
        if self.store:
            self.store.close()
        if self.watchdog:  # a shutdown that hangs is ended by it instead
            self.watchdog.stop()


def _recovered(message: dict) -> bool:
    """A message recover() added: answers to the tool calls a restart left open."""
    blocks = message["content"]
    return (
        message["role"] == "user"
        and bool(blocks)
        and all(
            block["type"] == "tool_result"
            and bool(block["content"])
            and block["content"][0].get("text") == RESTARTED
            for block in blocks
        )
    )


def prune_files(roots: list[Path], days: float) -> int:
    """Delete the files under `roots` older than `days` (never following symlinks)."""
    files = (
        Path(folder) / name
        for root in roots
        for folder, _, names in os.walk(root)
        for name in names
    )
    return delete_older_than(files, days * 86400)
