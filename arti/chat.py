"""Conversations: what a turn works on.

`Conversation` holds what a turn reads and changes: the history, the inputs waiting to be read,
host notes, tools' state, and the flags that steer a running turn. Two kinds exist:

- `Chat` lives in a chat, named by its key (channel.py): the owner's one-on-one, or one person's
  conversation in a place the channel knows (a group chat, say). Its session and history are
  stored, and its `runner` task takes its turns one at a time; conversations run independently.
  `/stop` and `/new` are `stop`.
- `Headless` is a run with no chat (a plugin's background run, --once): the same turn loop with
  in-memory history that is never stored, and nothing is sent.
"""

import asyncio
import logging
import threading
import time
import uuid

from arti import media, secret
from arti.store import Input
from arti.turn import run_turn

log = logging.getLogger(__name__)

STOP_WAIT = 10  # seconds to wait for a cancelled turn to repair its history
SAVE_ATTEMPTS = 3


class Conversation:
    """The state a turn reads and changes; Chat and Headless fill in storage and sending."""

    headless = False
    allowed: set[str] | None = None  # headless tool allowlist; None = all
    excluded: frozenset[str] | set[str] = frozenset()
    instructions = ""  # a headless run's extra system prompt text (App.run_headless)

    def __init__(
        self, app, key: str, scope: str, route: tuple[str, str, str], trusted: bool = True
    ):
        self.app, self.key, self.scope = app, key, scope
        # The owner's own conversation (their one-on-one, their conversations elsewhere, their
        # jobs): all tools and approvals. False for anyone else's: guest tools, no private context.
        self.trusted = trusted
        # Whose conversation it is (Channel.person_of): "" for the owner's one-on-one.
        self.person = app.person_of(key) or ""
        self.person_name = app.owner_name if trusted else (self.person or "someone")
        self.context_dir = app.context_dir(scope)
        self.route = route  # (provider, model, effort)
        self.session_id = ""
        self.messages: list[dict] = []  # the active history (after the compaction point)
        self.summary = ""  # the summary of what came before the compaction point
        self.state: dict = {}  # per-session state: approval grants, cwd, and the tools' own
        self.pending: list[Input] = []  # inputs not yet in history
        self.notes: list[dict] = []  # host notes (text blocks) for the next user message
        # Steering. Chat.add counts input as it arrives:
        #   person_inputs  inputs from this conversation's own person: a message, or their
        #                  reaction, edit or vote (a tool's ctx.wait wakes up when it moves)
        #   messages_in    messages in words only (a reply in words ends a pending approval)
        # and drain() notes `person_inputs` when it takes the inputs for a model request:
        #   person_inputs_asked
        # While the two differ the person said something the model has not seen, so the reply's
        # remaining tool calls are skipped (person_spoke_since_request, turn.run_calls).
        self.person_inputs = self.messages_in = self.person_inputs_asked = 0
        self.unanswered = False  # the latest drained message has no reply yet
        self.cancel = threading.Event()  # the current turn's; set by /stop, /new, shutdown
        self.offered: dict = {}  # name -> Tool offered in the current model request
        self.running_call = self.running_tool = ""  # the tool call running now
        self.rate_limit: dict = {}  # the provider's latest usage report (/status)
        self.last_error = ""  # the latest turn failure (/status)
        # The system prompt is built once per conversation and rebuilt only after a compaction
        # or a new session, so what a tool saves mid-conversation for later prompts (already
        # visible in the history) does not break the provider's prompt cache.
        self.system: str | None = None
        self.waits: dict[int, tuple] = {}  # inbox seq -> (Input, timer) of a pending wait
        # A plugin's background run whose `prompt` calls run without asking (App.run_headless).
        self.trusted_prompt_writes = False

    # --- what the turn loop calls, in order ---------------------------------------------------

    def begin_turn(self) -> None:
        self.cancel = threading.Event()
        self.running_call = self.running_tool = ""

    async def drain(self) -> tuple[list[dict], list[int]]:
        """Take the waiting inputs and notes off their queues, as content blocks for the next user
        message, plus the inbox rows they came from. Nothing is stored here: the caller appends
        the blocks with `done=` those rows, and that one append also records the inputs as
        done."""
        inputs, seen = list(self.pending), self.person_inputs
        blocks = []
        for inp in inputs:
            blocks += await asyncio.to_thread(
                media.input_blocks, self.app, inp, self.cancel, self.context_dir
            )
        # Taken off the queue only now, so /stop during a slow attachment still drops them.
        self.pending = self.pending[len(inputs) :]
        # Input added during the awaits above is not in this request: it still counts as new.
        self.person_inputs_asked = seen
        self.unanswered |= any(inp.kind == "message" for inp in inputs)
        blocks, self.notes = blocks + self.notes, []
        return blocks, [inp.seq for inp in inputs if inp.seq]

    def append(self, message: dict, done=()) -> None:
        """Add a message to history (secrets redacted); `done` = inbox rows it consumed."""
        message = secret.redact_obj(message)
        message["id"] = self._store(message, done)
        self.messages.append(message)

    async def send(self, text: str) -> str:
        return "sent"

    def typing(self, on: bool) -> None:
        pass

    def note(self, text: str, unknown: bool = False) -> None:
        """Queue a host note for the next user message."""
        self.notes.append({"type": "text", "text": text, **({"unknown": True} if unknown else {})})

    def person_spoke_since_request(self) -> bool:
        """Something from the conversation's person arrived that the last model request did
        not include (steering): the reply's remaining tool calls must not run."""
        return self.person_inputs != self.person_inputs_asked

    def mark_answered(self) -> None:
        """A reply (or a reaction) answered the latest message."""
        self.unanswered = False

    def end_turn(self) -> None:
        if self.notes:  # e.g. a failure note: keep it in history for the next turn
            self.append({"role": "user", "content": self.notes})
            self.notes = []
        # The turn is over (answered, left silent, or out of nudges), so a later turn, say for a
        # notice, must not nudge about this message. (The notes appended above answer nothing.)
        self.unanswered = False

    # --- storage (Chat stores; Headless keeps everything in memory) ---------------------------

    def _store(self, message: dict, done) -> int:
        return 0

    def compact(self, summary: str, cut: int) -> None:
        """Replace messages[:cut] by `summary` (the archive keeps the originals). The system
        prompt is rebuilt for the next request: the cache breaks here anyway."""
        self.summary, self.messages = summary, self.messages[cut:]
        self.system = None

    def save_state(self) -> None:
        pass


class Chat(Conversation):
    """A conversation that lives in a chat (the home chat, or one person's conversation in another
    place). It keeps its stored session; turns run one at a time in `runner`."""

    def __init__(self, app, key: str):
        super().__init__(app, key, app.scope_for(key), app.default_route(), app.owns(key))
        self.runner: asyncio.Task | None = None
        self.halting = False  # inside stop(): no new turn starts
        self.last_used = time.time()  # the latest input or turn end (App.housekeep unloads)
        self._load(app.store.current_session(key, app.default_route()))

    @property
    def busy(self) -> bool:
        return self.runner is not None and not self.runner.done()

    def add(self, inp: Input) -> None:
        """Queue an input; a message or notice starts a turn when none is running. A delayed
        notice (a wait) is held until it is due, checked at least every minute: the event
        loop's clock stops while the Mac sleeps, the wall clock does not."""
        self.waits.pop(inp.seq, None)
        self.last_used = time.time()
        if (delay := inp.due - time.time()) > 0:
            self.waits[inp.seq] = (inp, self.app.loop.call_later(min(delay, 60), self.add, inp))
            return
        self.pending.append(inp)
        # Only the conversation's own person steers it: a reaction or vote by someone else on
        # its message is news, not a correction. (The one-on-one admits only the owner.)
        own = inp.sender and (not self.person or inp.sender == self.person)
        if inp.kind == "message" or own:
            self.person_inputs += 1
        self.messages_in += inp.kind == "message"
        self._start()

    def _start(self) -> None:
        # Notes alone start no turn; they wait for the next one.
        if self.busy or self.halting or self.app.stopping:
            return
        if any(inp.kind != "note" for inp in self.pending):
            self.runner = self.app.spawn(self._run())
            # A stopped runner may wind down late (STOP_WAIT passed): input that came in
            # meanwhile starts the next turn when it ends.
            self.runner.add_done_callback(lambda task: task.cancelled() and self._start())

    async def _run(self) -> None:
        while any(inp.kind != "note" for inp in self.pending) and not self.app.stopping:
            waiting = list(self.pending)
            await run_turn(self)
            if self._still_queued(waiting):  # the turn failed before it took them
                self.app.store.mark_inputs([i.seq for i in waiting if i.seq], "cancelled")
                self.pending = self.pending[len(waiting) :]  # (its failure was reported)

    def _still_queued(self, inputs: list[Input]) -> bool:
        """True when `inputs` are still the head of the queue: no drain took them. (Input that
        arrived during the turn queues after them, so compare only the head.)"""
        return self.pending[: len(inputs)] == inputs

    def begin_turn(self) -> None:
        super().begin_turn()
        self.app.store.update_session(self.session_id, turn_open=True)
        self.typing(True)

    def end_turn(self) -> None:
        super().end_turn()
        self.last_used = time.time()
        # A turn cut short by shutdown stays open, so the restart resumes it with a notice.
        if not self.app.stopping:
            self.app.store.update_session(self.session_id, turn_open=False)
        self.typing(False)

    async def send(self, text: str) -> str:
        status = await self.app.send(self.key, text)
        if status != "failed":  # "unknown" may have arrived: never nudged into sending it again
            self.mark_answered()
        return status

    def typing(self, on: bool) -> None:
        async def best_effort():
            try:
                await asyncio.wait_for(self.app.channel.typing(self.key, on), 5)
            except Exception:
                log.debug("typing indicator failed", exc_info=True)

        self.app.spawn(best_effort())

    # --- /stop, /new and shutdown -------------------------------------------------------------

    def halt(self, drop_pending: bool = True) -> None:
        """The instant part of /stop: drop the queued inputs and waits (if asked), cancel the
        running turn and start no new one until stop() finishes. Input queued after this call
        is kept."""
        self.halting = True
        if drop_pending:
            for _, timer in self.waits.values():
                timer.cancel()
            # Notes (e.g. a record of a scheduled message that went out) are context: kept.
            dropped = [i for i in self.pending if i.kind != "note"]
            seqs = [i.seq for i in dropped if i.seq] + list(self.waits)
            self.app.store.mark_inputs(seqs, "cancelled")
            self.pending = [i for i in self.pending if i.kind == "note"]
            self.waits = {}
        self.cancel.set()
        if self.busy:
            self.runner.cancel()

    async def stop(self, drop_pending: bool = True, fresh: bool = False) -> None:
        """Halt (unless already halted), wait for the turn to repair its history, then start
        on any input that arrived meanwhile. `fresh` starts a new session first (/new: history
        and session state reset). Shutdown passes `drop_pending=False`: queued input stays
        in the inbox for the next start."""
        if not self.halting:
            self.halt(drop_pending)
        try:
            if self.busy:
                await asyncio.wait({self.runner}, timeout=STOP_WAIT)
            if fresh:
                self._load(self.app.store.new_session(self.key, self.route))
                self.notes, self.unanswered = [], False
        finally:
            self.halting = False
        self._start()

    # --- storage ------------------------------------------------------------------------------

    def _load(self, session: dict) -> None:
        self.session_id = session["id"]
        self.route = (session["provider"], session["model"], session["effort"])
        self.summary, self.state = session["summary"], session["state"]
        self.messages = self.app.store.history(self.session_id)
        self.system = None

    def _store(self, message: dict, done) -> int:
        return self.app.store.append(self.session_id, self.key, message, done)

    def compact(self, summary: str, cut: int) -> None:
        start_id = self.messages[cut]["id"]
        self.app.store.update_session(
            self.session_id, summary=summary, start_id=start_id, state=self.state
        )
        super().compact(summary, cut)

    def save_state(self) -> None:
        """Store the tools' state; never raises: a lost save must not fail the turn."""
        # A timed-out tool thread may still change the state while it is serialised; that
        # makes json.dumps fail ("changed size during iteration"), so try again.
        for attempt in range(SAVE_ATTEMPTS):
            try:
                self.app.store.update_session(self.session_id, state=self.state)
                return
            except Exception:
                if attempt == SAVE_ATTEMPTS - 1:
                    log.exception("the session state of %s could not be saved", self.key)

    def set_route(self, route: tuple[str, str, str]) -> None:
        """/model and /effort: used from the next model request on."""
        provider, model, effort = self.route = route
        self.app.store.update_session(
            self.session_id, provider=provider, model=model, effort=effort
        )


class Headless(Conversation):
    """A run with no chat: sends nothing and keeps no stored history."""

    headless = True

    def __init__(
        self,
        app,
        scope,
        route,
        *,
        tools=None,
        exclude=(),
        instructions="",
        trusted=True,
        trusted_prompt_writes=False,
    ):
        super().__init__(app, "", scope, route, trusted)
        self.session_id = uuid.uuid4().hex
        self.allowed = set(tools) if tools is not None else None
        self.excluded, self.instructions = set(exclude), instructions
        self.trusted_prompt_writes = trusted_prompt_writes

    def end_turn(self) -> None:
        self.notes = []
