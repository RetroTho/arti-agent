"""The host: runs on the machine, owns the chat platform, and keeps one agent per user running,
each in that user's sandbox (sandbox.py).

  channel ──▶ `receive` ──▶ the sender's agent (its link/in) ··· the agent's link/out ──▶ `_pump`
          ◀── send / typing ◀───────────────────────────────────────────────────────────────┘

`receive` finds the user by the sender's handle (users.py) and drops a message from anyone else;
it names the conversation for that agent ("home", the user's own one-on-one with Arti, or
"<place>|<handle>" in a group chat) and remembers which chat that is (routes.json), then leaves
the message in the agent's link/in (arti/link.py describes the files). `_pump` carries out what
the agents ask, one user at a time in order. An agent is not trusted: a request is read with a
size limit, may only name that user's own conversations, and may only send files inside that
user's home folder. A send the host took but did not finish (a crash) is answered "unknown",
never done twice.
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

from arti import link, secret
from arti.channel import Part
from arti.fsutil import private_dir, read_json, write_json
from host.channel import Channel, Message
from host.sandbox import Sandboxes
from host.users import SANDBOX_HOME, User, Users

log = logging.getLogger(__name__)

POLL = 0.1  # seconds between looks at the agents' requests
MAX_REQUEST = 1_000_000  # bytes
SEND_TIMEOUT = 300  # seconds for the channel to take one send
STARTED_AFTER = 5  # seconds after which a sandbox that started should still be running


class Host:
    def __init__(self, data_dir: Path, channel: Channel, sandboxes: Sandboxes | None):
        """`sandboxes`: None when the agents run elsewhere (tests)."""
        self.data_dir, self.channel, self.sandboxes = Path(data_dir), channel, sandboxes
        self.users = Users(data_dir)
        self.routes_path = self.data_dir / "routes.json"
        self._pumper: asyncio.Task | None = None
        self._looked = 0.0

    # --- life ---------------------------------------------------------------------------------

    async def start(self) -> None:
        """Start every user's sandbox afresh, then the channel and the pump."""
        if self.sandboxes:
            await asyncio.to_thread(self.sandboxes.ready)
        for user in self.users.all():
            self.prepare(user)
            if self.sandboxes:
                await asyncio.to_thread(self.sandboxes.start, user)
                log.info("started %s's agent", user.id)
        if self.sandboxes:
            await asyncio.sleep(STARTED_AFTER)
            for user in self.users.all():
                if not await asyncio.to_thread(self.sandboxes.running, user):
                    log.error("%s's agent stopped right after starting; see: podman logs %s",
                              user.id, self.sandboxes.name(user))  # fmt: skip
        await self.channel.start(self)
        self._pumper = asyncio.create_task(self._pump(), name="host-pump")

    async def stop(self) -> None:
        if self._pumper:
            self._pumper.cancel()
            await asyncio.gather(self._pumper, return_exceptions=True)
        await self.channel.stop()
        if self.sandboxes:
            users = self.users.all()
            await asyncio.gather(*(asyncio.to_thread(self.sandboxes.stop, u) for u in users))

    def alive(self) -> bool:
        return self.channel.alive() and time.monotonic() - self._looked < 60

    def prepare(self, user: User) -> None:
        """Make the user's link folders, tell the agent whom it works for, and answer the sends
        a crash left unfinished."""
        paths = link.folders(user.agent_data(self.data_dir))
        for name in ("in", "out", "done"):
            private_dir(paths[name])
        handle = user.handles.get(self.channel.name, "")
        write_json(paths["me"], {"owner": handle, "notes": self.channel.notes})
        for taken in paths["out"].glob("*.taken"):
            request_id = taken.name.removesuffix(".taken")
            write_json(paths["done"] / f"{request_id}.json", {"status": "unknown"})
            taken.unlink(missing_ok=True)

    # --- in -----------------------------------------------------------------------------------

    async def receive(self, message: Message) -> bool:
        """Pass a message to its sender's agent; False (dropped) when the sender is no user."""
        user = self.users.by_handle(self.channel.name, message.sender)
        if user is None:
            log.info("dropped a message from %s, who is not a user", message.sender)
            return False
        if message.place:
            key = f"{message.place.replace('|', '_')}|{message.sender}"
            self._remember(user, key, message.chat)
        else:
            key = link.HOME
        data = user.agent_data(self.data_dir)
        attachments = [self._hand_over(user, path) for path in message.files]
        value = {
            "event_id": f"{self.channel.name}:{message.event_id}",
            "key": key,
            "kind": "message",
            "text": message.text,
            "attachments": [a for a in attachments if a],
            "sender": message.sender,
            "sender_name": message.sender_name,
            "ts": message.ts or time.time(),
            "place": message.place_name,
        }
        name = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}.json"  # in arrival order
        write_json(link.folders(data)["in"] / name, value)
        return True

    def _hand_over(self, user: User, path: Path) -> str:
        """Move a downloaded attachment into the user's home; its path in the sandbox."""
        folder = private_dir(user.agent_data(self.data_dir) / "media" / "incoming")
        target = folder / f"{uuid.uuid4().hex[:8]}-{Path(path).name[-80:]}"
        try:
            Path(path).replace(target)
        except OSError as error:
            log.warning("an attachment was not handed over: %s", error)
            return ""
        return self.sandbox_path(user, target)

    def sandbox_path(self, user: User, path: Path) -> str:
        relative = Path(path).resolve().relative_to(user.home(self.data_dir).resolve())
        return f"{SANDBOX_HOME}/{relative}"

    def host_path(self, user: User, sandbox_path: str) -> Path | None:
        """The host path of a file in the user's sandbox, or None when it is not a regular file
        inside their home (a link out of it is judged by where it points)."""
        if not sandbox_path.startswith(SANDBOX_HOME + "/"):
            return None
        home = user.home(self.data_dir).resolve()
        path = (home / sandbox_path.removeprefix(SANDBOX_HOME + "/")).resolve()
        return path if path.is_relative_to(home) and path.is_file() else None

    # --- routes -------------------------------------------------------------------------------

    def _routes(self) -> dict:
        return read_json(self.routes_path, {})

    def _remember(self, user: User, key: str, chat: str) -> None:
        routes = self._routes()
        if routes.get(user.id, {}).get(key) != chat:
            routes.setdefault(user.id, {})[key] = chat
            write_json(self.routes_path, routes)

    def chat_of(self, user: User, key: str) -> str | None:
        """The chat a user's conversation key names, or None (not theirs)."""
        if key == link.HOME:
            handle = user.handles.get(self.channel.name)
            return self.channel.home_chat(handle) if handle else None
        return self._routes().get(user.id, {}).get(key)

    # --- out ----------------------------------------------------------------------------------

    async def _pump(self) -> None:
        while True:
            self._looked = time.monotonic()
            for user in self.users.all():
                try:
                    await self.serve(user)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("serving %s's agent failed", user.id)
            await asyncio.sleep(POLL)

    async def serve(self, user: User) -> None:
        """Carry out the user's agent's requests, oldest first."""
        paths = link.folders(user.agent_data(self.data_dir))
        requests = sorted(paths["out"].glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
        for path in requests:
            taken = path.with_suffix(".taken")
            try:
                path.replace(taken)  # from here on a crash answers "unknown"
                if taken.stat().st_size > MAX_REQUEST:
                    raise ValueError("too large")
                request = json.loads(taken.read_text())
                request_id = str(request["id"])
                if request_id != path.stem:
                    raise ValueError("its id is not its name")
            except (OSError, ValueError, KeyError, TypeError) as error:
                log.warning("dropped a bad request from %s's agent: %s", user.id, error)
                taken.unlink(missing_ok=True)
                continue
            if request.get("type") == "send":
                status = await self._send(user, request)
                write_json(paths["done"] / f"{request_id}.json", {"status": status})
            elif request.get("type") == "typing" and (chat := self.chat_of(user, request["key"])):
                await self._quietly(self.channel.typing(chat, bool(request.get("on"))))
            taken.unlink(missing_ok=True)

    async def _send(self, user: User, request: dict) -> str:
        chat = self.chat_of(user, str(request.get("key", "")))
        if chat is None:
            log.warning("%s's agent asked to send to a conversation not theirs", user.id)
            return "failed"
        parts = []
        for raw in request.get("parts") or []:
            kind, value = raw.get("kind"), str(raw.get("value", ""))
            if kind == "file":
                path = self.host_path(user, value)
                parts.append(Part("file", str(path)) if path else Part("text", "[file not sent]"))
            elif kind == "text":
                parts.append(Part("text", secret.redact(value)))  # the host's secrets, too
        try:
            status = await asyncio.wait_for(self.channel.send(chat, parts), SEND_TIMEOUT)
        except TimeoutError:
            log.warning("a send to %s timed out; outcome unknown", user.id)
            return "unknown"
        except Exception:
            log.exception("a send to %s raised; outcome unknown", user.id)
            return "unknown"
        return status if status in ("sent", "failed") else "unknown"

    async def _quietly(self, coro) -> None:
        try:
            await asyncio.wait_for(coro, 10)
        except Exception as error:
            log.debug("typing not shown: %s", error)
