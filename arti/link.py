"""The link between the host and an agent: files in the agent's data directory.

The host (host/) runs on the machine and owns the chat platform; each user's agent runs in that
user's sandbox, with its data directory mounted from the host. They talk only through these
files, so there is no port to guard and a crash on either side loses nothing:

  link/me.json      whom the agent works for, written by the host: {"owner": the owner's handle,
                    "notes": the platform's notes for the system prompt}
  link/in/          the host's inputs: one JSON file each (an Input, plus "place": the place's
                    name). The agent admits them (the store drops an event it already has, by its
                    id), and only then deletes them.
  link/out/         the agent's requests: {"id", "type": "send", "key", "parts"} or {"id", "type":
                    "typing", "key", "on"}. The host takes one by renaming it (`.taken`), so a
                    crash in between leaves a send "unknown", never sent twice.
  link/done/        the host's answer to a send: {"status": "sent" | "failed" | "unknown"}.

Every file is written whole (a temporary name, then a rename), so a reader never sees half of one.

Keys: "home" is the owner's own one-on-one chat; "<place>|<person>" is one person's conversation
in a place (a group chat), whose scope is "chat-" and 16 hex digits of the place's sha256. The
host alone knows which real chat a key is; the agent never sees a platform's ids.

The agent side is `Link`, the agent's one channel. The host side is host/host.py.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path

from arti.channel import Channel, Part
from arti.fsutil import delete_older_than, private_dir, write_json
from arti.store import Input

log = logging.getLogger(__name__)

HOME = "home"
POLL = 0.2  # seconds between looks at a folder
STALE = 3600  # seconds after which an answer nobody waited for is deleted
INPUT_FIELDS = {f.name for f in dataclasses.fields(Input)} - {"seq", "due"}
PLACES = "link.places"  # kv: {scope: the place's name}


def folders(data_dir: Path) -> dict[str, Path]:
    root = Path(data_dir) / "link"
    return {"in": root / "in", "out": root / "out", "done": root / "done", "me": root / "me.json"}


def scope(place: str) -> str:
    return "chat-" + hashlib.sha256(place.encode("utf-8")).hexdigest()[:16]


def input_of(value: dict) -> Input:
    """An Input from the host's JSON (unknown fields ignored, the kinds checked)."""
    fields = {k: v for k, v in value.items() if k in INPUT_FIELDS}
    inp = Input(**fields)
    if inp.kind not in ("message", "notice", "note") or not inp.event_id or not inp.key:
        raise ValueError(f"not an input: {inp.event_id or '?'}")
    return inp


class Link(Channel):
    """The agent's channel: inputs from link/in, sends through link/out (the host does them)."""

    home = HOME

    def __init__(self, data_dir: Path):
        self.paths = folders(data_dir)
        try:
            me = json.loads(self.paths["me"].read_text())
        except (OSError, ValueError):
            me = {}
        self.owner, self.notes = me.get("owner", ""), me.get("notes", "")
        self.app = None
        self._watcher: asyncio.Task | None = None
        self._looked = 0.0

    async def start(self, app) -> None:
        self.app = app
        for name in ("in", "out", "done"):
            private_dir(self.paths[name])
        self._watcher = asyncio.create_task(self._watch(), name="link-watch")

    async def stop(self) -> None:
        if self._watcher:
            self._watcher.cancel()
            await asyncio.gather(self._watcher, return_exceptions=True)

    def alive(self) -> bool:
        return time.monotonic() - self._looked < 60

    async def _watch(self) -> None:
        pruned = 0.0
        while True:
            self._looked = time.monotonic()
            try:
                if self._looked - pruned > STALE:  # answers that came after the App gave up
                    delete_older_than(self.paths["done"].glob("*.json"), STALE)
                    pruned = self._looked
                await self.take_inputs()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("reading the host's inputs failed; trying again")
            await asyncio.sleep(POLL)

    async def take_inputs(self) -> None:
        """Admit what the host left in link/in, then delete it (in name order: the host names
        them in the order they arrived)."""
        files = sorted(p for p in self.paths["in"].glob("*.json"))
        if not files:
            return
        inputs, places = [], {}
        for path in files:
            try:
                value = json.loads(path.read_text())
                inputs.append(input_of(value))
                if value.get("place"):
                    places[self.scope_of(value["key"])] = str(value["place"])[:100]
            except (OSError, ValueError, TypeError) as error:
                log.warning("dropped a bad input from the host (%s): %s", path.name, error)
        if places:
            self.app.store.kv_set(PLACES, {**self.app.store.kv_get(PLACES, {}), **places})
        await self.app.admit(inputs)
        for path in files:
            path.unlink(missing_ok=True)

    # --- what a key means ---------------------------------------------------------------------

    def scope_of(self, key: str) -> str:
        return "home" if key == HOME else scope(key.partition("|")[0])

    def person_of(self, key: str) -> str | None:
        if key == HOME:
            return ""
        return key.partition("|")[2] or None

    def place_name(self, scope_: str) -> str:
        if scope_ == "home" or self.app is None:
            return ""
        return self.app.store.kv_get(PLACES, {}).get(scope_, "a group chat")

    def prompt_notes(self, key: str) -> str:
        return self.notes

    # --- sending ------------------------------------------------------------------------------

    async def send(self, key: str, parts: list[Part]) -> str:
        """Ask the host to send, and wait for its answer (the App bounds the wait)."""
        request_id = uuid.uuid4().hex
        request = {"id": request_id, "type": "send", "key": key}
        request["parts"] = [dataclasses.asdict(part) for part in parts]
        write_json(self.paths["out"] / f"{request_id}.json", request)
        answer = self.paths["done"] / f"{request_id}.json"
        try:
            while not answer.exists():
                await asyncio.sleep(POLL / 2)
            status = json.loads(answer.read_text()).get("status")
        finally:  # an answer that comes after the App gave up is pruned later (STALE)
            answer.unlink(missing_ok=True)
        return status if status in ("sent", "failed") else "unknown"

    async def typing(self, key: str, on: bool) -> None:
        request_id = uuid.uuid4().hex
        request = {"id": request_id, "type": "typing", "key": key, "on": on}
        try:
            write_json(self.paths["out"] / f"{request_id}.json", request)
        except OSError as error:
            log.warning("typing not shown: %s", error)
