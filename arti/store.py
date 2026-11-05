"""The SQLite database, state.db: everything Arti must not forget across a restart.

Each table is described in SCHEMA below. Sessions and history belong to a conversation key (the
channel's, channel.py). The inbox and outbox are what make
"exactly once in, at most once out" work: an input is in the inbox before Arti acts on it and is
marked done in the same transaction that puts it into history (`append`), and a send is recorded
as `sending` before the channel is called, so after a crash it is reported, never repeated.

Every method is synchronous and safe from any thread. Two connections: `db`, guarded by `lock`,
for every write and the quick reads the event loop makes; and `reader`, a read-only one with its
own lock, for the reads that can be slow (full-text search, reading the archive), which tools
make from worker threads. WAL lets the reader see the last commit while a write goes on, so a
slow search never holds the loop's lock. WAL with synchronous=FULL so a
committed row survives a crash or power loss. Images in messages are stored as file paths, never
as base64.
"""

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = """
-- A conversation's stored state: its model route, compaction summary and point (start_id), the
-- tools' state, and whether a turn was running (turn_open: recovered after a crash). /new starts
-- a fresh session; `conversations` says which session of each key is current.
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, key TEXT NOT NULL, provider TEXT, model TEXT, effort TEXT,
  summary TEXT NOT NULL DEFAULT '', start_id INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL DEFAULT '{}', turn_open INTEGER NOT NULL DEFAULT 0, created REAL);
CREATE TABLE IF NOT EXISTS conversations (key TEXT PRIMARY KEY, session TEXT NOT NULL);
-- History: every message of every session (`body` is its JSON, `text` what search indexes).
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, session TEXT NOT NULL, key TEXT NOT NULL, role TEXT NOT NULL,
  body TEXT NOT NULL, text TEXT NOT NULL, ts REAL);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session, id);
CREATE INDEX IF NOT EXISTS messages_key ON messages(key, id);
-- The full-text index over `messages.text`, for recall (`search`).
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(text, content='messages',
  content_rowid='id');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text); END;
-- Inbound events (store.Input), deduplicated by event_id: pending until they are in history.
CREATE TABLE IF NOT EXISTS inbox (
  seq INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL,
  body TEXT NOT NULL, state TEXT NOT NULL, ts REAL);
-- Sends: `sending` before the channel is called, then sent, failed or unknown.
CREATE TABLE IF NOT EXISTS outbox (
  id INTEGER PRIMARY KEY, key TEXT, session TEXT, kind TEXT, body TEXT, state TEXT, ts REAL);
-- Small named values: the channel's read cursor, plugins' state and positions.
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""

SESSION_FIELDS = {"provider", "model", "effort", "summary", "start_id", "state", "turn_open"}
SEARCH_TEXT_CHARS = 20_000


@dataclass
class Input:
    """One inbound event: a message to answer, a notice that starts a turn, or a context note."""

    event_id: str  # stable id from the channel (from the platform's message id); the dedupe key
    key: str  # the conversation key (the channel's)
    kind: str  # "message" | "notice" | "note"
    text: str = ""
    attachments: list[str] = field(default_factory=list)  # file paths: images, audio, other
    sender: str = ""  # handle; "" for host-made inputs
    sender_name: str = ""  # how the sender is shown ("" in the owner's own chat)
    guid: str = ""  # platform message id (reply threading)
    ts: float = 0.0  # when it was sent (unix time)
    seq: int = 0  # inbox row id, set by the store
    due: float = 0.0  # a delayed notice (a wait) is held until this unix time; 0 = at once


def searchable_text(message: dict) -> str:
    """The visible text of a message, for the full-text index: text, tool names, result text."""
    parts = []
    for block in message["content"]:
        if block["type"] == "text":
            parts.append(block["text"])
        elif block["type"] == "tool_use":
            parts.append(f"[{block['name']}]")
        elif block["type"] == "tool_result":
            parts += [b["text"] for b in block["content"] if b["type"] == "text"]
    return "\n".join(parts)[:SEARCH_TEXT_CHARS]


def _session(row: sqlite3.Row) -> dict:
    return {**dict(row), "state": json.loads(row["state"])}


class Store:
    def __init__(self, path: Path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        # synchronous=FULL makes every commit (one per history append, on the event loop) wait
        # for the disk itself. That is deliberate: persist before act (docs/architecture.md).
        for pragma in ("journal_mode=WAL", "synchronous=FULL"):
            self.db.execute(f"PRAGMA {pragma}")
        self.db.executescript(SCHEMA)
        # The slow reads' own connection (see the module docstring); it can never write.
        self.read_lock = threading.Lock()
        self.reader = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.reader.row_factory = sqlite3.Row
        self.reader.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        with self.read_lock:
            self.reader.close()
        with self.lock:
            self.db.close()

    @contextmanager
    def _write(self):
        """One transaction: committed when the block ends, rolled back on any error."""
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")

    def _query(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def _slow_query(self, sql: str, args=()) -> list[sqlite3.Row]:
        """A read that may take a while, on the reader: it never waits for or blocks `lock`."""
        with self.read_lock:
            return self.reader.execute(sql, args).fetchall()

    # --- sessions ---------------------------------------------------------------------------

    def current_session(self, key: str, route: tuple[str, str, str]) -> dict:
        """The conversation's current session, created with `route` (provider, model, effort) if
        none."""
        rows = self._query(
            "SELECT s.* FROM conversations c JOIN sessions s ON s.id = c.session WHERE c.key = ?",
            (key,),
        )
        return _session(rows[0]) if rows else self.new_session(key, route)

    def new_session(self, key: str, route: tuple[str, str, str]) -> dict:
        session_id = uuid.uuid4().hex
        with self._write() as db:
            db.execute(
                "INSERT INTO sessions (id, key, provider, model, effort, created)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, key, *route, time.time()),
            )
            db.execute("INSERT OR REPLACE INTO conversations VALUES (?, ?)", (key, session_id))
        return self.session(session_id)

    def update_session(self, session_id: str, **fields) -> None:
        if not fields.keys() <= SESSION_FIELDS:
            raise ValueError(f"unknown session fields: {sorted(fields.keys() - SESSION_FIELDS)}")
        if "state" in fields:
            fields["state"] = json.dumps(fields["state"])
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._write() as db:
            db.execute(
                f"UPDATE sessions SET {assignments} WHERE id = ?", (*fields.values(), session_id)
            )

    def session(self, session_id: str) -> dict | None:
        rows = self._query("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return _session(rows[0]) if rows else None

    def conversation_keys(self) -> list[str]:
        """The key of every conversation that has had a session."""
        return [r["key"] for r in self._query("SELECT DISTINCT key FROM sessions ORDER BY key")]

    def open_sessions(self) -> list[dict]:
        """Sessions whose turn was running when Arti stopped."""
        return [_session(r) for r in self._query("SELECT * FROM sessions WHERE turn_open = 1")]

    # --- history ----------------------------------------------------------------------------

    def history(self, session_id: str) -> list[dict]:
        """The active history: the session's messages from its compaction point on."""
        rows = self._query(
            "SELECT m.id, m.body FROM messages m JOIN sessions s ON s.id = m.session"
            " WHERE m.session = ? AND m.id >= s.start_id ORDER BY m.id",
            (session_id,),
        )
        return [{**json.loads(row["body"]), "id": row["id"]} for row in rows]

    def append(self, session_id: str, key: str, message: dict, done=()) -> int:
        """Store a message and mark the inbox rows `done` it came from, in one transaction."""
        stored = {k: v for k, v in message.items() if k != "id"}
        body, text = json.dumps(stored), searchable_text(stored)
        with self._write() as db:
            row = db.execute(
                "INSERT INTO messages (session, key, role, body, text, ts)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, key, stored["role"], body, text, time.time()),
            )
            db.executemany("UPDATE inbox SET state = 'done' WHERE seq = ?", [(s,) for s in done])
            return row.lastrowid

    # --- inbox ------------------------------------------------------------------------------

    def admit(self, inputs: list[Input]) -> list[Input]:
        """Record inputs not seen before (by event_id), in one transaction. Returns the new
        inputs, with their `seq` set; duplicates are left out."""
        new = []
        with self._write() as db:
            for inp in inputs:
                body = asdict(inp)
                for name in ("event_id", "key", "kind", "seq"):
                    del body[name]
                row = db.execute(
                    "INSERT OR IGNORE INTO inbox (event_id, key, kind, body, state, ts)"
                    " VALUES (?, ?, ?, ?, 'pending', ?)",
                    (inp.event_id, inp.key, inp.kind, json.dumps(body), time.time()),
                )
                if row.rowcount:
                    inp.seq = row.lastrowid
                    new.append(inp)
        return new

    def pending_inputs(self) -> list[Input]:
        rows = self._query("SELECT * FROM inbox WHERE state = 'pending' ORDER BY seq")
        return [
            Input(r["event_id"], r["key"], r["kind"], seq=r["seq"], **json.loads(r["body"]))
            for r in rows
        ]

    def mark_inputs(self, seqs, state: str) -> None:
        with self._write() as db:
            db.executemany("UPDATE inbox SET state = ? WHERE seq = ?", [(state, s) for s in seqs])

    # --- outbox -----------------------------------------------------------------------------

    def outbox_begin(self, key: str, session: str, kind: str, body: str) -> int:
        """Record a send as `sending` before the channel is called (so it is never repeated)."""
        with self._write() as db:
            return db.execute(
                "INSERT INTO outbox (key, session, kind, body, state, ts)"
                " VALUES (?, ?, ?, ?, 'sending', ?)",
                (key, session, kind, body, time.time()),
            ).lastrowid

    def outbox_end(self, row_id: int, state: str) -> None:
        with self._write() as db:
            db.execute("UPDATE outbox SET state = ? WHERE id = ?", (state, row_id))

    def outbox_inflight(self) -> list[dict]:
        return [dict(r) for r in self._query("SELECT * FROM outbox WHERE state = 'sending'")]

    # --- kv ---------------------------------------------------------------------------------

    def kv_get(self, key: str, default=None):
        rows = self._query("SELECT value FROM kv WHERE key = ?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def kv_set(self, key: str, value) -> None:
        with self._write() as db:
            db.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value))
            )

    # --- the archive ------------------------------------------------------------------------

    def messages_since(self, key: str, after_id: int, limit: int = 1000) -> list[dict]:
        """A conversation's stored messages after `after_id`, oldest first: every session, and
        also what compaction took out of the active history (the archive keeps it)."""
        rows = self._slow_query(
            "SELECT id, ts, body FROM messages WHERE key = ? AND id > ? ORDER BY id LIMIT ?",
            (key, after_id, limit),
        )
        return [{**json.loads(row["body"]), "id": row["id"], "ts": row["ts"]} for row in rows]

    # --- recall (a plugin's search tool) ----------------------------------------------------

    def search(self, query: str, keys: list[str], limit: int) -> list[dict]:
        """Full-text search over the messages of the conversations `keys`, best matches first.

        A query that is not valid FTS5 syntax is retried as one quoted phrase.
        """
        if not keys:
            return []
        sql = (
            "SELECT m.id, m.session, m.key, m.role, m.ts,"
            " snippet(messages_fts, 0, '[', ']', ' ... ', 24) AS snippet"
            " FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid"
            f" WHERE messages_fts MATCH ? AND m.key IN ({','.join('?' * len(keys))})"
            " ORDER BY rank LIMIT ?"
        )
        try:
            rows = self._slow_query(sql, (query, *keys, limit))
        except sqlite3.OperationalError:
            rows = self._slow_query(sql, ('"' + query.replace('"', '""') + '"', *keys, limit))
        return [dict(r) for r in rows]

    def read(self, session_id: str, around_id: int | None, limit: int) -> list[dict]:
        """Up to `limit` messages of a session around `around_id` (the newest when None)."""
        select = "SELECT id, role, text, ts FROM messages WHERE session = ?"
        if around_id is None:
            rows = self._slow_query(f"{select} ORDER BY id DESC LIMIT ?", (session_id, limit))
        else:
            half = max(1, limit // 2)
            before = f"{select} AND id < ? ORDER BY id DESC LIMIT ?"
            after = f"{select} AND id >= ? ORDER BY id LIMIT ?"
            rows = self._slow_query(before, (session_id, around_id, half))
            rows += self._slow_query(after, (session_id, around_id, limit - half))
        return sorted((dict(r) for r in rows), key=lambda r: r["id"])

    def prune(self, days: int) -> None:
        """Forget handled inbox rows and finished sends older than `days`. History is kept."""
        cutoff = time.time() - days * 86400
        with self._write() as db:
            db.execute("DELETE FROM inbox WHERE state != 'pending' AND ts < ?", (cutoff,))
            db.execute("DELETE FROM outbox WHERE state != 'sending' AND ts < ?", (cutoff,))
