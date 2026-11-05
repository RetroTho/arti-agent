import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from arti.store import Input, Store

ROUTE = ("anthropic", "m", "high")
DAY = 86400


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "state.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_admit_dedupes(self):
        first = self.store.admit([Input("e1", "c", "message", text="hi")])
        again = self.store.admit([Input("e1", "c", "message", text="hi"), Input("e2", "c", "note")])
        self.assertEqual([i.event_id for i in first], ["e1"])
        self.assertEqual([i.event_id for i in again], ["e2"])
        pending = self.store.pending_inputs()
        self.assertEqual([(i.event_id, i.text) for i in pending], [("e1", "hi"), ("e2", "")])

    def test_append_marks_inputs_done_in_the_same_transaction(self):
        [inp] = self.store.admit([Input("e1", "c", "message", text="hi")])
        session = self.store.current_session("c", ROUTE)
        message = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        self.store.append(session["id"], "c", message, done=[inp.seq])
        self.assertEqual(self.store.pending_inputs(), [])
        self.assertEqual(self.store.history(session["id"])[0]["content"], message["content"])

    def test_a_failed_append_leaves_the_input_pending(self):
        [inp] = self.store.admit([Input("e1", "c", "message")])
        with self.assertRaises(KeyError):
            self.store.append("s", "c", {"content": []}, done=[inp.seq])  # no role
        self.assertEqual(len(self.store.pending_inputs()), 1)

    def test_sessions_compaction_point_and_state(self):
        session = self.store.current_session("c", ROUTE)
        self.assertEqual(self.store.current_session("c", ROUTE)["id"], session["id"])
        ids = [
            self.store.append(
                session["id"], "c", {"role": "user", "content": [{"type": "text", "text": str(n)}]}
            )
            for n in range(3)
        ]
        self.store.update_session(session["id"], start_id=ids[1], summary="s", state={"todo": [1]})
        again = self.store.current_session("c", ROUTE)
        self.assertEqual((again["summary"], again["state"]), ("s", {"todo": [1]}))
        self.assertEqual([m["id"] for m in self.store.history(session["id"])], ids[1:])
        fresh = self.store.new_session("c", ROUTE)
        self.assertEqual(self.store.current_session("c", ROUTE)["id"], fresh["id"])
        self.assertEqual(fresh["state"], {})

    def test_outbox_and_search(self):
        row = self.store.outbox_begin("c", "s", "text", "hello")
        self.assertEqual([r["id"] for r in self.store.outbox_inflight()], [row])
        self.store.outbox_end(row, "sent")
        self.assertEqual(self.store.outbox_inflight(), [])
        session = self.store.current_session("c", ROUTE)
        text = {"role": "user", "content": [{"type": "text", "text": "the purple elephant"}]}
        self.store.append(session["id"], "c", text)
        self.assertEqual(len(self.store.search("elephant", ["c"], 5)), 1)
        self.assertEqual(self.store.search("elephant", ["other"], 5), [])
        self.assertEqual(len(self.store.search('purple "ele', ["c"], 5)), 0)  # bad syntax: no crash
        self.assertEqual(len(self.store.read(session["id"], None, 5)), 1)

    def age(self, table: str, days: float) -> None:
        """Make the rows of `table` `days` old."""
        with self.store.lock:
            self.store.db.execute(f"UPDATE {table} SET ts = ?", (time.time() - days * DAY,))

    def test_prune_forgets_handled_rows_after_days_but_never_pending_work(self):
        inputs = [Input(f"e{n}", "c", "message") for n in range(3)]
        done, cancelled, pending = self.store.admit(inputs)
        self.store.mark_inputs([done.seq], "done")
        self.store.mark_inputs([cancelled.seq], "cancelled")
        sent = self.store.outbox_begin("c", "s", "text", "sent")
        self.store.outbox_end(sent, "sent")
        sending = self.store.outbox_begin("c", "s", "text", "in flight")
        self.age("inbox", 8)
        self.age("outbox", 8)
        self.store.prune(7)
        with self.store.lock:
            inbox = [r[0] for r in self.store.db.execute("SELECT event_id FROM inbox ORDER BY seq")]
            outbox = [r[0] for r in self.store.db.execute("SELECT id FROM outbox")]
        self.assertEqual(inbox, ["e2"])  # pending: kept however old
        self.assertEqual(outbox, [sending])  # a send in flight is kept for recovery
        self.assertEqual([i.seq for i in self.store.pending_inputs()], [pending.seq])
        self.store.prune(9)  # younger than the cutoff: nothing more goes
        self.assertEqual(len(self.store.pending_inputs()), 1)

    def test_slow_reads_have_their_own_connection(self):
        def note(text):
            return {"role": "user", "content": [{"type": "text", "text": text}]}

        def in_thread(work):
            out = []
            thread = threading.Thread(target=lambda: out.append(work()), daemon=True)
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive(), "blocked on the other lock")
            return out[0]

        session = self.store.current_session("c", ROUTE)
        self.store.append(session["id"], "c", note("pelican one"))
        with self.store.read_lock:  # a slow search runs on a worker thread...
            # ...while the loop still appends and reads.
            in_thread(lambda: self.store.append(session["id"], "c", note("pelican two")))
            self.assertEqual(len(in_thread(lambda: self.store.history(session["id"]))), 2)
        with self.store._write() as db:  # the loop mid-write: a search still runs, and sees
            db.execute(  # only what is committed
                "INSERT INTO messages (session, key, role, body, text) VALUES (?, ?, ?, ?, ?)",
                (session["id"], "c", "user", "{}", "pelican three"),
            )
            hits = in_thread(lambda: self.store.search("pelican", ["c"], 10))
            self.assertEqual(len(hits), 2)
            self.assertEqual(len(in_thread(lambda: self.store.messages_since("c", 0))), 2)
        self.assertEqual(len(self.store.search("pelican", ["c"], 10)), 3)
        with self.assertRaises(sqlite3.OperationalError):  # the reader can never write
            self.store.reader.execute("DELETE FROM messages")

    def test_a_send_cut_short_by_a_crash_is_found_and_settled_as_unknown(self):
        path = Path(self.tmp.name) / "state.db"
        row = self.store.outbox_begin("c", "s", "text", "half sent")
        self.store.close()  # the crash: outbox_end never ran
        self.store = Store(path)
        self.assertEqual([r["body"] for r in self.store.outbox_inflight()], ["half sent"])
        self.store.outbox_end(row, "unknown")  # what App.recover does after its notice
        self.assertEqual(self.store.outbox_inflight(), [])
        with self.store.lock:
            state = self.store.db.execute("SELECT state FROM outbox WHERE id = ?", (row,))
            self.assertEqual(state.fetchone()[0], "unknown")


if __name__ == "__main__":
    unittest.main()
