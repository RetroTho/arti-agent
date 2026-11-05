import asyncio
import json
import logging
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from arti import app as app_module
from arti import lock, log, secret
from arti.chat import Chat
from arti.store import Store
from arti.tools import ToolError
from plugins import safety
from plugins.safety import proc
from tests.support import (
    GROUP,
    HOME,
    OWNER,
    lane,
    make_app,
    make_config,
    message,
    run_message,
    settle,
    text_reply,
    tool_reply,
)

release = threading.Event()


def register(api):
    def slow(args, ctx):
        release.wait(5)
        return "done"

    api.tool("slow", "Slow.", {"properties": {}}, slow, review=None)


class AppTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        release.clear()
        self.app = None

    async def asyncTearDown(self):
        release.set()
        if self.app:
            await self.app.shutdown()
        self.tmp.cleanup()

    async def test_the_watchdog_is_fed_and_a_past_hang_is_reported(self):
        logs = Path(self.tmp.name) / "logs"
        logs.mkdir(mode=0o700)
        (logs / "hang.log").write_text("=== stacks ===")  # the last run hung
        with mock.patch.object(app_module.watchdog, "BEAT", 0.01):
            self.app = app = await make_app(self.tmp.name, cfg=make_config(hang_timeout=300))
            beaten = app.watchdog.last
            await asyncio.sleep(0.05)
        self.assertGreater(app.watchdog.last, beaten)  # the heartbeat runs
        self.assertEqual((logs / "hang.log.1").read_text(), "=== stacks ===")
        self.assertFalse((logs / "hang.log").exists())
        await settle(app)
        self.assertIn("I got stuck and restarted myself", app.channel.texts()[0])
        await app.shutdown()
        self.assertTrue(app.watchdog.stopped.is_set())  # a finished shutdown is not a hang

    async def test_no_heartbeat_while_the_channel_is_stuck(self):
        with mock.patch.object(app_module.watchdog, "BEAT", 0.01):
            self.app = app = await make_app(self.tmp.name, cfg=make_config(hang_timeout=300))
            app.channel.alive = lambda: False
            await asyncio.sleep(0.03)
            beaten = app.watchdog.last
            await asyncio.sleep(0.05)
        self.assertEqual(app.watchdog.last, beaten)

    async def test_admit_dedupes(self):
        self.app = app = await make_app(self.tmp.name, replies=[text_reply("hi")])
        inp = message("hello")
        await app.admit([inp])
        await app.admit([inp])  # the same event again: ignored
        await settle(app)
        self.assertEqual(app.channel.texts(), ["hi"])
        self.assertEqual(len(app.fake.requests), 1)

    async def test_commands_run_at_once_even_mid_turn(self):
        self.app = app = await make_app(
            self.tmp.name, plugins=[register], replies=[tool_reply("slow"), text_reply("ok")]
        )
        await app.admit([message("work")])
        await asyncio.sleep(0.2)
        await app.admit([message("/status")])
        for _ in range(100):
            await asyncio.sleep(0.01)
            if app.channel.texts():
                break
        [status] = app.channel.texts()
        self.assertIn("Busy: running slow", status)
        release.set()
        await settle(app)
        self.assertEqual(app.channel.texts()[-1], "ok")
        self.assertEqual(app.store.pending_inputs(), [])

    async def test_owner_only_commands(self):
        self.app = app = await make_app(self.tmp.name)
        sent = await run_message(app, "/model gpt-x", key=lane(OWNER), sender=OWNER)
        self.assertIn("works only in Owner's own chat", sent[0])
        sent = await run_message(app, "/model openai/gpt-x")
        self.assertEqual(app.conversations[HOME].route[:2], ("openai", "gpt-x"))
        self.assertIn("openai/gpt-x", sent[0])

    async def test_send_redacts_and_records_the_outbox(self):
        self.app = app = await make_app(self.tmp.name, plugins=[safety.register])
        secret.remember(["sup3r-secret-value"])
        self.assertEqual(await app.send(HOME, "key is sup3r-secret-value"), "sent")
        self.assertNotIn("sup3r-secret-value", app.channel.sent[-1][1])
        self.assertEqual(app.store.outbox_inflight(), [])

    async def test_a_raising_channel_makes_the_send_unknown(self):
        self.app = app = await make_app(self.tmp.name)

        async def broken(key, text):
            raise OSError("helper died")

        app.channel.send = broken
        self.assertEqual(await app.send(HOME, "hi"), "unknown")
        self.assertEqual(app.store.outbox_inflight(), [])

    async def test_recovery_after_a_crash(self):
        # Simulate a crash: a turn left open with an unanswered call, and a send in flight.
        data = Path(self.tmp.name)
        os.makedirs(data, exist_ok=True)
        store = Store(data / "state.db")
        session = store.current_session(HOME, ("anthropic", "fake-model", "high"))
        store.append(
            session["id"], HOME, {"role": "user", "content": [{"type": "text", "text": "go"}]}
        )
        call = {"type": "tool_use", "id": "c1", "name": "terminal", "input": {}}
        store.append(session["id"], HOME, {"role": "assistant", "content": [call]})
        store.update_session(session["id"], turn_open=True)
        store.outbox_begin(HOME, session["id"], "text", "half sent")
        store.close()

        self.app = app = await make_app(self.tmp.name, replies=[text_reply("I'm back")])
        await settle(app)
        history = app.conversations[HOME].messages
        result = history[2]["content"][0]
        self.assertEqual((result["tool_use_id"], result["content"][0]["unknown"]), ("c1", True))
        notice = history[3]["content"][0]["text"]
        self.assertIn("Arti restarted", notice)
        self.assertIn("half sent", notice)
        self.assertEqual(app.channel.texts(), ["I'm back"])
        self.assertEqual(app.store.outbox_inflight(), [])
        self.assertEqual(app.store.open_sessions(), [])

    async def test_recovery_admits_the_notice_before_it_changes_anything(self):
        data = Path(self.tmp.name)
        store = Store(data / "state.db")
        session = store.current_session(HOME, ("anthropic", "fake-model", "high"))
        call = {"type": "tool_use", "id": "c1", "name": "terminal", "input": {}}
        store.append(session["id"], HOME, {"role": "assistant", "content": [call]})
        store.update_session(session["id"], turn_open=True)
        app = await make_app(self.tmp.name, start=False)
        app.store = store
        try:
            with mock.patch.object(store, "update_session", side_effect=RuntimeError("crash")):
                with self.assertRaises(RuntimeError):
                    app.recover()
            app.recover()  # the restart after that crash
            notices = [i for i in store.pending_inputs() if i.kind == "notice"]
            self.assertEqual(len(notices), 1)
            self.assertEqual(store.open_sessions(), [])
        finally:
            store.close()

    async def test_prune_covers_every_contexts_media_and_cache(self):
        self.app = app = await make_app(self.tmp.name)
        app.context_dir("chat-0123")
        roots = app.prune_roots()
        for base in (app.data_dir, app.data_dir / "contexts" / "chat-0123"):
            self.assertIn(base / "media", roots)
            self.assertIn(base / "cache", roots)

    async def test_housekeeping_is_daily_by_the_wall_clock(self):
        # The event loop's clock stops while the Mac sleeps; housekeeping goes by time.time().
        now = [1_000_000.0]
        clock = mock.Mock(time=lambda: now[0])
        with (
            mock.patch.object(app_module, "time", clock),
            mock.patch.object(app_module, "WAKE_EVERY", 0.01),
            mock.patch.object(app_module.App, "housekeep", mock.AsyncMock()) as housekeep,
        ):
            self.app = await make_app(self.tmp.name)
            await asyncio.sleep(0.1)
            self.assertEqual(housekeep.await_count, 1)  # at start
            now[0] += 23 * 3600  # many wake-ups, less than a day of wall-clock time
            await asyncio.sleep(0.1)
            self.assertEqual(housekeep.await_count, 1)
            now[0] += 2 * 3600  # e.g. the Mac slept through the night
            await asyncio.sleep(0.1)
            self.assertEqual(housekeep.await_count, 2)
        await self.app.housekeep()  # the real one runs (prunes the store and files)

    async def test_housekeeping_unloads_only_lanes_quiet_for_a_while(self):
        self.app = app = await make_app(self.tmp.name)
        home = app.conversation(HOME)
        quiet, recent, running = (app.conversation(lane(f"U900{n}")) for n in range(3))
        old = app_module.time.time() - app_module.IDLE_UNLOAD - 1
        home.last_used = quiet.last_used = running.last_used = old
        running.running_call = "call_1"  # e.g. a tool call still running in its thread
        await app.housekeep()
        self.assertNotIn(quiet.key, app.conversations)
        for kept in (home, recent, running):  # home always; the others are not quiet enough
            self.assertIs(app.conversations[kept.key], kept)
        quiet.add(message("later", key=quiet.key, kind="note"))  # input counts as use
        self.assertGreater(quiet.last_used, old + 1)

    async def test_shutdown_stops_the_chats_at_once(self):
        self.app = app = await make_app(self.tmp.name)
        for key in (HOME, GROUP, "chat0003"):
            app.conversation(key)

        async def slow_stop(conv, drop_pending=True, fresh=False):
            await asyncio.sleep(0.3)

        started = asyncio.get_running_loop().time()
        with mock.patch.object(Chat, "stop", slow_stop):
            await app.shutdown()
        self.assertLess(asyncio.get_running_loop().time() - started, 0.8)

    async def test_shutdown_keeps_queued_input_and_leaves_the_turn_open(self):
        self.app = app = await make_app(
            self.tmp.name, plugins=[register], replies=[tool_reply("slow")]
        )
        await app.admit([message("work")])
        await asyncio.sleep(0.2)
        await app.admit([message("and this")])
        await app.shutdown()
        store = Store(Path(self.tmp.name) / "state.db")
        self.assertEqual([i.text for i in store.pending_inputs()], ["and this"])
        [session] = store.open_sessions()  # a restart resumes it with a notice
        result = store.history(session["id"])[-1]["content"][0]
        self.assertTrue(result["content"][0]["unknown"])
        store.close()

    async def test_a_send_cut_short_by_shutdown_is_reported_after_restart(self):
        self.app = app = await make_app(self.tmp.name)
        hang = asyncio.Event()

        async def stuck(key, text):
            await hang.wait()

        app.channel.send = stuck
        stopped = app.spawn(app.send(HOME, "stopped by /stop"))
        await asyncio.sleep(0.05)
        stopped.cancel()  # /stop: not shutting down, so the outcome is settled as unknown now
        await asyncio.sleep(0.05)
        self.assertEqual(app.store.outbox_inflight(), [])
        app.spawn(app.send(HOME, "cut by shutdown"))
        await asyncio.sleep(0.05)
        await app.shutdown()
        store = Store(Path(self.tmp.name) / "state.db")
        self.assertEqual([r["body"] for r in store.outbox_inflight()], ["cut by shutdown"])
        store.close()

    async def test_once_mode_leaves_queued_input_and_hooks_alone(self):
        started = []

        def plugin(api):
            api.on_start(lambda app: started.append(app))

        data = Path(self.tmp.name)
        store = Store(data / "state.db")
        store.admit([message("queued while Arti was down")])
        store.close()
        self.app = app = await make_app(self.tmp.name, plugins=[plugin], start=False)
        with mock.patch.object(lock, "acquire"):
            await app.start(serve=False)
        await asyncio.sleep(0.05)
        self.assertEqual((started, app.conversations, app.fake.requests), ([], {}, []))
        self.assertEqual(len(app.store.pending_inputs()), 1)

    async def test_headless_run_keeps_no_history(self):
        self.app = app = await make_app(self.tmp.name, replies=[text_reply("result")])
        self.assertEqual(await app.run_headless("do it"), "result")
        self.assertEqual(app.store.conversation_keys(), [])
        self.assertEqual(app.channel.sent, [])

    async def test_shutdown_cancels_headless_runs_started_from_threads(self):
        self.app = app = await make_app(
            self.tmp.name, plugins=[register], replies=[tool_reply("slow")]
        )
        outcome = []

        def worker():
            try:
                asyncio.run_coroutine_threadsafe(app.run_headless("work"), app.loop).result(10)
            except BaseException as error:
                outcome.append(type(error).__name__)

        thread = threading.Thread(target=worker)
        thread.start()
        await asyncio.sleep(0.3)
        await app.shutdown()
        await asyncio.to_thread(thread.join, 5)
        self.assertEqual(outcome, ["CancelledError"])

    async def test_a_failed_background_task_is_logged(self):
        self.app = app = await make_app(self.tmp.name)

        async def broken():
            raise RuntimeError("background bug")

        async def stopped():
            await asyncio.sleep(10)

        with self.assertLogs("arti.app", "ERROR") as logged:
            failed = app.spawn(broken())
            failed.set_name("broken-job")
            cancelled = app.spawn(stopped())
            await asyncio.sleep(0.01)
            cancelled.cancel()
            await asyncio.sleep(0.01)
            app.spawn(asyncio.sleep(0))  # a clean finish is not logged either
            await asyncio.sleep(0.01)
        self.assertEqual(len(logged.records), 1)
        self.assertIn("background task broken-job failed", logged.output[0])
        self.assertIn("RuntimeError: background bug", logged.output[0])
        self.assertNotIn(failed, app._tasks)
        self.assertNotIn(cancelled, app._tasks)

    async def test_a_failing_start_hook_does_not_stop_the_others(self):
        started = []

        def plugin(api):
            def broken(app):
                raise RuntimeError("start bug")

            async def broken_async(app):
                raise OSError("async start bug")

            api.on_start(broken)
            api.on_start(broken_async)
            api.on_start(lambda app: started.append(app))

        with self.assertLogs("arti.app", "ERROR") as logged:
            self.app = app = await make_app(
                self.tmp.name, plugins=[plugin], replies=[text_reply("yes")]
            )
        self.assertEqual(started, [app])
        text = "\n".join(logged.output)
        self.assertIn("start bug", text)
        self.assertIn("async start bug", text)
        self.assertEqual(await run_message(app, "still there?"), ["yes"])  # and Arti runs

    async def test_a_failing_shutdown_step_does_not_stop_the_rest(self):
        stopped = []

        def plugin(api):
            def broken(app):
                raise RuntimeError("stop bug")

            async def hangs(app):
                await asyncio.sleep(10)

            api.on_stop(broken)
            api.on_stop(hangs, timeout=0.05)
            api.on_stop(lambda app: stopped.append("plugin"))

        self.app = app = await make_app(self.tmp.name, plugins=[safety.register, plugin])
        channel_stop = app.channel.stop

        async def stop_channel():
            stopped.append("channel")
            await channel_stop()

        async def broken_close():
            raise OSError("provider close bug")

        app.channel.stop = stop_channel
        app.fake.aclose = broken_close
        with (
            mock.patch.object(proc, "kill_all", side_effect=OSError("kill bug")),
            self.assertLogs("arti.app", "ERROR") as logged,
        ):
            await app.shutdown()
        self.assertEqual(stopped, ["plugin", "channel"])
        text = "\n".join(logged.output)
        for error in ("stop bug", "TimeoutError", "kill bug", "provider close bug"):
            self.assertIn(error, text)
        with self.assertRaises(Exception):  # the store was still closed at the end
            app.store.conversation_keys()

    async def test_deliver_sends_and_records_a_note(self):
        self.app = app = await make_app(self.tmp.name, replies=[text_reply("seen")])
        await app.deliver(HOME, "Your daily summary")
        await settle(app)
        self.assertEqual(app.channel.texts(), ["Your daily summary"])
        self.assertEqual(app.fake.requests, [])  # a note starts no turn
        await run_message(app, "what did you send?")
        first = app.fake.requests[0]["messages"][0]["content"]
        self.assertIn("Your daily summary", first[0]["text"])


HIDDEN = "hush-hush-value-42"


def hiding(api):
    """A redactor that hides HIDDEN, and tools whose result and error carry it."""

    def text(value: str) -> str:
        return value.replace(HIDDEN, "[HIDDEN]")

    def leak_error(args, ctx):
        raise ToolError(f"failed on {HIDDEN}")

    api.redactor(text, lambda value: json.loads(text(json.dumps(value))))
    empty = {"properties": {}}
    api.tool("leak", "Leak.", empty, lambda args, ctx: f"found {HIDDEN}", review=None)
    api.tool("leak_error", "Leak an error.", empty, leak_error, review=None)


class RedactorHookTest(unittest.IsolatedAsyncioTestCase):
    """The core writes everything through the registered redactor (api.redactor)."""

    async def test_inputs_history_tool_results_errors_sends_and_logs_go_through_it(self):
        self.addCleanup(secret.set_redactor, secret._redactor)
        replies = [tool_reply("leak"), tool_reply("leak_error"), text_reply(f"it is {HIDDEN}")]
        app = await make_app(tempfile.mkdtemp(), plugins=[hiding], replies=replies)
        self.addAsyncCleanup(app.shutdown)
        self.assertEqual(await run_message(app, f"mine is {HIDDEN}"), ["it is [HIDDEN]"])
        history = json.dumps(app.store.history(app.conversations[HOME].session_id))
        for seen in (history, json.dumps(app.fake.requests), json.dumps(app.channel.sent)):
            self.assertNotIn(HIDDEN, seen)
        for part in ("mine is [HIDDEN]", "found [HIDDEN]", "failed on [HIDDEN]", "it is [HIDDEN]"):
            self.assertIn(part, history)
        record = logging.LogRecord("arti.x", logging.INFO, __file__, 1, HIDDEN, None, None)
        self.assertEqual(log.RedactingFormatter().format(record), "[HIDDEN]")


if __name__ == "__main__":
    unittest.main()
