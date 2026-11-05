import asyncio
import tempfile
import threading
import time
import unittest
from unittest import mock

from arti import media, tools, turn
from arti.providers import ProviderError
from arti.tools import UnknownOutcome
from tests.support import (
    HOME,
    make_app,
    message,
    run_message,
    settle,
    text_reply,
    tool_reply,
)

calls: list[dict] = []
release = threading.Event()
waiting = threading.Event()


def register(api):
    def echo(args, ctx):
        calls.append(args)
        return f"echo {args['text']}"

    def slow(args, ctx):
        release.wait(5)
        return "slow done"

    def hang(args, ctx):  # ignores its cancel
        release.wait(10)
        return "never seen"

    def stoppable(args, ctx):
        ctx.cancel.wait(10)
        return "stopped cleanly"

    def react(args, ctx):
        ctx.mark_answered()
        return "reacted"

    def maybe(args, ctx):
        raise UnknownOutcome("the order may have been placed")

    def wait(args, ctx):
        waiting.set()
        return ctx.wait(5)

    def raw_image(args, ctx):
        return [{"type": "image", "media_type": "image/png", "data": "QUJD" * 1000}]

    text = {"properties": {"text": {"type": "string"}}, "required": ["text"]}
    api.tool("echo", "Echo.", text, echo, review=None)
    api.tool("slow", "Slow.", {"properties": {}}, slow, review=None)
    api.tool("hang", "Hang.", {"properties": {}}, hang, review=None, timeout=0.2)
    api.tool("stoppable", "Stoppable.", {"properties": {}}, stoppable, review=None, timeout=0.2)
    api.tool("react", "React.", {"properties": {}}, react, review=None)
    api.tool("maybe", "Maybe.", {"properties": {}}, maybe, review=None)
    api.tool("wait", "Wait.", {"properties": {}}, wait, review=None)
    api.tool("raw_image", "Raw image.", {"properties": {}}, raw_image, review=None)


def results(request) -> list[dict]:
    """The tool_result blocks of the last user message of a recorded request."""
    return [b for b in request["messages"][-1]["content"] if b.get("type") == "tool_result"]


class TurnTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        calls.clear()
        release.clear()
        waiting.clear()

    async def asyncTearDown(self):
        release.set()
        await self.app.shutdown()
        self.tmp.cleanup()

    async def start(self, *replies):
        self.app = await make_app(self.tmp.name, plugins=[register], replies=list(replies))
        return self.app

    async def test_tool_loop_runs_calls_and_sends_text(self):
        app = await self.start(tool_reply("echo", {"text": "a"}, text="on it"), text_reply("done"))
        sent = await run_message(app, "hello")
        self.assertEqual(sent, ["on it", "done"])
        self.assertEqual(calls, [{"text": "a"}])
        [result] = results(app.fake.requests[1])
        self.assertEqual(result["content"][0]["text"], "echo a")
        first = app.fake.requests[0]["messages"][0]["content"][0]["text"]
        self.assertRegex(first, r"^\[\w{3} \w{3} \d+ \d+:\d\d [AP]M\] hello$")
        self.assertIn("echo", app.fake.requests[0]["tools"])

    async def test_the_system_prompt_is_built_once_per_conversation(self):
        # A memory saved mid-conversation is already in the history; changing the system prompt
        # for it would break the provider's prompt cache, so the prompt stays as it was.
        app = await self.start(
            tool_reply("echo", {"text": "a"}), text_reply("one"), text_reply("two")
        )
        prompt_text = "first version"
        with mock.patch("arti.prompt.build", side_effect=lambda conv: prompt_text) as build:
            await run_message(app, "hello")
            prompt_text = "second version"
            await run_message(app, "again")
        self.assertEqual(build.call_count, 1)
        self.assertEqual({r["system"] for r in app.fake.requests}, {"first version"})
        app.conversation(HOME).compact("summary", 1)  # a compaction (or /new) rebuilds it
        self.assertIsNone(app.conversation(HOME).system)

    async def test_unknown_tool_and_bad_arguments_are_error_results(self):
        app = await self.start(tool_reply("nope"), tool_reply("echo", {"x": 1}), text_reply("ok"))
        await run_message(app, "hi")
        self.assertIn("unknown tool", results(app.fake.requests[1])[0]["content"][0]["text"])
        self.assertIn("$.text: required", results(app.fake.requests[2])[0]["content"][0]["text"])
        self.assertEqual(calls, [])

    async def test_steering_skips_stale_calls_and_joins_the_turn(self):
        app = await self.start(tool_reply("slow"), _two_calls(), text_reply("final"))
        await app.admit([message("first")])
        for _ in range(100):  # wait until the slow tool runs
            await asyncio.sleep(0.01)
            if app.conversations[HOME].running_tool == "slow":
                break
        await app.admit([message("actually, stop that")])
        release.set()
        await settle(app)
        # The second reply was asked for after the new message arrived, so its calls run.
        second = app.fake.requests[1]["messages"][-1]["content"]
        self.assertEqual(second[0]["type"], "tool_result")
        self.assertIn("actually, stop that", second[1]["text"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(app.channel.texts(), ["final"])

    async def test_a_message_during_the_request_skips_that_replys_calls(self):
        app = None

        def reply_after_message(messages):
            app.conversations[HOME].add(message("wait!"))
            return tool_reply("echo", {"text": "x"})

        app = await self.start(reply_after_message, text_reply("ok"))
        await run_message(app, "go")
        await settle(app)
        self.assertEqual(calls, [])
        content = app.fake.requests[1]["messages"][-1]["content"]
        self.assertIn("not run: something new arrived", content[0]["content"][0]["text"])
        self.assertIn("wait!", content[1]["text"])

    async def test_a_message_during_a_slow_drain_skips_the_replys_calls(self):
        app = None
        drain = media.input_blocks

        def slow_blocks(app_, inp, cancel, folder):  # a slow transcription, in its thread
            if inp.text == "go":
                app.loop.call_soon_threadsafe(app.conversations[HOME].add, message("stop!"))
                time.sleep(0.1)
            return drain(app_, inp, cancel, folder)

        app = await self.start(tool_reply("echo", {"text": "x"}), text_reply("ok"))
        with mock.patch.object(media, "input_blocks", slow_blocks):
            await run_message(app, "go")
            await settle(app)
        self.assertEqual(calls, [])
        first = app.fake.requests[0]["messages"][-1]["content"]
        self.assertNotIn("stop!", str(first))
        content = app.fake.requests[1]["messages"][-1]["content"]
        self.assertIn("not run: something new arrived", content[0]["content"][0]["text"])
        self.assertIn("stop!", content[1]["text"])

    async def test_silent_without_an_answer_is_nudged_at_most_twice(self):
        app = await self.start(
            text_reply("[SILENT]"), text_reply("[SILENT]"), text_reply("[SILENT]")
        )
        sent = await run_message(app, "hi")
        self.assertEqual(sent, [])
        self.assertEqual(len(app.fake.requests), 3)
        self.assertIn("not answered", app.fake.requests[1]["messages"][-1]["content"][-1]["text"])

    async def test_a_reply_of_unknown_outcome_is_not_nudged_into_a_second_one(self):
        app = await self.start(text_reply("the answer"), text_reply("the answer (again)"))
        app.channel.statuses = ["unknown"]
        await run_message(app, "question")
        self.assertEqual(len(app.fake.requests), 1)

    async def test_a_silence_is_not_nudged_again_in_a_later_turn(self):
        app = await self.start(*[text_reply("[SILENT]")] * 4)
        await run_message(app, "thanks!")  # nudged twice, then left silent on purpose
        await app.admit([message("[Arti: your wait ended]", kind="notice")])
        await settle(app)
        self.assertEqual(len(app.fake.requests), 4)  # the notice turn: one request, no nudge

    async def test_a_call_in_a_group_is_nudged_naming_the_caller(self):
        app = await self.start(text_reply("[SILENT]"), text_reply("hi"))
        group = "chat777|U3"
        self.assertEqual(await run_message(app, "@Arti hey", group, "U3"), ["hi"])
        self.assertIn("answered U3's", str(app.fake.requests[1]["messages"][-1]))

    async def test_silent_after_a_reaction_is_not_nudged(self):
        app = await self.start(tool_reply("react"), text_reply("[SILENT]"))
        self.assertEqual(await run_message(app, "thanks"), [])
        self.assertEqual(len(app.fake.requests), 2)

    async def test_timeout_gives_an_unknown_result_and_the_turn_goes_on(self):
        app = await self.start(tool_reply("hang"), text_reply("ok"))
        with mock.patch.object(tools, "CANCEL_GRACE", 0.1):
            self.assertEqual(await run_message(app, "go"), ["ok"])
        [result] = results(app.fake.requests[1])
        self.assertTrue(result["content"][0]["unknown"])
        self.assertIn("timed out", result["content"][0]["text"])

    async def test_a_tool_that_stops_within_the_grace_keeps_its_result(self):
        app = await self.start(tool_reply("stoppable"), text_reply("ok"))
        await run_message(app, "go")
        [result] = results(app.fake.requests[1])
        self.assertEqual(result["content"][0]["text"], "stopped cleanly")
        self.assertNotIn("unknown", result["content"][0])

    async def test_unknown_outcome_is_flagged(self):
        app = await self.start(tool_reply("maybe"), text_reply("ok"))
        await run_message(app, "buy")
        [result] = results(app.fake.requests[1])
        self.assertTrue(result["content"][0]["unknown"])
        self.assertIn("may have been placed", result["content"][0]["text"])

    async def test_failed_send_ends_the_turn_with_a_note_not_a_second_reply(self):
        app = await self.start(
            tool_reply("echo", {"text": "a"}, text="working"), text_reply("next")
        )
        app.channel.statuses = ["failed"]
        await run_message(app, "go")
        self.assertEqual(calls, [])
        history = app.conversations[HOME].messages
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "user", "user"])
        self.assertIn(
            "not run: your message failed to send", history[2]["content"][0]["content"][0]["text"]
        )
        self.assertIn("failed", history[3]["content"][0]["text"])
        self.assertFalse(app.store.open_sessions())

    @mock.patch.object(turn, "RETRY_DELAYS", (0, 0, 0))
    async def test_provider_error_is_retried_with_backoff_then_reported(self):
        def busy():
            return ProviderError("busy", "server", retryable=True)

        app = await self.start(busy(), busy(), busy(), text_reply("fine"))
        self.assertEqual(await run_message(app, "a"), ["fine"])  # the third retry succeeds
        self.assertEqual(len(app.fake.requests), 4)
        app.fake.replies = [busy(), busy(), busy(), busy(), text_reply("never asked")]
        sent = await run_message(app, "b")
        self.assertEqual(len(app.fake.requests), 8)  # one request and three retries, no more
        self.assertEqual(len(sent), 1)
        self.assertIn("busy", sent[0])
        auth = ProviderError("The Claude login expired: run `claude auth login`.", "auth")
        app.fake.replies = [auth]
        sent = await run_message(app, "c")
        self.assertEqual(len(app.fake.requests), 9)  # not retryable: asked once
        self.assertEqual(len(sent), 1)
        self.assertIn("claude auth login", sent[0])

    async def test_retries_wait_ever_longer_and_honour_retry_after(self):
        def busy(retry_after=0.0):
            return ProviderError("busy", "server", retryable=True, retry_after=retry_after)

        app = await self.start(busy(), busy(20), busy(), text_reply("fine"))
        slept, real_sleep = [], asyncio.sleep

        async def sleep(seconds, *args, **kwargs):  # records the retry waits and skips them
            if 1 <= seconds <= turn.MAX_RETRY_AFTER:  # (housekeeping's hourly sleep is longer)
                slept.append(seconds)
                seconds = 0
            return await real_sleep(seconds, *args, **kwargs)

        with mock.patch.object(asyncio, "sleep", sleep):
            self.assertEqual(await run_message(app, "a"), ["fine"])
        self.assertEqual(slept, [2, 20, 30])  # the provider's 20 s beats the schedule's 10 s
        app.fake.replies = [busy(turn.MAX_RETRY_AFTER + 1), text_reply("never asked")]
        sent = await run_message(app, "b")
        self.assertIn("busy", sent[0])  # a wait longer than MAX_RETRY_AFTER is not retried

    async def test_a_model_that_does_not_answer_in_time_fails_the_turn(self):
        app = await self.start()
        app.config["agent"]["request_timeout"] = 0.2

        async def hangs(messages, **options):
            await asyncio.sleep(10)

        with mock.patch.object(app.fake, "complete", hangs):
            sent = await run_message(app, "hello")
            self.assertEqual(
                sent,
                ["Sorry, I hit a problem and stopped: The model did not answer within 0 seconds."],
            )
            with self.assertRaisesRegex(turn.TurnError, "did not answer within 0 seconds"):
                await app.run_headless("work")  # a headless run raises it to its caller

    async def test_context_too_long_compacts_once_and_asks_again(self):
        def too_long():
            return ProviderError("prompt is too long", "context_length")

        app = await self.start(text_reply("r1"), text_reply("r2"))
        await run_message(app, "one")
        await run_message(app, "two")
        app.fake.replies = [too_long(), text_reply("they said one and two"), text_reply("three")]
        self.assertEqual(await run_message(app, "three"), ["three"])
        failed, summary, retry = app.fake.requests[2:]
        self.assertEqual(summary["tools"], [])  # the compaction's own call
        self.assertIn("they said one and two", str(retry["messages"]))
        self.assertNotIn("r1", str(retry["messages"]))  # summarised away, not sent again
        self.assertIn("r1", str(failed["messages"]))
        # Still too long after compacting: reported, not compacted again in a loop.
        app.fake.replies = [too_long(), text_reply("summary two"), too_long(), text_reply("x")]
        sent = await run_message(app, "four")
        self.assertEqual(sent, ["Sorry, I hit a problem and stopped: prompt is too long"])
        self.assertEqual(len(app.fake.requests), 8)
        self.assertEqual(app.fake.replies, [text_reply("x")])

    async def test_context_too_long_with_nothing_to_compact_is_reported(self):
        app = await self.start(ProviderError("prompt is too long", "context_length"))
        sent = await run_message(app, "a huge first message")
        self.assertEqual(sent, ["Sorry, I hit a problem and stopped: prompt is too long"])
        self.assertEqual(len(app.fake.requests), 1)  # no summary call, no retry

    async def test_stop_mid_tool_repairs_the_open_call(self):
        app = await self.start(tool_reply("slow"), text_reply("never"))
        await app.admit([message("go")])
        for _ in range(100):
            await asyncio.sleep(0.01)
            if app.conversations[HOME].running_tool == "slow":
                break
        await app.admit([message("/stop")])
        await settle(app)
        history = app.conversations[HOME].messages
        result = history[-2]["content"][0]
        self.assertEqual(result["type"], "tool_result")
        self.assertTrue(result["content"][0]["unknown"])
        self.assertIn("sent /stop here", history[-1]["content"][0]["text"])
        self.assertIn("Stopped.", app.channel.texts())

    async def test_a_wait_wakes_on_a_person_notice_not_on_a_host_notice(self):
        app = await self.start(tool_reply("wait"), text_reply("ok"), text_reply("[SILENT]"))
        await app.admit([message("go")])
        for _ in range(200):
            await asyncio.sleep(0.01)
            if waiting.is_set():
                break
        await app.notify(HOME, "[Arti: a job finished]")
        await asyncio.sleep(0.5)
        self.assertEqual(len(app.fake.requests), 1)  # still waiting
        await app.admit([message("Owner loved a message", kind="notice")])
        await settle(app)
        [result] = results(app.fake.requests[1])
        self.assertEqual(result["content"][0]["text"], "message")

    async def test_a_person_notice_skips_stale_calls_a_host_notice_does_not(self):
        app = None

        def reply_after(inp):
            def reply(messages):
                app.conversations[HOME].add(inp)
                return tool_reply("echo", {"text": "x"})

            return reply

        reaction = message("[Sam reacted 👎 to your message]", kind="notice")
        host = message("[Arti: a job finished]", kind="notice", sender="")
        app = await self.start(
            reply_after(reaction), text_reply("ok"), reply_after(host), text_reply("ok")
        )
        await run_message(app, "go")
        self.assertEqual(calls, [])  # skipped: the reaction came after the request
        self.assertIn(
            "not run: something new", results(app.fake.requests[1])[0]["content"][0]["text"]
        )
        await run_message(app, "again")
        self.assertEqual(calls, [{"text": "x"}])  # a host notice skips nothing

    async def test_an_image_without_a_file_is_an_error_never_stored_as_base64(self):
        app = await self.start(tool_reply("raw_image"), text_reply("ok"))
        await run_message(app, "show me")
        [result] = results(app.fake.requests[1])
        self.assertTrue(result["is_error"])
        rows = app.store.db.execute("SELECT body FROM messages").fetchall()
        self.assertFalse(any("QUJDQUJD" in row["body"] for row in rows))


class SilentTest(unittest.TestCase):
    def test_only_markers_count(self):
        from arti.turn import is_silent

        for text in ("[SILENT]", " SILENT ", "NO_REPLY", "[SILENT] nothing new", "ok [SILENT]"):
            self.assertTrue(is_silent(text), text)
        for text in ("Keep it SILENT", "NO_REPLY needed?", "silent", ""):
            self.assertFalse(is_silent(text), text)


def _two_calls():
    reply = tool_reply("echo", {"text": "1"})
    reply.content += tool_reply("echo", {"text": "2"}).content
    return reply


if __name__ == "__main__":
    unittest.main()
