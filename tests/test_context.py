import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from arti import context
from arti.chat import Headless
from plugins import anthropic, files, openai, todo
from tests.support import make_app

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 100


def user(*blocks):
    return {"role": "user", "content": list(blocks)}


def text(t, **extra):
    return {"type": "text", "text": t, **extra}


def call(i, name="terminal", **args):
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": f"c{i}", "name": name, "input": args}],
    }


def result(i, t="ok", **extra):
    return user(
        {
            "type": "tool_result",
            "tool_use_id": f"c{i}",
            "is_error": False,
            "content": [text(t, **extra)],
        }
    )


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def conv(self, messages, summary=""):
        return SimpleNamespace(messages=messages, summary=summary, route=("anthropic", "m", "high"))

    def test_the_default_budget_is_capped_and_the_config_overrides_it(self):
        windows = {"claude-subscription": anthropic.context_window, "openai": openai.api_window}

        def conv(model, configured=0):
            app = SimpleNamespace(
                config={"agent": {"context_window": configured}},
                provider_spec=lambda name: SimpleNamespace(context_window=windows[name]),
            )
            return SimpleNamespace(app=app, route=("claude-subscription", model, "high"))

        # A 1M-token model is compacted as if it had 200K, so requests stay small...
        self.assertEqual(context.budget(conv("claude-opus-5-5")), context.DEFAULT_WINDOW)
        self.assertEqual(context.budget(conv("claude-new-9")), 200_000)
        # ...unless the config asks for the larger window.
        self.assertEqual(context.budget(conv("claude-opus-5-5", 1_000_000)), 1_000_000)
        smaller = SimpleNamespace(app=conv("x").app, route=("openai", "gpt-4.1", "high"))
        self.assertEqual(context.budget(smaller), 128_000)  # a smaller window is kept as it is

    def test_images_beyond_the_budget_become_notes_newest_kept(self):
        paths = []
        for i in range(context.MAX_IMAGES + 2):
            path = Path(self.tmp.name) / f"{i}.png"
            path.write_bytes(PNG)
            paths.append(str(path))
        messages = [
            user({"type": "image", "path": p, "media_type": "image/png"}, text(str(i)))
            for i, p in enumerate(paths)
        ]
        messages[1:1] = [{"role": "assistant", "content": [text("x")]}]
        out = context.for_request(self.conv(messages))
        kinds = [
            b["type"]
            for m in out
            for b in m["content"]
            if b["type"] != "text" or "omitted" in b["text"]
        ]
        self.assertEqual(kinds.count("image"), context.MAX_IMAGES)
        self.assertEqual(kinds[:2], ["text", "text"])  # the two oldest
        self.assertIn("data", [b for b in out[-1]["content"] if b["type"] == "image"][-1])
        self.assertNotIn("data", messages[-1]["content"][0])  # history itself is untouched

    def test_missing_images_and_merging_and_summary(self):
        messages = [
            user(text("go")),
            call(1),
            result(1),
            user(text("steer")),
            user({"type": "image", "path": "/gone.png", "media_type": "image/png"}),
        ]
        out = context.for_request(self.conv(messages, summary="SUMMARY"))
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
        self.assertEqual(out[0]["content"][0]["text"], "SUMMARY")
        last = out[2]["content"]
        self.assertEqual([b["type"] for b in last], ["tool_result", "text", "text"])
        self.assertEqual(last[2]["text"], "[image no longer available]")


class CompactionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # The todo and files plugins keep their state through compaction (api.on_compact).
        self.app = await make_app(self.tmp.name, plugins=[todo.register, files.register])

    async def asyncTearDown(self):
        await self.app.shutdown()
        self.tmp.cleanup()

    def test_cut_points_never_split_a_call_from_its_result(self):
        messages = [user(text("a")), call(1), result(1), user(text("b")), call(2), result(2)]
        sizes = [10] * len(messages)
        self.assertEqual(context.choose_cut(messages, sizes, 1000), 1)
        self.assertEqual(
            context.choose_cut(messages, sizes, 25), 4
        )  # the tail of 2 does not start safely
        self.assertEqual(context.choose_cut(messages, sizes, 0), 4)  # always summarise something
        self.assertIsNone(context.choose_cut([user(text("only"))], [10], 0))

    async def test_compaction_keeps_unknown_warnings_and_todo(self):
        conv = Headless(self.app, "home", self.app.default_route())
        item = {"id": "1", "content": "ship it", "status": "pending"}
        conv.state = {"todo": [item], "seen_files": {"/x": 1}}
        conv.waits = {7: (SimpleNamespace(text="[Arti: the wait is over: check the poll]"), None)}
        for message in [
            user(text("a")),
            call(1),
            result(1, "[Arti: timed out; outcome unknown]", unknown=True),
            user(text("b" * 3000)),
            {"role": "assistant", "content": [text("done")]},
        ]:
            conv.append(message)
        self.app.fake.replies = [lambda m: _reply("They worked on a.")]
        self.assertTrue(await context.maybe_compact(conv, "system", [], force=True))
        self.assertEqual(conv.messages[0]["content"][0]["text"], "done")  # forced: about half kept
        self.assertIn("They worked on a.", conv.summary)
        self.assertIn("outcome unknown", conv.summary)
        self.assertIn("1. ship it", conv.summary)
        self.assertIn("check the poll", conv.summary)  # pending waits survive the summary
        self.assertNotIn("seen_files", conv.state)
        summary_call = self.app.fake.requests[0]
        self.assertEqual(
            (summary_call["model"], summary_call["effort"]), ("claude-sonnet-5-5", "high")
        )
        summarised = self.app.fake.requests[0]["messages"][0]["content"][0]["text"]
        self.assertIn("user: a", summarised)
        # A later compaction keeps the warning word for word, though the summariser drops it.
        for message in [user(text("c" * 3000)), {"role": "assistant", "content": [text("ok")]}]:
            conv.append(message)
        self.app.fake.replies = [lambda m: _reply("They worked on c.")]
        self.assertTrue(await context.maybe_compact(conv, "system", [], force=True))
        self.assertIn("They worked on c.", conv.summary)
        self.assertEqual(conv.summary.count("[Arti: timed out; outcome unknown]"), 1)

    async def test_a_long_stretch_is_summarised_in_passes_that_build_on_each_other(self):
        conv = Headless(self.app, "home", self.app.default_route())
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            conv.append({"role": role, "content": [text(f"part {i} " + "x" * 3000)]})
        kept_from = conv.messages[5]  # forced: about half is kept
        self.app.fake.replies = [_reply(f"summary {n}") for n in range(1, 6)]
        with mock.patch.object(context, "MAX_SPAN_TOKENS", 2500):  # two messages a pass
            self.assertTrue(await context.maybe_compact(conv, "system", [], force=True))
        inputs = [r["messages"][0]["content"][0]["text"] for r in self.app.fake.requests]
        self.assertEqual(len(inputs), 3)  # parts 0-1, 2-3, then 4
        self.assertTrue(all(len(i) < 8000 for i in inputs))  # each pass reads only its span
        self.assertIn("summary 1", inputs[1])  # ...and the summary so far
        self.assertIn("summary 3", conv.summary)
        self.assertIs(conv.messages[0], kept_from)  # what was to be kept is untouched

    async def test_a_failed_summary_changes_nothing(self):
        conv = Headless(self.app, "home", self.app.default_route())
        for message in [
            user(text("a")),
            {"role": "assistant", "content": [text("x")]},
            user(text("b")),
        ]:
            conv.append(message)
        before = list(conv.messages)
        self.app.fake.replies = [RuntimeError("down")]
        self.assertFalse(await context.maybe_compact(conv, "system", [], force=True))
        self.assertEqual(conv.messages, before)
        self.assertFalse(await context.maybe_compact(conv, "system", []))  # far below the window


def _reply(t):
    from tests.support import text_reply

    return text_reply(t)


if __name__ == "__main__":
    unittest.main()
