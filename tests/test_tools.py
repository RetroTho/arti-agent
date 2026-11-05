import asyncio
import struct
import tempfile
import threading
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

from arti import tools
from arti.chat import Conversation, Headless
from tests.support import HOME, make_app

ran = []


def register(api):
    def big(args, ctx):
        return "x" * 30_000 + "THE END"

    def fails(args, ctx):
        raise tools.ToolError("bad input, try again")

    def bug(args, ctx):
        raise SystemExit(2)

    def image(args, ctx):
        return [{"type": "image", "path": "/nonexistent.png"}]

    def record(args, ctx):
        ran.append(args)
        return {"ok": True}

    empty = {"properties": {}}
    api.tool("big", "Big.", empty, big, review=None)
    api.tool("fails", "Fails.", empty, fails, review=None)
    api.tool("bug", "Bug.", empty, bug, review=None)
    api.tool(
        "record", "Record.", {"properties": {"n": {"type": "integer"}}}, record, review="shell"
    )
    api.tool("quick", "Quick.", empty, record, review="shell", timeout=0.2)
    api.tool("chatty", "Needs a chat.", empty, record, review=None, interactive=True)
    api.tool("open", "For anyone.", empty, record, review=None, guest=True)
    api.tool("missing", "Unavailable.", empty, record, review=None, available=lambda app: False)


def tiny_png() -> bytes:
    """A real 1x1 PNG."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"\x00\xff\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
    )


def broken_register(api):
    api.tool("half", "Half.", {"properties": {}}, lambda a, c: "", review=None)
    raise RuntimeError("plugin bug")


class ToolsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = await make_app(self.tmp.name, plugins=[register, broken_register])
        self.conv = Headless(self.app, "home", self.app.default_route())
        self.conv.offered = {t.name: t for t in tools.tools_for(self.conv)}
        ran.clear()

    async def asyncTearDown(self):
        await self.app.shutdown()
        self.tmp.cleanup()

    async def call(self, name, args=None):
        result = await tools.run_tool(self.conv, {"id": "c1", "name": name, "input": args or {}})
        return result, result["content"][0]["text"]

    def test_a_failing_plugin_adds_nothing_and_filters_apply(self):
        self.assertNotIn("half", self.app.tools)
        self.assertEqual(
            sorted(self.conv.offered), ["big", "bug", "fails", "open", "quick", "record"]
        )
        guest = Headless(self.app, "chat-0123", self.app.default_route(), trusted=False)
        self.assertEqual([t.name for t in tools.tools_for(guest)], ["open"])

    async def test_validation_comes_first(self):
        result, text = await self.call("record", {"n": "five"})
        self.assertTrue(result["is_error"])
        self.assertIn("$.n: expected integer", text)
        self.assertEqual(ran, [])

    async def test_with_no_gate_every_call_runs_unreviewed(self):
        self.assertIsNone(self.app.gate)
        result, text = await self.call("record", {"n": 1})  # review kind "shell"
        self.assertEqual((ran, result["is_error"]), ([{"n": 1}], False))

    async def test_a_gate_refusal_runs_nothing(self):
        self.app.gate = mock.Mock(return_value="BLOCKED: Owner denied it")
        result, text = await self.call("record", {"n": 1})
        [(ctx, tool, args)] = [c.args for c in self.app.gate.call_args_list]
        self.assertEqual((ctx.call_id, tool.name, args), ("c1", "record", {"n": 1}))
        self.assertEqual(text, "BLOCKED: Owner denied it")
        self.assertEqual(ran, [])
        self.app.gate = mock.Mock(return_value=None)
        result, text = await self.call("record", {"n": 1})
        self.assertEqual((ran, result["is_error"]), ([{"n": 1}], False))
        self.assertIn('"ok": true', text)

    async def test_the_tool_timeout_starts_after_the_gate(self):
        def late_approval(ctx, tool, args):
            time.sleep(0.5)  # the owner answers after the tool's own 0.2 s timeout
            return None

        self.app.gate = late_approval
        result, text = await self.call("quick")
        self.assertEqual((ran, result["is_error"]), ([{}], False))

    async def test_a_failing_gate_is_a_plain_error(self):
        self.app.gate = mock.Mock(side_effect=RuntimeError("gate bug"))
        with self.assertLogs("arti.tools", "ERROR"):
            result, text = await self.call("record", {"n": 1})
        self.assertEqual(text, "Error: RuntimeError: gate bug")
        self.assertNotIn("unknown", result["content"][0])
        self.assertEqual(ran, [])

    async def test_huge_results_are_capped_and_spilled(self):
        result, text = await self.call("big")
        self.assertLess(len(text), 21_000)
        self.assertTrue(text.endswith("THE END"))
        spill = self.app.data_dir / "cache" / "tool-output" / "c1-0.txt"
        self.assertIn(str(spill), text)
        self.assertEqual(len(spill.read_text()), 30_007)

    async def test_a_model_call_id_is_made_safe_for_the_spill_file_name(self):
        call = {"id": "../../evil/" + "x" * 100, "name": "big", "input": {}}
        text = (await tools.run_tool(self.conv, call))["content"][0]["text"]
        spill = self.app.data_dir / "cache" / "tool-output" / ("evil" + "x" * 60 + "-0.txt")
        self.assertIn(str(spill), text)
        self.assertTrue(spill.is_file())
        self.assertEqual(spill.parent.parent.name, "cache")

    def test_wait_watches_another_conversation_from_when_it_starts(self):
        ctx = tools.ToolContext(self.conv, "c1", self.conv.cancel)
        other = Headless(self.app, "home", self.app.default_route())
        other.messages_in = 3  # a message before the wait is not news
        self.assertEqual(ctx.wait(0.05, watch=other), "timeout")
        other.person_inputs += 1  # nor is a tapback there (only a message in words counts)
        self.conv.person_inputs += 1  # nor this conversation's input, while watching another
        self.assertEqual(ctx.wait(0.05, watch=other), "timeout")
        self.assertEqual(ctx.wait(0.05), "message")
        threading.Timer(0.05, lambda: setattr(other, "messages_in", 4)).start()
        self.assertEqual(ctx.wait(5, watch=other), "message")

    def test_unattended(self):
        self.assertTrue(tools.ToolContext(self.conv, "c1", self.conv.cancel).unattended)
        self.assertEqual(self.conv.key, "")  # a headless run has no chat
        chat = Conversation(self.app, "home", "home", self.app.default_route())
        self.assertFalse(tools.ToolContext(chat, "c2", self.conv.cancel).unattended)

    async def test_errors_and_bugs_become_error_results(self):
        result, text = await self.call("fails")
        self.assertEqual((result["is_error"], text), (True, "Error: bad input, try again"))
        result, text = await self.call("bug")  # it may have done part of its work
        self.assertTrue(result["content"][0]["unknown"])
        self.assertIn("failed with SystemExit: 2; it may have partly run", text)
        result, text = await self.call("missing")
        self.assertIn("unknown tool", text)

    async def test_a_bad_image_is_an_error_not_a_crash(self):
        self.app.tools["image"] = tools.Tool(
            "image",
            "",
            {"type": "object"},
            lambda a, c: [{"type": "image", "path": "/no.png"}],
            None,
        )
        self.conv.offered["image"] = self.app.tools["image"]
        result, text = await self.call("image")
        self.assertTrue(result["is_error"])
        self.assertTrue(result["content"][0]["unknown"])  # the handler did run
        self.assertIn("the tool ran, but its result image could not be loaded", text)

    def test_cache_dirs_are_per_context(self):
        home = tools.ToolContext(self.conv, "c1", self.conv.cancel)
        self.assertEqual(home.cache_dir("web"), self.app.data_dir / "cache" / "web")
        guest = Headless(self.app, "chat-0123", self.app.default_route())
        folder = tools.ToolContext(guest, "c2", guest.cancel).cache_dir("web")
        self.assertEqual(folder, self.app.data_dir / "contexts" / "chat-0123" / "cache" / "web")
        self.assertTrue(folder.is_dir())

    async def test_a_contacts_result_image_is_kept_in_its_own_folder(self):
        outside = Path(self.tmp.name) / "elsewhere.png"
        outside.write_bytes(tiny_png())
        guest = Headless(self.app, "chat-0123", self.app.default_route())
        guest.offered = {
            "image": tools.Tool(
                "image",
                "",
                {"type": "object"},
                lambda a, c: [{"type": "image", "path": str(outside)}],
                None,
            )
        }
        result = await tools.run_tool(guest, {"id": "c1", "name": "image", "input": {}})
        path = Path(result["content"][0]["path"])
        self.assertEqual(path.parent, self.app.data_dir / "contexts" / "chat-0123" / "media")

    def chat_ctx(self):
        """A context in a conversation with a chat (Conversation, not Headless)."""
        chat = Conversation(self.app, HOME, "home", self.app.default_route())
        return chat, tools.ToolContext(chat, "c1", threading.Event())

    async def test_call_times_out_and_cancels_the_coroutine(self):
        ctx = tools.ToolContext(self.conv, "c1", threading.Event())
        cancelled = asyncio.Event()

        async def hangs():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def answers():
            return 42

        self.assertEqual(await asyncio.to_thread(ctx.call, answers()), 42)
        started = time.monotonic()
        with self.assertRaisesRegex(tools.ToolError, "timed out after 0s"):
            await asyncio.to_thread(ctx.call, hangs(), 0.3)
        self.assertLess(time.monotonic() - started, 2)
        await asyncio.wait_for(cancelled.wait(), 1)  # not left running on the loop

    async def test_call_stops_waiting_on_cancel(self):
        ctx = tools.ToolContext(self.conv, "c1", threading.Event())
        cancelled, ran = asyncio.Event(), []

        async def hangs():
            ran.append(True)
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        threading.Timer(0.1, ctx.cancel.set).start()
        with self.assertRaisesRegex(tools.ToolError, "^cancelled$"):
            await asyncio.to_thread(ctx.call, hangs(), 30)
        await asyncio.wait_for(cancelled.wait(), 1)
        # Already cancelled: the coroutine is closed, never started.
        with self.assertRaisesRegex(tools.ToolError, "^cancelled$"):
            await asyncio.to_thread(ctx.call, hangs())
        self.assertEqual(ran, [True])

    async def test_notify_times_out_instead_of_waiting_forever(self):
        chat, ctx = self.chat_ctx()
        cancelled = asyncio.Event()

        async def notify(key, text, at=None):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        real = asyncio.run_coroutine_threadsafe

        def quick(coro, loop):  # notify waits 30 s; this test waits 0.1 s
            future = real(coro, loop)
            result = future.result
            future.result = lambda timeout=None: result(min(timeout, 0.1))
            return future

        with (
            mock.patch.object(self.app, "notify", notify),
            mock.patch.object(tools.asyncio, "run_coroutine_threadsafe", quick),
        ):
            with self.assertRaises(TimeoutError):  # not a ToolError: it may still arrive
                await asyncio.to_thread(ctx.notify, "job done")
        await asyncio.wait_for(cancelled.wait(), 1)
        headless = tools.ToolContext(self.conv, "c2", threading.Event())
        with self.assertRaisesRegex(tools.ToolError, "no chat to notify"):
            headless.notify("job done")

    async def test_a_sub_agent_gets_only_tools_its_parent_was_offered(self):
        runs = []

        async def run_headless(prompt, **options):
            runs.append(options)
            return "child done"

        self.conv.offered = {name: self.app.tools[name] for name in ("record", "open")}
        guest = Headless(self.app, "chat-0123456789abcdef", self.app.default_route(), trusted=False)
        guest.offered = {"open": self.app.tools["open"]}
        ctx = tools.ToolContext(self.conv, "c1", threading.Event())
        with mock.patch.object(self.app, "run_headless", run_headless):
            wanted = ["record", "bug", "big"]  # bug and big exist, but were not offered here
            self.assertEqual(
                await asyncio.to_thread(ctx.run_headless, "p", tools=wanted), "child done"
            )
            await asyncio.to_thread(ctx.run_headless, "p")
            await asyncio.to_thread(ctx.run_headless, "p", tools=[])
            guest_ctx = tools.ToolContext(guest, "c2", threading.Event())
            await asyncio.to_thread(guest_ctx.run_headless, "p", tools=["record", "open"])
        self.assertEqual(
            [run["tools"] for run in runs], [["record"], ["open", "record"], [], ["open"]]
        )
        self.assertEqual(runs[0]["scope"], "home")
        self.assertTrue(runs[0]["trusted"])
        self.assertEqual((runs[3]["scope"], runs[3]["trusted"]), ("chat-0123456789abcdef", False))


if __name__ == "__main__":
    unittest.main()
