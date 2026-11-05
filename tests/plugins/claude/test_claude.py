import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import plugins.anthropic as anthropic
import plugins.claude as claude_plugin
from arti.providers import ProviderError
from plugins.anthropic.models import claude_route, context_window
from plugins.claude import OPTIONS as CLI_OPTIONS
from plugins.claude import claude
from plugins.claude.claude import ClaudeProvider, cli_env, inventory
from tests.plugins.heavy_support import register
from tests.providers.fakes import FakeServer, messages_stream, response, stream_response

FAKE = Path(__file__).with_name("fake_claude.py")


class RoutingTest(unittest.TestCase):
    def test_windows_and_routes(self):
        self.assertEqual(claude_route("claude-opus-5-5"), "claude-opus-5-5[1m]")
        self.assertEqual(claude_route("opus"), "claude-opus-5-5[1m]")
        self.assertEqual(claude_route("claude-new-9"), "claude-new-9")
        self.assertEqual(context_window("claude-new-9"), 200_000)
        self.assertEqual(context_window("claude-opus-5-5"), 1_000_000)

    def test_the_cli_is_the_bin_option_else_the_usual_places(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "claude"
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)
            self.assertEqual(claude.find_claude(str(binary)), str(binary))
            with mock.patch.dict(os.environ, {"HOME": tmp, "PATH": tmp + "/none"}):
                with self.assertRaisesRegex(ProviderError, "providers.claude-subscription.bin"):
                    claude.find_claude(str(Path(tmp) / "missing"))

    def test_inventory_limits_names_to_fit_the_prefix(self):
        tool = {"description": "d", "input_schema": {"type": "object"}}
        manifest, native = inventory([{**tool, "name": "a" * 53}])
        self.assertEqual(native[0]["name"], "mcp__arti__" + "a" * 53)
        self.assertEqual(manifest[0]["inputSchema"], {"type": "object"})
        for bad in ("a" * 54, "bad name"):
            with self.assertRaises(ProviderError):
                inventory([{**tool, "name": bad}])

    def test_cli_env_is_a_whitelist(self):
        leaked = {
            "ANTHROPIC_API_KEY": "sk",
            "ANTHROPIC_BASE_URL": "https://evil",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "HTTPS_PROXY": "http://p",
            "LC_ALL": "C",
        }
        with mock.patch.dict(os.environ, leaked):
            env = cli_env("http://127.0.0.1:1/admit/x")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:1/admit/x")
        for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK", "HTTPS_PROXY"):
            self.assertNotIn(name, env)
        self.assertEqual(
            (env["LC_ALL"], env["CLAUDE_CODE_MAX_RETRIES"], env["DISABLE_FEEDBACK_COMMAND"]),
            ("C", "0", "1"),
        )


class ProviderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.record = self.dir / "record.json"

    def provider(self, upstream, **config):
        wrapper = self.dir / "claude"
        settings = self.dir / "fake.json"
        settings.write_text(
            json.dumps({"record": str(self.record), "pid_file": str(self.dir / "pid"), **config})
        )
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{FAKE}" "{settings}" "$@"\n')
        wrapper.chmod(0o755)
        return ClaudeProvider(
            {**CLI_OPTIONS, "bin": str(wrapper)}, self.dir / "data", upstream=upstream
        )

    async def complete(self, provider, messages=None):
        messages = messages or [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "probe", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "out"}],
                        "is_error": False,
                    }
                ],
            },
        ]
        tools = [{"name": "probe", "description": "p", "input_schema": {"type": "object"}}]
        return await provider.complete(
            messages,
            system="sys",
            tools=tools,
            model="claude-opus-5-5",
            effort="high",
            session_id="s",
        )

    async def test_tool_call_through_the_cli_and_relay(self):
        call = {"type": "tool_use", "id": "t2", "name": "mcp__arti__probe", "input": {"x": 1}}
        unknown = {"type": "tool_use", "id": "t3", "name": "mcp__arti__nope", "input": {}}
        upstream = stream_response(
            messages_stream([{"type": "text", "text": "ok"}, call, unknown], "tool_use")
        )
        async with FakeServer([upstream]) as server:
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-leak"}):
                reply = await self.complete(self.provider(server.url, retry=True))
        self.assertEqual(
            [c["name"] for c in reply.tool_calls], ["probe", "nope"]
        )  # core answers "unknown tool"
        self.assertEqual(reply.rate_limit["windows"]["5h"]["used"], 0.25)
        self.assertEqual(reply.rate_limit["windows"]["7d"]["used"], 0.1)
        self.assertEqual(reply.native["items"][1]["name"], "mcp__arti__probe")
        # one upstream request; the CLI's retry got a local 400
        self.assertEqual(len(server.requests), 1)
        self.assertEqual(Path(str(self.record) + ".retry").read_text(), "400")
        sent = server.requests[0]
        self.assertEqual(sent["path"], "/v1/messages?beta=true")
        self.assertEqual(sent["headers"]["authorization"], "Bearer cli-oauth-token")
        body = json.loads(sent["body"])
        newest = body["messages"][2]["content"]
        self.assertEqual(newest[0]["content"], [{"type": "text", "text": "out"}])  # restored
        self.assertIn("system-reminder", newest[-1]["text"])  # the CLI's context, moved after it
        record = json.loads(self.record.read_text())
        self.assertNotIn("ANTHROPIC_API_KEY", record["env"])
        self.assertEqual(record["cwd"], str((self.dir / "data" / "claude" / "cwd").resolve()))
        argv = record["argv"]
        self.assertEqual(argv[argv.index("--model") + 1], "claude-opus-5-5[1m]")
        for flag in ("--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--max-turns") + 1], "1")

    async def test_only_offered_tool_names_are_prefixed_on_replay(self):
        upstream = stream_response(messages_stream([{"type": "text", "text": "ok"}]))
        calls = [
            {"type": "tool_use", "id": "t1", "name": "probe", "input": {}},
            {"type": "tool_use", "id": "t2", "name": "Bash", "input": {}},
        ]
        answers = [
            {"type": "tool_result", "tool_use_id": f"t{n}", "content": [], "is_error": True}
            for n in (1, 2)
        ]
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": calls},
            {"role": "user", "content": answers},
        ]
        async with FakeServer([upstream]) as server:
            await self.complete(self.provider(server.url), messages)
        replayed = json.loads(server.requests[0]["body"])["messages"][1]["content"]
        self.assertEqual([b["name"] for b in replayed], ["mcp__arti__probe", "Bash"])

    async def test_plain_reply(self):
        upstream = stream_response(messages_stream([{"type": "text", "text": "hello"}]))
        async with FakeServer([upstream]) as server:
            messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
            reply = await self.complete(self.provider(server.url), messages)
        self.assertEqual((reply.text, reply.stop_reason), ("hello", "stop"))

    async def test_upstream_error_is_classified(self):
        error = b'{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}'
        async with FakeServer([response(429, error)]) as server:
            with self.assertRaises(ProviderError) as caught:
                await self.complete(self.provider(server.url))
        self.assertEqual(caught.exception.code, "rate_limit")
        self.assertIn("slow down", caught.exception.message)

    async def test_logged_out_cli(self):
        async with FakeServer([]) as server:
            with self.assertRaises(ProviderError) as caught:
                await self.complete(self.provider(server.url, mode="logged_out"))
        self.assertEqual(caught.exception.code, "auth")
        self.assertIn("claude auth login", caught.exception.message)

    async def test_a_cli_that_never_calls_the_relay_is_a_retryable_timeout(self):
        async with FakeServer([]) as server:
            with mock.patch.object(claude, "START_IDLE", 0.5):
                with self.assertRaises(ProviderError) as caught:
                    await self.complete(self.provider(server.url, mode="hang"))
        self.assertEqual((caught.exception.code, caught.exception.retryable), ("timeout", True))
        self.assertEqual(server.requests, [])

    async def test_a_cli_that_stalls_after_its_request_is_a_retryable_timeout(self):
        upstream = stream_response(messages_stream([{"type": "text", "text": "hi"}]))
        async with FakeServer([upstream]) as server:
            with mock.patch.object(claude, "IDLE", 0.5):
                with self.assertLogs(claude.log, logging.WARNING) as logs:
                    with self.assertRaises(ProviderError) as caught:
                        await self.complete(self.provider(server.url, mode="stall"))
        self.assertEqual((caught.exception.code, caught.exception.retryable), ("timeout", True))
        self.assertEqual(len(server.requests), 1)
        self.assertIn("error=timeout status=200 denied=0", logs.output[0])

    async def test_each_replayed_turn_restarts_the_idle_count(self):
        upstream = stream_response(messages_stream([{"type": "text", "text": "done"}]))
        messages = []
        for n in range(5):  # five acknowledgements of 0.3 s: longer than START_IDLE in all
            messages.append({"role": "user", "content": [{"type": "text", "text": f"q{n}"}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": "a"}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "last"}]})
        async with FakeServer([upstream]) as server:
            with mock.patch.object(claude, "START_IDLE", 1.0):
                provider = self.provider(server.url, ack_delay=0.3)
                reply = await self.complete(provider, messages)
        self.assertEqual(reply.text, "done")

    async def test_the_cli_own_usage_limit_is_not_retried(self):
        async with FakeServer([]) as server:
            with self.assertRaises(ProviderError) as caught:
                await self.complete(self.provider(server.url, mode="limited"))
        self.assertEqual((caught.exception.code, caught.exception.retryable), ("rate_limit", False))
        self.assertEqual(server.requests, [])

    async def test_stale_request_folders_are_removed_at_start(self):
        stale = self.dir / "data" / "claude" / "tmpold"
        stale.mkdir(parents=True)
        (stale / "system.md").write_text("private memory")
        self.provider("http://127.0.0.1:9")
        self.assertFalse(stale.exists())

    async def test_cancellation_kills_the_process_group(self):
        async with FakeServer([]) as server:
            task = asyncio.create_task(self.complete(self.provider(server.url, mode="hang")))
            pid_file = self.dir / "pid"
            for _ in range(100):
                if pid_file.exists() and pid_file.read_text():
                    break
                await asyncio.sleep(0.05)
            grandchild = int(pid_file.read_text())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        for _ in range(40):
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail("the CLI's child survived the cancel")


class RegistrationTest(unittest.TestCase):
    def test_each_plugin_registers_its_route_with_its_own_options(self):
        cli, api = register(claude_plugin), register(anthropic)
        self.assertEqual(
            (set(cli.providers), set(api.providers)), ({"claude-subscription"}, {"anthropic"})
        )
        self.assertEqual(cli.providers["claude-subscription"].options["bin"], "")
        self.assertNotIn("bin", api.providers["anthropic"].options)
        self.assertEqual(api.providers["anthropic"].default_model, "claude-opus-5-5")
        self.assertEqual(claude_plugin.REQUIRES, ("safety", "anthropic"))
        self.assertFalse(hasattr(anthropic, "REQUIRES"))  # the API route runs without safety


class RateTest(unittest.TestCase):
    def test_headers_and_cli_event_merge_per_window(self):
        headers = {"status": "allowed", "windows": {"5h": {"used": 0.66, "resets_at": 1}}}
        cli = claude._cli_rate(
            {
                "status": "allowed",
                "unifiedWindows": {
                    "five_hour": {"utilization": 0.5},
                    "seven_day": {"utilization": 0.12, "resetsAt": 2},
                },
            }
        )
        merged = claude.merge_rates(headers, cli)
        self.assertEqual(merged["windows"]["5h"]["used"], 0.66)  # the headers win
        self.assertEqual(merged["windows"]["7d"], {"used": 0.12, "resets_at": 2})


if __name__ == "__main__":
    unittest.main()
