"""The inert MCP stub the Claude CLI runs: it lists Arti's tools and denies every call."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from plugins.claude import mcp_stub

MANIFEST = [{"name": "terminal", "description": "Run a command.", "inputSchema": {}}]


class RespondTest(unittest.TestCase):
    def ask(self, method, **params):
        return mcp_stub.respond(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, MANIFEST
        )

    def test_initialize_echoes_the_protocol_version_and_offers_tools(self):
        result = self.ask("initialize", protocolVersion="2025-06-18")["result"]
        self.assertEqual(result["protocolVersion"], "2025-06-18")
        self.assertEqual(result["capabilities"], {"tools": {}})
        self.assertEqual(self.ask("initialize")["result"]["protocolVersion"], "2024-11-05")

    def test_lists_the_manifest_and_denies_every_call(self):
        self.assertEqual(self.ask("tools/list")["result"], {"tools": MANIFEST})
        denied = self.ask("tools/call", name="terminal", arguments={"command": "rm -rf ~"})
        self.assertTrue(denied["result"]["isError"])
        self.assertIn("Denied", denied["result"]["content"][0]["text"])
        self.assertEqual(self.ask("ping"), {"jsonrpc": "2.0", "id": 1, "result": {}})

    def test_notifications_get_no_reply_and_unknown_methods_an_error(self):
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self.assertIsNone(mcp_stub.respond(notification, MANIFEST))
        self.assertEqual(self.ask("resources/list")["error"]["code"], -32601)


class StdioTest(unittest.TestCase):
    def test_runs_as_the_cli_starts_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "tools.json"
            manifest.write_text(json.dumps(MANIFEST), encoding="utf-8")
            lines = [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                "not json",
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            ]
            done = subprocess.run(
                [sys.executable, "-I", mcp_stub.__file__, str(manifest)],
                input="\n".join(lines) + "\n",
                capture_output=True,
                text=True,
                timeout=30,
            )
        replies = [json.loads(line) for line in done.stdout.splitlines()]
        self.assertEqual(done.returncode, 0)
        self.assertEqual([r.get("id") for r in replies], [1, None, 2])  # no reply to the notice
        self.assertEqual(replies[1]["error"]["code"], -32700)  # a bad line is answered, not fatal
        self.assertEqual(replies[2]["result"]["tools"], MANIFEST)


if __name__ == "__main__":
    unittest.main()
