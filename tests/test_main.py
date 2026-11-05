"""arti/__main__.py: the command line, run as `python -m arti` in a child process."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.providers.fakes import FakeServer, messages_stream, stream_response

ROOT = Path(__file__).resolve().parent.parent
PROVIDER = '[agent]\nprovider = "anthropic"\n'  # more than one is registered: name one


def arti(*args: str, home: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "ARTI_HOME": home}
    argv = [sys.executable, "-m", "arti", *args]
    return subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)


class MainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config(PROVIDER)

    def config(self, text: str) -> None:
        Path(self.tmp.name, "config.toml").write_text(text)

    def test_bad_arguments_are_refused_before_anything_starts(self):
        for args in (["--image", "x.png"], ["--once", " "], ["--provider", "nope"]):
            result = arti(*args, home=self.tmp.name)
            self.assertEqual(result.returncode, 2, args)
            self.assertIn("usage: arti", result.stderr)

    def test_a_bad_config_exits_with_its_message(self):
        self.config('[agent]\neffort = "huge"\n')
        result = arti("--once", "hi", home=self.tmp.name)
        self.assertEqual(result.returncode, 1)
        self.assertIn("invalid configuration: agent.effort", result.stderr)

    def test_auth_status_needs_no_login(self):
        result = arti("auth-status", home=self.tmp.name)  # a command of the openai plugin
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"logged_in": false', result.stdout)
        self.assertFalse(Path(self.tmp.name, "arti.lock").exists())  # the App never started

    def test_help_lists_the_plugins_commands(self):
        result = arti("--help", home=self.tmp.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: arti", result.stdout)
        for name in ("login", "logout", "auth-status", "browser-login"):
            self.assertRegex(result.stdout, rf"\n  {name} +\S", name)

    def test_an_unknown_command_or_a_plugin_commands_bad_arguments(self):
        result = arti("nope", home=self.tmp.name)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown command 'nope'", result.stderr)
        self.assertIn("auth-status", result.stderr)
        result = arti("auth-status", "--bogus", home=self.tmp.name)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage: python -m arti auth-status", result.stderr)

    def test_a_disabled_plugins_commands_and_section_are_gone(self):
        self.config(PROVIDER + '\n[plugins]\ndisabled = ["openai"]\n\n[memory]\nuser_chars = 99\n')
        result = arti("auth-status", home=self.tmp.name)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown command 'auth-status'", result.stderr)
        self.config(PROVIDER + "\n[nobody]\nx = 1\n")
        result = arti("auth-status", home=self.tmp.name)
        self.assertEqual(result.returncode, 1)
        self.assertIn("invalid configuration: [nobody] is not a known section", result.stderr)


class OnceTest(unittest.IsolatedAsyncioTestCase):
    async def test_once_runs_a_task_against_a_loopback_model_and_prints_the_reply(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        reply = messages_stream([{"type": "text", "text": "Hello from the fake model"}])
        async with FakeServer([stream_response(reply)]) as server:
            Path(tmp.name, "config.toml").write_text(
                f'[agent]\nprovider = "anthropic"\n\n[providers.anthropic]\n'
                f'base_url = "{server.url}"\napi_key_env = "ARTI_TEST_KEY"\n'
            )
            env = {**os.environ, "ARTI_HOME": tmp.name, "ARTI_TEST_KEY": "test-key-123"}
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "arti",
                "--once",
                "say hello",
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(child.communicate(), 60)
        self.assertEqual(child.returncode, 0, err.decode())
        self.assertEqual(out.decode().strip(), "Hello from the fake model")
        [request] = server.requests
        self.assertEqual(request["path"], "/messages")
        self.assertEqual(request["headers"]["x-api-key"], "test-key-123")
        body = json.loads(request["body"])
        self.assertIn("say hello", json.dumps(body["messages"]))
        self.assertTrue(body["tools"])  # the plugins were loaded and offered
        self.assertTrue(Path(tmp.name, "state.db").exists())


if __name__ == "__main__":
    unittest.main()
