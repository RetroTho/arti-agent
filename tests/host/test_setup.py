"""The host's users, settings and sandbox options."""

import io
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import host.__main__ as host_main
from arti.config import ConfigError
from host import config, sandbox
from host.users import Users


class UsersTest(unittest.TestCase):
    def setUp(self):
        self.data = Path(tempfile.mkdtemp())
        self.users = Users(self.data)

    def test_add_find_and_remove(self):
        sam = self.users.add("sam", 'Sam "J"', "chat", "U1", "Europe/Berlin", "anthropic")
        self.assertEqual(self.users.by_handle("chat", "U1"), sam)
        self.assertIsNone(self.users.by_handle("chat", ""))
        self.assertIsNone(self.users.by_handle("other", "U1"))
        agent = tomllib.loads((sam.agent_data(self.data) / "config.toml").read_text())
        expected = {"user_name": 'Sam "J"', "timezone": "Europe/Berlin", "provider": "anthropic"}
        self.assertEqual(agent["agent"], expected)
        self.assertEqual((self.data / "users.json").stat().st_mode & 0o077, 0)
        self.users.remove("sam")
        self.assertEqual(self.users.all(), [])
        self.assertTrue(sam.home(self.data).is_dir())  # the folder is kept

    def test_ids_and_handles_are_unique_and_ids_plain(self):
        self.users.add("sam", "Sam", "chat", "U1", "UTC", "anthropic")
        for uid, handle, problem in (
            ("sam", "U2", "already a user"),
            ("alex", "U1", "already has that chat handle"),
            ("../x", "U3", "lower-case"),
            ("Alex", "U3", "lower-case"),
        ):
            with self.subTest(uid):
                with self.assertRaisesRegex(ValueError, problem):
                    self.users.add(uid, "X", "chat", handle, "UTC", "anthropic")


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.data = Path(tempfile.mkdtemp())

    def load(self, text: str, channels=("chat",)):
        (self.data / "host.toml").write_text(text)
        return config.load(self.data, list(channels))

    def test_defaults_and_the_only_channel(self):
        cfg = self.load("[chat]\nport = 9000\n")
        self.assertEqual(cfg["host"]["channel"], "chat")
        self.assertEqual(cfg["host"]["memory"], "2g")
        self.assertEqual(cfg["chat"], {"port": 9000})  # the channel checks its own section

    def test_with_no_channel_installed_only_running_and_adding_users_are_refused(self):
        cfg = self.load("[host]\nmemory = '1g'\n", channels=())
        self.assertEqual(cfg["host"]["channel"], "")
        with self.assertRaisesRegex(ConfigError, "no chat channel is installed"):
            host_main.make_channel(cfg, None)
        with mock.patch("sys.stderr", io.StringIO()) as err, self.assertRaises(SystemExit):
            host_main.user_command(
                self.data, cfg, ["add", "sam", "--name", "Sam", "--handle", "U1"]
            )
        self.assertIn("no chat channel is installed", err.getvalue())
        self.assertEqual(Users(self.data).all(), [])
        with mock.patch("sys.stdout", io.StringIO()) as out:
            self.assertEqual(host_main.user_command(self.data, cfg, ["list"]), 0)
        self.assertEqual(out.getvalue(), "")

    def test_mistakes_stop_startup(self):
        for text, channels, problem in (
            ("[host]\nchannel = 'chat'\n", (), r"must be one of \(none installed\)"),
            ("[chat]\n", (), r"\[chat\] is not a known section; known: host$"),
            ("", ("chat", "other"), "host.channel must be one of chat, other"),
            ("[host]\nchannel = 'nope'\n", ("chat",), "host.channel must be one of chat"),
            ("[host]\nmemroy = '1g'\n", ("chat",), "host.memroy is not a known setting"),
            ("[other]\n", ("chat",), r"\[other\] is not a known section"),
            ("[host\n", ("chat",), "invalid configuration"),
        ):
            with self.subTest(text):
                with self.assertRaisesRegex(ConfigError, problem):
                    self.load(text, channels)


class SandboxTest(unittest.TestCase):
    def test_a_sandbox_is_locked_down_and_mounts_only_its_home_and_the_code(self):
        data = Path(tempfile.mkdtemp())
        user = Users(data).add("sam", "Sam", "chat", "U1", "UTC", "anthropic")
        settings = {"memory": "2g", "cpus": 2.0}
        with mock.patch.object(sandbox.shutil, "which", return_value="/usr/bin/podman"):
            boxes = sandbox.Sandboxes(settings, Path("/repo"), data)
        args = boxes.run_args(user)
        joined = " ".join(args)
        for part in (
            "--name arti-sam",
            "--userns keep-id",
            "--read-only",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "--memory 2g",
            "--pids-limit 512",
            "--restart unless-stopped",
            f"--volume {user.home(data).resolve()}:/home/arti",
            "--volume /repo/arti:/opt/arti/arti:ro",
            "--volume /repo/plugins:/opt/arti/plugins:ro",
        ):
            self.assertIn(part, joined)
        self.assertEqual(args[-4:], ["arti-sandbox:latest", "python", "-m", "arti"])
        volumes = [args[i + 1] for i, a in enumerate(args) if a == "--volume"]
        self.assertEqual(len(volumes), 3)  # nothing else of the machine, the host's .env least

    def test_on_macos_a_stopped_podman_vm_is_started(self):
        data = Path(tempfile.mkdtemp())
        with mock.patch.object(sandbox.shutil, "which", return_value="/usr/bin/podman"):
            boxes = sandbox.Sandboxes({"memory": "1g", "cpus": 1.0}, Path("/repo"), data)
        answers = iter(["", "", "linux"])  # info fails, machine start, info works
        calls = []

        def run(*args, timeout=0, check=True):
            calls.append(args[:2])
            return next(answers)

        with (
            mock.patch.object(boxes, "_run", run),
            mock.patch.object(sandbox.sys, "platform", "darwin"),
        ):
            boxes.ready()
        self.assertEqual(calls, [("info", "--format"), ("machine", "start"), ("info", "--format")])
        with mock.patch.object(boxes, "_run", lambda *a, **k: ""):
            with mock.patch.object(sandbox.sys, "platform", "linux"):
                with self.assertRaisesRegex(sandbox.SandboxError, "does not answer"):
                    boxes.ready()

    def test_podman_is_required(self):
        with mock.patch.object(sandbox.shutil, "which", return_value=None):
            with self.assertRaisesRegex(sandbox.SandboxError, "Podman is needed"):
                sandbox.podman()


if __name__ == "__main__":
    unittest.main()
