"""host.service: the launchd agent and the systemd unit that keep the host running."""

import contextlib
import io
import os
import plistlib
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from host import service


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        data = self.home / "arti data"
        patches = (
            mock.patch.dict(os.environ, {"ARTI_HOME": str(data)}),
            mock.patch.object(service.sys, "executable", "/r & d/arti/.venv/bin/python"),
            mock.patch.object(service.config, "code_dir", return_value=Path("/r & d/arti")),
            contextlib.redirect_stdout(io.StringIO()),
        )
        for patch in patches:
            patch.__enter__()
            self.addCleanup(patch.__exit__, None, None, None)
        self.logs = data.resolve() / "logs"

    def test_macos_gets_a_launch_agent(self):
        start = service.install(self.home, "darwin")
        path = self.home / "Library/LaunchAgents/ai.arti.host.plist"
        agent = plistlib.loads(path.read_bytes())
        # `-m arti` and the os.execv restart both need the repository as working directory.
        self.assertEqual(agent["ProgramArguments"], ["/r & d/arti/.venv/bin/python", "-m", "host"])
        self.assertEqual(agent["WorkingDirectory"], "/r & d/arti")
        self.assertTrue(agent["KeepAlive"])
        env = agent["EnvironmentVariables"]
        self.assertTrue(env["PATH"].startswith(f"{self.home}/.local/bin:/opt/homebrew/bin:"))
        self.assertEqual(env["ARTI_HOME"], str(self.logs.parent))
        self.assertEqual(agent["StandardErrorPath"], str(self.logs / "daemon.err.log"))
        self.assertEqual(start, [f"launchctl bootstrap gui/$(id -u) {shlex.quote(str(path))}"])

    def test_linux_gets_a_systemd_user_unit(self):
        start = service.install(self.home, "linux")
        unit = (self.home / ".config/systemd/user/arti-host.service").read_text()
        for line in (
            'ExecStart="/r & d/arti/.venv/bin/python" -m host',
            'WorkingDirectory="/r & d/arti"',
            f'Environment="PATH={self.home}/.local/bin:/usr/local/bin:',
            f'Environment="ARTI_HOME={self.logs.parent}"',
            "Restart=always",
            "RestartSec=30",
            f"StandardError=append:{self.logs}/daemon.err.log",
            "WantedBy=default.target",
        ):
            self.assertIn(line, unit)
        self.assertEqual(start[1], "systemctl --user enable --now arti-host.service")

    def test_systemd_words_are_quoted_and_expansions_escaped(self):
        self.assertEqual(service.quote('a "b" 100% $HOME \\'), '"a \\"b\\" 100%% $$HOME \\\\"')

    def test_another_system_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "no service manager"):
            service.install(self.home, "plan9")


if __name__ == "__main__":
    unittest.main()
