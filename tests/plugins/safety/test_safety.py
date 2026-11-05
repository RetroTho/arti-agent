"""The safety plugin (plugins/safety/__init__.py): what it registers, its settings, and its start
and stop hooks around the other plugins'."""

import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from arti import lock, secret
from arti.app import App
from arti.channel import FakeChannel
from arti.config import ConfigError
from plugins import safety
from plugins.safety import approvals, proc, redact
from tests.support import fake_providers, make_app, make_config

ENV = {"PATH": "/usr/bin:/bin"}


def configured(sections: dict | None = None) -> App:
    """An App with the safety plugin loaded (not started)."""
    cfg = make_config()
    cfg.update(sections or {})
    app = App(
        cfg, Path(tempfile.mkdtemp()), FakeChannel(), plugins=[fake_providers, safety.register]
    )
    app.load_plugins()
    return app


class RegisterTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(secret.set_redactor, secret._redactor)

    def test_it_registers_the_gate_the_redactor_and_its_commands(self):
        app = configured()
        self.assertIs(app.gate, approvals.check)
        self.assertEqual(app.redactor, (redact.redact, redact.redact_obj))
        self.assertEqual(secret.redact("key sk-ant-0123456789abcdefgh"), "key [REDACTED]")
        self.assertEqual(list(app.chat_commands), ["approve", "deny"])
        self.assertTrue(all(c.home_only for c in app.chat_commands.values()))

    def test_its_sections_have_their_defaults_and_are_validated(self):
        app = configured()
        self.assertEqual(app.config["approvals"], safety.APPROVALS)
        self.assertEqual(app.config["network"], {"allow_private_urls": False})
        self.assertEqual(
            app.plugin_sections, {"approvals": "plugins.safety", "network": "plugins.safety"}
        )
        allowed = configured({"network": {"allow_private_urls": True}})
        self.assertTrue(allowed.config["network"]["allow_private_urls"])
        self.assertEqual(
            configured({"approvals": {"provider": "openai"}}).config["approvals"]["provider"],
            "openai",
        )
        for sections, fragment in (
            (
                {"network": {"allow_private_urls": 1}},
                "network.allow_private_urls must be true or false",
            ),
            ({"approvals": {"allow": "x"}}, "approvals.allow must be a list"),
            ({"approvals": {"mode": "yolo"}}, "approvals.mode must be one of off, manual, smart"),
            ({"approvals": {"provider": "nope"}}, "approvals.provider must be one of"),
        ):
            with self.subTest(fragment), self.assertRaises(ConfigError) as caught:
                configured(sections)
            self.assertIn(f"invalid configuration: {fragment}", str(caught.exception))

    def test_without_it_its_sections_are_unknown(self):
        cfg = make_config()
        cfg["approvals"] = {"mode": "off"}
        app = App(cfg, Path(tempfile.mkdtemp()), FakeChannel(), plugins=[fake_providers])
        with self.assertRaisesRegex(ConfigError, r"\[approvals\] is not a known section"):
            app.load_plugins()


class HooksTest(unittest.IsolatedAsyncioTestCase):
    """The leftover sweep at start, the kill-all at shutdown, and their order."""

    def setUp(self):
        self.addCleanup(proc.keep_records, None)

    def leftover(self, data: Path) -> subprocess.Popen:
        """A process group a crashed Arti left behind, recorded under data's run/children."""
        child = subprocess.Popen(["/bin/sleep", "999"], start_new_session=True)
        threading.Thread(target=child.wait, daemon=True).start()  # launchd would reap it
        self.addCleanup(child.wait)
        self.addCleanup(lambda: proc._signal(child.pid, 9))
        records = data / proc.CHILDREN_DIR
        records.mkdir(parents=True, exist_ok=True)
        (records / str(child.pid)).touch()
        return child

    def test_start_sweeps_leftovers_and_records_children_and_stop_kills_them(self):
        data = Path(tempfile.mkdtemp())
        child = self.leftover(data)
        app = App(make_config(), data, FakeChannel())
        with self.assertLogs("plugins.safety", "WARNING"):
            safety.start(app)
        child.wait(timeout=5)
        spawned = proc.spawn(["/bin/sleep", "999"], env=ENV, stdout=proc.DEVNULL)
        record = data / proc.CHILDREN_DIR / str(spawned.pgid)
        self.assertTrue(record.exists())  # recorded, for the next start's sweep
        safety.stop(app)
        self.assertIsNotNone(spawned.proc.poll())
        self.assertFalse(record.exists())

    async def test_it_starts_before_and_stops_after_the_plugins_that_require_it(self):
        order = []

        def dependent(api):  # what a plugin that requires safety does: start a child, stop it
            api.on_start(lambda app: order.append(("dependent starts", proc._records)))
            api.on_stop(lambda app: order.append(("dependent stops", dict(proc._live))))

        live = proc.spawn(["/bin/sleep", "999"], env=ENV, stdout=proc.DEVNULL)
        data = Path(tempfile.mkdtemp())
        app = await make_app(data, plugins=[safety.register, dependent], start=False)
        with mock.patch.object(lock, "acquire"):
            await app.start()
        await app.shutdown()
        (first, records), (second, alive) = order
        self.assertEqual((first, records), ("dependent starts", data / proc.CHILDREN_DIR))
        self.assertEqual(second, "dependent stops")
        self.assertIn(live.pgid, alive)  # safety's kill-all had not run yet
        self.assertIsNotNone(live.proc.poll())  # and then it did

    async def test_once_also_sweeps(self):
        data = Path(tempfile.mkdtemp())
        child = self.leftover(data)
        app = await make_app(data, plugins=[safety.register], start=False)
        with mock.patch.object(lock, "acquire"), self.assertLogs("plugins.safety", "WARNING"):
            await app.start(serve=False)  # --once
        await app.shutdown()
        child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
