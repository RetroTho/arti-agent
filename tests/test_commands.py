import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from arti import commands
from plugins import safety
from plugins.safety import approvals
from tests.support import GUEST, OWNER, lane, make_app, run_message


def ping(api):
    """A plugin with two chat commands: one for anyone, one for the owner's chat alone."""

    async def pong(app, inp, arg):
        return f"pong {arg} from {inp.sender}"

    api.chat_command("ping", pong, "[text] - answer pong", home_only=False)
    api.chat_command("secretping", lambda app, inp, arg: "owner pong", "- answer the owner")


class ParseTest(unittest.TestCase):
    def test_parse(self):
        app = SimpleNamespace(chat_commands={})
        self.assertEqual(commands.parse(app, " /Model  openai/x "), ("model", "openai/x"))
        self.assertEqual(commands.parse(app, "/reasoning low"), ("effort", "low"))
        self.assertIsNone(commands.parse(app, "/usr/bin is a path"))
        self.assertIsNone(commands.parse(app, "hello /stop"))
        self.assertIsNone(commands.parse(app, "/ping x"))  # nobody registered it
        app.chat_commands["ping"] = object()
        self.assertEqual(commands.parse(app, "/PING x"), ("ping", "x"))


class CommandsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = await make_app(self.tmp.name)

    async def asyncTearDown(self):
        await self.app.shutdown()
        self.tmp.cleanup()

    async def test_effort_and_model(self):
        self.assertEqual(await run_message(self.app, "/effort"), ["Effort: high"])
        self.assertIn("must be one of", (await run_message(self.app, "/effort huge"))[0])
        self.assertEqual(await run_message(self.app, "/reasoning low"), ["Effort set to low."])
        self.assertEqual(self.app.conversation(self.app.home).route[2], "low")
        self.assertEqual(await run_message(self.app, "/model"), ["Model: anthropic/fake-model"])

    async def test_status_shows_usage_in_percent(self):
        conv = self.app.conversation(self.app.home)
        conv.rate_limit = {
            "status": "allowed",
            "windows": {"5h": {"used": 0.66, "resets_at": 0}, "7d": {"used": 0.12}},
        }
        [reply] = await run_message(self.app, "/status")
        self.assertIn("Usage (5h window): 66% used", reply)
        self.assertIn("Usage (7d window): 12% used", reply)

    async def test_approve_and_deny_are_the_safety_plugins(self):
        self.assertIsNone(commands.parse(self.app, "/approve ab12"))  # no safety plugin here
        app = await make_app(tempfile.mkdtemp(), plugins=[safety.register])
        self.addAsyncCleanup(app.shutdown)
        with mock.patch.object(approvals, "handle_command", return_value="Approved.") as handle:
            self.assertEqual(await run_message(app, "/approve ab12 always"), ["Approved."])
            self.assertEqual(await run_message(app, "/deny ab12"), ["Approved."])
        self.assertEqual(
            handle.call_args_list, [mock.call("approve", "ab12 always"), mock.call("deny", "ab12")]
        )
        sent = await run_message(app, "/approve", key=lane(OWNER), sender=OWNER)
        self.assertIn("works only in Owner's own chat", sent[0])
        [help_text] = await run_message(app, "/help")
        self.assertIn(
            "/approve [id] [once|session|always] - answer an approval question", help_text
        )
        self.assertIn("/deny [id] - ", help_text)

    async def test_a_plugins_chat_commands_run_where_they_may_and_are_in_help(self):
        app = await make_app(tempfile.mkdtemp(), plugins=[ping])
        self.addAsyncCleanup(app.shutdown)
        self.assertEqual(await run_message(app, "/ping hi"), [f"pong hi from {OWNER}"])
        sent = await run_message(app, "/ping there", key=lane(GUEST), sender=GUEST)
        self.assertEqual(sent, [f"pong there from {GUEST}"])  # home_only=False: anyone's lane
        self.assertEqual(await run_message(app, "/secretping"), ["owner pong"])
        sent = await run_message(app, "/secretping", key=lane(GUEST), sender=GUEST)
        self.assertIn("works only in Owner's own chat", sent[0])
        [help_text] = await run_message(app, "/help")
        self.assertTrue(help_text.startswith(commands.HELP))
        self.assertTrue(
            help_text.endswith("/ping [text] - answer pong\n/secretping - answer the owner")
        )
        self.assertEqual(app.store.pending_inputs(), [])  # a command never reaches a turn
        self.assertEqual(app.fake.requests, [])

    async def test_restart_is_the_owners_and_help_is_everyones(self):
        sent = await run_message(self.app, "/restart", key=lane(GUEST), sender=GUEST)
        self.assertIn("works only in Owner's own chat", sent[0])
        self.assertFalse(self.app.restart)
        self.assertIn("/stop", (await run_message(self.app, "/help", lane(GUEST), GUEST))[0])
        self.assertEqual(await run_message(self.app, "/restart"), ["Restarting."])
        self.assertTrue(self.app.restart and self.app.exit_requested.is_set())


if __name__ == "__main__":
    unittest.main()
