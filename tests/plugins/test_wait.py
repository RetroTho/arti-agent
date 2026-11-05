"""The wait tool and the delayed notices behind it, against a real App."""

import asyncio
import tempfile
import time
import unittest

from plugins import wait as plugin
from tests.support import (
    GUEST,
    HOME,
    lane,
    make_app,
    message,
    run_message,
    settle,
    text_reply,
    tool_reply,
)


def texts(request) -> str:
    return str(request["messages"][-1]["content"])


class WaitTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = None

    async def asyncTearDown(self):
        if self.app:
            await self.app.shutdown()
        self.tmp.cleanup()

    async def start(self, *replies):
        self.app = await make_app(self.tmp.name, plugins=[plugin.register], replies=list(replies))
        return self.app

    async def test_wait_schedules_one_notice_and_returns_at_once(self):
        app = await self.start(
            tool_reply("wait", {"minutes": 30, "note": "check the oven"}), text_reply("[SILENT]")
        )
        self.assertEqual(await run_message(app, "remind me in 30 min"), [])
        self.assertEqual(len(app.fake.requests), 2)  # no nudge: the wait answered the message
        self.assertIn("End your turn now", texts(app.fake.requests[1]))
        [due] = [i for i in app.store.pending_inputs() if i.kind == "notice"]
        self.assertAlmostEqual(due.due, time.time() + 1800, delta=5)
        self.assertIn("): check the oven]", due.text)
        self.assertEqual(list(app.conversations[HOME].waits), [due.seq])

    async def test_a_wait_in_a_group_comes_back_to_the_callers_lane(self):
        app = await self.start(
            tool_reply("wait", {"minutes": 5, "note": "ask Riley again"}), text_reply("[SILENT]")
        )
        await run_message(app, "@Arti ping me in 5", lane(GUEST), GUEST)
        [due] = [i for i in app.store.pending_inputs() if i.kind == "notice"]
        self.assertEqual(due.key, lane(GUEST))

    async def test_a_due_notice_fires_into_the_chat(self):
        app = await self.start(text_reply("the oven!"))
        await app.notify(HOME, "[Arti: the wait is over: oven]", at=time.time() + 0.2)
        await asyncio.sleep(0.1)
        self.assertEqual(app.fake.requests, [])  # not yet
        await asyncio.sleep(0.3)
        await settle(app)
        self.assertIn("the wait is over: oven", texts(app.fake.requests[0]))
        self.assertEqual(app.channel.texts(), ["the oven!"])

    async def test_a_wait_survives_a_restart(self):
        app = await self.start()
        await app.notify(HOME, "[Arti: the wait is over: oven]", at=time.time() + 0.3)
        await app.shutdown()
        self.app = app = await self.start(text_reply("[SILENT]"))
        self.assertEqual(len(app.conversations[HOME].waits), 1)  # re-armed from the inbox
        await asyncio.sleep(0.5)
        await settle(app)
        self.assertIn("the wait is over: oven", texts(app.fake.requests[0]))

    async def test_new_cancels_the_chats_waits(self):
        app = await self.start()
        await app.notify(HOME, "[Arti: the wait is over: oven]", at=time.time() + 0.2)
        await app.admit([message("/new")])
        await settle(app)
        self.assertEqual(app.conversations[HOME].waits, {})
        await asyncio.sleep(0.4)
        await settle(app)
        self.assertEqual(app.fake.requests, [])
        self.assertEqual(app.store.pending_inputs(), [])


if __name__ == "__main__":
    unittest.main()
