import asyncio
import tempfile
import threading
import unittest
from unittest import mock

from arti import media
from arti.secret import remember
from plugins import memory, safety, session_search, skills
from tests.support import (
    GUEST,
    HOME,
    OWNER,
    lane,
    make_app,
    message,
    run_message,
    settle,
    text_reply,
    tool_reply,
)

release = threading.Event()
stopped_jobs = []
preparing = threading.Event()


def register(api):
    def slow(args, ctx):
        release.wait(5)
        return "done"

    api.tool("slow", "Slow.", {"properties": {}}, slow, review=None, guest=True)
    api.on_new_session(lambda app, key: stopped_jobs.append(key))


async def until_running(app, key=HOME):
    for _ in range(200):
        await asyncio.sleep(0.01)
        if key in app.conversations and app.conversations[key].running_tool:
            return
    raise AssertionError("the tool never started")


class ChatTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        release.clear()
        stopped_jobs.clear()
        preparing.clear()

    async def asyncTearDown(self):
        release.set()
        await self.app.shutdown()
        self.tmp.cleanup()

    async def test_stop_cancels_the_turn_and_drops_queued_input(self):
        self.app = app = await make_app(
            self.tmp.name, plugins=[register], replies=[tool_reply("slow")]
        )
        await app.admit([message("work")])
        await until_running(app)
        await app.admit([message("queued note", kind="note"), message("/stop")])
        await settle(app)
        self.assertEqual(app.channel.texts(), ["Stopped."])
        # The message is dropped; a note (context, e.g. a scheduled message that went out) is kept.
        self.assertEqual([i.text for i in app.store.pending_inputs()], ["queued note"])
        self.assertFalse(app.conversations[HOME].busy)
        self.assertEqual(app.store.open_sessions(), [])

    async def test_input_that_cannot_be_read_fails_once_not_forever(self):
        self.app = app = await make_app(self.tmp.name, plugins=[register], replies=[])
        with mock.patch.object(media, "input_blocks", side_effect=OSError("disk gone")):
            await app.admit([message("hello")])
            await settle(app)
        self.assertEqual(len(app.channel.texts()), 1)  # one failure report
        self.assertEqual(app.store.pending_inputs(), [])
        self.assertFalse(app.conversations[HOME].busy)

    async def test_input_after_stop_in_the_same_batch_is_kept(self):
        self.app = app = await make_app(
            self.tmp.name, plugins=[register], replies=[tool_reply("slow"), text_reply("doing B")]
        )
        await app.admit([message("do A")])
        await until_running(app)
        await app.admit([message("/stop"), message("do B instead")])
        await settle(app)
        self.assertEqual(app.channel.texts(), ["Stopped.", "doing B"])
        last = str(app.fake.requests[-1]["messages"][-1])
        self.assertIn("do B instead", last)
        # The next turn is told the earlier task was stopped, so it does not resume it.
        self.assertIn("Owner sent /stop here, so the task above", str(app.fake.requests[-1]))

    async def test_stop_names_who_sent_it(self):
        group = lane("U3", "chat123")
        self.app = app = await make_app(
            self.tmp.name, plugins=[register], replies=[tool_reply("slow"), text_reply("ok")]
        )
        await app.admit([message("do A", key=group, sender="U3")])
        await until_running(app, group)
        await app.admit(
            [
                message("/stop", key=group, sender="U3"),
                message("hi", key=group, sender="U3"),
            ]
        )
        await settle(app)
        self.assertIn("U3 sent /stop here", str(app.fake.requests[-1]))
        self.assertEqual(app.store.pending_inputs(), [])

    async def test_stop_while_an_attachment_is_prepared_drops_the_message(self):
        def slow_blocks(app_, inp, cancel, folder):  # a large file, being read
            preparing.set()
            cancel.wait(5)
            return [{"type": "text", "text": "file"}]

        self.app = app = await make_app(self.tmp.name, plugins=[register])
        with mock.patch.object(media, "input_blocks", slow_blocks):
            await app.admit([message("", attachments=["/tmp/big.pdf"])])
            for _ in range(200):
                await asyncio.sleep(0.01)
                if preparing.is_set():
                    break
            await app.admit([message("/stop")])
            await settle(app)
        self.assertEqual(app.fake.requests, [])
        self.assertEqual(app.store.pending_inputs(), [])  # not replayed after a restart
        self.assertEqual(app.conversations[HOME].pending, [])

    async def test_new_starts_a_fresh_session(self):
        self.app = app = await make_app(
            self.tmp.name, plugins=[register], replies=[text_reply("hi")]
        )
        await app.admit([message("hello")])
        await settle(app)
        old = app.conversations[HOME].session_id
        app.conversations[HOME].state["todo"] = ["x"]
        app.conversations[HOME].save_state()
        await app.admit([message("/new")])
        await settle(app)
        conv = app.conversations[HOME]
        self.assertNotEqual(conv.session_id, old)
        self.assertEqual((conv.messages, conv.state), ([], {}))
        self.assertEqual(stopped_jobs, [HOME])
        self.assertEqual(app.channel.texts()[-1], "Started a new conversation.")

    async def test_inputs_are_handled_in_order_one_turn_at_a_time(self):
        self.app = app = await make_app(self.tmp.name, replies=[text_reply("ok")])
        await app.admit([message("one"), message("two")])
        await settle(app)
        # Both inputs were waiting when the turn started, so they form one user message.
        self.assertEqual(len(app.fake.requests), 1)
        texts = [b["text"] for b in app.fake.requests[0]["messages"][0]["content"]]
        self.assertTrue(texts[0].endswith("one") and texts[1].endswith("two"))

    async def test_a_secret_in_the_providers_replay_copy_is_never_stored(self):
        secret = "q7W-replay-Secret/value+99"
        remember([secret])
        clean, leaky = text_reply("plain answer"), text_reply("plain answer")
        clean.native = {"provider": "p", "items": [{"type": "text", "text": "plain answer"}]}
        leaky.native = {"provider": "p", "items": [{"type": "reasoning", "text": secret}]}
        self.app = app = await make_app(
            self.tmp.name, plugins=[safety.register], replies=[clean, leaky]
        )
        await run_message(app, "one")
        await run_message(app, "two")
        replies = [
            m
            for m in app.store.history(app.conversations[HOME].session_id)
            if m["role"] == "assistant"
        ]
        self.assertEqual(replies[0]["native"], clean.native)
        self.assertNotIn("native", replies[1])
        self.assertNotIn(secret, str(replies))

    async def test_a_failing_state_save_never_fails_the_turn(self):
        self.app = app = await make_app(self.tmp.name)
        conv = app.conversation(HOME)
        conv.state["todo"] = "x"
        real = app.store.update_session
        failures = [RuntimeError("dictionary changed size during iteration")]

        def flaky(*args, **fields):  # an abandoned tool thread changed the state mid-dump
            if failures:
                raise failures.pop()
            return real(*args, **fields)

        with mock.patch.object(app.store, "update_session", side_effect=flaky):
            conv.save_state()
        self.assertEqual(app.store.session(conv.session_id)["state"], {"todo": "x"})
        with mock.patch.object(app.store, "update_session", side_effect=RuntimeError("down")):
            conv.save_state()  # logged, not raised

    async def test_chats_are_independent(self):
        self.app = app = await make_app(
            self.tmp.name,
            plugins=[register],
            replies=[tool_reply("slow"), text_reply("group reply")],
        )
        await app.admit([message("work")])
        await until_running(app)
        await app.admit([message("hello", key=lane(GUEST), sender=GUEST)])
        for _ in range(200):
            await asyncio.sleep(0.01)
            if app.channel.texts(lane(GUEST)):
                break
        self.assertEqual(app.channel.texts(lane(GUEST)), ["group reply"])
        self.assertNotEqual(app.conversations[lane(GUEST)].scope, "home")


class LaneTest(unittest.IsolatedAsyncioTestCase):
    """A group has one conversation per caller: its own history, runner, trust and /stop."""

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        release.clear()

    async def asyncTearDown(self):
        release.set()
        await self.app.shutdown()
        self.tmp.cleanup()

    async def test_trust_is_by_lane(self):
        search = tool_reply("session_search", {"action": "search", "query": "pelican"})
        self.app = app = await make_app(
            self.tmp.name,
            plugins=[skills.register, session_search.register, memory.register],
            replies=[search, text_reply("ok"), search, text_reply("ok")],
        )
        home = app.store.current_session(HOME, app.default_route())
        app.store.append(
            home["id"],
            HOME,
            {"role": "user", "content": [{"type": "text", "text": "the pelican code is 42"}]},
        )
        await app.admit([message("@Arti pelican code?", key=lane(OWNER), sender=OWNER)])
        await settle(app)
        await app.admit([message("@Arti pelican code?", key=lane(GUEST), sender=GUEST)])
        await settle(app)
        owner, owner_found, guest, guest_found = app.fake.requests
        self.assertIn("skill_manage", owner["tools"])
        self.assertIn("code is 42", str(owner_found["messages"][-1]))
        self.assertEqual(sorted(guest["tools"]), ["memory", "session_search"])
        self.assertNotIn("code is 42", str(guest_found["messages"][-1]))
        self.assertTrue(app.conversations[lane(OWNER)].trusted)
        self.assertFalse(app.conversations[lane(GUEST)].trusted)

    async def test_a_strangers_stop_never_stops_the_owners_lane(self):
        self.app = app = await make_app(
            self.tmp.name,
            plugins=[register],
            replies=[
                tool_reply("slow"),
                text_reply("I can't stop Owner's task"),
                text_reply("owner task done"),
            ],
        )
        await app.admit([message("@Arti do the slow thing", key=lane(OWNER), sender=OWNER)])
        await until_running(app, lane(OWNER))
        await app.admit([message("@Arti stop that", key=lane(GUEST), sender=GUEST)])
        for _ in range(200):
            await asyncio.sleep(0.01)
            if app.channel.texts(lane(GUEST)):
                break
        await app.admit([message("/stop", key=lane(GUEST), sender=GUEST)])
        await asyncio.sleep(0.1)
        self.assertEqual(app.channel.texts(lane(GUEST)), ["I can't stop Owner's task", "Stopped."])
        self.assertTrue(app.conversations[lane(OWNER)].busy)  # both lanes ran; the owner's goes on
        release.set()
        await settle(app)
        self.assertEqual(app.channel.texts(lane(OWNER)), ["owner task done"])
        owner_requests = [r for r in app.fake.requests if "do the slow thing" in str(r)]
        self.assertNotIn("sent /stop", str(owner_requests[-1]))


if __name__ == "__main__":
    unittest.main()
