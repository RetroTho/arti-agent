"""The host (host/host.py): routing by the sender's handle, and carrying out an agent's requests
without trusting it. The last test runs a real agent on the link, in this process."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from arti import link
from arti.fsutil import write_json
from host.channel import Channel, Message
from host.host import Host
from host.users import Users
from tests.support import make_app, settle, text_reply


class FakeChannel(Channel):
    name = "fake"
    notes = "# Fake notes"

    def __init__(self):
        self.sent, self.typed, self.status = [], [], "sent"

    async def send(self, chat, parts):
        self.sent.append((chat, [(p.kind, p.value) for p in parts]))
        return self.status

    async def typing(self, chat, on):
        self.typed.append((chat, on))


class HostTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.users = Users(self.data)
        self.sam = self.users.add("sam", "Sam", "fake", "U-SAM", "UTC", "anthropic")
        self.alex = self.users.add("alex", "Alex", "fake", "U-ALEX", "UTC", "anthropic")
        self.channel = FakeChannel()
        self.host = Host(self.data, self.channel, None)
        for user in (self.sam, self.alex):
            self.host.prepare(user)

    def paths(self, user):
        return link.folders(user.agent_data(self.data))

    def inputs(self, user) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.paths(user)["in"].glob("*.json"))]

    def request(self, user, value: dict) -> str:
        request_id = value.setdefault("id", os.urandom(8).hex())
        write_json(self.paths(user)["out"] / f"{request_id}.json", value)
        return request_id

    def answer(self, user, request_id: str) -> str:
        return json.loads((self.paths(user)["done"] / f"{request_id}.json").read_text())["status"]

    async def test_a_message_reaches_only_its_senders_agent(self):
        self.assertTrue(await self.host.receive(Message("m1", "chat-j", "U-SAM", "Sam", "hi")))
        group = Message("m2", "room@g", "U-ALEX", "Alex", "yo", place="room", place_name="Room")
        self.assertTrue(await self.host.receive(group))
        with self.assertLogs("host.host", "INFO"):
            self.assertFalse(await self.host.receive(Message("m3", "chat-x", "U-STRANGER")))
        [mine] = self.inputs(self.sam)
        self.assertEqual((mine["event_id"], mine["key"], mine["text"]), ("fake:m1", "home", "hi"))
        [theirs] = self.inputs(self.alex)
        self.assertEqual((theirs["key"], theirs["place"]), ("room|U-ALEX", "Room"))
        self.assertEqual(self.host.chat_of(self.alex, "room|U-ALEX"), "room@g")
        self.assertIsNone(self.host.chat_of(self.sam, "room|U-ALEX"))  # not Sam's
        me = json.loads(self.paths(self.sam)["me"].read_text())
        self.assertEqual(me, {"owner": "U-SAM", "notes": "# Fake notes"})

    async def test_an_attachment_moves_into_the_users_home(self):
        download = self.data / "download.png"
        download.write_bytes(b"png")
        await self.host.receive(Message("m1", "c", "U-SAM", files=[download]))
        [inp] = self.inputs(self.sam)
        [path] = inp["attachments"]
        self.assertTrue(path.startswith("/home/arti/.arti/media/incoming/"))
        self.assertTrue(self.host.host_path(self.sam, path))
        self.assertFalse(download.exists())

    async def test_sends_go_only_to_the_users_own_chats_and_files_only_from_their_home(self):
        home = self.sam.home(self.data)
        (home / "a.png").write_bytes(b"x")
        (home / "escape.png").symlink_to(self.data / "users.json")  # a link out of the home
        parts = [
            {"kind": "text", "value": "look"},
            {"kind": "file", "value": "/home/arti/a.png"},
            {"kind": "file", "value": "/home/arti/escape.png"},
            {"kind": "file", "value": "/etc/passwd"},
        ]
        ok = self.request(self.sam, {"type": "send", "key": "home", "parts": parts})
        stolen = self.request(self.sam, {"type": "send", "key": "room|U-ALEX", "parts": []})
        await self.host.serve(self.sam)
        self.assertEqual((self.answer(self.sam, ok), self.answer(self.sam, stolen)),
                         ("sent", "failed"))  # fmt: skip
        [(chat, sent)] = self.channel.sent
        self.assertEqual(chat, "U-SAM")  # FakeChannel's home_chat is the handle
        not_sent = ("text", "[file not sent]")
        self.assertEqual(sent, [("text", "look"), ("file", str((home / "a.png").resolve())),
                                not_sent, not_sent])  # fmt: skip

    async def test_bad_requests_are_dropped_and_typing_shown(self):
        out = self.paths(self.sam)["out"]
        (out / "bad.json").write_text("{not json")
        write_json(out / "x1.json", {"id": "x2", "type": "send", "key": "home", "parts": []})
        self.request(self.sam, {"type": "typing", "key": "home", "on": True})
        with self.assertLogs("host.host", "WARNING"):
            await self.host.serve(self.sam)
        self.assertEqual(list(out.iterdir()), [])
        self.assertEqual(self.channel.typed, [("U-SAM", True)])
        self.assertEqual(self.channel.sent, [])

    async def test_a_send_a_crash_left_unfinished_is_answered_unknown(self):
        out = self.paths(self.sam)["out"]
        (out / "abc.taken").write_text("{}")
        self.host.prepare(self.sam)
        self.assertEqual(self.answer(self.sam, "abc"), "unknown")
        self.assertFalse((out / "abc.taken").exists())

    async def test_a_channel_that_raises_or_hangs_is_unknown(self):
        async def broken(chat, parts):
            raise RuntimeError("network")

        self.channel.send = broken
        sent = self.request(self.sam, {"type": "send", "key": "home", "parts": []})
        with self.assertLogs("host.host", "ERROR"):
            await self.host.serve(self.sam)
        self.assertEqual(self.answer(self.sam, sent), "unknown")


class EndToEndTest(unittest.IsolatedAsyncioTestCase):
    """The host and a real agent (in this process, on the same folders): a message in, a turn,
    the reply out through the channel."""

    async def test_a_message_comes_back_answered(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data = Path(tmp.name)
        sam = Users(data).add("sam", "Sam", "fake", "U-SAM", "UTC", "anthropic")
        channel = FakeChannel()
        host = Host(data, channel, None)
        await host.start()
        self.addAsyncCleanup(host.stop)
        agent_dir = sam.agent_data(data)
        agent = await make_app(
            str(agent_dir), replies=[text_reply("Hello Sam")], channel=link.Link(agent_dir)
        )
        self.addAsyncCleanup(agent.shutdown)
        self.assertEqual(agent.owner, "U-SAM")
        await host.receive(Message("m1", "U-SAM", "U-SAM", "Sam", "hi"))
        deadline = time.monotonic() + 5
        while not channel.sent and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        await settle(agent)
        self.assertEqual(channel.sent, [("U-SAM", [("text", "Hello Sam")])])
        [request] = agent.fake.requests
        self.assertIn("hi", str(request["messages"][-1]))


if __name__ == "__main__":
    unittest.main()
