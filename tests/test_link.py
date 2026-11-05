"""The agent's side of the link (arti/link.py): inputs from link/in, sends through link/out."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from arti import link
from arti.channel import Part
from arti.fsutil import write_json
from tests.support import make_app, settle, text_reply


class LinkTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.paths = link.folders(self.data)
        for name in ("in", "out", "done"):
            self.paths[name].mkdir(parents=True)
        write_json(self.paths["me"], {"owner": "U1", "notes": "# Platform notes"})

    async def app(self, replies=()):
        app = await make_app(self.tmp.name, replies=replies, channel=link.Link(self.data))
        self.addAsyncCleanup(app.shutdown)
        return app

    def put(self, name: str, value: dict) -> None:
        write_json(self.paths["in"] / name, value)

    async def test_who_and_what_a_key_means(self):
        channel = link.Link(self.data)
        self.assertEqual((channel.home, channel.owner), ("home", "U1"))
        self.assertEqual(channel.prompt_notes("home"), "# Platform notes")
        self.assertEqual((channel.scope_of("home"), channel.person_of("home")), ("home", ""))
        self.assertEqual(channel.scope_of("C9|U1"), link.scope("C9"))
        self.assertEqual(channel.person_of("C9|U1"), "U1")
        self.assertIsNone(channel.person_of("C9"))

    async def test_inputs_are_admitted_in_order_then_deleted_and_a_bad_one_dropped(self):
        self.put("1-a.json", {"event_id": "chat:1", "key": "home", "kind": "message", "text": "hi"})
        self.put("2-b.json", {"event_id": "", "key": "home", "kind": "message"})  # no id
        self.put("3-c.json", {"event_id": "chat:2", "key": "C9|U1", "kind": "message",
                              "text": "there", "place": "Team"})  # fmt: skip
        with self.assertLogs("arti.link", "WARNING"):
            app = await self.app(replies=[text_reply("hello Sam"), text_reply("hi team")])
            sent = asyncio.create_task(self.answer_sends(app, "sent"))
            for _ in range(50):
                if not list(self.paths["in"].iterdir()):
                    break
                await asyncio.sleep(0.05)
        self.assertEqual(list(self.paths["in"].iterdir()), [])
        self.assertEqual(app.place_name(link.scope("C9")), "Team")
        await settle(app)
        sent.cancel()
        said = [r["parts"][0]["value"] for r in self.requests if r["type"] == "send"]
        self.assertEqual(sorted(said), ["hello Sam", "hi team"])
        self.assertEqual(
            sorted(r["key"] for r in self.requests if r["type"] == "send"), ["C9|U1", "home"]
        )

    async def answer_sends(self, app, status):
        """Stand in for the host: answer every send request with `status`."""
        self.requests = []
        while True:
            for path in self.paths["out"].glob("*.json"):
                request = json.loads(path.read_text())
                self.requests.append(request)
                path.unlink()
                if request["type"] == "send":
                    write_json(self.paths["done"] / f"{request['id']}.json", {"status": status})
            await asyncio.sleep(0.02)

    async def test_a_send_waits_for_the_hosts_answer(self):
        app = await self.app()
        host = asyncio.create_task(self.answer_sends(app, "failed"))
        parts = [Part("text", "hi"), Part("file", "/home/arti/a.png")]
        self.assertEqual(await app.channel.send("home", parts), "failed")
        await app.channel.typing("home", True)
        await asyncio.sleep(0.1)
        host.cancel()
        send, typing = self.requests
        self.assertEqual(send["parts"], [{"kind": "text", "value": "hi"},
                                         {"kind": "file", "value": "/home/arti/a.png"}])  # fmt: skip
        self.assertEqual((typing["type"], typing["on"]), ("typing", True))
        self.assertEqual(list(self.paths["done"].iterdir()), [])  # the answer was taken

    async def test_an_odd_answer_is_unknown(self):
        app = await self.app()
        host = asyncio.create_task(self.answer_sends(app, "maybe"))
        self.assertEqual(await app.channel.send("home", [Part("text", "x")]), "unknown")
        host.cancel()

    async def test_the_app_makes_the_link_when_given_no_channel(self):
        app = await make_app(self.tmp.name, channel=None, start=False)
        self.assertIsInstance(app.make_channel(), link.Link)


if __name__ == "__main__":
    unittest.main()
