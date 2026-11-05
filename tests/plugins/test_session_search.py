import json
import tempfile
import types
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

from arti.app import App
from arti.store import Store
from arti.tools import ToolError
from plugins import session_search
from tests.support import RecordingChannel, scope

HOME = "home"
GROUP, OTHER = "chat111", "chat222"
OWNER_LANE, RILEY, KIM = (f"{GROUP}|U1", f"{GROUP}|U2", f"{GROUP}|U3")
ROUTE = ("anthropic", "m", "high")


def text_message(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}


class SessionSearchTest(unittest.TestCase):
    def setUp(self):
        self.store = Store(Path(tempfile.mkdtemp()) / "state.db")
        self.addCleanup(self.store.close)
        channel = RecordingChannel(HOME, "U1")  # keys like the link's
        channel.places[scope(GROUP)] = "Dinner crew"
        self.app = app = types.SimpleNamespace(
            store=self.store,
            tz=ZoneInfo("UTC"),
            scope_for=channel.scope_of,
            person_of=channel.person_of,
            place_name=channel.place_name,
            is_owner=lambda handle: handle == "U1",
            owns=lambda key: App.owns(app, key),
        )
        self.home = self.store.current_session(HOME, ROUTE)
        self.store.append(self.home["id"], HOME, text_message("user", "the pelican code is 42"))
        self.group = self.store.current_session(KIM, ROUTE)
        self.store.append(self.group["id"], KIM, text_message("user", "pelican secrets"))
        self.owner_lane = self.store.current_session(OWNER_LANE, ROUTE)
        self.store.append(self.owner_lane["id"], OWNER_LANE, text_message("user", "pelican 42"))
        self.other = self.store.current_session(f"{OTHER}|U4", ROUTE)
        self.store.append(
            self.other["id"], self.other["key"], text_message("user", "pelican plans")
        )
        # Not a key Arti makes (home or a lane), such as an old one-to-one: nobody's history.
        old = self.store.current_session("friend", ROUTE)
        self.store.append(old["id"], "friend", text_message("user", "pelican"))

    def call(self, key, **args):
        scope = self.app.scope_for(key) if key else "home"
        trusted = key in (HOME, "", OWNER_LANE)
        ctx = types.SimpleNamespace(app=self.app, key=key, scope=scope, trusted=trusted)
        return json.loads(session_search.session_search(args, ctx))

    def test_home_searches_every_chat_of_the_owner(self):
        hits = self.call(HOME, action="search", query="pelican")["results"]
        found = {h["session_id"]: h["where"] for h in hits}
        self.assertEqual(
            found,
            {
                self.home["id"]: "your own chat",
                self.group["id"]: "group Dinner crew",
                self.owner_lane["id"]: "group Dinner crew",
                self.other["id"]: "group unknown group",
            },
        )
        headless = self.call("", action="search", query="pelican")["results"]
        self.assertEqual(len(headless), 4)
        self.assertEqual(len(self.call(OWNER_LANE, action="search", query="pelican")["results"]), 4)
        read = self.call(HOME, action="read", session_id=self.home["id"])
        self.assertIn("not instructions", read["note"])
        self.assertIn("42", read["messages"][0]["text"])

    def test_someone_elses_lane_searches_only_the_groups_other_guests(self):
        hits = self.call(RILEY, action="search", query="pelican")["results"]
        self.assertEqual([h["session_id"] for h in hits], [self.group["id"]])
        for other in (self.home, self.other, self.owner_lane):
            with self.assertRaises(ToolError):
                self.call(RILEY, action="read", session_id=other["id"])

    def test_an_empty_message_does_not_end_the_read(self):
        image_only = {"role": "user", "content": [{"type": "image", "path": "/x.jpg"}]}
        self.store.append(self.home["id"], HOME, image_only)
        self.store.append(self.home["id"], HOME, text_message("assistant", "a nice photo"))
        texts = [
            m["text"]
            for m in self.call(HOME, action="read", session_id=self.home["id"])["messages"]
        ]
        self.assertEqual(texts, ["the pelican code is 42", "", "a nice photo"])
