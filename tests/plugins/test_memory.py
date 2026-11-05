import tempfile
import types
import unittest
from pathlib import Path

from arti.config import ConfigError
from arti.tools import ToolError
from plugins import memory
from tests.support import make_app, make_config, scope

CONFIG = {"memory": {"user_chars": 60, "notes_chars": 2200}}
HOME = "home"
PLACES = {  # the channel's names of the groups' places
    scope("chat1"): "Dinner crew",
    scope("chat2"): "Work",
    scope("chat3"): "Climbing",
    scope("chat4"): "Book club",
}


def scope_for(chat: str) -> str:
    return "home" if chat == HOME else scope(chat)


CONV = types.SimpleNamespace(trusted_prompt_writes=False)  # an ordinary chat, not the reviewer


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        app = types.SimpleNamespace(
            config=CONFIG, data_dir=self.dir, owner_name="Sam", place_name=PLACES.get
        )
        self.ctx = self.in_chat(HOME, app)

    def in_chat(self, chat, app=None):
        app = app or self.ctx.app
        scope = scope_for(chat)
        # A group's own folder is elsewhere: memory must still be the one in the data dir.
        return types.SimpleNamespace(
            app=app,
            scope=scope,
            trusted=scope == "home",
            context_dir=self.dir / "contexts" / scope,
            conv=CONV,
        )

    def call(self, target, *ops, ctx=None):
        return memory.memory({"target": target, "operations": list(ops)}, ctx or self.ctx)

    def test_batch_add_replace_remove(self):
        self.call(
            "notes",
            {"action": "add", "content": "likes tea"},
            {"action": "add", "content": "lives in LA"},
        )
        self.call(
            "notes",
            {"action": "replace", "old_text": "tea", "content": "likes green tea"},
            {"action": "remove", "old_text": "LA"},
        )
        self.assertEqual(memory.load(self.dir, "notes"), ["likes green tea"])
        self.assertIn("likes green tea", memory.memory({"target": "notes"}, self.ctx))
        self.assertTrue((self.dir / "memories" / "MEMORY.md").is_file())

    def test_bad_batch_changes_nothing(self):
        self.call("notes", {"action": "add", "content": "one"})
        with self.assertRaises(ToolError) as caught:
            self.call(
                "notes",
                {"action": "add", "content": "two"},
                {"action": "remove", "old_text": "zzz"},
            )
        self.assertIn("op 2", str(caught.exception))
        self.assertIn("- one", str(caught.exception))
        self.assertEqual(memory.load(self.dir, "notes"), ["one"])

    def test_budget_is_checked_on_the_final_result(self):
        self.call("user", {"action": "add", "content": "x" * 50})
        with self.assertRaises(ToolError):
            self.call("user", {"action": "add", "content": "y" * 20})
        # Removing the old entry in the same batch makes room.
        self.call(
            "user", {"action": "remove", "old_text": "xxx"}, {"action": "add", "content": "y" * 20}
        )
        self.assertEqual(memory.load(self.dir, "user"), ["y" * 20])

    def test_shrinking_is_allowed_over_a_lowered_budget(self):
        path = self.dir / "memories" / "USER.md"
        path.parent.mkdir()
        path.write_text("a" * 50 + memory.SEP + "b" * 50)  # written under a larger budget
        self.call("user", {"action": "remove", "old_text": "aaa"})
        self.assertEqual(memory.load(self.dir, "user"), ["b" * 50])

    def test_all_groups_together_fill_at_most_a_quarter(self):
        share = CONFIG["memory"]["notes_chars"] // memory.GROUP_SHARE
        lanes = [self.in_chat(f"chat{n}") for n in range(1, 5)]
        for n, ctx in enumerate(lanes[:3]):  # three groups, well within a quarter each...
            self.call("notes", {"action": "add", "content": f"{n} " + "x" * 140}, ctx=ctx)
        with self.assertRaisesRegex(ToolError, "all groups together are over their budget"):
            self.call("notes", {"action": "add", "content": "y" * 140}, ctx=lanes[3])
        entries = memory.load(self.dir, "notes")
        self.assertLessEqual(len(memory.SEP.join(entries)), share)  # ...but not a fourth
        self.call("notes", {"action": "add", "content": "z" * 400})  # the owner still has room
        self.call("notes", {"action": "remove", "old_text": "0 x"}, ctx=lanes[0])  # shrinking: ok
        self.call("notes", {"action": "add", "content": "y" * 140}, ctx=lanes[3])  # room again
        self.assertIn("[in Book club] " + "y" * 140, memory.load(self.dir, "notes"))

    def test_ambiguous_and_separator(self):
        self.call(
            "notes", {"action": "add", "content": "cat a"}, {"action": "add", "content": "cat b"}
        )
        with self.assertRaises(ToolError):
            self.call("notes", {"action": "remove", "old_text": "cat"})
        with self.assertRaises(ToolError):
            self.call("notes", {"action": "add", "content": "a § b"})

    def test_prompt_section(self):
        info = types.SimpleNamespace(app=self.ctx.app, scope="home", trusted=True)
        self.assertIsNone(memory.prompt_section(info))
        # In a group the sharing rule is there before anything is saved.
        scope = scope_for("chat1")
        guest = types.SimpleNamespace(app=self.ctx.app, scope=scope, trusted=False)
        owner = types.SimpleNamespace(app=self.ctx.app, scope=scope, trusted=True)
        self.assertIn("never someone's word", memory.prompt_section(guest))
        self.assertIn("shared: true", memory.prompt_section(owner))
        self.call("user", {"action": "add", "content": "Name: Alex"})
        section = memory.prompt_section(info)
        self.assertIn("USER PROFILE [10/60 chars]", section)
        self.assertIn("Name: Alex", section)
        self.assertIn("untagged entries come from Sam directly", section)
        self.assertIn("treat them as hearsay", section)
        self.assertIn("with shared: true", section)
        scope = scope_for("chat1")
        owner = types.SimpleNamespace(app=self.ctx.app, scope=scope, trusted=True, conv=CONV)
        self.assertIn("Name: Alex", memory.prompt_section(owner))  # one memory everywhere
        self.call("notes", {"action": "add", "content": "Band: Pelican", "shared": True})
        self.call("notes", {"action": "add", "content": "Kim is vegan"}, ctx=self.in_chat("chat2"))
        self.call(
            "notes",
            {"action": "add", "content": "Riley plays bass"},
            ctx=self.in_chat("chat1"),
        )
        guest = memory.prompt_section(
            types.SimpleNamespace(app=self.ctx.app, scope=scope, trusted=False, conv=CONV)
        )
        self.assertIn("[shared] Band: Pelican", guest)
        self.assertIn("[in Dinner crew] Riley plays bass", guest)
        for hidden in ("Name: Alex", "Kim is vegan", "USER PROFILE"):
            self.assertNotIn(hidden, guest)
        self.assertIn("change only the [in Dinner crew] ones", guest)

    def test_sharing_is_the_owners_and_guests_see_only_what_they_may(self):
        dinner = self.in_chat("chat1")
        self.call(
            "notes",
            {"action": "add", "content": "Sam lives at 1 Main St"},
            {"action": "add", "content": "[shared] Sam's band is Pelican", "shared": True},
        )
        self.assertIn("[shared] Sam's band is Pelican", memory.load(self.dir, "notes"))
        with self.assertRaisesRegex(ToolError, "only the owner can mark"):
            self.call("notes", {"action": "add", "content": "x", "shared": True}, ctx=dinner)
        shown = memory.memory({"target": "notes"}, dinner)
        self.assertIn("Pelican", shown)
        self.assertNotIn("Main St", shown)
        for old in ("Main St", "Pelican"):  # neither found nor revealed
            with self.assertRaises(ToolError) as caught:
                self.call("notes", {"action": "remove", "old_text": old}, ctx=dinner)
            self.assertNotIn("Main St", str(caught.exception).split("entries:")[1])

    def test_group_entries_are_tagged_and_only_they_can_change_there(self):
        dinner, work = self.in_chat("chat1"), self.in_chat("chat2")
        self.call("notes", {"action": "add", "content": "Sam is vegetarian"})
        self.call("notes", {"action": "add", "content": "Alex said Sam eats fish now"}, ctx=dinner)
        self.call("notes", {"action": "add", "content": "Kim said standup is at 9"}, ctx=work)
        self.assertEqual(
            memory.load(self.dir, "notes"),
            [
                "Sam is vegetarian",
                "[in Dinner crew] Alex said Sam eats fish now",
                "[in Work] Kim said standup is at 9",
            ],
        )
        for old in ("vegetarian", "standup"):  # the owner's and another group's: out of reach
            with self.assertRaises(ToolError) as caught:
                self.call("notes", {"action": "remove", "old_text": old}, ctx=dinner)
            self.assertIn("no entry contains", str(caught.exception))
            self.assertNotIn(old, str(caught.exception).split("entries:")[1])
        # A tag the model writes itself is dropped; the entry keeps the group's.
        self.call(
            "notes",
            {
                "action": "replace",
                "old_text": "fish",
                "content": "[in Work] Alex said Sam eats meat",
            },
            ctx=dinner,
        )
        self.assertIn("[in Dinner crew] Alex said Sam eats meat", memory.load(self.dir, "notes"))
        self.call("notes", {"action": "remove", "old_text": "Alex said"}, ctx=dinner)
        # Home may change anything.
        self.call("notes", {"action": "remove", "old_text": "standup"})
        self.assertEqual(memory.load(self.dir, "notes"), ["Sam is vegetarian"])
        # So may the owner's own lane of a group: what it adds is tagged with the group, while a
        # replaced entry keeps its own provenance (a private one does not leak to the group).
        dinner.trusted = True
        self.call(
            "notes",
            {"action": "replace", "old_text": "vegetarian", "content": "Sam is vegan"},
            ctx=dinner,
        )
        self.call("notes", {"action": "add", "content": "[shared] Alex likes jazz"}, ctx=dinner)
        self.assertEqual(
            memory.load(self.dir, "notes"), ["Sam is vegan", "[in Dinner crew] Alex likes jazz"]
        )


class SettingsTest(unittest.IsolatedAsyncioTestCase):
    async def test_memory_declares_its_sizes_and_nothing_of_the_reviewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = await make_app(tmp, plugins=[memory.register], start=False)
            app.load_plugins()
            self.assertEqual(app.config["memory"], {"user_chars": 1375, "notes_chars": 2200})
            cfg = make_config()
            cfg["memory"] = {"review": True}  # the reviewer's switch is [review] enabled
            app = await make_app(tmp, plugins=[memory.register], cfg=cfg, start=False)
            with self.assertRaisesRegex(ConfigError, "memory.review is not a known setting"):
                app.load_plugins()
