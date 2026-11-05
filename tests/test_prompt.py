import tempfile
import unittest

from arti import prompt, tools
from arti.chat import Headless
from plugins import memory, skills
from tests.support import GUEST, OWNER, lane, make_app, message, scope, settle, text_reply

SKILL = '---\nname: deploy\ndescription: "Use when deploying."\n---\nStep 1: do it.\n'


def register(api):
    api.prompt_section(lambda info: f"## SECTION of {info.scope}")
    api.prompt_section(lambda info: 1 / 0)  # a broken section is skipped


class PromptTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = await make_app(
            self.tmp.name, plugins=[register, memory.register, skills.register]
        )
        self.app.channel.places[scope()] = "Crew"
        skill = self.app.data_dir / "skills" / "deploy" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(SKILL)
        (self.app.data_dir / "memories").mkdir()
        (self.app.data_dir / "memories" / "USER.md").write_text(
            "Owner likes tea\n§\n[in Crew] Riley plays bass\n§\n[in Other] Kim is vegan"
            "\n§\n[shared] Owner's band is called Pelican"
        )

    async def asyncTearDown(self):
        await self.app.shutdown()
        self.tmp.cleanup()

    def test_home_chat_prompt(self):
        conv = self.app.conversation(self.app.home)
        text = prompt.build(conv)
        self.assertTrue(text.startswith("You are Arti, Owner's"))
        self.assertTrue((self.app.data_dir / "SOUL.md").is_file())
        for part in (
            "# Working",
            "# platform test notes",
            "# Chat",
            "put MEDIA:/absolute/path on its own line",  # every channel sends files this way
            "## SECTION of home",
            "Owner likes tea",
            "- deploy: Use when deploying.",
            "This conversation started on",
        ):
            self.assertIn(part, text)
        self.assertLess(text.index("# Working"), text.index("## SECTION"))
        self.assertLess(text.index("## SECTION"), text.index("This conversation started on"))

    def test_the_owners_conversation_in_a_group_has_everything(self):
        conv = self.app.conversation(lane(OWNER))
        text = prompt.build(conv)
        self.assertTrue(text.startswith("You are Arti, Owner's"))  # the one SOUL
        self.assertFalse((conv.context_dir / "SOUL.md").exists())
        for part in ("Owner likes tea", "Kim is vegan", "- deploy: Use when deploying."):
            self.assertIn(part, text)
        self.assertIn("with shared: true", text)
        self.assertIn("skill_manage", {t.name for t in tools.tools_for(conv)})

    async def test_someone_elses_conversation_sees_no_private_context(self):
        self.app.fake.replies = [text_reply("hi")]
        await self.app.admit([message("@Arti hi", key=lane(GUEST), sender=GUEST)])
        await settle(self.app)
        request = self.app.fake.requests[0]
        text = request["system"] + str(request["messages"])
        self.assertIn("You are Arti, Owner's", text)
        self.assertIn("[in Crew] Riley plays bass", text)
        self.assertIn("[shared] Owner's band is called Pelican", text)
        for private in ("Owner likes tea", "Kim is vegan", "deploy", "with shared: true"):
            self.assertNotIn(private, text)
        offered = set(request["tools"])
        self.assertIn("memory", offered)
        self.assertFalse(offered & {"skill_view", "skill_manage"})

    def test_headless_prompt(self):
        extra = "# Job notes\nBe brief."
        conv = Headless(self.app, "chat-abc", self.app.default_route(), instructions=extra)
        text = prompt.build(conv)
        self.assertIn("# Background task", text)
        self.assertNotIn("# Chat", text)
        self.assertIn(extra, text)
        self.assertIn("Owner likes tea", text)


if __name__ == "__main__":
    unittest.main()
