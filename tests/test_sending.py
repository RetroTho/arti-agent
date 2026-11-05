"""What Arti sends: a reply's MEDIA lines become files, and every file passes App.file_problem
(readable, no raw tool output, a place's own files only, every plugin's file check)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from arti.channel import Part
from plugins import safety
from plugins.safety import risk
from tests.support import GROUP, HOME, OWNER, lane, make_app, scope

GUEST = lane("U2")


class SendingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.app = await make_app(self.tmp.name, plugins=[safety.register])
        self.addAsyncCleanup(self.app.shutdown)
        self.data = Path(self.tmp.name)

    def file(self, *parts: str) -> Path:
        path = self.data.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        return path

    async def test_a_reply_goes_out_as_text_and_file_parts_in_order(self):
        picture = self.file("artifacts", "a.png")
        self.assertEqual(await self.app.send(HOME, f"here\nMEDIA:{picture}\nbye"), "sent")
        [(key, parts)] = self.app.channel.parts
        self.assertEqual(key, HOME)
        self.assertEqual(
            parts, [Part("text", "here"), Part("file", str(picture)), Part("text", "bye")]
        )

    async def test_a_file_that_may_not_go_out_is_a_note_and_the_rest_still_goes(self):
        key = self.file("secrets", "id_rsa")
        with mock.patch.object(risk, "is_secret_path", side_effect=lambda p: "id_rsa" in str(p)):
            await self.app.send(HOME, f"here\nMEDIA:{key}\nMEDIA:/no/such.png\nbye")
        [(_, parts)] = self.app.channel.parts
        self.assertEqual(
            [p.value for p in parts],
            [
                "here",
                "[file not sent: id_rsa is a secret file]",
                "[file not sent: such.png is not a readable file]",
                "bye",
            ],
        )

    async def test_raw_tool_output_is_never_sent(self):
        spill = self.file("cache", "tool-output", "out-1.txt")
        self.assertEqual(self.app.file_problem(HOME, spill)[:28], "out-1.txt is raw tool output")

    async def test_a_place_gets_only_its_own_files_and_home_any(self):
        place = ("contexts", scope(GROUP))
        own = self.file(*place, "workspace", "a.txt")
        generated = self.file(*place, "artifacts", "images", "b.png")
        spill = self.file(*place, "cache", "tool-output", "e.txt")
        others = self.file("artifacts", "images", "c.png")  # made in the owner's chat
        private = Path(tempfile.mkdtemp()) / "taxes.pdf"
        private.write_text("x")
        for key in (GUEST, lane(OWNER)):  # the owner's lane of a group is in the group too
            with self.subTest(key):
                self.assertIsNone(self.app.file_problem(key, own))
                self.assertIsNone(self.app.file_problem(key, generated))
                self.assertIn("raw tool output", self.app.file_problem(key, spill))
                for outside in (others, private):
                    problem = self.app.file_problem(key, outside)
                    self.assertEqual(problem, f"{outside.name} is outside this chat's files")
        self.assertIsNone(self.app.file_problem(HOME, private))

    async def test_a_link_into_a_place_is_judged_by_where_it_points(self):
        place = self.data / "contexts" / scope(GROUP)
        place.mkdir(parents=True)
        link = place / "innocent.png"
        link.symlink_to(self.file("artifacts", "c.png"))
        self.assertEqual(
            self.app.file_problem(GUEST, link), "innocent.png is outside this chat's files"
        )


if __name__ == "__main__":
    unittest.main()
