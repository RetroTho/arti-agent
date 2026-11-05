"""reply.parts: a reply's MEDIA lines become file parts; everything else stays text as written."""

import unittest

from arti.channel import Part
from arti.reply import parts


class PartsTest(unittest.TestCase):
    def test_media_lines_become_files_in_order(self):
        self.assertEqual(
            parts("Look\nMEDIA:/tmp/a b.png\n\n**nice**"),
            [Part("text", "Look"), Part("file", "/tmp/a b.png"), Part("text", "**nice**")],
        )

    def test_a_media_line_in_code_is_text_and_fences_are_kept(self):
        text = "```\nMEDIA:/etc/x\n```"
        self.assertEqual(parts(text), [Part("text", text)])

    def test_only_a_whole_line_with_an_absolute_path_is_a_file(self):
        for text in ("see MEDIA:/a.png", "MEDIA:a.png", "MEDIA: /a.png"):
            with self.subTest(text):
                self.assertEqual(parts(text), [Part("text", text)])

    def test_nothing_but_blank_lines_is_no_part(self):
        self.assertEqual(parts("\n \n"), [])


if __name__ == "__main__":
    unittest.main()
