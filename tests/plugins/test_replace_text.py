"""replace_text: the patch rule of the files and skills plugins (each keeps its own copy, so both
are checked here)."""

import unittest

from arti.tools import ToolError
from plugins import files, skills


class FilesReplace(unittest.TestCase):
    replace = staticmethod(files.replace_text)

    def test_exact_and_ambiguous(self):
        self.assertEqual(self.replace("a b a", "b", "c"), ("a c a", "exact"))
        with self.assertRaisesRegex(ToolError, "matches 2 times"):
            self.replace("a b a", "a", "c")
        self.assertEqual(self.replace("a b a", "a", "c", replace_all=True), ("c b c", "exact"))

    def test_trimmed_reindents(self):
        text = "def f():\n    if x:\n        a()\n        b()\n"
        new, how = self.replace(text, "if x:\n    a()\n    b()", "if y:\n    c()")
        self.assertEqual(how, "trimmed")
        self.assertEqual(new, "def f():\n    if y:\n        c()\n")

    def test_not_found(self):
        with self.assertRaisesRegex(ToolError, "not found"):
            self.replace("abc", "xyz", "q")


class SkillsReplace(FilesReplace):
    replace = staticmethod(skills.replace_text)


if __name__ == "__main__":
    unittest.main()
