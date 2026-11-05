"""files plugin: paging, key redaction across pages, read-before-overwrite, patch, search."""

import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from plugins import files
from plugins.files import ToolError
from plugins.safety import risk


def make_ctx() -> SimpleNamespace:
    return SimpleNamespace(state={}, cancel=threading.Event(), app=SimpleNamespace())


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp()).resolve()
        self.ctx = make_ctx()

    def write(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text)
        return path

    def test_resolve_relative_to_the_session_cwd_like_the_review(self):
        self.assertEqual(
            risk.resolve_path("notes.txt", self.ctx.state), (Path.home() / "notes.txt").resolve()
        )
        self.assertEqual(risk.resolve_path("~/a", self.ctx.state), (Path.home() / "a").resolve())
        # After `cd /some/dir` the approval review reads ".zshrc" as /some/dir/.zshrc; the tool
        # must write that same file, not ~/.zshrc (which would dodge the review).
        self.ctx.state["cwd"] = str(self.dir)
        files.write_file({"path": ".zshrc", "content": "x\n"}, self.ctx)
        self.assertEqual((self.dir / ".zshrc").read_text(), "x\n")
        reviewed = risk._resolve(".zshrc", self.dir)  # what approvals.check assesses
        self.assertEqual(str(risk.resolve_path(".zshrc", self.ctx.state)).casefold(), reviewed)
        # search_files without a path searches the cwd, which is what the review looked at.
        self.write("here.txt", "needle\n")
        self.assertIn("here.txt", files.search_files({"pattern": "needle"}, self.ctx))

    def test_read_pages_with_line_numbers(self):
        path = self.write("a.txt", "".join(f"line {n}\n" for n in range(1, 11)))
        out = files.read_file({"path": str(path), "offset": 3, "limit": 2}, self.ctx)
        self.assertEqual(out, "3|line 3\n4|line 4\n[lines 3-4 of 10; continue with offset=5]")
        out = files.read_file({"path": str(path), "offset": 9}, self.ctx)
        self.assertTrue(out.endswith("[end of file, 10 lines]"))

    def test_page_char_cap_and_long_lines(self):
        path = self.write("big.txt", ("x" * 3000 + "\n") * 20)
        out = files.read_file({"path": str(path)}, self.ctx)
        self.assertLessEqual(len(out), files.PAGE_CHARS + 200)
        self.assertIn("… [line cut]", out)
        self.assertIn("of 20; continue with offset=", out)

    def test_refuses_binary_and_directories(self):
        path = self.dir / "b.bin"
        path.write_bytes(b"\x89PNG\0\0data")
        with self.assertRaisesRegex(ToolError, "binary"):
            files.read_file({"path": str(path)}, self.ctx)
        with self.assertRaisesRegex(ToolError, "directory"):
            files.read_file({"path": str(self.dir)}, self.ctx)

    def test_private_key_redacted_even_mid_block(self):
        key = "-----BEGIN RSA PRIVATE KEY-----\nAAAA\nBBBB\nCCCC\n-----END RSA PRIVATE KEY-----\n"
        path = self.write("k.pem", "intro\n" + key + "after\n")
        whole = files.read_file({"path": str(path)}, self.ctx)
        page = files.read_file({"path": str(path), "offset": 4}, self.ctx)
        for out in (whole, page):
            self.assertNotIn("BBBB", out)
            self.assertNotIn("CCCC", out)
        self.assertIn("4|[REDACTED]", page)
        self.assertIn("7|after", page)

    def test_overwrite_needs_a_fresh_read(self):
        path = self.write("w.txt", "old\n")
        with self.assertRaisesRegex(ToolError, "Read"):
            files.write_file({"path": str(path), "content": "new\n"}, self.ctx)
        files.read_file({"path": str(path)}, self.ctx)
        files.write_file({"path": str(path), "content": "new\n"}, self.ctx)
        self.assertEqual(path.read_text(), "new\n")
        # Changed on disk since: refused again.
        path.write_text("someone else's edit, longer\n")
        with self.assertRaisesRegex(ToolError, "Read"):
            files.write_file({"path": str(path), "content": "mine\n"}, self.ctx)

    def test_write_creates_parents_and_keeps_mode(self):
        path = self.dir / "new" / "deep" / "f.sh"
        files.write_file({"path": str(path), "content": "echo hi\n"}, self.ctx)
        os.chmod(path, 0o755)
        files.read_file({"path": str(path)}, self.ctx)
        files.write_file({"path": str(path), "content": "echo bye\n"}, self.ctx)
        self.assertEqual(path.stat().st_mode & 0o777, 0o755)
        self.assertEqual(path.read_text(), "echo bye\n")

    def test_patch_exact_and_diff(self):
        path = self.write("p.py", "a = 1\nb = 2\n")
        out = files.patch(
            {"path": str(path), "old_string": "b = 2", "new_string": "b = 3"}, self.ctx
        )
        self.assertEqual(path.read_text(), "a = 1\nb = 3\n")
        self.assertIn("-b = 2\n+b = 3", out)

    def test_patch_unread_does_not_grant_overwrite(self):
        path = self.write("p.txt", "x\n")
        files.patch({"path": str(path), "old_string": "x", "new_string": "y"}, self.ctx)
        with self.assertRaises(ToolError):
            files.write_file({"path": str(path), "content": "z"}, self.ctx)

    def test_patch_ambiguous_and_missing(self):
        path = self.write("p.txt", "x\nx\n")
        with self.assertRaisesRegex(ToolError, "2 times"):
            files.patch({"path": str(path), "old_string": "x", "new_string": "y"}, self.ctx)
        files.patch(
            {"path": str(path), "old_string": "x", "new_string": "y", "replace_all": True}, self.ctx
        )
        self.assertEqual(path.read_text(), "y\ny\n")
        with self.assertRaisesRegex(ToolError, "not found"):
            files.patch({"path": str(path), "old_string": "zzz", "new_string": "y"}, self.ctx)

    def test_patch_keeps_crlf(self):
        path = self.dir / "w.txt"
        path.write_bytes(b"one\r\ntwo\r\n")
        files.patch({"path": str(path), "old_string": "one\ntwo", "new_string": "1\n2"}, self.ctx)
        self.assertEqual(path.read_bytes(), b"1\r\n2\r\n")

    def test_search_content_files_and_secret_rows(self):
        self.write("a.txt", "needle one\nhay\n")
        (self.dir / "sub").mkdir()
        self.write("sub/b.md", "needle two\n")
        (self.dir / ".hidden").mkdir()
        self.write(".hidden/c.txt", "needle hidden\n")
        out = files.search_files({"pattern": "needle", "path": str(self.dir)}, self.ctx)
        self.assertIn("a.txt:1:needle one", out)
        self.assertIn("b.md:1:needle two", out)
        self.assertNotIn("hidden", out)
        names = files.search_files(
            {"pattern": "*.md", "path": str(self.dir), "target": "files"}, self.ctx
        )
        self.assertEqual(names.strip(), str(self.dir / "sub" / "b.md"))
        secret = str(self.dir / "a.txt")
        with mock.patch.object(files.risk, "is_secret_path", lambda p: str(p) == secret):
            out = files.search_files({"pattern": "needle", "path": str(self.dir)}, self.ctx)
        self.assertNotIn("needle one", out)
        self.assertIn("needle two", out)

    def test_search_paging_and_count(self):
        self.write("many.txt", "".join(f"hit {n}\n" for n in range(30)))
        out = files.search_files(
            {"pattern": "hit", "path": str(self.dir), "limit": 10, "offset": 10}, self.ctx
        )
        count = files.search_files(
            {"pattern": "hit", "path": str(self.dir), "output_mode": "count"}, self.ctx
        )
        self.assertIn("many.txt:11:hit 10", out)
        self.assertIn("[30 rows; more with offset=20]", out)
        self.assertTrue(count.endswith("many.txt:30"))


if __name__ == "__main__":
    unittest.main()
