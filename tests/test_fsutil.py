import os
import stat
import tempfile
import unittest
from pathlib import Path

from arti import fsutil


class FsutilTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_private_dir_is_owner_only(self):
        path = fsutil.private_dir(self.dir / "a" / "b")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)

    def test_atomic_write_replaces_whole_file_privately(self):
        path = self.dir / "f.txt"
        fsutil.atomic_write(path, "old")
        fsutil.atomic_write(path, b"new")
        self.assertEqual(path.read_text(), "new")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(os.listdir(self.dir), ["f.txt"])  # no temp file left behind

    def test_json_round_trip_missing_and_corrupt(self):
        path = self.dir / "j.json"
        self.assertEqual(fsutil.read_json(path, {"x": 1}), {"x": 1})
        fsutil.write_json(path, {"jobs": ["é"]})
        self.assertEqual(fsutil.read_json(path), {"jobs": ["é"]})
        path.write_text("{broken")
        with self.assertRaises(ValueError):
            fsutil.read_json(path)

    def test_delete_older_than_takes_old_files_and_folders_never_through_a_link(self):
        old_file, new_file, old_dir, link = (self.dir / n for n in ("old", "new", "dir", "link"))
        old_file.write_text("x")
        new_file.write_text("x")
        (old_dir / "inside").mkdir(parents=True)
        link.symlink_to(new_file)
        for path in (old_file, old_dir, link):
            os.utime(path, (0, 0), follow_symlinks=False)
        removed = fsutil.delete_older_than([old_file, new_file, old_dir, link], 3600)
        self.assertEqual((removed, sorted(os.listdir(self.dir))), (3, ["new"]))
