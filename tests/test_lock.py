"""lock: one Arti per data directory."""

import os
import tempfile
import unittest
from pathlib import Path

from arti import lock


class LockTest(unittest.TestCase):
    def test_lock_refuses_a_second_holder(self):
        data = Path(tempfile.mkdtemp())
        lock.acquire(data)
        # flock belongs to the open file, so even a second open in this process is refused.
        with self.assertRaises(lock.AlreadyRunning):
            lock.acquire(data)
        os.close(lock._fd)  # what process death does
        lock.acquire(data)
        os.close(lock._fd)


if __name__ == "__main__":
    unittest.main()
