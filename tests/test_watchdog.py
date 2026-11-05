"""arti/watchdog.py: no heartbeat for too long means stacks in the log and an exit."""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from arti import watchdog


class WatchdogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "hang.log"
        patcher = mock.patch.object(watchdog, "CHECK", 0.01)
        patcher.start()
        self.addCleanup(patcher.stop)

    def watch(self, hang_after: float) -> tuple[watchdog.Watchdog, threading.Event]:
        dog, fired = watchdog.Watchdog(self.path, hang_after), threading.Event()
        dog.hung = lambda quiet: fired.set()
        dog.start()
        self.addCleanup(dog.stop)
        return dog, fired

    def test_no_heartbeat_is_a_hang(self):
        _, fired = self.watch(0.05)
        self.assertTrue(fired.wait(2))

    def test_heartbeats_keep_it_quiet_and_stop_ends_it(self):
        dog, fired = self.watch(0.1)
        for _ in range(20):  # 0.2 s of beats, twice the limit
            dog.beat()
            time.sleep(0.01)
        self.assertFalse(fired.is_set())
        dog.stop()
        time.sleep(0.2)
        self.assertFalse(fired.is_set())

    def test_a_hang_writes_every_threads_stack_and_exits(self):
        dog = watchdog.Watchdog(self.path, 300)
        with mock.patch.object(watchdog.os, "_exit") as exit_now:
            dog.hung(301)
        exit_now.assert_called_once_with(70)
        text = self.path.read_text()
        self.assertIn("no progress for 301 s", text)
        self.assertIn("test_a_hang_writes_every_threads_stack_and_exits", text)  # this stack


if __name__ == "__main__":
    unittest.main()
