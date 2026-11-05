"""The hang watchdog: when Arti stops making progress, record where it is stuck and exit, so
its service (launchd or systemd: service.py) starts it again.

Everything Arti does runs through one asyncio event loop, so the loop is what is watched. A
heartbeat task on the loop (App._heartbeat) calls `beat()` every BEAT seconds, but only while the
channel still reads new messages (`Channel.alive`). A plain thread here checks the time since
the last beat. A blocked loop (a call that should never block it, a deadlock, code that never
returns) or a dead message watcher stops the beats; after `hang_after` seconds the thread writes
every thread's stack to logs/hang.log and ends the process. The service starts Arti again, and
the usual recovery takes over: inbox rows are handled once, sends that were in flight become
"outcome unknown", an interrupted turn resumes with a notice.

Time is measured with time.monotonic, which stands still while the computer sleeps (on macOS
and Linux alike), so a night asleep is not a hang. Not covered: a C call that holds the
interpreter lock forever (this thread cannot run either) and a process stopped from outside.
"""

import faulthandler
import os
import threading
import time
from datetime import datetime
from pathlib import Path

BEAT = 30  # seconds between heartbeats
CHECK = 10  # seconds between the watchdog's looks


class Watchdog:
    def __init__(self, path: Path, hang_after: float):
        self.path, self.hang_after = Path(path), hang_after
        self.last = time.monotonic()
        self.stopped = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._watch, name="watchdog", daemon=True).start()

    def beat(self) -> None:
        """Arti made progress: the countdown starts again."""
        self.last = time.monotonic()

    def stop(self) -> None:
        """A clean shutdown finished: nothing more to watch."""
        self.stopped.set()

    def _watch(self) -> None:
        while not self.stopped.wait(CHECK):
            quiet = time.monotonic() - self.last
            if quiet > self.hang_after:
                self.hung(quiet)

    def hung(self, quiet: float) -> None:
        """Write every thread's stack and end the process at once. Not through logging: a stuck
        thread may hold its lock. The stacks name files, lines and functions, never values."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S}: no progress for {quiet:.0f} s; ")
            f.write("Arti exits so its service starts it again ===\n")
            f.flush()
            faulthandler.dump_traceback(f, all_threads=True)
        os._exit(70)  # EX_SOFTWARE: the service restarts it
