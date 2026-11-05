"""One Arti per data directory: an exclusive lock on <data>/arti.lock, held for life.

Two Arti processes on one data directory would race on its database, inbox and files, so
`App.start` takes this lock first (and a command that must not run beside a live Arti takes it
itself). The kernel drops it when the process dies, however it dies.
"""

import fcntl
import os
from pathlib import Path

LOCK_NAME = "arti.lock"

_fd: int | None = None  # held for life


class AlreadyRunning(Exception):
    """Another Arti holds the lock on this data directory."""


def acquire(data_dir: Path) -> None:
    """Take the lock on data_dir/arti.lock (Python fds are not inherited by children, so no
    child can keep it held)."""
    global _fd
    fd = os.open(Path(data_dir) / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise AlreadyRunning(f"another Arti is already running on {data_dir}") from None
    _fd = fd
