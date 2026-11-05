"""Logging: one rotating, owner-only, redacted log file at <data_dir>/logs/arti.log.

Every record from every logger (and warnings, and uncaught exceptions in any thread) goes
through a formatter that removes secrets from the finished line (through the registered
redactor, secret.redact, from the moment a plugin registers one), because the log is where a
secret would otherwise leak.
"""

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path

from arti import secret
from arti.fsutil import private_dir


class RedactingFormatter(logging.Formatter):
    """Formats a record as usual, then redacts the whole line (traceback included)."""

    def format(self, record: logging.LogRecord) -> str:
        return secret.redact(super().format(record))


def setup(data_dir: Path, verbose: bool = False) -> None:
    """Send every log record to the redacted rotating file (and stderr when `verbose`)."""
    path = private_dir(Path(data_dir) / "logs") / "arti.log"
    path.touch(mode=0o600, exist_ok=True)
    os.chmod(path, 0o600)
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(
            path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
    ]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    formatter = RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
        old.close()
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.captureWarnings(True)

    uncaught = logging.getLogger("arti.uncaught")
    sys.excepthook = lambda *exc: uncaught.critical("uncaught exception", exc_info=exc)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not SystemExit:
            uncaught.error("uncaught in thread %s", args.thread, exc_info=args[:3])

    threading.excepthook = thread_hook
    sys.unraisablehook = lambda u: uncaught.error(
        "%s: %r", u.err_msg or "Exception ignored in", u.object, exc_info=u[:3]
    )
