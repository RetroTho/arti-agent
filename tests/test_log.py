import logging
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from arti import log, secret


class LogTest(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        saved = (list(root.handlers), root.level, sys.excepthook, threading.excepthook)

        def restore():
            for handler in list(root.handlers):
                root.removeHandler(handler)
                handler.close()
            for handler in saved[0]:
                root.addHandler(handler)
            root.setLevel(saved[1])
            sys.excepthook, threading.excepthook = saved[2], saved[3]
            logging.captureWarnings(False)

        self.addCleanup(restore)

    def test_log_file_is_private_and_redacted_through_the_redactor(self):
        data_dir = Path(tempfile.mkdtemp())
        value = "sk-test-very-secret-value-123456"
        log.setup(data_dir)
        self.addCleanup(secret.set_redactor, secret._redactor)
        secret.set_redactor(None)
        logging.getLogger("arti.test").info("before a redactor: %s", value)
        secret.set_redactor((lambda text: text.replace(value, "[HIDDEN]"), lambda obj: obj))
        logging.getLogger("arti.test").info("token is %s", value)
        try:
            raise ValueError(value)
        except ValueError:
            logging.getLogger("arti.test").exception("failed")
        path = data_dir / "logs" / "arti.log"
        text = path.read_text()
        self.assertIn("token is [HIDDEN]", text)
        self.assertEqual(text.count(value), 1)  # only the line written with no redactor
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
