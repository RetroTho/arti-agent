"""secret: loading .env, and redacting through the registered redactor (identity without one)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from arti import secret


class EnvTest(unittest.TestCase):
    def test_values_stay_out_of_the_environment_and_the_mapping_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("SECRET_TEST_KEY=dotenv-value-1234\n")
            with mock.patch.dict(os.environ, {}, clear=True):
                secret.load_env(path)
                self.assertNotIn("SECRET_TEST_KEY", os.environ)
                self.assertEqual(secret.get("SECRET_TEST_KEY"), "dotenv-value-1234")
        self.assertEqual(secret.env()["SECRET_TEST_KEY"], "dotenv-value-1234")
        with self.assertRaises(TypeError):
            secret.env()["SECRET_TEST_KEY"] = "changed"
        secret.load_env(Path(tmp) / "missing.env")  # no file: nothing to load

    def test_bad_line_names_the_line_not_the_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("GOOD_LINE_KEY=1\nnot a line secretvalue\n")
            with self.assertRaises(ValueError) as caught:
                secret.load_env(path)
            self.assertIn("line 2", str(caught.exception))
            self.assertNotIn("secretvalue", str(caught.exception))
        self.assertNotIn("GOOD_LINE_KEY", secret.env())  # nothing of a bad file is loaded


class RedactorTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(secret.set_redactor, secret._redactor)

    def test_without_a_redactor_the_known_values_are_masked_as_written(self):
        # Before the plugins load (a broken one's traceback), or with no redactor at all.
        secret.set_redactor(None)
        secret.remember(["learned-token-value-42"])
        text = "token learned-token-value-42, key sk-ant-0123456789abcdef, pw short"
        self.assertEqual(
            secret.redact(text), "token [REDACTED], key sk-ant-0123456789abcdef, pw short"
        )  # (shapes are the redactor's job; a value under 8 characters is never masked)
        message = {"role": "user", "content": [{"type": "text", "text": text}]}
        self.assertIn("[REDACTED]", secret.redact_obj(message)["content"][0]["text"])

    def test_the_redactor_is_used_once_set(self):
        secret.set_redactor((lambda text: text.replace("pw", "**"), lambda obj: {"redacted": obj}))
        self.assertEqual(secret.redact("my pw"), "my **")
        self.assertEqual(secret.redact(""), "")
        self.assertEqual(secret.redact_obj([1]), {"redacted": [1]})


if __name__ == "__main__":
    unittest.main()
