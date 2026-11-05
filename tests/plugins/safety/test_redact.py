"""Redaction and child environments (plugins/safety/redact.py), also of the core's .env."""

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote

from arti import secret
from plugins.safety import redact
from plugins.safety.redact import REDACTED

KNOWN = "zq8V-known-Secret/value+42"


class RedactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        secret.remember([KNOWN, "short"])

    def test_known_value_in_every_encoding(self):
        raw = KNOWN.encode()
        cases = [
            f"value: {KNOWN} end",
            json.dumps({"x": KNOWN}),
            "url?q=" + quote(KNOWN, safe=""),
            base64.b64encode(raw).decode(),
            base64.urlsafe_b64encode(raw).decode(),
        ]
        # base64 of a longer text, with the secret at each byte offset.
        for prefix in ("", "a", "ab", "abc", "user:"):
            cases.append(base64.b64encode(prefix.encode() + raw + b"\n").decode())
        for text in cases:
            with self.subTest(text=text):
                out = redact.redact(text)
                self.assertIn(REDACTED, out)
                self.assertNotIn(KNOWN, out)
                self.assertNotIn(base64.b64encode(raw).decode()[4:16], out)

    def test_short_values_are_not_redacted(self):
        self.assertEqual(redact.redact("a short word"), "a short word")

    def test_shapes(self):
        cases = {
            "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\ndef\n-----END OPENSSH PRIVATE KEY-----": REDACTED,
            "Authorization: Basic dXNlcjpwYXNz": "Authorization: " + REDACTED,
            "curl -H 'Bearer abcdefghijklmnop123'": "curl -H 'Bearer " + REDACTED + "'",
            "postgres://me:hunter22pw@db/x": "postgres://me:" + REDACTED + "@db/x",
            "key sk-ant-api03-abcdefghijklmnopqrstu": "key " + REDACTED,
            "ghp_abcdefghijklmnopqrstuvwxyz0123": REDACTED,
            "AKIAABCDEFGHIJKLMNOP": REDACTED,
            "GITHUB_TOKEN=abc123def": "GITHUB_TOKEN=" + REDACTED,
            'export API_KEY="x y z"': f'export API_KEY="{REDACTED}"',
            '{"password": "pa ss word"}': f'{{"password": "{REDACTED}"}}',
            "client_secret: s3cr3tvalue": "client_secret: " + REDACTED,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(redact.redact(text), expected)

    def test_left_alone(self):
        for text in [
            "API_KEY=$API_KEY",
            "token=None",
            "password == other",
            "input_tokens=1234567",
            "api_key_env=OPENAI_API_KEY",
            "TOKEN_PATH=/tmp/x",
            "API_KEY=[REDACTED]",
            "Bearer token",
        ]:
            with self.subTest(text=text):
                self.assertEqual(redact.redact(text), text)

    def test_registering_rebuilds_the_matcher(self):
        later = "registered-later-0987"
        self.assertEqual(redact.redact(later), later)
        secret.remember([later])
        self.assertEqual(redact.redact(later), REDACTED)
        # The codex login re-registers the same tokens on every request: no rebuild then.
        matcher = redact._known_matcher()
        secret.remember([later])
        self.assertIs(redact._known_matcher(), matcher)

    def test_redact_obj_keeps_images(self):
        message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "the key is " + KNOWN},
                {"type": "image", "media_type": "image/png", "data": KNOWN},
                {"type": "tool_use", "id": "1", "name": "x", "input": {"api_key": "plain"}},
            ],
            "native": {"blob": KNOWN},
        }
        out = redact.redact_obj(message)
        self.assertEqual(out["content"][0]["text"], "the key is " + REDACTED)
        self.assertEqual(out["content"][1]["data"], KNOWN)
        self.assertEqual(out["content"][2]["input"]["api_key"], REDACTED)
        # The replay copy would carry the secret the visible content no longer does: dropped.
        self.assertNotIn("native", out)

    def test_native_is_kept_only_when_nothing_needed_redacting(self):
        clean = {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}
        native = {"provider": "p", "items": [{"type": "text", "text": "hello"}]}
        self.assertEqual(redact.redact_obj({**clean, "native": native})["native"], native)
        # A secret only in the replay copy (hidden reasoning, say) is never stored.
        leaky = {"provider": "p", "items": [{"type": "reasoning", "text": "key " + KNOWN}]}
        out = redact.redact_obj({**clean, "native": leaky})
        self.assertEqual(out, clean)
        # Changed visible content: the copy no longer matches it, so it goes too.
        secret = {"role": "assistant", "content": [{"type": "text", "text": KNOWN}]}
        self.assertNotIn("native", redact.redact_obj({**secret, "native": native}))


class NamesTest(unittest.TestCase):
    def test_secret_names(self):
        for name in [
            "OPENAI_API_KEY",
            "apiKey",
            "client_secret",
            "SMTP_PASS",
            "PGPASSWORD",
            "GITHUB_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "BW_SESSION",
            "authorization",
        ]:
            self.assertTrue(redact.is_secret_name(name), name)
        for name in [
            "PWD",
            "OLDPWD",
            "api_key_env",
            "TOKEN_PATH",
            "input_tokens",
            "public_key",
            "PATH",
            "HOME",
            "SSH_AUTH_SOCK",
            "max_tokens",
        ]:
            self.assertFalse(redact.is_secret_name(name), name)


class EnvTest(unittest.TestCase):
    def test_load_env_and_child_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# comment\nexport ODDLY_NAMED='from-dotenv-1234'\nPLAIN=bare-value-99 # note\n"
                'QUOTED="a \\"q\\" b"\n'
            )
            env = {
                "PATH": "/bin",
                "OPENAI_BASE_URL": "x",
                "MY_TOKEN": "tttttttttt",
                "BASH_ENV": "/x",
                "DB": "postgres://u:pw@h/db",
                "PLAIN": "kept-existing",
            }
            with mock.patch.dict(os.environ, env, clear=True):
                secret.load_env(path)
                self.assertNotIn("ODDLY_NAMED", os.environ)  # never in Arti's own environment
                self.assertEqual(secret.get("ODDLY_NAMED"), "from-dotenv-1234")
                self.assertEqual(secret.get("PLAIN"), "kept-existing")  # the environment wins
                self.assertEqual(secret.get("QUOTED"), 'a "q" b')
                self.assertEqual(secret.get("MISSING"), "")
                self.assertEqual(redact.redact("x from-dotenv-1234"), "x " + REDACTED)
                child = redact.child_env({"WANTED_KEY": "on purpose"})
            self.assertEqual(child, {"PATH": "/bin", "WANTED_KEY": "on purpose"})

    def test_bare_env(self):
        env = redact.bare_env(X="1")
        self.assertEqual(env["PATH"], "/usr/bin:/bin:/usr/sbin:/sbin")
        self.assertEqual(env["X"], "1")
        self.assertEqual(set(env), {"PATH", "HOME", "LANG", "X"})


if __name__ == "__main__":
    unittest.main()
