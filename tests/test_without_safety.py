"""The stock plugins with the safety plugin turned off: Arti still loads, unguarded, and every
plugin that requires safety (the risky tools among them) is left out."""

import copy
import tempfile
import unittest
from pathlib import Path

from arti import config
from arti.app import App
from arti.channel import FakeChannel


def loaded(disabled: list[str], provider: str) -> App:
    cfg = copy.deepcopy(config.DEFAULTS)
    cfg["providers"] = {}
    cfg["agent"]["provider"] = provider
    cfg["plugins"]["disabled"] = disabled
    app = App(cfg, Path(tempfile.mkdtemp()), FakeChannel())
    app.load_plugins()
    return app


class WithoutSafetyTest(unittest.TestCase):
    def test_only_what_needs_no_safety_is_loaded(self):
        app = loaded(["safety"], "anthropic")
        self.assertIsNone(app.gate)
        self.assertEqual(set(app.provider_specs), {"anthropic", "openai", "openai-codex"})
        for risky in ("terminal", "read_file", "write_file", "browser", "computer_use",
                      "web_extract", "image_generate", "cronjob"):  # fmt: skip
            self.assertNotIn(risky, app.tools)
        self.assertIn("todo", app.tools)

    def test_a_provider_that_needs_safety_is_gone_with_it(self):
        with self.assertRaisesRegex(config.ConfigError, "agent.provider"):
            loaded(["safety"], "claude-subscription")

    def test_with_safety_everything_loads_behind_the_gate(self):
        app = loaded([], "claude-subscription")
        self.assertIsNotNone(app.gate)
        self.assertIn("claude-subscription", app.provider_specs)
        self.assertIn("terminal", app.tools)


if __name__ == "__main__":
    unittest.main()
