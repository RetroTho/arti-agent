import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from arti import config
from arti.config import ConfigError

OPTIONS = {"api_key_env": "", "base_url": "", "timeout": 0.0, "max_output_tokens": 0}
TWO = {
    name: SimpleNamespace(default_model=f"default-{name}", options=OPTIONS)
    for name in ("local", "openai")
}
PROVIDERS = {"openai": TWO["openai"]}  # one provider: agent.provider "" picks it


class ConfigTest(unittest.TestCase):
    """config.load (core sections), then check_plugins as App.load_plugins runs it."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def load(
        self, text="", declared=(), skipped=(), providers=PROVIDERS, provider_keys=(), **overrides
    ):
        (self.dir / "config.toml").write_text(text)
        cfg = config.load(self.dir, **overrides)
        config.check_plugins(cfg, set(declared), set(skipped), providers, provider_keys)
        return cfg

    def assertInvalid(self, text, fragment, **kwargs):
        with self.assertRaises(ConfigError) as caught:
            self.load(text, **kwargs)
        self.assertIn(fragment, str(caught.exception))

    def test_defaults_and_provider_default_model(self):
        cfg = self.load()
        self.assertEqual(cfg["agent"]["provider"], "openai")  # the only one registered
        self.assertEqual(cfg["agent"]["model"], "default-openai")
        self.assertEqual(cfg["storage"]["cache_max_age_days"], 7)
        # Every registered provider's table is there, its options filled in.
        self.assertEqual(cfg["providers"]["openai"], OPTIONS)

    def test_overrides(self):
        cfg = self.load('[agent]\nmodel = "m1"\nrequest_timeout = 60\n')
        self.assertEqual(cfg["agent"]["model"], "m1")
        self.assertEqual(cfg["agent"]["request_timeout"], 60)
        # A command-line provider resets the file's model: a model belongs to its provider.
        text = '[agent]\nprovider = "local"\nmodel = "m1"\n'
        cfg = self.load(text, providers=TWO, provider="openai", effort="low")
        self.assertEqual(cfg["agent"]["model"], "default-openai")
        self.assertEqual(cfg["agent"]["effort"], "low")

    def test_rejections(self):
        self.assertInvalid("[agent]\nmodle = 'x'\n", "agent.modle is not a known setting; known: ")
        self.assertInvalid("[agent]\nmodle = 'x'\n", "provider, model, effort")
        self.assertInvalid("[agnet]\n", "[agnet] is not a known section")
        self.assertInvalid("[agent]\neffort = 'huge'\n", "agent.effort must be one of")
        self.assertInvalid("[agent]\nmax_iterations = 2.5\n", "must be a whole number")
        self.assertInvalid("[agent]\ntool_timeout = 0\n", "must be greater than 0")
        self.assertInvalid("[agent]\ntimezone = 'Mars/Base'\n", "time zone")
        self.assertInvalid("[agent]\nprovider = 'nope'\n", "agent.provider must be one of")
        self.assertInvalid("[providers.nope]\n", "not a known provider")
        self.assertInvalid("[providers.openai]\nkey = 'x'\n", "providers.openai.key")
        self.assertInvalid("[providers]\nopenai = 1\n", "providers.openai must be a table")
        self.assertInvalid("agent = 1\n", "agent must be a table")
        self.assertInvalid("not toml [", "invalid configuration")
        with self.assertRaises(ConfigError):
            self.load(effort="huge")

    def test_the_core_holds_only_core_sections(self):
        core = ["agent", "storage", "plugins", "providers"]
        self.assertEqual(list(config.CORE), core)
        # Plugin settings are the plugins' own: unknown to the core until a plugin declares them.
        self.assertInvalid("[agent]\nclaude_bin = 'x'\n", "agent.claude_bin is not a known setting")
        for section in (
            "approvals",
            "network",
            "web",
            "browser",
            "voice",
            "image",
            "cron",
            "review",
        ):
            self.assertInvalid(f"[{section}]\n", f"[{section}] is not a known section")

    def test_a_plugin_setting_naming_a_provider_must_name_a_registered_one(self):
        text = "[demo]\nprovider = 'nope'\n"
        keys = {"declared": ["demo"], "provider_keys": ["demo.provider"]}
        self.assertInvalid(text, "demo.provider must be one of openai", **keys)
        self.assertEqual(
            self.load("[demo]\nprovider = 'openai'\n", **keys)["demo"]["provider"], "openai"
        )
        self.assertEqual(self.load("[demo]\nprovider = ''\n", **keys)["demo"]["provider"], "")

    def test_an_empty_provider_needs_exactly_one_registered(self):
        self.assertInvalid("", "agent.provider must be set: one of local, openai", providers=TWO)
        self.assertInvalid(
            "", "must be set: one of (none: no provider plugin is loaded)", providers={}
        )
        self.assertInvalid("[agent]\nprovider = 'nope'\n", "agent.provider must be one of openai")
        self.assertEqual(
            self.load("[agent]\nprovider = 'local'\n", providers=TWO)["agent"]["provider"], "local"
        )
        self.assertEqual(config.pick("agent.provider", "", {"only": 1}, "provider"), "only")

    def test_context_window_may_be_zero(self):
        self.assertEqual(self.load("[agent]\ncontext_window = 0\n")["agent"]["context_window"], 0)

    def test_base_url_must_be_https_or_loopback(self):
        self.assertInvalid("[providers.openai]\nbase_url = 'http://example.com'\n", "base_url")
        self.assertInvalid("[providers.openai]\nbase_url = 'https://u:p@x.com'\n", "base_url")
        for url in ("https://api.example.com/v1", "http://127.0.0.1:8080", "http://localhost:1"):
            cfg = self.load(f"[providers.openai]\nbase_url = '{url}'\ntimeout = 30\n")
            self.assertEqual(cfg["providers"]["openai"]["base_url"], url)
            self.assertEqual(cfg["providers"]["openai"]["max_output_tokens"], 0)  # filled in

    def test_a_plugin_section_is_kept_raw_until_its_plugin_checks_it(self):
        (self.dir / "config.toml").write_text("[memory]\nuser_chars = 'lots'\n")
        cfg = config.load(self.dir)  # the core does not know [memory]
        self.assertEqual(cfg["memory"], {"user_chars": "lots"})

    def test_plugin_sections_declared_unknown_and_skipped(self):
        self.assertEqual(self.load("[memory]\n", declared=["memory"])["memory"], {})
        self.assertInvalid("[memory]\n", "[memory] is not a known section; known: agent")
        # A disabled (or broken) plugin's section is ignored, and dropped.
        self.assertNotIn("memory", self.load("[memory]\nx = 1\n", skipped=["memory"]))
        # ...also one not named after it ([image] is the openai plugin's), and its provider table.
        cfg = self.load("[image]\nmodel = 'x'\n[providers.openai-codex]\ntimeout = 5\n",
                        skipped=["openai"])  # fmt: skip
        self.assertNotIn("image", cfg)
        self.assertNotIn("openai-codex", cfg["providers"])

    def test_check_section_fills_defaults_and_applies_the_rules(self):
        defaults = {"size": 10, "mode": "a", "on": True}
        table = config.check_section("demo", {"size": 3}, defaults, {"mode": ("a", "b")})
        self.assertEqual(table, {"size": 3, "mode": "a", "on": True})
        for bad, fragment in (
            ({"sise": 1}, "demo.sise is not a known setting; known: size, mode, on"),
            ({"size": 0}, "demo.size must be greater than 0"),
            ({"mode": "c"}, "demo.mode must be one of a, b"),
            ({"on": "yes"}, "demo.on must be true or false"),
        ):
            with self.assertRaises(ConfigError) as caught:
                config.check_section("demo", bad, defaults, {"mode": ("a", "b")})
            self.assertIn(f"invalid configuration: {fragment}", str(caught.exception))

    def test_data_dir_follows_arti_home(self):
        with mock.patch.dict("os.environ", {"ARTI_HOME": str(self.dir)}):
            self.assertEqual(config.data_dir(), self.dir)
        self.assertTrue((config.code_dir() / "arti" / "config.py").is_file())
