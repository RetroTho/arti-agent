"""The plugin API (arti/plugin.py): finding and ordering plugins, settings, hooks, providers,
commands, and the boundaries between the core, the plugins and the host."""

import ast
import importlib
import re
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

from arti import commands, context, lock, plugin, secret
from arti.app import App
from arti.channel import FakeChannel
from arti.chat import Headless
from arti.config import ConfigError
from tests.support import fake_providers, make_app, make_config

ROOT = Path(__file__).resolve().parent.parent
# Prefixes no module of the core (arti/) may import.
FORBIDDEN = ("plugins", "host", "channels")  # the agent core runs in the sandbox, alone
# Names the core may not mention (test_the_core_names_no_plugin_tool_vendor_or_model): plugins,
# their tools and data, model vendors and models, and any path to a plugin. "memory" alone means
# the plugin ("in memory" and "in-memory" are plain words), and ".opus" is an audio file type.
CORE_NAMES = re.compile(
    r"(?i)(?<![\w.-])(?:"
    r"cron|cronjob|skills?|skill_view|skill_manage|(?-i:todo)|session_search|terminal|browser"
    r"|delegate_task|vision|image_generate|reviewer|review\.py|load_skill|stop_jobs"
    r"|claude|codex|chatgpt|anthropic|openai|azure_openai|gpt|opus|sonnet|haiku|fable"
    r")(?![a-z])"
    r"|(?<!in )(?<!in-)\bmemor(?:y|ies)\b"
    r"|\bplugins[./]\w"
)
# The safety plugin's modules and commands: the core reaches them only through the hooks it
# offers (api.gate, api.redactor, api.chat_command, start and stop hooks).
SAFETY_NAMES = re.compile(
    r"(?<![\w.])(?:approvals|risk|redact|net|proc)\.[a-z_]+|/(?:approve|deny)\b|\bsafety plugin"
)

# Group chats are the platform's: the core knows only places (Channel.scope_of) and whose a
# conversation is (Channel.person_of), never lanes, a group log, the groups or their commands.
GROUP_NAMES = re.compile(
    r"(?i)\blanes?\b|\bgroup[ _-]?log\b|/(?:allow|leave|groups)\b|\bgroups\b|\bscopes\.py\b"
)


def plugin_files(name: str) -> list[Path]:
    """The source files of plugin `name`: plugins/<name>.py, or every .py in plugins/<name>/."""
    single = ROOT / "plugins" / f"{name}.py"
    return [single] if single.is_file() else sorted((ROOT / "plugins" / name).rglob("*.py"))


def requires_of(name: str) -> tuple:
    """A plugin's REQUIRES as its source spells it (a module-level tuple), without running it."""
    single = ROOT / "plugins" / f"{name}.py"
    main = single if single.is_file() else ROOT / "plugins" / name / "__init__.py"
    for node in ast.parse(main.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets] == ["REQUIRES"]:
            return ast.literal_eval(node.value)
    return ()


def imports(path: Path) -> list[tuple[str, int]]:
    """(module, line) of every import in a file, relative ones made absolute."""
    package = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    if path.name != "__init__.py":
        package = package.rpartition(".")[0]
    else:
        package = package.removesuffix(".__init__")
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*parent, *([node.module] if node.module else [])])
            found.append((base, node.lineno))
    return found


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


REGISTER = "def register(api):\n    api.service({name!r}, lambda app: {name!r})\n"


class FindAndLoadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # A package name of its own, so it never meets the real `plugins` in sys.modules.
        self.dir = Path(self.tmp.name) / f"plugins_{uuid.uuid4().hex[:8]}"
        write(self.dir / "single.py", REGISTER.format(name="single"))
        # A package: its helper module is imported by it alone, never loaded as a plugin.
        write(
            self.dir / "pack" / "__init__.py",
            "from . import helper\n\n\ndef register(api):\n"
            "    api.service('pack', lambda app: helper.VALUE)\n",
        )
        write(self.dir / "pack" / "helper.py", "VALUE = 'from the helper'\n")
        write(self.dir / "off.py", REGISTER.format(name="off"))
        write(self.dir / "broken.py", "raise ImportError('missing dependency')\n")
        write(self.dir / "_private.py", REGISTER.format(name="_private"))
        write(self.dir / ".hidden.py", REGISTER.format(name="hidden"))
        write(self.dir / "notes" / "README.md", "not a package\n")
        (self.dir / "__pycache__").mkdir()
        self.addCleanup(
            lambda: str(self.dir.parent) in sys.path and sys.path.remove(str(self.dir.parent))
        )

    def test_modules_and_packages_are_plugins_and_nothing_else_is(self):
        self.assertEqual(plugin.find(self.dir), ["broken", "off", "pack", "single"])

    def test_load_skips_disabled_and_broken_ones(self):
        app = App(make_config(), Path(self.tmp.name), FakeChannel())
        with self.assertLogs("arti.plugin", "ERROR"):
            skipped = plugin.load_plugins(app, self.dir, {"off"})
        self.assertEqual(skipped, {"off", "broken"})
        self.assertEqual(sorted(app.services), ["pack", "single"])
        self.assertEqual(app.services["pack"](app), "from the helper")


# A plugin that records when it registered and may require others (and fail on purpose).
NEEDS = (
    "REQUIRES = {requires!r}\n"
    "ORDER = []\n\n\n"
    "def register(api):\n"
    "    if {fails!r}:\n"
    "        raise RuntimeError('register failed')\n"
    "    api.service({name!r}, lambda app: {name!r})\n"
)


class DependencyTest(unittest.TestCase):
    """REQUIRES: the load order, and skipping what cannot be loaded, transitively."""

    def load(self, plugins: dict, disabled=()) -> tuple[list[str], set[str]]:
        """Write `plugins` ({name: (requires, fails)}), load them; (the order, the skipped)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name) / f"plugins_{uuid.uuid4().hex[:8]}"
        for name, (requires, fails) in plugins.items():
            write(folder / f"{name}.py", NEEDS.format(requires=requires, fails=fails, name=name))
        self.addCleanup(lambda: sys.path.remove(str(folder.parent)))
        app = App(make_config(), Path(tmp.name), FakeChannel())
        order = []
        real = plugin.add_plugin

        def recording(app, name, register):
            order.append(name)
            return real(app, name, register)

        with mock.patch.object(plugin, "add_plugin", recording):
            skipped = plugin.load_plugins(app, folder, set(disabled))
        self.assertEqual(sorted(app.services), sorted(set(order) - skipped))
        return order, skipped

    def test_a_plugin_loads_after_what_it_requires_and_otherwise_in_name_order(self):
        order, skipped = self.load(
            {
                "alpha": (("zeta",), False),
                "beta": ((), False),
                "gamma": (("alpha", "beta"), False),
                "zeta": ((), False),
            }
        )
        self.assertEqual((order, skipped), (["beta", "zeta", "alpha", "gamma"], set()))

    def test_a_disabled_missing_or_failed_requirement_skips_its_dependents_transitively(self):
        for case, plugins, disabled in (
            ("disabled", {"base": ((), False)}, {"base"}),
            ("missing", {}, set()),
            ("failed", {"base": ((), True)}, set()),
        ):
            with self.subTest(case):
                plugins = {
                    **plugins,
                    "mid": (("base",), False),
                    "top": (("mid",), False),
                    "free": ((), False),
                }
                with self.assertLogs("arti.plugin", "ERROR") as logged:
                    order, skipped = self.load(plugins, disabled)
                self.assertEqual(skipped, {"mid", "top", *([] if case == "missing" else ["base"])})
                self.assertEqual(order, (["base"] if case == "failed" else []) + ["free"])
                self.assertIn("plugin mid skipped: it requires base, not loaded", logged.output[-2])
                self.assertIn("plugin top skipped: it requires mid, not loaded", logged.output[-1])

    def test_a_cycle_is_skipped_with_what_requires_it(self):
        plugins = {
            "one": (("two",), False),
            "two": (("one",), False),
            "self_made": (("self_made",), False),
            "after": (("one",), False),
            "fine": ((), False),
        }
        with self.assertLogs("arti.plugin", "ERROR") as logged:
            order, skipped = self.load(plugins)
        self.assertEqual((order, skipped), (["fine"], {"one", "two", "self_made", "after"}))
        self.assertEqual(len([line for line in logged.output if "cycle" in line]), 4)

    def test_load_order(self):
        self.assertEqual(
            plugin.load_order({"b": ("a", "nowhere"), "a": (), "c": ("b",), "d": ("d",)}),
            (["a", "b", "c"], ["d"]),
        )


def settings_plugin(api):
    api.settings("demo", {"size": 10, "mode": "a", "on": True}, {"mode": ("a", "b")})


def failing_plugin(api):
    raise RuntimeError("plugin bug")


failing_plugin.__module__ = "failing"  # tests name a plugin by its register's module


class SettingsTest(unittest.TestCase):
    def app(self, sections: dict, *plugins) -> App:
        cfg = make_config()
        cfg.update(sections)
        app = App(cfg, Path(tempfile.mkdtemp()), FakeChannel(), plugins=[fake_providers, *plugins])
        app.load_plugins()
        return app

    def assertInvalid(self, fragment, sections, *plugins):
        with self.assertRaises(ConfigError) as caught:
            self.app(sections, *plugins)
        self.assertIn(f"invalid configuration: {fragment}", str(caught.exception))

    def test_a_declared_section_is_validated_with_defaults_filled_in(self):
        app = self.app({"demo": {"size": 3}}, settings_plugin)
        self.assertEqual(app.config["demo"], {"size": 3, "mode": "a", "on": True})
        self.assertEqual(app.plugin_sections, {"demo": __name__})
        self.assertEqual(self.app({}, settings_plugin).config["demo"]["size"], 10)

    def test_bad_values_stop_startup(self):
        self.assertInvalid(
            "demo.size must be greater than 0", {"demo": {"size": 0}}, settings_plugin
        )
        self.assertInvalid(
            "demo.mode must be one of a, b", {"demo": {"mode": "c"}}, settings_plugin
        )
        self.assertInvalid(
            "demo.sise is not a known setting", {"demo": {"sise": 1}}, settings_plugin
        )
        self.assertInvalid("demo must be a table", {"demo": 1}, settings_plugin)

    def test_a_section_nobody_declared_is_an_error_but_a_skipped_plugins_is_ignored(self):
        self.assertInvalid("[demo] is not a known section; known: agent", {"demo": {}})
        app = self.app({"failing": {"x": 1}}, failing_plugin)
        self.assertNotIn("failing", app.config)

    def test_a_plugin_may_not_take_a_core_or_a_taken_section(self):
        def core(api):
            api.settings("agent", {})

        def again(api):
            api.settings("demo", {})

        with self.assertLogs("arti.plugin", "ERROR"):
            app = self.app({}, core, settings_plugin, again)
        self.assertEqual(app.plugin_sections, {"demo": __name__})


class HooksTest(unittest.IsolatedAsyncioTestCase):
    """api.gate, api.redactor, api.chat_command, the start and stop hooks,
    api.on_new_session and api.on_compact."""

    def app(self, *plugins) -> App:
        app = App(make_config(), Path(tempfile.mkdtemp()), FakeChannel(), plugins=list(plugins))
        app.load_plugins()
        return app

    def test_one_gate_and_one_redactor_at_most(self):
        def first(api):
            api.gate(lambda ctx, tool, args: None)
            api.redactor(str.upper, lambda obj: obj)

        def second(api):
            api.gate(lambda ctx, tool, args: "no")

        def third(api):
            api.redactor(str.lower, lambda obj: obj)

        second.__module__, third.__module__ = "second", "third"
        self.addCleanup(secret.set_redactor, secret._redactor)
        with self.assertLogs("arti.plugin", "ERROR") as logged:
            app = self.app(fake_providers, first, second, third)
        self.assertEqual(len(logged.output), 2)
        self.assertIsNone(app.gate(None, None, {}))
        self.assertEqual(secret.redact("loaded"), "LOADED")  # installed once loaded
        bare = self.app(fake_providers)
        self.assertIsNone(bare.gate)
        self.assertIsNone(bare.redactor)
        self.assertEqual(secret.redact("as is"), "as is")  # the App loaded last has none

    def test_chat_command_names_are_checked(self):
        def good(api):
            api.chat_command("hello", lambda app, inp, arg: "hi", "- say hi")

        def core(api):
            api.chat_command("stop", lambda app, inp, arg: "", "- not mine")

        def taken(api):
            api.chat_command("hello", lambda app, inp, arg: "", "- again")

        def bad(api):
            api.chat_command("Two Words", lambda app, inp, arg: "", "- bad")

        for fn in (core, taken, bad):
            fn.__module__ = fn.__name__
        with self.assertLogs("arti.plugin", "ERROR") as logged:
            app = self.app(fake_providers, good, core, taken, bad)
        self.assertEqual(list(app.chat_commands), ["hello"])
        self.assertEqual(app.chat_commands["hello"].home_only, True)  # the default
        self.assertEqual(len(logged.output), 3)

    async def test_start_hooks_run_in_order_and_once_ones_also_for_once(self):
        ran = []

        def first(api):
            api.on_start(lambda app: ran.append("first"), once=True)
            api.on_stop(lambda app: ran.append("first stopped"))

        def second(api):
            async def start(app):
                ran.append("second")

            async def stop(app):
                ran.append("second stopped")

            api.on_start(start)
            api.on_stop(stop)

        app = await make_app(tempfile.mkdtemp(), plugins=[first, second])
        await app.shutdown()
        # A plugin stops before the plugins it requires: stop hooks run in reverse order.
        self.assertEqual(ran, ["first", "second", "second stopped", "first stopped"])
        ran.clear()
        app = await make_app(tempfile.mkdtemp(), plugins=[first, second], start=False)
        with mock.patch.object(lock, "acquire"):
            await app.start(serve=False)  # --once
        await app.shutdown()
        self.assertEqual(ran, ["first", "second stopped", "first stopped"])

    async def test_new_session_hooks_run_in_threads_within_the_bound(self):
        seen, release = [], threading.Event()

        def plugin_(api):
            api.on_new_session(lambda app, key: seen.append(key))
            api.on_new_session(lambda app, key: 1 / 0)  # logged; the others still run

        app = self.app(fake_providers, plugin_)
        with self.assertLogs("arti.commands", "ERROR"):
            self.assertTrue(await commands.new_session_hooks(app, "k"))
        self.assertEqual(seen, ["k"])
        app.new_session_hooks = [lambda app, key: release.wait(5)]
        with mock.patch.object(commands, "NEW_SESSION_TIMEOUT", 0.05):
            self.assertFalse(await commands.new_session_hooks(app, "k"))
        release.set()

    def test_compact_hooks_add_text_and_a_failing_one_is_skipped(self):
        def plugin_(api):
            api.on_compact(lambda conv, span: f"Kept: {len(span)} messages")
            api.on_compact(lambda conv, span: None)
            api.on_compact(lambda conv, span: 1 / 0)

        app = self.app(fake_providers, plugin_)
        conv = Headless(app, "home", app.default_route())
        with self.assertLogs("arti.context", "ERROR"):
            note = context.summary_note(conv, "They talked.", [{"role": "user", "content": []}])
        self.assertEqual(
            note, "[Arti: summary of the earlier conversation]\nThey talked.\n\nKept: 1 messages"
        )


class ProvidersTest(unittest.IsolatedAsyncioTestCase):
    async def test_the_registry_gives_defaults_and_knows_only_registered_names(self):
        app = await make_app(tempfile.mkdtemp(), cfg=make_config(model=""))
        self.addAsyncCleanup(app.shutdown)
        self.assertEqual(app.config["agent"]["model"], "claude-opus-5-5")  # anthropic's default
        self.assertEqual(app.default_route("openai"), ("openai", "gpt-5.4", "high"))
        self.assertEqual(app.provider_spec("anthropic").judge_model, "claude-haiku-5-5")
        self.assertEqual(app.config["providers"]["openai"], {})  # registered with no options
        with self.assertRaisesRegex(ValueError, "unknown provider: nope"):
            app.provider_spec("nope")

    def test_unknown_providers_in_the_config_are_errors(self):
        for agent, fragment in (
            ({"provider": "nope"}, "agent.provider must be one of claude-subscription, "),
            ({}, "providers.nope is not a known provider"),
        ):
            cfg = make_config(**agent)
            if not agent:
                cfg["providers"]["nope"] = {}
            app = App(cfg, Path(tempfile.mkdtemp()), FakeChannel(), plugins=[fake_providers])
            with self.assertRaises(ConfigError) as caught:
                app.load_plugins()
            self.assertIn(fragment, str(caught.exception))

    def test_options_are_declared_checked_and_filled_and_the_factory_gets_them(self):
        made = []

        def register(api):
            def factory(app, options):
                made.append(options)
                return object()

            api.provider(
                "local",
                factory,
                default_model="m",
                judge_model="j",
                summary_model="s",
                context_window=lambda model: 1000,
                options={"base_url": "", "timeout": 0.0},
            )

        cfg = make_config(provider="local")
        cfg["providers"]["local"] = {"timeout": 5}
        app = App(cfg, Path(tempfile.mkdtemp()), FakeChannel(), plugins=[register])
        app.load_plugins()
        app.provider("local")
        self.assertEqual(made, [{"base_url": "", "timeout": 5}])
        cfg["providers"]["local"] = {"base_url": "http://example.com"}  # the core's safety rule
        app = App(cfg, Path(tempfile.mkdtemp()), FakeChannel(), plugins=[register])
        with self.assertRaisesRegex(ConfigError, "providers.local.base_url must be https://"):
            app.load_plugins()


class CommandsTest(unittest.TestCase):
    def test_commands_are_registered_by_name_and_checked(self):
        ran = []

        def register(api):
            api.command("say-hi", lambda app, argv: ran.append(argv) or 3, "say hi")

        def bad(api):
            api.command("Say Hi", lambda app, argv: 0, "bad name")

        app = App(
            make_config(),
            Path(tempfile.mkdtemp()),
            FakeChannel(),
            plugins=[fake_providers, register],
        )
        app.load_plugins()
        command = app.commands["say-hi"]
        self.assertEqual((command.help, command.plugin), ("say hi", __name__))
        self.assertEqual(command.func(app, ["x"]), 3)
        self.assertEqual(ran, [["x"]])
        app = App(
            make_config(), Path(tempfile.mkdtemp()), FakeChannel(), plugins=[fake_providers, bad]
        )
        with self.assertLogs("arti.plugin", "ERROR"):
            app.load_plugins()
        self.assertEqual(app.commands, {})


class BoundaryTest(unittest.TestCase):
    def test_the_core_imports_no_plugin(self):
        """No module under arti/ imports a FORBIDDEN prefix: the core knows plugins only
        through what they register, and the host only through the link's files."""
        found = []
        for path in sorted((ROOT / "arti").rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                else:
                    continue
                for name in names:
                    for prefix in FORBIDDEN:
                        if name == prefix or name.startswith(prefix + "."):
                            found.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {name}")
        self.assertEqual(found, [])

    def test_the_core_names_no_plugin_tool_vendor_or_model(self):
        """The core (arti/) knows plugins only by what they register: no code, default, comment
        or string in it names a plugin, a plugin's tool, a model vendor or a model."""
        found = []
        for path in sorted((ROOT / "arti").rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for match in (*CORE_NAMES.finditer(line), *SAFETY_NAMES.finditer(line)):
                    found.append(f"{rel}:{number} names {match.group(0)!r}")
        self.assertEqual(found, [])

    def test_the_core_knows_no_group_chats(self):
        """No code, comment or string in arti/ names lanes, a group log, the groups or their
        commands: all of that is the chat platform's."""
        found = []
        for path in sorted((ROOT / "arti").rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                found += [
                    f"{rel}:{number} names {m.group(0)!r}" for m in GROUP_NAMES.finditer(line)
                ]
        self.assertEqual(found, [])

    def test_a_plugin_imports_only_the_core_itself_and_what_it_requires(self):
        """A plugin's modules import from `arti.*`, from their own plugin, and from the plugins
        in its REQUIRES (read from the source); never from another plugin or from tests."""
        found = []
        for name in plugin.find(ROOT / "plugins"):
            allowed = {name, *requires_of(name)}
            for path in plugin_files(name):
                for imported, line in imports(path):
                    top, _, rest = imported.partition(".")
                    if top in ("tests", "host", "channels") or (
                        top == "plugins" and rest.partition(".")[0] not in allowed
                    ):
                        found.append(f"{path.relative_to(ROOT)}:{line} imports {imported}")
        self.assertEqual(found, [])

    def test_the_host_and_its_channels_never_import_the_agents_plugins(self):
        """The host runs on the machine and the plugins in the sandboxes: host/ and channels/
        import the core's small helpers (arti.*), the host and themselves, never a plugin."""
        found = []
        for folder in ("host", "channels"):
            for path in sorted((ROOT / folder).rglob("*.py")):
                for imported, line in imports(path):
                    if imported.partition(".")[0] in ("plugins", "tests"):
                        found.append(f"{path.relative_to(ROOT)}:{line} imports {imported}")
        self.assertEqual(found, [])

    def test_every_plugin_with_a_reviewed_tool_requires_the_safety_plugin(self):
        """A tool with a review kind (or a function choosing one) is only safe behind the gate,
        so its plugin must be skipped whenever the safety plugin is not loaded."""
        app = App(make_config(), Path(tempfile.mkdtemp()), FakeChannel())
        modules = {
            name: importlib.import_module(f"plugins.{name}")
            for name in plugin.find(ROOT / "plugins")
        }
        requires = {name: getattr(module, "REQUIRES", ()) for name, module in modules.items()}
        self.assertEqual({name: requires_of(name) for name in modules}, requires)  # as read
        reviewed = {}
        order, cyclic = plugin.load_order(requires)
        self.assertEqual(cyclic, [])
        for name in order:
            before = set(app.tools)
            self.assertTrue(plugin.add_plugin(app, name, modules[name].register), name)
            for tool in set(app.tools) - before:
                if app.tools[tool].review is not None:
                    reviewed.setdefault(name, []).append(tool)
        self.assertIn("terminal", reviewed)  # the check sees reviewed tools at all
        unguarded = [n for n in reviewed if n != "safety" and "safety" not in requires[n]]
        self.assertEqual(unguarded, [])
