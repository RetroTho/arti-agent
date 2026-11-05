"""The plugin API: finding and loading plugins, and everything their register(api) can add.

A plugin is `plugins/<name>.py`, or a package `plugins/<name>/` (its `__init__.py`; its other
modules are its own helpers, imported only by it), with a `register(api)` function. Through
`api` it adds:

  api.tool(...)                        a tool the model can call (tools.py runs it)
  api.prompt_section(fn)               text for the system prompt
  api.service(name, fn)                a function other parts call by name
  api.on_start(fn), api.on_stop(fn)    hooks run with the App at startup and shutdown
  api.on_new_session(fn)               a hook run when /new starts a conversation afresh
  api.on_compact(fn)                   text to keep word for word when history is summarised
  api.gate(fn)                         reviews every tool call before it runs (at most one)
  api.redactor(text, obj)              hides secrets in everything the core writes (at most one)
  api.file_check(fn)                   refuses files that must not be sent (App.file_problem)
  api.chat_command(name, fn, help)     a slash command, like /stop
  api.settings(section, defaults)      its own config.toml section, validated
  api.provider(name, factory, ...)     a model provider (agent.provider, --provider, /model)
  api.command(name, func, help)        a `python -m arti <name>` command

The core never imports a plugin: it finds them all here, by name. A plugin may require other
plugins, `REQUIRES = ("<name>", ...)` at the top of its module, and then import from them (`from
plugins.<name> import ...`). Plugins register in dependency order, then name order
(`load_order`); one whose requirement is disabled, missing or failed is skipped, and so is every
plugin that requires it. A plugin that fails to import or register is logged and skipped, and
Arti still starts; its config section, like a disabled plugin's, is ignored. A bad value in a
section a plugin declared stops startup, as any configuration error does.

With no gate, tool calls run unreviewed; with no redactor, nothing is redacted. That is why a
plugin with a reviewed tool requires the plugin that registers them (tests/test_plugin.py).
"""

import importlib
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from arti import commands, config, schema
from arti.config import ConfigError
from arti.tools import PromptInfo, Tool

log = logging.getLogger(__name__)

COMMAND_NAME = re.compile(r"[a-z][a-z0-9-]*")
CHAT_COMMAND_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass
class ProviderSpec:
    """A model provider as a plugin registered it (api.provider)."""

    name: str
    factory: Callable[[Any, dict], Any]  # (app, its [providers.<name>] options) -> Provider
    default_model: str
    judge_model: str  # small and fast: a plugin's quick checks (the approval judge)
    summary_model: str  # summarises old history (context.py); plugins may use it too
    context_window: Callable[[str], int]  # model -> its window in tokens, conservatively
    options: dict = field(default_factory=dict)  # the [providers.<name>] keys and defaults


@dataclass
class Command:
    """A `python -m arti <name> [args...]` command (api.command)."""

    name: str
    func: Callable[[Any, list[str]], int]  # (app, argv) -> exit code
    help: str
    plugin: str


@dataclass
class ChatCommand:
    """A slash command a plugin added (api.chat_command)."""

    name: str
    func: Callable[[Any, Any, str], Any]  # (app, the Input, its argument) -> the reply
    help: str  # what follows "/<name> " in /help
    home_only: bool  # only in the owner's one-on-one chat
    plugin: str


class Api:
    """What a plugin's register(api) gets. Registrations are kept until register returns, so a
    plugin that fails half-way adds nothing (its settings excepted: they are only values)."""

    def __init__(self, app, plugin: str):
        self.app, self.config, self.plugin = app, app.config, plugin
        self.tools: dict[str, Tool] = {}
        self.sections, self.services, self.starts, self.stops = [], {}, [], []
        self.new_sessions, self.compacts = [], []
        self.gates, self.redactors, self.file_checks, self.provider_keys = [], [], [], []
        self.chat_commands: dict[str, ChatCommand] = {}
        self.providers: dict[str, ProviderSpec] = {}
        self.commands: dict[str, Command] = {}

    def tool(self, name, description, schema_, handler, *, review, **options) -> None:
        full = {"type": "object", **schema_, "additionalProperties": False}
        schema.check(full)
        if name in self.tools or name in self.app.tools:
            raise ValueError(f"tool {name!r} is already registered")
        self.tools[name] = Tool(name, description, full, handler, review, **options)

    def prompt_section(self, fn: Callable[[PromptInfo], str | None]) -> None:
        self.sections.append(fn)

    def service(self, name: str, fn: Callable) -> None:
        self.services[name] = fn

    def on_start(self, fn: Callable, once: bool = False) -> None:
        """`fn(app)` (plain or async) runs when Arti starts, in load order: after recovery,
        before the channel starts. With `once` it also runs before a --once task, which runs
        no other start hook."""
        self.starts.append((fn, once))

    def on_stop(self, fn: Callable, timeout: float = 5) -> None:
        """`fn(app)` (plain or async) runs at every shutdown, --once's too, in reverse load
        order: a plugin stops before the plugins it requires. It is given `timeout` seconds."""
        self.stops.append((fn, timeout))

    def on_new_session(self, fn: Callable[[Any, str], None]) -> None:
        """`fn(app, key)` runs in a thread when /new starts conversation `key` afresh (to stop
        what it left running, say). All of them together get commands.NEW_SESSION_TIMEOUT
        seconds."""
        self.new_sessions.append(fn)

    def on_compact(self, fn: Callable[[Any, list[dict]], str | None]) -> None:
        """`fn(conv, span)` runs when the messages `span` of conversation `conv` are summarised
        (context.py); the text it returns is kept word for word under the summary (None: none).
        It may also drop what it keeps in `conv.state` about the span."""
        self.compacts.append(fn)

    def gate(self, fn: Callable[[Any, Tool, dict], str | None]) -> None:
        """`fn(ctx, tool, args)` reviews every tool call before its handler runs, in the call's
        worker thread: None lets the call run, a string refuses it (the model gets the string
        as the call's result). It may take minutes (asking the owner, say): the tool's deadline
        starts once it has passed. One plugin at most; with none, tool calls run unreviewed."""
        if self.gates or self.app.gate is not None:
            raise ValueError("a gate is already registered")
        self.gates.append(fn)

    def redactor(self, text: Callable[[str], str], obj: Callable[[Any], Any]) -> None:
        """Hide secrets in everything the core writes (arti/secret.py, from the moment the
        plugins are loaded): `text(s)` returns the string `s` redacted, `obj(value)` a redacted
        copy of a message or other JSON-like value. One plugin at most; with none, nothing is
        redacted."""
        if self.redactors or self.app.redactor is not None:
            raise ValueError("a redactor is already registered")
        self.redactors.append((text, obj))

    def file_check(self, fn: Callable[[str], str | None]) -> None:
        """`fn(path)` says why a file must not be sent (a reply's MEDIA line, a tool's file), or
        returns None; every file Arti sends passes all of them (App.file_problem). It runs in a
        thread and must be quick."""
        self.file_checks.append(fn)

    def chat_command(
        self, name: str, fn: Callable[[Any, Any, str], Any], help: str, *, home_only: bool = True
    ) -> None:
        """A slash command: `/name arg` runs `fn(app, inp, arg)` at once (`inp` is the message,
        a store.Input; plain or async) and sends the reply it returns. `home_only` (the default)
        keeps it to the owner's one-on-one chat; otherwise it works in any conversation. `help`
        follows "/name " in /help, e.g. "[id] - answer a question"."""
        if not CHAT_COMMAND_NAME.fullmatch(name):
            raise ValueError(f"chat command name {name!r} must be one lower-case word")
        if name in commands.CORE or name in self.chat_commands or name in self.app.chat_commands:
            raise ValueError(f"chat command /{name} is already registered")
        self.chat_commands[name] = ChatCommand(name, fn, help, home_only, self.plugin)

    def settings(
        self, section: str, defaults: dict, choices: dict | None = None, provider_keys=()
    ) -> dict:
        """Declare this plugin's config.toml section and return it, validated, with every
        default filled in (also at app.config[section]). `defaults` gives each key's default,
        whose type is the rule (numbers > 0); `choices` maps a key to its allowed strings;
        `provider_keys` names the keys whose value, unless "", must be a model provider's name
        (checked once every plugin is loaded). Raises ConfigError for a bad value, which stops
        startup."""
        if section in config.CORE:
            raise ValueError(f"[{section}] is a core section")
        if (owner := self.app.plugin_sections.get(section)) is not None:
            raise ValueError(f"[{section}] is already declared by plugin {owner}")
        table = config.check_section(section, self.config.get(section, {}), defaults, choices)
        self.config[section] = table
        self.app.plugin_sections[section] = self.plugin
        self.provider_keys += [f"{section}.{key}" for key in provider_keys]
        return table

    def provider(
        self,
        name: str,
        factory: Callable,
        *,
        default_model: str,
        judge_model: str,
        summary_model: str,
        context_window: Callable[[str], int],
        options: dict | None = None,
    ) -> None:
        """A model provider. `factory(app, options)` builds its client (a providers.Provider)
        on first use; `options` declares the keys of its optional [providers.<name>] table
        with their defaults (0 allowed in numbers: "the default")."""
        if name in self.providers or name in self.app.provider_specs:
            raise ValueError(f"provider {name!r} is already registered")
        spec = ProviderSpec(
            name, factory, default_model, judge_model, summary_model, context_window, options or {}
        )
        self.providers[name] = spec

    def command(self, name: str, func: Callable[[Any, list[str]], int], help: str) -> None:
        """`python -m arti <name> [args...]` runs `func(app, args)` and exits with what it
        returns. The App is configured and its plugins loaded, but not started: no lock, no
        store, no channel (a command that must not run beside a live Arti takes the lock
        itself, lock.acquire). `help` is its line in `python -m arti --help`."""
        if not COMMAND_NAME.fullmatch(name):
            raise ValueError(f"command name {name!r} must be lower-case words joined by -")
        if name in self.commands or name in self.app.commands:
            raise ValueError(f"command {name!r} is already registered")
        self.commands[name] = Command(name, func, help, self.plugin)


def add_plugin(app, name: str, register: Callable[[Api], None]) -> bool:
    """Register one plugin; False when it failed (logged, and the plugin skipped). A
    ConfigError is not a plugin failure: it stops startup."""
    api = Api(app, name)
    try:
        register(api)
    except ConfigError:
        raise
    except Exception:
        log.exception("plugin %s failed to register; skipped", name)
        return False
    app.tools.update(api.tools)
    app.services.update(api.services)
    app.prompt_sections += api.sections
    app.start_hooks += api.starts
    app.stop_hooks += api.stops
    app.new_session_hooks += api.new_sessions
    app.compact_hooks += api.compacts
    if api.gates:
        app.gate = api.gates[0]
    if api.redactors:
        app.redactor = api.redactors[0]
    app.file_checks += api.file_checks
    app.chat_commands.update(api.chat_commands)
    app.provider_keys += api.provider_keys
    app.provider_specs.update(api.providers)
    app.commands.update(api.commands)
    return True


def find(plugins_dir: Path) -> list[str]:
    """The plugin names in `plugins_dir`: <name>.py files and <name>/ packages, in name order.
    A name starting with "_" or "." is never a plugin (__pycache__, editor files)."""
    names = set()
    for path in plugins_dir.iterdir():
        if path.name.startswith(("_", ".")):
            continue
        if path.suffix == ".py" and path.is_file():
            names.add(path.stem)
        elif path.is_dir() and (path / "__init__.py").is_file():
            names.add(path.name)
    return sorted(names)


def load_order(requires: dict[str, tuple[str, ...]]) -> tuple[list[str], list[str]]:
    """The order to register plugins in, and the plugins that can never be registered because
    they are in a cycle of requirements or require one. `requires` maps every plugin to the
    plugins it requires.

    A plugin comes after every plugin it requires; of those free to go, the first by name goes
    next. A requirement that is not in `requires` (a disabled or missing plugin) does not hold
    the order up: the plugin needing it is skipped when its turn comes (load_plugins)."""
    waiting = {name: {r for r in needs if r in requires} for name, needs in requires.items()}
    order = []
    while ready := sorted(name for name, needs in waiting.items() if not needs):
        order.append(ready[0])
        del waiting[ready[0]]
        for needs in waiting.values():
            needs.discard(ready[0])
    return order, sorted(waiting)


def load_plugins(app, plugins_dir: Path, disabled: set[str]) -> set[str]:
    """Import every plugin in `plugins_dir` that is not disabled (to read its REQUIRES), then
    register them in load_order. Returns the names skipped: disabled, failed to load, caught in
    a cycle of requirements, or requiring a plugin that is not loaded (because of any of these,
    or because there is no such plugin)."""
    if str(plugins_dir.parent) not in sys.path:
        sys.path.insert(0, str(plugins_dir.parent))
    skipped = set(disabled)
    modules = {}
    for name in find(plugins_dir):
        if name in disabled:
            continue
        try:
            modules[name] = importlib.import_module(f"{plugins_dir.name}.{name}")
        except Exception:
            log.exception("plugin %s failed to import; skipped", name)
            skipped.add(name)
    requires = {name: tuple(getattr(module, "REQUIRES", ())) for name, module in modules.items()}
    order, cyclic = load_order(requires)
    for name in cyclic:
        log.error("plugin %s skipped: it is in, or requires, a cycle of requirements", name)
        skipped.add(name)
    loaded = set()
    for name in order:
        if missing := [needed for needed in requires[name] if needed not in loaded]:
            log.error("plugin %s skipped: it requires %s, not loaded", name, ", ".join(missing))
            skipped.add(name)
        elif add_plugin(app, name, modules[name].register):
            loaded.add(name)
        else:
            skipped.add(name)
    return skipped
