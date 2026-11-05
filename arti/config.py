"""Configuration: ~/.arti/config.toml merged over the defaults, validated, kept as a plain dict.

Two steps, because plugins own most sections:

1. `load` reads the file and validates the core sections (DEFAULTS, and the shape of every
   [providers.<id>] table). Any other section is kept as written, for the plugin that owns it.
2. Once the plugins are loaded (plugin.py), each plugin section has been checked against what
   its plugin declared (`check_section`, from `api.settings`), and `check_plugins` checks the
   rest: no section nobody declared, the providers named exist, their tables, the default model.

A default's type is the rule for its value (a float setting also accepts an integer, a number
must be > 0 unless 0 is allowed); a few keys also have a fixed set of choices. Unknown keys,
wrong types and bad choices stop startup with `invalid configuration: <key> ...`. Secrets live in
~/.arti/.env, never here (secret.load_env).
"""

import copy
import ipaddress
import os
import tomllib
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

EFFORTS = ("low", "medium", "high", "xhigh", "max")

DEFAULTS = {
    "agent": {
        "provider": "",  # a provider a plugin registered (api.provider); "" = the only one
        "model": "",  # "" = the provider's default model
        "effort": "high",
        "timezone": "America/Los_Angeles",
        "user_name": "the user",
        "name": "Arti",  # how people call Arti in a group: "@Arti", or a real mention
        "context_window": 0,  # tokens; 0 = the model's window, at most 200K (context.py)
        "request_timeout": 900.0,
        "tool_timeout": 3600.0,
        "max_iterations": 100,
        "hang_timeout": 300.0,  # seconds without progress before Arti restarts itself; 0 = off
    },
    "storage": {"cache_max_age_days": 7},
    "plugins": {"disabled": []},
}
CORE = (*DEFAULTS, "providers")  # sections no plugin may declare

CHOICES = {"agent.effort": EFFORTS}
# Numbers must be > 0, except these (and every [providers.<id>] number), where 0 means "default"
# or "off".
NONNEGATIVE = {"agent.context_window", "agent.hang_timeout"}


class ConfigError(ValueError):
    """The configuration cannot be used; the message names the key and the problem."""


def data_dir() -> Path:
    """~/.arti, or $ARTI_HOME when set (tests, a second instance)."""
    return Path(os.environ.get("ARTI_HOME") or "~/.arti").expanduser()


def code_dir() -> Path:
    """The repository root: the directory holding the `arti` package and `plugins/`."""
    return Path(__file__).resolve().parent.parent


def load(data_dir: Path, *, provider: str = "", model: str = "", effort: str = "") -> dict:
    """Read config.toml, merge it over DEFAULTS, apply command-line overrides and validate the
    core sections. Plugin sections stay as written until their plugins check them."""
    path = Path(data_dir) / "config.toml"
    settings = {}
    if path.is_file():
        try:
            settings = tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
            raise ConfigError(f"invalid configuration: {path}: {e}") from None
    cfg = copy.deepcopy(DEFAULTS)
    cfg["providers"] = {}
    for section, table in settings.items():
        if not isinstance(table, dict):
            fail(section, "must be a table")
        if section in DEFAULTS:
            cfg[section].update(table)
        else:  # [providers], or a plugin's: checked once the plugins are loaded
            cfg[section] = table
    agent = cfg["agent"]
    if provider and provider != agent["provider"]:
        # A model name only means something on its own provider.
        agent["provider"], agent["model"] = provider, ""
    agent["model"] = model or agent["model"]
    agent["effort"] = effort or agent["effort"]
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    """Raise ConfigError for an unknown key, a wrong type, a bad choice or a bad value in a
    core section."""
    for section, defaults in DEFAULTS.items():
        for key, value in cfg[section].items():
            name = f"{section}.{key}"
            if key not in defaults:
                fail(name, "is not a known setting; known: " + ", ".join(defaults))
            _check_value(name, value, defaults[key], CHOICES.get(name), name in NONNEGATIVE)
    try:
        ZoneInfo(cfg["agent"]["timezone"])
    except (ValueError, ZoneInfoNotFoundError):
        fail("agent.timezone", "is not a known time zone (e.g. America/Los_Angeles)")
    for name, table in cfg["providers"].items():
        if not isinstance(table, dict):
            fail(f"providers.{name}", "must be a table")


def check_section(section: str, table, defaults: dict, choices: dict | None = None) -> dict:
    """A plugin's section (api.settings): validated like a core one, defaults filled in."""
    if not isinstance(table, dict):
        fail(section, "must be a table")
    for key, value in table.items():
        if key not in defaults:
            fail(f"{section}.{key}", "is not a known setting; known: " + ", ".join(defaults))
        _check_value(f"{section}.{key}", value, defaults[key], (choices or {}).get(key))
    return {**copy.deepcopy(defaults), **table}


def check_plugins(
    cfg: dict, declared: set[str], skipped: set[str], providers: dict, provider_keys=()
) -> None:
    """After the plugins are loaded: every section belongs to the core or to a plugin, the
    providers named exist (agent.provider, and the plugins' `provider_keys`, "<section>.<key>"),
    each [providers.<id>] table holds only that provider's options (all filled in, so
    `cfg["providers"][name]` is complete), and agent.model has its default.

    While a plugin is disabled or failed to load (`skipped`), a section or provider table nobody
    declared is ignored and dropped: it may be that plugin's (a plugin may declare sections not
    named after it), and turning a plugin off should not mean deleting its settings too. With
    every plugin loaded, it is a mistake and stops startup. `providers` is App.provider_specs
    (plugin.ProviderSpec by name)."""
    for section in list(cfg):
        if section in CORE or section in declared:
            continue
        if skipped:
            del cfg[section]
            continue
        known = ", ".join([*CORE, *sorted(declared)])
        fail(f"[{section}]", f"is not a known section; known: {known}")
    names = ", ".join(providers) or "(none: no provider plugin is loaded)"
    agent = cfg["agent"]
    agent["provider"] = pick("agent.provider", agent["provider"], providers, "provider", skipped)
    for key in provider_keys:
        section, name = key.split(".", 1)
        if (value := cfg[section][name]) and value not in providers:
            fail(key, "must be one of " + names)
    for name in list(cfg["providers"]):
        if name not in providers:
            if skipped:
                del cfg["providers"][name]
                continue
            fail(f"providers.{name}", "is not a known provider; known: " + names)
    for name, spec in providers.items():
        table = cfg["providers"].get(name, {})
        cfg["providers"][name] = check_provider_table(name, table, spec.options)
    if not agent["model"]:
        agent["model"] = providers[agent["provider"]].default_model


def pick(key: str, value: str, registered, kind: str, skipped=()) -> str:
    """The setting `key`'s choice among the `registered` names: `value`, or when it is "" the
    only one registered. ConfigError when it names none of them, or is "" with no single one
    (naming the plugins that were skipped: the one it wants may be among them)."""
    names = ", ".join(registered) or f"(none: no {kind} plugin is loaded)"
    if skipped:
        names += f" (not loaded: {', '.join(sorted(skipped))}, or what they require)"
    if not value and len(registered) == 1:
        return next(iter(registered))
    if not value:
        fail(key, "must be set: one of " + names)
    if value not in registered:
        fail(key, "must be one of " + names)
    return value


def check_provider_table(name: str, table: dict, options: dict) -> dict:
    """A [providers.<id>] table against the options its provider declared (0 allowed: it means
    "the default"), defaults filled in. A base_url must be safe, whoever declared it."""
    for key, value in table.items():
        if key not in options:
            known = ", ".join(options) or "(this provider has no options)"
            fail(f"providers.{name}.{key}", "is not a known setting; known: " + known)
        _check_value(f"providers.{name}.{key}", value, options[key], zero_ok=True)
    if "base_url" in table and not _safe_base_url(table["base_url"]):
        fail(f"providers.{name}.base_url", "must be https://, or http:// on loopback")
    return {**copy.deepcopy(options), **table}


def _check_value(key: str, value, default, choices=None, zero_ok: bool = False) -> None:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            fail(key, "must be true or false")
    elif isinstance(default, (int, float)):
        allowed = int if isinstance(default, int) else (int, float)
        if isinstance(value, bool) or not isinstance(value, allowed):
            fail(key, "must be a whole number" if allowed is int else "must be a number")
        if value < 0 or (value == 0 and not zero_ok):
            fail(key, "must be >= 0" if zero_ok else "must be greater than 0")
    elif isinstance(default, str):
        if not isinstance(value, str):
            fail(key, "must be a string")
        if choices and value not in choices:
            fail(key, "must be one of " + ", ".join(choices))
    elif isinstance(default, list):
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            fail(key, "must be a list of nonempty strings")


def _safe_base_url(url: str) -> bool:
    """HTTPS anywhere, or plain HTTP only to this machine (a local proxy); no credentials."""
    parts = urlsplit(url)
    if not parts.hostname or parts.username or parts.password:
        return False
    if parts.scheme == "https":
        return True
    if parts.scheme != "http":
        return False
    if parts.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        return False


def fail(key: str, problem: str):
    """Raise the ConfigError for `key` (a plugin's own checks use it too)."""
    raise ConfigError(f"invalid configuration: {key} {problem}")
