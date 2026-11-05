"""The host's settings: <data>/host.toml, with [host] and the channel's own section (named after
it); its secrets (the channel's credentials) in <data>/.env. Each agent has its own settings,
in its home folder (.arti/config.toml), which the host never reads.

The data directory is ~/.arti, or $ARTI_HOME.
"""

import tomllib
from pathlib import Path

from arti.config import ConfigError, check_section, fail

HOST = {
    "channel": "",  # the chat platform, a package in channels/; "" = the only one there
    "memory": "2g",  # each sandbox's memory limit
    "cpus": 2.0,  # and its CPU limit
    "hang_timeout": 300.0,  # seconds without progress before the host restarts itself
}


def load(data_dir: Path, channels: list[str]) -> dict:
    """host.toml, checked; [host] filled in, `channel` resolved among `channels` ("" when none is
    installed: the host can't run then, but its other commands can)."""
    path = Path(data_dir) / "host.toml"
    try:
        raw = tomllib.loads(path.read_text()) if path.exists() else {}
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"invalid configuration: {path}: {error}") from None
    cfg = {"host": check_section("host", raw.pop("host", {}), HOST)}
    name = cfg["host"]["channel"] or (channels[0] if len(channels) == 1 else "")
    if name not in channels and (name or channels):
        known = ", ".join(sorted(channels)) or "(none installed)"
        fail("host.channel", "must be one of " + known)
    cfg["host"]["channel"] = name
    sections = ["host", name] if name else ["host"]
    if unknown := sorted(set(raw) - set(sections)):
        fail(f"[{unknown[0]}]", f"is not a known section; known: {', '.join(sections)}")
    if name:
        cfg[name] = raw.get(name, {})
    return cfg
