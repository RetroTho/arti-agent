"""Keep the host running: write the service that starts it at login and again whenever it exits.

    python -m host service [--home DIR]

macOS: a LaunchAgent, ~/Library/LaunchAgents/ai.arti.host.plist (launchd). Linux: a systemd user
unit, ~/.config/systemd/user/arti-host.service. Either runs this Python (`sys.executable`, the
repository's .venv when run from it) as `-m host` in the repository, starts it again 30 s after
it exits for any reason (the hang watchdog's exit too), keeps ARTI_HOME when it is set, and logs
to <data>/logs/daemon.{out,err}.log. Nothing is loaded or started: the command prints how.
"""

import argparse
import os
import plistlib
import shlex
import sys
from pathlib import Path

from arti import config

LABEL = "ai.arti.host"
UNIT = "arti-host.service"
RESTART_AFTER = 30  # seconds
PATHS = {  # the PATH the service runs with; ~ is the target home
    "darwin": "~/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "linux": "~/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin",
}


def environment(home: Path, platform: str) -> dict[str, str]:
    env = {"PATH": PATHS[platform].replace("~", str(home))}
    if data := os.environ.get("ARTI_HOME"):
        env["ARTI_HOME"] = str(Path(data).expanduser().resolve())
    return env


def launch_agent(home: Path, python: str, repo: Path, logs: Path) -> bytes:
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": [python, "-m", "host"],
            "WorkingDirectory": str(repo),
            "EnvironmentVariables": environment(home, "darwin"),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ThrottleInterval": RESTART_AFTER,
            "StandardOutPath": str(logs / "daemon.out.log"),
            "StandardErrorPath": str(logs / "daemon.err.log"),
        }
    )


def quote(value: str) -> str:
    """One word of a systemd unit line, quoted (and % and $ escaped, which systemd expands)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$")
    return f'"{escaped}"'


def systemd_unit(home: Path, python: str, repo: Path, logs: Path) -> str:
    env = "\n".join(
        f"Environment={quote(f'{k}={v}')}" for k, v in environment(home, "linux").items()
    )
    return f"""\
[Unit]
Description=Arti's host: the chat channel and every user's agent
Wants=network-online.target
After=network-online.target

[Service]
ExecStart={quote(python)} -m host
WorkingDirectory={quote(str(repo))}
{env}
Restart=always
RestartSec={RESTART_AFTER}
# The host stops the sandboxes when told to stop; anything left after the timeout is killed.
KillMode=mixed
TimeoutStopSec=120
StandardOutput=append:{logs / "daemon.out.log"}
StandardError=append:{logs / "daemon.err.log"}

[Install]
WantedBy=default.target
"""


def install(home: Path, platform: str = sys.platform) -> list[str]:
    """Write the service file for `platform`; returns the commands that start it."""
    repo, python = config.code_dir(), sys.executable
    logs = config.data_dir().resolve() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if platform == "darwin":
        path = home / "Library/LaunchAgents" / f"{LABEL}.plist"
        text = launch_agent(home, python, repo, logs)
        start = [f"launchctl bootstrap gui/$(id -u) {shlex.quote(str(path))}"]
    elif platform == "linux":
        path = home / ".config/systemd/user" / UNIT
        text = systemd_unit(home, python, repo, logs).encode()
        start = [
            "systemctl --user daemon-reload",
            f"systemctl --user enable --now {UNIT}",
            "loginctl enable-linger $USER  # keeps it running while you are logged out",
        ]
    else:
        raise SystemExit(f"host service: no service manager known for {platform}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text)
    print(f"Written: {path}")
    return start


def command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m host service",
        description="Write the service that keeps the host running (launchd on macOS, systemd "
        "on Linux). Nothing is started.",
    )
    parser.add_argument("--home", type=Path, default=Path.home(), help="target home directory")
    args = parser.parse_args(argv)
    start = install(args.home.expanduser().resolve())
    print("Nothing was started. To start the host, run:")
    for line in start:
        print(f"  {line}")
    return 0
