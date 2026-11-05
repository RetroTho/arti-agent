"""The host's command line: what the operator runs on the machine.

python -m host                              run the host: the channel and every user's agent
python -m host user add ID --name NAME --handle HANDLE [--provider P]
                                            add a user (HANDLE: their id on the channel; P:
                                            their model provider, claude-subscription at first)
python -m host user list | remove ID        list users; forget one (their folder is kept)
python -m host user exec ID [COMMAND...]    run a command in a user's sandbox (default: bash),
                                            e.g. `claude` to sign in, `python -m arti login`
python -m host build                        build the sandbox image
python -m host test                         run the tests, in the sandbox image
python -m host service [--home DIR]         write the service that keeps the host running

The data directory is ~/.arti, or $ARTI_HOME (host.toml, .env, users.json, homes/, logs/).
"""

import argparse
import asyncio
import importlib
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from arti import config as arti_config
from arti import lock, secret, watchdog
from arti import log as log_setup
from arti.config import ConfigError
from host import config, service
from host.host import Host
from host.sandbox import CODE, IMAGE, SandboxError, Sandboxes, podman
from host.users import Users

log = logging.getLogger("host")
REPO = arti_config.code_dir()
CHANNELS = REPO / "channels"


def channel_names() -> list[str]:
    return sorted(p.name for p in CHANNELS.iterdir() if (p / "__init__.py").is_file())


NO_CHANNEL = "no chat channel is installed (a package in channels/; see host/channel.py)"


def make_channel(cfg: dict, host_ref):
    """The configured channel (channels/<name>/), its section checked."""
    name = cfg["host"]["channel"]
    if not name:
        raise ConfigError(NO_CHANNEL)
    package = importlib.import_module(f"channels.{name}")
    settings = arti_config.check_section(name, cfg[name], package.SETTINGS)
    return package.make(host_ref, settings)


def local_timezone() -> str:
    """The machine's time zone name, for a new user's agent ("UTC" when it can't be told)."""
    if tz := os.environ.get("TZ"):
        return tz
    try:
        return str(Path("/etc/localtime").resolve()).split("zoneinfo/", 1)[1]
    except (OSError, IndexError):
        return "UTC"


# --- running ---------------------------------------------------------------------------------


async def run(data_dir: Path, cfg: dict) -> None:
    sandboxes = Sandboxes(cfg["host"], REPO, data_dir)
    host = Host(data_dir, None, sandboxes)
    host.channel = make_channel(cfg, host)
    lock.acquire(data_dir)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    await host.start()
    log.info("host started: %s, %d user(s)", host.channel.name, len(host.users.all()))
    dog = watchdog.Watchdog(data_dir / "logs" / "hang.log", cfg["host"]["hang_timeout"])
    dog.start()
    try:
        while not stop.is_set():
            if host.alive():
                dog.beat()
            try:
                await asyncio.wait_for(stop.wait(), watchdog.BEAT)
            except TimeoutError:
                pass
    finally:
        log.info("host shutting down")
        dog.stop()
        await asyncio.wait_for(host.stop(), 90)


# --- the commands ----------------------------------------------------------------------------


def user_command(data_dir: Path, cfg: dict, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m host user")
    sub = parser.add_subparsers(dest="action", required=True)
    add = sub.add_parser("add", help="add a user")
    add.add_argument("id")
    add.add_argument("--name", required=True, help="how Arti calls them")
    add.add_argument("--handle", required=True, help="their id on the channel")
    add.add_argument("--provider", default="claude-subscription", help="their model provider")
    sub.add_parser("list", help="list the users")
    remove = sub.add_parser("remove", help="forget a user (their folder is kept)")
    remove.add_argument("id")
    run_in = sub.add_parser("exec", help="run a command in a user's sandbox")
    run_in.add_argument("id")
    run_in.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    users, channel = Users(data_dir), cfg["host"]["channel"]
    if args.action == "add":
        if not channel:
            parser.error(NO_CHANNEL + ", and a user is added with their handle on it")
        try:
            user = users.add(
                args.id, args.name, channel, args.handle, local_timezone(), args.provider
            )
        except ValueError as error:
            parser.error(str(error))
        print(f"Added {user.id}; their folder is {user.home(data_dir)}.")
        print("Their agent starts with the host. To sign it in to a model provider, run:")
        print(f"  python -m host user exec {user.id} claude")
        print(f"  (or, for ChatGPT: python -m host user exec {user.id} python -m arti login)")
        return 0
    if args.action == "list":
        for user in users.all():
            handles = ", ".join(f"{name}: {h}" for name, h in sorted(user.handles.items()))
            print(f"{user.id}  {user.name}  {handles or '-'}")
        return 0
    user = users.get(args.id)
    if user is None:
        parser.error(f"no user {args.id!r}")
    if args.action == "remove":
        Sandboxes(cfg["host"], REPO, data_dir).remove(user)
        users.remove(user.id)
        print(f"Removed {user.id}. Their folder is kept: {user.home(data_dir)}")
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    argv_ = Sandboxes(cfg["host"], REPO, data_dir).exec_argv(user, command or ["bash"])
    return subprocess.run(argv_).returncode


def test_command() -> int:
    """The whole test suite, in the sandbox image (the system the agents run on); the tests
    themselves mounted read only beside the code."""
    argv = [podman(), "run", "--rm", "--init", "--userns", "keep-id", "--workdir", CODE]
    argv += ["--tmpfs", "/home/arti:rw,mode=1777", "--env", "HOME=/home/arti"]
    argv += ["--env", "PYTHONDONTWRITEBYTECODE=1"]
    for part in ("arti", "plugins", "host", "channels", "tests"):
        argv += ["--volume", f"{REPO / part}:{CODE}/{part}:ro"]
    argv += [f"{IMAGE}:latest", "python", "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    return subprocess.run(argv).returncode


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    os.umask(0o077)
    data_dir = arti_config.data_dir()
    data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    command = argv[0] if argv else ""
    try:
        if command in ("-h", "--help"):
            print(__doc__)
            return 0
        if command == "service":
            return service.command(argv[1:])
        if command == "test":
            return test_command()
        cfg = config.load(data_dir, channel_names())
        if command == "build":
            Sandboxes(cfg["host"], REPO, data_dir).build()
            return 0
        if command == "user":
            return user_command(data_dir, cfg, argv[1:])
        if command:
            print(f"host: unknown command {command!r} (see --help)", file=sys.stderr)
            return 2
        secret.load_env(data_dir / ".env")
        log_setup.setup(data_dir)
        asyncio.run(run(data_dir, cfg))
        return 0
    except (ConfigError, SandboxError, lock.AlreadyRunning) as error:
        print(f"host: {secret.redact(str(error))}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
