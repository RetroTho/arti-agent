"""The agent's command line, run inside a user's sandbox (the host starts it there): run the
agent, run one task (--once), or run a plugin's command.

python -m arti                         the always-on agent, on its link to the host (link.py)
python -m arti --once "PROMPT" [--image PATH] [--provider P] [--model M] [--effort E] [-v]
python -m arti NAME [ARGS...]          a plugin's command; `python -m arti --help` lists them

Every mode first loads config.toml and the plugins, and checks one against the other. A plugin
command then runs on the App without starting it: no lock, no store, no channel (see
plugin.Api.command). Nothing here knows a plugin, a provider or a platform by name.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys

from arti import config, secret
from arti import log as log_setup
from arti.app import App
from arti.channel import FakeChannel
from arti.config import ConfigError

log = logging.getLogger("arti")


def parser(commands: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arti",
        usage="%(prog)s [--once PROMPT [options]] | %(prog)s COMMAND [ARGS...]",
        epilog=commands or None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--once", metavar="PROMPT", help="run one task without a chat, print the reply")
    p.add_argument("--image", action="append", default=[], help="image for --once (repeatable)")
    p.add_argument("--provider", help="the model provider (a plugin's), for this run")
    p.add_argument("--model", default="")
    p.add_argument("--effort", choices=config.EFFORTS)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def parse(argv: list[str]) -> argparse.Namespace:
    p = parser()
    args = p.parse_args(argv)
    if args.image and args.once is None:
        p.error("--image needs --once")
    if args.once is not None and not args.once.strip():
        p.error("--once needs a prompt")
    return args


def configure(data_dir, args: argparse.Namespace | None) -> App:
    """The App with its config loaded and its plugins registered and checked (not started)."""
    overrides = {}
    if args is not None:
        overrides = dict(provider=args.provider or "", model=args.model, effort=args.effort or "")
    app = App(config.load(data_dir, **overrides), data_dir)
    try:
        app.load_plugins()
    except ConfigError:
        if args is not None and args.provider and args.provider not in app.provider_specs:
            known = ", ".join(app.provider_specs)
            parser().error(f"argument --provider: invalid choice: {args.provider!r} ({known})")
        raise
    return app


def show_help(data_dir) -> int:
    """--help, with the commands the plugins registered (when the configuration loads; the
    secrets in .env are not read for it)."""
    try:
        app = configure(data_dir, None)
        lines = [f"  {name:<16} {c.help}" for name, c in sorted(app.commands.items())]
        epilog = "commands (from plugins):\n" + "\n".join(lines or ["  (none)"])
    except Exception as error:
        epilog = f"plugin commands are unavailable: {secret.redact(str(error))}"
    parser(epilog).print_help()
    return 0


def run_command(app: App, name: str, argv: list[str]) -> int:
    command = app.commands.get(name)
    if command is None:
        known = ", ".join(sorted(app.commands)) or "none"
        print(f"arti: unknown command {name!r} (commands: {known}; see --help)", file=sys.stderr)
        return 2
    return command.func(app, argv)


def once_channel(app: App) -> FakeChannel:
    """A printing channel that still knows the owner, as the link to the host names them (a job
    --once creates is the owner's): the link is made, never started."""
    real = app.make_channel()
    return FakeChannel(home=real.home, owner=real.owner, like=real)


async def once(app: App, args: argparse.Namespace) -> int:
    """One headless task with a printing channel; prints the final reply."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, asyncio.current_task().cancel)
    try:
        await app.start(serve=False)
        print(await app.run_headless(args.once, images=args.image) or "[SILENT]")
        return 0
    except asyncio.CancelledError:
        return 130
    finally:
        await app.shutdown()


async def daemon(app: App) -> bool:
    """Run the agent until SIGTERM/SIGINT or /restart; True when a restart was asked."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, app.exit_requested.set)
    try:
        await app.start()
        await app.exit_requested.wait()
    finally:
        await app.shutdown()
    return app.restart


def run(coro):
    """Run `coro` on a fresh event loop. Not asyncio.run: on the way out it waits without limit
    for worker threads, and one hung plugin thread must never keep a stopped Arti alive."""
    return asyncio.new_event_loop().run_until_complete(coro)


def exit_now(code: int):
    """Exit at once after a clean shutdown: threads abandoned by tools or hooks are not awaited."""
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    os._exit(code)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    os.umask(0o077)
    data_dir = config.data_dir()
    data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if argv[:1] in (["-h"], ["--help"]):
        return show_help(data_dir)
    command = argv[0] if argv and not argv[0].startswith("-") else None
    args = None if command else parse(argv)
    try:
        secret.load_env(data_dir / ".env")  # every value a secret (a redactor hides them)
        if command:
            return run_command(configure(data_dir, None), command, argv[1:])
        log_setup.setup(data_dir, verbose=args.verbose)
        app = configure(data_dir, args)
        if args.once is not None:
            app.channel = once_channel(app)
            exit_now(run(once(app, args)))
        restart = run(daemon(app))
    except KeyboardInterrupt:
        exit_now(130)
    except Exception as error:
        print(f"arti: {secret.redact(str(error))}", file=sys.stderr)
        exit_now(1)  # not sys.exit: a stuck worker thread must not keep a dead Arti alive
    if restart:
        log.info("restarting")
        try:
            os.execv(sys.executable, [sys.executable, "-m", "arti", *argv])
        except OSError:
            log.exception("re-exec failed; exiting so the sandbox starts Arti again")
            exit_now(1)
    exit_now(0)


if __name__ == "__main__":
    sys.exit(main())
