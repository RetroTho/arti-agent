"""A fake App and ToolContext for the heavy plugins' tests (web, browser, vision, image): just
the attributes those plugins use."""

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from arti.tools import ToolError
from plugins import browser

CODE_DIR = Path(__file__).resolve().parents[2]


def make_app(**sections) -> SimpleNamespace:
    data_dir = Path(tempfile.mkdtemp(prefix="arti-heavy-"))
    # The safety plugin's [network], and each plugin's own section with its defaults.
    config = {
        "network": {"allow_private_urls": False},
        "browser": dict(browser.SETTINGS),
        "image": {"model": "gpt-image-2-medium"},
    }
    for name, values in sections.items():
        config[name].update(values)

    def cache_dir(name):
        path = data_dir / "cache" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    return SimpleNamespace(config=config, data_dir=data_dir, code_dir=CODE_DIR, cache_dir=cache_dir)


def make_ctx(app=None, scope="home") -> SimpleNamespace:
    app = app or make_app()
    cancel = threading.Event()

    def check():
        if cancel.is_set():
            raise ToolError("cancelled")

    context_dir = app.data_dir if scope == "home" else app.data_dir / "contexts" / scope
    context_dir.mkdir(parents=True, exist_ok=True)

    def cache_dir(name):  # like ToolContext.cache_dir: each context has its own
        if scope == "home":
            return app.cache_dir(name)
        path = context_dir / "cache" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    return SimpleNamespace(
        app=app,
        cancel=cancel,
        check=check,
        scope=scope,
        key="home",
        trusted=scope == "home",
        context_dir=context_dir,
        state={},
        headless=False,
        cache_dir=cache_dir,
    )


def register(module, app=None):
    """Run module.register through the real Api; returns the Api with its tools and hooks."""
    from arti.plugin import Api

    app = app or make_app()
    app.tools, app.commands, app.channels, app.provider_specs, app.plugin_sections = (
        {},
        {},
        {},
        {},
        {},
    )
    api = Api(app, module.__name__)
    module.register(api)
    return api
