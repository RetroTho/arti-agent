"""openai: the OpenAI model providers and Arti's own ChatGPT login.

  openai        the Responses API, with OPENAI_API_KEY from .env
  openai-codex  the same API on the ChatGPT backend, signed in with `python -m arti login`

Commands: `login` (ChatGPT device login), `logout`, `auth-status`. responses.py is both routes,
codex.py the login and the ChatGPT image client (the image plugin's image_generate uses it).
"""

import argparse
import json

from . import codex
from .responses import CodexProvider, OpenAIProvider

# The optional [providers.<id>] keys, with their defaults standing in for the type (0 and ""
# mean "the default"). The ChatGPT backend takes no API key and rejects max_output_tokens.
API_OPTIONS = {"api_key_env": "", "base_url": "", "timeout": 0.0, "max_output_tokens": 0}
CHATGPT_OPTIONS = {"base_url": "", "timeout": 0.0}
MODELS = dict(default_model="gpt-5.4", judge_model="gpt-5.4-mini", summary_model="gpt-5.4")
LONG_GPT = ("gpt-5.4", "gpt-5.5", "gpt-5.6", "gpt-6")


def api_window(model: str) -> int:
    """A conservative compaction budget in tokens; agent.context_window overrides it."""
    if model.startswith(("gpt-5.4-mini", "gpt-5.4-nano")):
        return 400_000
    return 1_050_000 if model.startswith(LONG_GPT) else 128_000


def chatgpt_window(model: str) -> int:
    return 272_000 if model.startswith(LONG_GPT) else 128_000


def auth_command(action: str, description: str):
    """`python -m arti login | logout | auth-status`."""

    def run(app, argv: list[str]) -> int:
        argparse.ArgumentParser(
            prog=f"python -m arti {action}", description=description
        ).parse_args(argv)
        auth = codex.CodexAuth(app.data_dir)
        if action == "auth-status":
            print(json.dumps(auth.status()))
        elif action == "logout":
            auth.logout()
            print("Arti's ChatGPT login was removed.")
        else:
            auth.login()
        return 0

    return run


def register(api) -> None:
    api.provider(
        "openai",
        lambda app, options: OpenAIProvider(options),
        context_window=api_window,
        options=API_OPTIONS,
        **MODELS,
    )
    api.provider(
        "openai-codex",
        lambda app, options: CodexProvider(options, codex.shared_auth(app.data_dir)),
        context_window=chatgpt_window,
        options=CHATGPT_OPTIONS,
        **MODELS,
    )
    for action, description in (
        ("login", "sign in to ChatGPT (for openai-codex and image_generate)"),
        ("logout", "remove Arti's ChatGPT login"),
        ("auth-status", "show Arti's ChatGPT login"),
    ):
        api.command(action, auth_command(action, description), description)
