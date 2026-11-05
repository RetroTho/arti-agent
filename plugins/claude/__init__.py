"""claude: the claude-subscription model provider, the official `claude` CLI on the owner's own
login (Arti never reads it).

claude.py runs the CLI for each request and judges the run; relay.py is the loopback relay the
CLI talks to (it admits exactly one upstream request); rewrite.py rewrites that request for the
prompt cache; mcp_stub.py is the inert tool server the CLI starts. The Messages API code and the
Claude models are the anthropic plugin's.
"""

from plugins.anthropic.models import MODELS, context_window

from .claude import ClaudeProvider

# The CLI is a child process (the safety plugin's proc tracks it); the wire format is anthropic's.
REQUIRES = ("safety", "anthropic")

# The optional [providers.claude-subscription] keys ("" and 0 mean "the default").
OPTIONS = {"bin": "", "timeout": 0.0, "max_output_tokens": 0}  # bin: the claude CLI


def register(api) -> None:
    api.provider(
        "claude-subscription",
        lambda app, options: ClaudeProvider(options, app.data_dir),
        context_window=context_window,
        options=OPTIONS,
        **MODELS,
    )
