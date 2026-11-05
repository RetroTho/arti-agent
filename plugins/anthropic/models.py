"""The Claude models: which ones have a 1M window, the short names, and the defaults (shared by
the anthropic route and the claude plugin's claude-subscription route)."""

# Claude models whose window is 1M. The CLI runs a plain id at 200K, so these are sent to it
# with "[1m]"; any other id is sent bare and reported as 200K (never pack more than it takes).
LONG = set(
    "claude-opus-5-5 claude-sonnet-5-5 claude-haiku-5-5 claude-opus-5 claude-sonnet-5"
    " claude-opus-4-8 claude-fable-5-1".split()
)
ALIASES = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}
ALIASES |= {"haiku": "claude-haiku-5-5", "fable": "claude-fable-5-1"}


def claude_route(model: str) -> str:
    """The model id the Claude CLI gets: aliases resolved, "[1m]" on the 1M models."""
    base = model.removesuffix("[1m]")
    base = ALIASES.get(base, base)
    return base + "[1m]" if base in LONG else model


MODELS = dict(  # the default model, the judge's, and the one that summarises old history
    default_model="claude-opus-5-5",
    judge_model="claude-haiku-5-5",
    summary_model="claude-sonnet-5-5",  # good at it, and lighter on quota
)


def context_window(model: str) -> int:
    """A conservative compaction budget in tokens; agent.context_window overrides it."""
    return 1_000_000 if claude_route(model).endswith("[1m]") else 200_000
