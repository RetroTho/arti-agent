"""anthropic: the Messages API as a model provider, with ANTHROPIC_API_KEY from .env.

messages.py speaks the Messages API (history encoding, stream assembly, its rate-limit headers);
models.py knows the Claude models. The claude plugin (claude-subscription) builds on both.
"""

from .messages import AnthropicProvider
from .models import MODELS, context_window

# The optional [providers.anthropic] keys, with their defaults standing in for the type (0 and ""
# mean "the default").
OPTIONS = {"api_key_env": "", "base_url": "", "timeout": 0.0, "max_output_tokens": 0}


def register(api) -> None:
    api.provider(
        "anthropic",
        lambda app, options: AnthropicProvider(options),
        context_window=context_window,
        options=OPTIONS,
        **MODELS,
    )
