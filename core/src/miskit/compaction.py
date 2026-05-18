from miskit.message import Message
from miskit.message import format_message

_COMPACT_TOOL_OUTPUT_CHARS = 1000
_BYTES_PER_TOKEN = 4


def estimate_tokens(text):
    if not text:
        return 0
    length = len(text.encode("utf-8")) if isinstance(text, str) else len(text)
    return (length + _BYTES_PER_TOKEN - 1) // _BYTES_PER_TOKEN


def estimate_message_tokens(message):
    text = format_message(message)
    return estimate_tokens(text)


def estimate_messages_tokens(messages):
    total = 0
    for message in messages:
        total += estimate_message_tokens(message)
    return total


class Compactor:
    def __init__(self, context_tokens=8000, compact_at=0.5, keep_recent=10):
        self.context_tokens = context_tokens
        self.compact_at = compact_at
        self.keep_recent = keep_recent

    @classmethod
    def from_config(cls, config):
        return cls(
            context_tokens=config.get("tokens", 8000),
            compact_at=config.get("compactAt", 0.5),
            keep_recent=config.get("keepRecent", 10),
        )

    def should_compact(self, prompt_tokens, message_count):
        if message_count <= self.keep_recent:
            return False
        return prompt_tokens >= self.context_tokens * self.compact_at

    def split(self, messages):
        messages = list(messages)
        if len(messages) <= self.keep_recent:
            return [], messages

        ideal = max(0, len(messages) - self.keep_recent)
        if ideal == 0:
            return [], messages

        start = ideal
        while start > 0 and not self._safe_start(messages[start]):
            start -= 1

        if start == 0 and not self._safe_start(messages[0]):
            start = ideal
            while start < len(messages) and not self._safe_start(messages[start]):
                start += 1

        if start >= len(messages):
            start = ideal

        if start == 0:
            return [], messages

        return messages[:start], messages[start:]

    def _safe_start(self, message):
        if message.role == "tool":
            return False
        if message.role == "assistant" and message.tool_calls:
            return False
        return True

    def compact_messages(self, messages, store=None):
        result = []
        for message in messages:
            if (message.role == "tool" and
                    isinstance(message.content, str) and
                    len(message.content) > _COMPACT_TOOL_OUTPUT_CHARS):
                if store is not None:
                    store_id = store.save(message.content)
                    content = (
                        message.content[:_COMPACT_TOOL_OUTPUT_CHARS] +
                        f"\n\n[truncated, id: {store_id}"
                        f" -- use read_more(id=\"{store_id}\", offset={_COMPACT_TOOL_OUTPUT_CHARS}) to retrieve the rest]"
                    )
                else:
                    content = message.content[:_COMPACT_TOOL_OUTPUT_CHARS] + "\n\n[truncated]"
                result.append(message.copy(content=content))
            else:
                result.append(message)
        return result

    def split_for_summary(self, messages, reserved_chars=0):
        max_chars = max(1, self.context_tokens * 5 // 2 - reserved_chars)
        chunks = []
        current = []
        current_chars = 0
        for message in messages:
            text = format_message(message)
            if current and message.role == "user" and current_chars + len(text) > max_chars:
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(message)
            current_chars += min(len(text), max_chars)
        if current:
            chunks.append(current)
        return chunks or [messages]

    def summary_prompt(self, messages, prior_summary=""):
        max_chars = self.context_tokens * 5 // 2
        parts = [
            "Summarize this conversation so Miskit can continue it later.\n\n"
            "Do not answer the conversation. Only write a compact summary for future turns.\n\n"
            "Include important names, facts, decisions, preferences, open tasks, "
            "and any details the user may expect Miskit to remember. Keep it concise."
        ]
        if prior_summary:
            parts.append(f"Summary so far:\n\n{prior_summary}")
        texts = []
        for message in messages:
            text = format_message(message)
            if len(text) > max_chars:
                text = text[:max_chars] + "\n[message truncated for summary]"
            texts.append(text)
        parts.append("\n\n".join(texts))
        return "\n\n".join(parts)

    def summary_message(self, summary, store_id=None):
        content = (
            "# Previous Conversation Summary\n\n"
            "This is a compact summary of an archived conversation. Use it as background context, "
            "but prefer newer messages when details conflict. "
            "Use recall(action=\"search\") when exact older details matter.\n\n"
            f"{summary}"
        )
        if store_id is not None:
            content += (
                f"\n\n[Full transcript archived, id: {store_id}"
                f" -- use read_more(id=\"{store_id}\") to retrieve the original messages]"
            )
        return Message("system", content)
