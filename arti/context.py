"""Context management: how big a request is, what a request contains, and compaction.

`for_request` turns stored history into what a provider receives: image paths become base64,
only the newest images are kept (a pruned file becomes a short note), and consecutive user
messages are merged. `maybe_compact` replaces older history by a model-written
summary when the history nears the context window. Warnings about unknown outcomes are carried
into the summary word for word, so the model never forgets that an effect may have happened;
plugins add what they must keep (api.on_compact).
"""

import json
import logging

from arti import media

log = logging.getLogger(__name__)

MAX_IMAGES = 16  # newest images kept in a request; older ones become a short note
MAX_IMAGE_BYTES = 24_000_000  # total image file bytes in one request
IMAGE_TOKENS = 1_600
COMPACT_AT = 0.75  # compact when the estimate passes this share of the context window
# The budget when agent.context_window is 0 (the default): the model's window, but at most this.
# Every request carries the whole history, so a 1M-token window used to the full would make
# each one slow and costly; a larger window is used only when the config asks for it.
DEFAULT_WINDOW = 200_000
KEEP_SHARE = 0.4  # the history kept verbatim after compaction fits in this share
SUMMARY_INPUT_CHARS = 500_000  # a safety cap; a pass never gives the summariser more than
MAX_SPAN_TOKENS = 100_000  # ...this much history, so it always reads all it summarises
MAX_KEPT = 20  # unknown outcomes carried from summary to summary (the latest ones)
MAX_PASSES = 8  # passes in one compaction, each building on the summary before
SUMMARY_TIMEOUT = 300
SUMMARY_SYSTEM = (
    "You summarise the earlier part of a conversation between a user and their AI assistant, "
    "so the assistant can carry on without it. Write plain prose and short lists: what the user "
    "asked for and prefers, decisions made, what was done (files, commands, sends) and what is "
    "still open. Keep names, paths, numbers and identifiers exactly. Do not address the user."
)


def estimate(messages: list[dict], system: str = "", tools: list[dict] = ()) -> int:
    """A rough token count: about 3 characters per token, plus a fixed cost per image."""
    images = 0
    chars = len(system) + len(json.dumps(list(tools)))
    for message in messages:
        for block in message["content"]:  # a tool result's own text is inside its JSON
            parts = block["content"] if block.get("type") == "tool_result" else [block]
            images += sum(part.get("type") == "image" for part in parts)
            text = [part for part in parts if part.get("type") != "image"]
            if block.get("type") == "tool_result":
                text = [{**block, "content": text}]
            chars += sum(len(json.dumps(part, ensure_ascii=False)) for part in text)
    return chars // 3 + images * IMAGE_TOKENS


def _all_blocks(message: dict):
    for block in message["content"]:
        yield block
        if block.get("type") == "tool_result":
            yield from block["content"]


# --- the request ---------------------------------------------------------------------------------


def for_request(conv) -> list[dict]:
    """The history as a provider receives it (runs in a worker thread: it reads image files)."""
    messages = []
    for stored in conv.messages:
        message = {"role": stored["role"], "content": [dict(b) for b in stored["content"]]}
        if stored.get("native"):  # the provider itself ignores data from another route
            message["native"] = stored["native"]
        if messages and messages[-1]["role"] == message["role"] == "user":
            _merge_user(messages[-1], message)
        else:
            messages.append(message)
    if conv.summary:
        note = {"type": "text", "text": conv.summary}
        if messages and messages[0]["role"] == "user":
            messages[0]["content"].insert(0, note)
        else:
            messages.insert(0, {"role": "user", "content": [note]})
    _load_images(messages)
    return messages


def _merge_user(into: dict, message: dict) -> None:
    """Join two user messages; tool results go first, as providers require."""
    blocks = into["content"] + message["content"]
    results = [b for b in blocks if b.get("type") == "tool_result"]
    into["content"] = results + [b for b in blocks if b.get("type") != "tool_result"]


def _load_images(messages: list[dict]) -> None:
    """Replace image paths by base64, newest first, within the count and byte budgets."""
    holders = []  # every list of blocks that may hold an image, oldest first
    for message in messages:
        for block in message["content"]:
            if block.get("type") == "tool_result":  # copied: the stored history stays as it is
                block["content"] = [dict(b) for b in block["content"]]
                holders.append(block["content"])
        holders.append(message["content"])
    slots = [(h, i) for h in holders for i, b in enumerate(h) if b.get("type") == "image"]
    kept, used = 0, 0
    for holder, index in reversed(slots):
        block = holder[index]
        size = media.file_size(block.get("path", ""))
        if size is None:
            holder[index] = {"type": "text", "text": "[image no longer available]"}
        elif kept >= MAX_IMAGES or used + size > MAX_IMAGE_BYTES:
            holder[index] = {"type": "text", "text": "[earlier image omitted to save space]"}
        else:
            try:
                media_type, data = media.encode(block["path"])
            except OSError:
                holder[index] = {"type": "text", "text": "[image no longer available]"}
                continue
            holder[index] = {"type": "image", "media_type": media_type, "data": data}
            kept, used = kept + 1, used + size


# --- compaction ----------------------------------------------------------------------------------


async def maybe_compact(conv, system: str, tools: list[dict], force: bool = False) -> bool:
    """Summarise older history if it is near the context window (or `force`); True if done.

    The newest history that fits in KEEP_SHARE of the window stays word for word. What comes
    before it is summarised in passes of at most MAX_SPAN_TOKENS, each folding the previous
    summary in, so the summariser always reads everything it summarises. A failed pass stops
    there and leaves the rest of history as it was.
    """
    limit = budget(conv)
    fixed = estimate([], system, tools) + estimate([{"content": [{"text": conv.summary}]}])
    sizes = [estimate([m]) for m in conv.messages]
    if not force and fixed + sum(sizes) < COMPACT_AT * limit:
        return False
    keep = KEEP_SHARE * limit - fixed
    if force:  # the provider said "too long", so the estimate is off: keep at most half
        keep = min(keep, sum(sizes) / 2)
    target = choose_cut(conv.messages, sizes, keep)
    if target is None:
        return False
    first_kept = conv.messages[target]  # stays the first message kept, pass after pass
    done = False
    for _ in range(MAX_PASSES):
        index = next(i for i, m in enumerate(conv.messages) if m is first_kept)
        if index == 0 or not await _summarise(conv, bounded_cut(conv.messages, index)):
            break
        done = True
    return done


def budget(conv) -> int:
    """The context window compaction works to, in tokens: agent.context_window if set, else the
    model's window capped at DEFAULT_WINDOW."""
    provider, model, _ = conv.route
    window = conv.app.provider_spec(provider).context_window(model)
    return conv.app.config["agent"]["context_window"] or min(window, DEFAULT_WINDOW)


async def _summarise(conv, cut: int) -> bool:
    """One pass: fold messages[:cut] (and the summary so far) into a new summary."""
    span = conv.messages[:cut]
    try:
        summary = await conv.app.complete(
            SUMMARY_SYSTEM,
            render(conv.summary, span),
            provider=conv.route[0],
            model=conv.app.provider_spec(conv.route[0]).summary_model,
            effort="high",
            timeout=SUMMARY_TIMEOUT,
        )
    except Exception as error:
        log.warning("compaction failed (%s); history unchanged", error)
        return False
    if not summary.strip():
        return False
    conv.compact(summary_note(conv, summary, span), cut)
    log.info("compacted %d messages in %s", cut, conv.key or "a headless run")
    return True


def safe_points(messages: list[dict]) -> list[int]:
    """Where history may be cut: messages with no tool results, so a call is never split from
    its result (the first message is never a cut: something must be summarised)."""
    return [
        i
        for i in range(1, len(messages))
        if not any(b.get("type") == "tool_result" for b in messages[i]["content"])
    ]


def choose_cut(messages: list[dict], sizes: list[int], keep_tokens: float) -> int | None:
    """Where the kept history starts: the earliest safe point whose tail fits `keep_tokens`
    (at least the last safe point, so something is always summarised); None if no point."""
    safe = safe_points(messages)
    if not safe:
        return None
    for i in safe:
        if sum(sizes[i:]) <= keep_tokens:
            return i
    return safe[-1]


def bounded_cut(messages: list[dict], target: int) -> int:
    """The latest safe point up to `target` whose span fits MAX_SPAN_TOKENS (else the first)."""
    points = [i for i in safe_points(messages) if i <= target]
    sizes = [estimate([m]) for m in messages[:target]]
    fitting = [i for i in points if sum(sizes[:i]) <= MAX_SPAN_TOKENS]
    return fitting[-1] if fitting else points[0]


def render(old_summary: str, span: list[dict]) -> str:
    """The span as plain text for the summariser; an earlier summary comes first."""
    lines = [old_summary] if old_summary else []
    for message in span:
        for block in message["content"]:
            kind = block.get("type")
            if kind == "text":
                lines.append(f"{message['role']}: {block['text']}")
            elif kind == "image":
                lines.append(f"{message['role']}: [image]")
            elif kind == "tool_use":
                lines.append(f"tool call {block['name']}({json.dumps(block['input'])[:500]})")
            elif kind == "tool_result":
                text = " ".join(b.get("text", "[image]") for b in block["content"])
                lines.append(f"tool result: {text[:2000]}")
    text = "\n".join(lines)
    if len(text) > SUMMARY_INPUT_CHARS:
        half = SUMMARY_INPUT_CHARS // 2
        text = text[:half] + "\n[...]\n" + text[-half:]
    return text


def summary_note(conv, summary: str, span: list[dict]) -> str:
    """The summary as the model sees it, plus what must survive word for word: unknown outcomes,
    pending waits and what plugins keep (api.on_compact)."""
    parts = [f"[Arti: summary of the earlier conversation]\n{summary.strip()}"]
    # Kept in the session state too, so they survive every later pass and compaction verbatim
    # (an earlier summary reaches the summariser only as text it may paraphrase).
    unknown = conv.state.get("unknown", [])
    unknown += [b["text"] for m in span for b in _all_blocks(m) if b.get("unknown")]
    unknown = conv.state["unknown"] = unknown[-MAX_KEPT:]
    if unknown:
        parts.append("Outcomes still unknown (check before repeating):\n" + "\n".join(unknown))
    waits = [inp.text for inp, _ in conv.waits.values()]
    if waits:
        parts.append("Waits still pending (each wakes you when it ends):\n" + "\n".join(waits))
    for hook in conv.app.compact_hooks:
        try:
            parts.append((hook(conv, span) or "").strip())
        except Exception:
            log.exception("compact hook %r failed", hook)
    return "\n\n".join(part for part in parts if part)
