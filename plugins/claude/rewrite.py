"""How the relay rewrites the Claude CLI's one request before it goes upstream (pure functions).

The CLI adds per-request context (the date, an account reminder) to the turn it answers and
puts a cache breakpoint on it. The next request replays that turn WITHOUT the context, so the
cached prefix would never recur and every request would pay for the whole history again. The
rewrite puts Arti's own frame (the newest user turn, as Arti sent it: `queried`) back in front
and pins the breakpoint on content the next request repeats byte for byte.

It never drops content, never adds a breakpoint, and forwards the body unchanged whenever it
cannot prove the request is shaped as the CLI builds it. relay.py calls `rewrite`.
"""

from __future__ import annotations

import copy
import json

from plugins.anthropic.messages import THINKING


def rewrite(body: bytes, queried: list[dict]) -> tuple[bytes, str]:
    """(the request body to send upstream, what restore_tool_results did)."""
    try:
        request = json.loads(body)
        messages = request["messages"]
        restored = restore_tool_results(messages, queried)
        reordered = frame_first(messages, queried)
        pinned = pin_breakpoint(messages, queried)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        return body, "unrestored"
    if restored != "restored" and not reordered and not pinned:
        return body, restored  # the original bytes, not a re-encoding of them
    return json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(), restored


def restore_tool_results(messages: list[dict], queried: list[dict]) -> str:
    """Put Arti's exact tool results back where the CLI appended its context to them; that
    text moves to the end of the turn. Returns "" (no tool results in the frame), "same",
    "restored", or "unrestored" (nothing changed: some result is not Arti's + plain text)."""
    ours = [b for b in queried if b.get("type") == "tool_result"]
    if not ours:
        return ""
    content = _newest_turn(messages).get("content")
    sent = [b for b in content or [] if isinstance(b, dict) and b.get("type") == "tool_result"]
    if [b.get("tool_use_id") for b in sent] != [b["tool_use_id"] for b in ours]:
        return "unrestored"
    notes = []
    for native, own in zip(sent, ours):
        note = _appended(native.get("content", ""), own["content"])
        if note is None or native.get("is_error", False) is not bool(own.get("is_error")):
            return "unrestored"
        notes.append(note)
    if all(native["content"] == own["content"] for native, own in zip(sent, ours)):
        return "same"
    for native, own in zip(sent, ours):
        native["content"] = copy.deepcopy(own["content"])
    content.extend({"type": "text", "text": note} for note in notes if note.strip())
    return "restored"


def frame_first(messages: list[dict], queried: list[dict]) -> bool:
    """On the first turn the CLI puts its context text AHEAD of Arti's frame (later turns put
    it after). Move the frame first so the next request, which replays the turn bare, finds
    it cached. Only when the frame occurs exactly once and only text precedes it."""
    if not queried or any(m.get("role") == "assistant" for m in messages):
        return False
    first = next((m for m in messages if m.get("role") == "user"), {})
    content, n = first.get("content"), len(queried)
    if not isinstance(content, list) or len(content) <= n:
        return False
    frame = [_unmarked(b) for b in queried]
    starts = [
        k
        for k in range(len(content) - n + 1)
        if [_unmarked(b) for b in content[k : k + n]] == frame
    ]
    if len(starts) != 1 or starts[0] == 0:
        return False
    k = starts[0]
    if not all(isinstance(b, dict) and b.get("type") == "text" for b in content[:k]):
        return False
    first["content"] = content[k : k + n] + content[:k] + content[k + n :]
    return True


def pin_breakpoint(messages: list[dict], queried: list[dict]) -> bool:
    """Move the breakpoints that sit after the last block the next request will replay onto
    that block (keeping the first marker's value). What recurs: everything through the last
    assistant message, plus the leading blocks of the newest turn that equal Arti's frame
    (unless the first block that differs is a tool_result: measured, that gave no cache hit)."""
    blocks = [  # ((message index, block index), block) for every block in a list content
        ((i, j), b)
        for i, m in enumerate(messages)
        if isinstance(m.get("content"), list)
        for j, b in enumerate(m["content"])
    ]
    last = _last_assistant(messages)
    recurring = [(at, b) for at, b in blocks if at[0] <= last]
    newest = _newest_turn(messages)
    if newest:
        same = 0
        for sent, own in zip(newest["content"], queried):
            if _unmarked(sent) != _unmarked(own):
                break
            same += 1
        rest = newest["content"][same:]
        if not (rest and isinstance(rest[0], dict) and rest[0].get("type") == "tool_result"):
            recurring += [((last + 1, j), newest["content"][j]) for j in range(same)]
    # A thinking block cannot carry a breakpoint, so the target is the last other block.
    targets = [
        (at, b) for at, b in recurring if isinstance(b, dict) and b.get("type") not in THINKING
    ]
    if not targets:
        return False
    target_at, target = targets[-1]
    later = [
        b for at, b in blocks if at > target_at and isinstance(b, dict) and "cache_control" in b
    ]
    if not later:
        return False
    markers = [b.pop("cache_control") for b in later]
    target.setdefault("cache_control", markers[0])
    return True


def _unmarked(block):
    """A block without its cache marker, to compare content only."""
    if not isinstance(block, dict):
        return block
    return {k: v for k, v in block.items() if k != "cache_control"}


def _last_assistant(messages: list[dict]) -> int:
    """The index of the last assistant message, -1 if there is none."""
    return max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)


def _newest_turn(messages: list[dict]) -> dict:
    """The user turn right after the last assistant message (the one being answered), or {}."""
    after = _last_assistant(messages) + 1
    newest = messages[after] if after < len(messages) else {}
    return newest if newest.get("role") == "user" else {}


def _texts_joined(content) -> list:
    """Content as a list in which each run of bare text blocks ({type, text} only) is one
    string, so text split differently across blocks compares equal."""
    if isinstance(content, str):
        return [content]
    items = []
    for block in content:
        bare_text = isinstance(block, dict) and set(block) == {"type", "text"}
        if bare_text and block["type"] == "text":
            if items and isinstance(items[-1], str):
                items[-1] += block["text"]
            else:
                items.append(block["text"])
        else:
            items.append(block)
    return items


def _appended(sent, ours) -> str | None:
    """The plain text the CLI appended to a tool result whose content was `ours`, or None when
    `sent` is anything other than `ours` followed by plain text."""
    s, o = _texts_joined(sent), _texts_joined(ours)
    if len(s) < len(o) or s[: len(o) - 1] != o[: len(o) - 1]:
        return None
    # All but our last item are equal. Our last item may be text the CLI extended in place.
    last, extra = s[len(o) - 1], s[len(o) :]
    if isinstance(o[-1], str):
        if not isinstance(last, str) or not last.startswith(o[-1]):
            return None
        extra = [last[len(o[-1]) :], *extra]
    elif last != o[-1]:
        return None
    return "".join(extra) if all(isinstance(x, str) for x in extra) else None
