"""todo: a per-session task list for multi-step work.

The list lives in the session's state (`ctx.state["todo"]`), which core saves after each call,
so it survives a restart, and a compaction carries it into the summary (`on_compact`); `/new`
starts a new session and so a new list.
"""

from arti.tools import ToolError

MAX_ITEMS = 50
MARKS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "cancelled": "[-]"}


def _new_item(item: dict) -> dict:
    if not item.get("content", "").strip():
        raise ValueError(f"new todo {item['id']!r} needs content")
    return {**item, "content": item["content"].strip(), "status": item.get("status", "pending")}


def update(current: list[dict], todos: list[dict], merge: bool) -> list[dict]:
    """The new list: `todos` replaces `current`, or with `merge` updates items by id and appends."""
    items = [dict(item) for item in current] if merge else []
    by_id = {item["id"]: item for item in items}
    for change in todos:
        item = by_id.get(change["id"])
        if item is None:
            item = by_id[change["id"]] = _new_item(change)
            items.append(item)
        elif not merge:
            raise ValueError(f"todo id {change['id']!r} is used twice")
        else:
            item["content"] = change.get("content", item["content"]).strip() or item["content"]
            item["status"] = change.get("status", item["status"])
    if len(items) > MAX_ITEMS:
        raise ValueError(f"a todo list holds at most {MAX_ITEMS} items; drop finished ones")
    return items


def render(items: list[dict]) -> str:
    if not items:
        return "Todo list is empty."
    done = sum(item["status"] == "completed" for item in items)
    lines = [f"{MARKS[item['status']]} {item['id']}. {item['content']}" for item in items]
    return f"Todo ({done}/{len(items)} done):\n" + "\n".join(lines)


def todo(args: dict, ctx) -> str:
    if "todos" in args:
        try:
            current = ctx.state.get("todo", [])
            ctx.state["todo"] = update(current, args["todos"], args.get("merge", False))
        except ValueError as e:
            raise ToolError(str(e)) from None
    return render(ctx.state.get("todo", []))


def on_compact(conv, span) -> str | None:
    """The current list, kept word for word under a summary of older history."""
    items = conv.state.get("todo")
    return "Current todo list:\n" + render(items) if items else None


def register(api) -> None:
    api.on_compact(on_compact)
    item = {
        "type": "object",
        "required": ["id"],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "content": {"type": "string"},
            "status": {"type": "string", "enum": list(MARKS)},
        },
    }
    api.tool(
        "todo",
        "Track a task list for multi-step work (3+ steps). No arguments reads it; `todos` "
        "replaces it; merge=true updates items by id and appends new ones. Keep one item "
        "in_progress and complete items only after verifying them. Returns the whole list.",
        {
            "properties": {
                "todos": {"type": "array", "maxItems": MAX_ITEMS, "items": item},
                "merge": {"type": "boolean"},
            }
        },
        todo,
        review=None,
        guest=True,
    )
