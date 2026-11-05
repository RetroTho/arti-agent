"""wait: come back to this conversation later, without blocking.

The wait is a notice the host delivers into this conversation (in a group: the caller's lane) at
the chosen time; it survives a restart, and /stop or /new cancels it. The tool returns at once
and the turn should end: anything new wakes Arti sooner anyway, and the notice still comes later
(the model ignores it if moot).
"""

import time

from arti.media import stamp
from arti.tools import ToolError

MAX_MINUTES = 7 * 24 * 60
MAX_PENDING = 20  # waits per conversation


def wait(args: dict, ctx) -> str:
    if len(ctx.conv.waits) >= MAX_PENDING:
        raise ToolError(f"this conversation already has {MAX_PENDING} waits pending")
    now = time.time()
    due = now + args["minutes"] * 60
    note = (
        f"[Arti: your wait (set {stamp(ctx.app, now)}, due {stamp(ctx.app, due)}): {args['note']}]"
    )
    ctx.notify(note, at=due)
    ctx.mark_answered()  # deliberate quiet is not an unanswered message
    return (
        f"Wait set until {stamp(ctx.app, due)}. End your turn now (reply [SILENT] if there is "
        "nothing to say). New messages wake you sooner; the wait still ends later, so ignore it "
        "then if it is moot."
    )


def register(api) -> None:
    api.tool(
        "wait",
        "Come back to this chat later: after `minutes`, a notice with your note arrives here and "
        "wakes you. Returns at once; then end your turn. Use it to check on something later or "
        "to give people time to answer (new messages wake you sooner anyway).",
        {
            "properties": {
                "minutes": {"type": "number", "minimum": 1, "maximum": MAX_MINUTES},
                "note": {"type": "string", "pattern": r"\S", "description": "what to do then"},
            },
            "required": ["minutes", "note"],
        },
        wait,
        review=None,
        guest=True,
        interactive=True,
    )
