"""delegate_task: run up to 3 self-contained tasks in fresh headless sessions, concurrently.

Each leaf gets this context's tools minus the ones that would let it message the user, schedule
work, change prompt memory or delegate again (interactive tools are hidden from every headless
run already). Leaves never talk to the owner, not even to ask for an approval: like any run with
no chat, a flagged call runs only when a grant covers it (or the judge approves it, with
approvals.unattended = "smart"), else it is blocked. A failed leaf becomes that task's result;
the others still report.
"""

from concurrent.futures import ThreadPoolExecutor

MAX_TASKS = 3
LEAF_TIMEOUT = 600  # seconds per leaf
LEAF_EXCLUDED = ["delegate_task", "cronjob", "memory", "skill_manage"]
PREAMBLE = (
    "You are a focused subagent working for another agent; you cannot message the user. Do the "
    "goal below, then end with a short factual summary: what you found or did, files touched, "
    "and anything left undone.\n\n"
)


def leaf_prompt(task: dict) -> str:
    text = PREAMBLE + "GOAL:\n" + task["goal"].strip()
    if task.get("context", "").strip():
        text += "\n\nCONTEXT:\n" + task["context"].strip()
    return text


def _run_one(ctx, task: dict) -> str:
    skills = (task.get("skills") or []) if ctx.trusted else []  # skills are the owner's
    preload = ctx.app.services.get("preload_skills")  # the skills plugin's, if it is loaded
    try:
        result = ctx.run_headless(
            leaf_prompt(task),
            instructions=preload(ctx.app, skills) if preload and skills else "",
            exclude=LEAF_EXCLUDED,
            timeout=LEAF_TIMEOUT,
        )
    except Exception as e:  # one leaf's failure must not hide the others' results
        return f"FAILED: {type(e).__name__}: {e}"
    return result.strip() or "(no summary)"


def delegate_task(args: dict, ctx) -> str:
    tasks = args["tasks"]
    with ThreadPoolExecutor(max_workers=MAX_TASKS) as pool:
        results = list(pool.map(lambda task: _run_one(ctx, task), tasks))
    return "\n\n".join(
        f"### Task {number}: {task['goal'].strip()[:120]}\n{result}"
        for number, (task, result) in enumerate(zip(tasks, results), 1)
    )


def register(api) -> None:
    task = {
        "type": "object",
        "required": ["goal"],
        "additionalProperties": False,
        "properties": {
            "goal": {"type": "string", "pattern": r"\S"},
            "context": {"type": "string"},
            "skills": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string"},
                "description": "skills to preload",
            },
        },
    }
    api.tool(
        "delegate_task",
        f"Run up to {MAX_TASKS} self-contained tasks at once in fresh sessions with your tools "
        "(minus messaging, memory, skills editing, cron and delegation). They see none of this "
        "conversation, so put everything they need in goal/context. Returns each one's summary.",
        {
            "properties": {
                "tasks": {"type": "array", "minItems": 1, "maxItems": MAX_TASKS, "items": task}
            },
            "required": ["tasks"],
        },
        delegate_task,
        review=None,
        timeout=LEAF_TIMEOUT + 60,
        interactive=True,  # never offered to headless runs, so a leaf cannot delegate again
    )
