"""terminal and process: bash commands, and background jobs.

A `cd` persists for the session (kept in the session state as "cwd", which the approval review
also reads); exports do not. Output is stdout and stderr merged, cut to its head and tail with the
full text saved to a file. Background jobs live in memory only: they keep running across /stop,
are killed by /new (that conversation's jobs), by `process kill` and at shutdown; after a crash the
startup sweep kills them (proc.sweep).
"""

import logging
import os
import select
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from arti.tools import ToolError
from plugins.safety import proc, redact

REQUIRES = ("safety",)  # imports from it, and its gate reviews this plugin's tools

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 180
MAX_OUTPUT = 30_000
MAX_JOBS = 4  # running background jobs per App
JOB_TTL = 3600  # finished jobs stay listed this long
LOG_CAP = 64 << 20  # a job writing more than this is killed
DEFAULT_WAIT = 60
TAIL_BYTES = 4096
EXTRA_PATH = ["~/.local/bin"]  # where `pip install --user` puts commands


def shell_env() -> dict:
    """The environment of every shell child: no secrets, ~/.local/bin on PATH, no pager."""
    env = redact.child_env()
    parts = env.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin").split(":")
    extra = [os.path.expanduser(p) for p in EXTRA_PATH]
    env["PATH"] = ":".join([p for p in extra if p not in parts] + parts)
    env.setdefault("LANG", "en_US.UTF-8")
    env["PAGER"] = env["GIT_PAGER"] = "cat"
    return env


def _workdir(args, ctx) -> Path:
    """Where this call runs: `workdir` (relative to the session cwd), else the session cwd."""
    cwd = Path(ctx.state.get("cwd") or Path.home())
    if not cwd.is_dir():
        # The review judged the command from this missing directory: never run it elsewhere.
        ctx.state.pop("cwd", None)
        raise ToolError(f"the working directory {cwd} is gone; the cwd is now home. Run it again.")
    path = (cwd / Path(args.get("workdir") or ".").expanduser()).resolve()
    if not path.is_dir():
        raise ToolError(f"working directory {path} does not exist")
    return path


def terminal(args: dict, ctx) -> str:
    cwd = _workdir(args, ctx)
    if args.get("background"):
        return start_job(args["command"], cwd, ctx, notify=bool(args.get("notify")))
    timeout = min(args.get("timeout", DEFAULT_TIMEOUT), ctx.app.config["agent"]["tool_timeout"])
    fd, pwd_file = tempfile.mkstemp(prefix="pwd-")
    os.close(fd)
    # An EXIT trap records the final directory even when the command ends with `exit N`.
    script = f"trap 'pwd > \"$__ARTI_PWD\"' EXIT\n{args['command']}"
    try:
        result = proc.run(
            ["/bin/bash", "-c", script],
            cwd=cwd,
            env={**shell_env(), "__ARTI_PWD": pwd_file},
            timeout=timeout,
            cancel=ctx.cancel,
            max_chars=MAX_OUTPUT,
            spill_dir=ctx.cache_dir("tool-output"),
        )
        new_cwd = Path(pwd_file).read_text().strip() or str(cwd)
    finally:
        os.unlink(pwd_file)
    if not args.get("workdir"):
        ctx.state["cwd"] = new_cwd
    if result.timed_out:
        end = f"timed out after {timeout:.0f}s; process group killed"
    elif result.cancelled:
        end = "cancelled; process group killed"
    else:
        end = "process group killed" if result.code is None else f"exit {result.code}"
    return f"{result.out.rstrip(chr(10)) or '(no output)'}\n[{end}; cwd {new_cwd}]"


# Background jobs


@dataclass
class Job:
    id: str
    command: str
    key: str  # the conversation that started it
    child: proc.Child
    log: Path
    notify: bool
    started: float = field(default_factory=time.time)
    ended: float | None = None
    code: int | None = None
    killed: str = ""  # why Arti killed it ("" = it exited by itself)
    done: threading.Event = field(default_factory=threading.Event)

    def status(self) -> str:
        if not self.done.is_set():
            state = f"running for {time.time() - self.started:.0f}s"
        else:
            state = f"killed ({self.killed})" if self.killed else f"exited with code {self.code}"
        return f"{self.id}: {state}; log {self.log}; command: {self.command[:200]}"

    def kill(self, why: str) -> None:
        if not self.done.is_set():
            self.killed = self.killed or why
            self.child.kill()


_jobs_lock = threading.Lock()


def _jobs(app) -> dict[str, Job]:
    return app.state.setdefault("terminal_jobs", {})


def start_job(command: str, cwd: Path, ctx, notify: bool) -> str:
    jobs = _jobs(ctx.app)
    with _jobs_lock:
        for job_id, job in list(jobs.items()):  # forget long-finished jobs
            if job.ended and time.time() - job.ended > JOB_TTL:
                del jobs[job_id]
        if sum(not job.done.is_set() for job in jobs.values()) >= MAX_JOBS:
            raise ToolError(f"{MAX_JOBS} background jobs are already running; kill one first")
        job_id = "bg_" + os.urandom(4).hex()
        log_path = ctx.cache_dir("tool-output") / f"{job_id}.log"
        log_path.touch(mode=0o600)
        child = proc.spawn(["/bin/bash", "-c", command], cwd=cwd, env=shell_env())
        job = jobs[job_id] = Job(job_id, command, ctx.key, child, log_path, notify)
    threading.Thread(target=_watch, args=(job, ctx), daemon=True, name=job_id).start()
    return (
        f"Started background job {job_id} (output: {log_path}). Use the process tool to poll, "
        "wait for or kill it."
    )


def _watch(job: Job, ctx) -> None:
    """Copy the job's output to its log until it exits, then clean up its group."""
    stream, written = job.child.proc.stdout, 0
    with open(job.log, "ab") as log_file:
        while True:
            ready, _, _ = select.select([stream], [], [], 0.25)
            data = os.read(stream.fileno(), 65536) if ready else None
            if data:
                written += len(data)
                if written > LOG_CAP:
                    job.killed = f"log passed {LOG_CAP >> 20} MiB"
                    break
                log_file.write(data)
                log_file.flush()
            elif data == b"":  # EOF: every writer is gone; wait for the leader
                job.child.proc.wait()
                break
            elif job.child.proc.poll() is not None:
                break
    job.child.kill()  # also ends anything it left running in its group
    stream.close()
    job.code, job.ended = job.child.proc.returncode, time.time()
    job.done.set()
    if job.notify and not job.killed and job.key:
        try:
            ctx.notify(
                f"[Arti: background job {job.id} exited with code {job.code}: "
                f"{job.command[:200]}. Log: {job.log}]"
            )
        except Exception:
            log.exception("could not queue the exit notice of %s", job.id)


def stop_jobs(app, key: str | None = None) -> None:
    """Kill the running jobs of one conversation (/new), or of every one (shutdown)."""
    with _jobs_lock:
        jobs = [job for job in _jobs(app).values() if key is None or job.key == key]
    for job in jobs:
        job.kill("stopped")
    for job in jobs:
        job.done.wait(5)


def process(args: dict, ctx) -> str:
    mine = {i: job for i, job in _jobs(ctx.app).items() if job.key == ctx.key}
    action = args["action"]
    if action == "list":
        return "\n".join(job.status() for job in mine.values()) or "No background jobs."
    job = mine.get(args.get("id") or "")
    if job is None:
        known = ", ".join(mine) or "none"
        raise ToolError(f"no background job {args.get('id')!r} in this chat (known: {known})")
    if action == "kill":
        job.kill("killed by request")
        job.done.wait(5)
        return job.status()
    if action == "wait":
        ctx.wait(args.get("timeout", DEFAULT_WAIT), until=job.done)
    if action == "log":
        return f"{job.status()}\n{_log_lines(job, args.get('offset'), args.get('limit', 200))}"
    with open(job.log, "rb") as f:  # poll and wait: the status and the output's tail
        f.seek(max(job.log.stat().st_size - TAIL_BYTES, 0))
        tail = f.read().decode("utf-8", "replace") or "(no output yet)"
    return f"{job.status()}\n--- output tail ---\n{tail}"


def _log_lines(job: Job, offset: int | None, limit: int) -> str:
    """Lines offset.. (1-based) of the log, or the last `limit` lines when offset is None."""
    lines = job.log.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(len(lines) - limit, 0) if offset is None else offset - 1
    shown = lines[start : start + limit]
    return "\n".join(shown + [f"[lines {start + 1}-{start + len(shown)} of {len(lines)}]"])


def _process_review(args: dict) -> str | None:
    """Killing a job is reviewed like a shell command; looking at one is not."""
    return "shell" if args.get("action") == "kill" else None


def register(api) -> None:
    max_timeout = api.app.config["agent"]["tool_timeout"]
    api.tool(
        "terminal",
        "Run a bash command (no PTY). `cd` persists in this session; exports do not. Long "
        "output is cut to head and tail, the rest saved to a file. background=true starts a "
        "job for the process tool; it survives /stop, /new ends it. Prefer read_file, "
        "search_files and patch over cat, grep, find and sed.",
        {
            "properties": {
                "command": {"type": "string", "pattern": r"\S"},
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_timeout,
                    "description": f"seconds (default {DEFAULT_TIMEOUT})",
                },
                "workdir": {"type": "string", "pattern": r"\S", "description": "this call only"},
                "background": {"type": "boolean"},
                "notify": {"type": "boolean", "description": "tell this chat when the job ends"},
            },
            "required": ["command"],
        },
        terminal,
        review="shell",
        timeout=max_timeout,
    )
    api.tool(
        "process",
        "This chat's background jobs: list, poll (status and output tail), log (lines of the "
        "output, offset/limit), wait (until it exits, up to timeout s) or kill.",
        {
            "properties": {
                "action": {"type": "string", "enum": ["list", "poll", "log", "wait", "kill"]},
                "id": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 1800},
                "offset": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
            },
            "required": ["action"],
        },
        process,
        review=_process_review,
        timeout=1860,
    )
    api.on_new_session(stop_jobs)  # /new: this conversation's jobs
    api.on_stop(stop_jobs)
