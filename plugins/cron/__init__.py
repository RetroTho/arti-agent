"""cron: scheduled jobs that run unattended and send their result where they were set.

A job is one entry in <data_dir>/cron/jobs.json (owner-readable): a schedule (schedule.py)
and a prompt (a fresh headless agent run; its final reply is the message), a script in
<data_dir>/scripts/ (its stdout is the message; empty sends nothing), or both (the script's
stdout is added to the prompt; a last line {"wakeAgent": false} skips the agent). Read top to
bottom: the job file, scripts, the Scheduler (a thread that runs due jobs) and the cronjob tool.

Mark before run: a due job's next_run_at, runs and running_since are saved before it does
anything, so a crash can lose a run but never repeat one; a run cut short is reported at the
next start, not re-run. A script runs only while its sha256 matches the one the owner approved.
A corrupt jobs.json stops the scheduler and is never overwritten.

A job records its conversation (`key`: the owner's chat, or the lane of a group it was set in;
the owner may say `private` to have a group job's results come to their own chat) and its
`creator`. Results go there and are recorded there as a note. Someone else's job runs like their
lane (guest tools, no private context) and is bounded: a prompt only, delivered to its group, at
most MAX_GUEST_JOBS active per person and group, recurring at most every GUEST_MIN_GAP. People
see and change only their own jobs; the owner sees and manages all.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from arti.fsutil import private_dir, read_json, write_json
from arti.tools import ToolError
from arti.turn import is_silent
from plugins.safety import approvals, proc

from . import schedule

REQUIRES = ("safety",)  # imports from it, and its gate reviews this plugin's tools

log = logging.getLogger(__name__)

TICK = 20  # seconds between schedule checks
MAX_RUNNING = 4
MAX_FAILURES = 3  # in a row, then the job is paused
RUN_TIMEOUT = 3600  # one headless agent run
DELIVER_TIMEOUT = 360  # longer than the App's own send timeout, so its status is the answer
SCRIPT_TIMEOUT = 300
SCRIPT_OUTPUT = 32_000  # chars
OUTPUTS_KEPT = 20  # per job
COMPLETED_KEPT = timedelta(days=7)
MIN_GRACE, MAX_GRACE = timedelta(minutes=2), timedelta(hours=2)
EDITABLE = ("name", "prompt", "skills", "tools", "model", "provider", "repeat", "deliver")
OWNER_FIELDS = ("script", "skills", "model", "provider", "deliver", "private")
MAX_GUEST_JOBS = 5  # active jobs per person per group
GUEST_MIN_GAP = timedelta(minutes=60)
SETTINGS = {"enabled": True}  # [cron]: false stops the scheduler and hides the tool


# --- the job file ------------------------------------------------------------------------------

_lock = threading.Lock()  # guards jobs.json


class CorruptJobs(Exception):
    """jobs.json cannot be read; nothing runs and nothing overwrites it until the owner fixes it."""


def jobs_path(app) -> Path:
    return Path(app.data_dir) / "cron" / "jobs.json"


def output_dir(app, job_id: str) -> Path:
    """Where a job's last OUTPUTS_KEPT messages are saved, one file per run."""
    return Path(app.data_dir) / "cron" / "output" / job_id


def load(app) -> list[dict]:
    try:
        data = read_json(jobs_path(app), {"jobs": []})
    except ValueError as e:
        raise CorruptJobs(str(e)) from None
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise CorruptJobs(f'{jobs_path(app)} must hold {{"jobs": [...]}}')
    return data["jobs"]


@contextlib.contextmanager
def edit(app):
    """Yield the job list under the lock and save it (atomically) when the block succeeds."""
    with _lock:
        jobs = load(app)
        yield jobs
        private_dir(jobs_path(app).parent)
        write_json(jobs_path(app), {"jobs": jobs})


def _find(jobs: list[dict], job_id: str) -> dict:
    for job in jobs:
        if job["id"] == job_id:
            return job
    raise ToolError(f"no cron job {job_id!r}; use action=list")


def where(app, job: dict) -> str:
    """The conversation a job's results go to."""
    return job.get("key") or app.home


# --- scripts -----------------------------------------------------------------------------------


class ScriptChanged(Exception):
    """The script no longer matches its approval: the job is paused at once."""


def scripts_dir(app) -> Path:
    return private_dir(Path(app.data_dir) / "scripts")


def script_path(app, name: str) -> Path:
    """The script `name`, which must be a file inside <data_dir>/scripts."""
    root = scripts_dir(app).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ToolError(f"script {name!r} must be an existing file in {root}")
    return path


def script_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_script(app, job: dict, cancel: threading.Event) -> str:
    """The script's stdout; an exception when it changed, fails, times out or is cancelled."""
    path = script_path(app, job["script"])
    if script_sha(path) != job["script_sha256"]:
        raise ScriptChanged(f"script {job['script']} changed since the owner approved it")
    runner = {".sh": ["/bin/bash"], ".bash": ["/bin/bash"], ".py": [sys.executable]}
    # proc.run's default env is the secret-free child one. Only stdout is sent, so stderr is
    # kept apart.
    result = proc.run(
        runner.get(path.suffix, []) + [str(path)],
        cwd=path.parent,
        timeout=SCRIPT_TIMEOUT,
        cancel=cancel,
        max_chars=SCRIPT_OUTPUT,
        merge_stderr=False,
    )
    if result.timed_out or result.cancelled:
        raise RuntimeError("script " + ("timed out" if result.timed_out else "cancelled"))
    if result.code != 0:
        raise RuntimeError(f"script exited with {result.code}: {(result.err or result.out)[-500:]}")
    return result.out


def _wake_agent(output: str) -> bool:
    """False when the script's last line is {"wakeAgent": false}."""
    try:
        last = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return True
    return not (isinstance(last, dict) and last.get("wakeAgent") is False)


# --- the scheduler -----------------------------------------------------------------------------


def grace(period: timedelta | None) -> timedelta | None:
    """How late a recurring slot may still run: half its period, within [2 min, 2 h]."""
    return None if period is None else min(max(period / 2, MIN_GRACE), MAX_GRACE)


class Scheduler:
    """A thread that claims due jobs every TICK seconds and runs them in a small pool. A run
    talks to the App (headless runs, sends) through the App's event loop."""

    def __init__(self, app):
        self.app = app
        self.stopping = threading.Event()
        self.running: set[str] = set()  # job ids with a run in flight
        self.futures: set = set()  # their coroutines on the App loop (stop cancels them)
        self.pool = ThreadPoolExecutor(MAX_RUNNING, thread_name_prefix="cron")
        self.first_tick = True  # the first one reports runs a restart cut short
        self.told_corrupt = False

    def start(self, app) -> None:
        # An on_start hook: App.start(serve=False), for --once, runs none, so no job fires there.
        threading.Thread(target=self._loop, name="cron-tick", daemon=True).start()

    def stop(self, app) -> None:
        """Stop ticking and cancel runs in flight; a run cut short is reported on next start."""
        self.stopping.set()
        for future in list(self.futures):
            future.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _loop(self) -> None:
        while not self.stopping.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("cron tick failed")
            self.stopping.wait(TICK)

    def now(self) -> datetime:
        return datetime.now(self.app.tz)

    def tick(self) -> None:
        """Claim every due job (saved before it runs) and hand it to the pool."""
        now = self.now()
        try:
            with edit(self.app) as jobs:
                interrupted = [j for j in jobs if j.get("running_since")] if self.first_tick else []
                self.first_tick = False
                for job in interrupted:
                    job.update(running_since=None, last_status="interrupted")
                    job["last_error"] = "Arti stopped while this ran; it was not re-run"
                jobs[:] = [job for job in jobs if not self._expired(job, now)]
                claimed, broken = [], []
                for job in jobs:
                    try:
                        if self._claim_if_due(job, now):
                            claimed.append(job)
                    except (ValueError, KeyError, TypeError) as e:  # one bad job stops only itself
                        job.update(enabled=False, last_status="error", last_error=str(e))
                        broken.append(job)
        except CorruptJobs as e:
            if not self.told_corrupt:
                self.told_corrupt = True
                log.error("cron: %s", e)
                self._tell(f"Cron is stopped: {e}. Fix or remove the file; no job runs until then.")
            return
        self.told_corrupt = False
        for job in interrupted:
            text = f"Cron job '{job['name']}' was interrupted by a restart and was not re-run."
            self._tell(text, where(self.app, job))
        for job in broken:
            reason = f"it could not be scheduled ({job['last_error']})"
            self._tell(f"Cron job '{job.get('name')}' is paused: {reason}.")
        for job in claimed:
            self._submit(job)

    def _expired(self, job: dict, now: datetime) -> bool:
        """A completed job is kept for a week (for `list`), then removed with its outputs."""
        done = job.get("completed_at")
        if done and now - datetime.fromisoformat(done) > COMPLETED_KEPT:
            shutil.rmtree(output_dir(self.app, job["id"]), ignore_errors=True)
            return True
        return False

    def _claim_if_due(self, job: dict, now: datetime) -> bool:
        """Advance a due job's schedule and mark it running; False if it is not to run now."""
        if not job["enabled"] or not job.get("next_run_at") or job["id"] in self.running:
            return False
        # In the zone, not the stored offset: a daily 9:00 stays 9:00 across a DST change.
        due = datetime.fromisoformat(job["next_run_at"]).astimezone(self.app.tz)
        if due > now:
            return False
        sched = job["sched"]
        following = schedule.next_run(sched, due)
        if following is not None and following <= now:
            following = schedule.next_run(sched, now)  # never schedule into the past
        job["next_run_at"] = following.isoformat() if following else None
        late = grace(schedule.period(sched, now))  # half the gap until the run after this one
        if late is not None and now - due > late:
            log.info("cron: skipped %s, due %s (missed by more than %s)", job["id"], due, late)
            job["last_status"] = "skipped"
            return False
        job.update(
            runs=job.get("runs", 0) + 1, running_since=now.isoformat(), due_at=due.isoformat()
        )
        if following is None or (job.get("repeat") and job["runs"] >= job["repeat"]):
            job.update(next_run_at=None, enabled=False, completed_at=now.isoformat())
        self.running.add(job["id"])
        return True

    def run_now(self, job_id: str, extra: str) -> None:
        """Run a job once now, outside its schedule."""
        with edit(self.app) as jobs:
            job = _find(jobs, job_id)
            if job_id in self.running:
                raise ToolError("that job is already running")
            job["running_since"] = self.now().isoformat()
            job.pop("due_at", None)  # not late: it was asked for now
            self.running.add(job_id)
        self._submit(dict(job), extra)

    def _submit(self, job: dict, extra: str = "") -> None:
        try:
            self.pool.submit(self._fire, job, extra)
        except RuntimeError:  # the pool is shut down: Arti is stopping
            self.running.discard(job["id"])

    def _fire(self, job: dict, extra: str) -> None:
        """Run one claimed job, deliver its message and record how it went."""
        error, pause = None, False
        try:
            text = self._produce(job, extra)
            self._save_output(job["id"], text)
            silent = not text or is_silent(text)
            if job["deliver"] != "local" and not silent:
                deliver = self.app.deliver(where(self.app, job), text)
                if self._call(deliver, DELIVER_TIMEOUT) == "failed":
                    raise RuntimeError("the message could not be sent")
        except Exception as e:
            if self.stopping.is_set():  # cut short by shutdown: reported as interrupted later
                self.running.discard(job["id"])
                return
            log.warning("cron job %s failed: %s", job["id"], e)
            pause = isinstance(e, ScriptChanged)
            error = str(e) if pause else f"{type(e).__name__}: {e}"
        self._finish(job["id"], error, pause)

    def _produce(self, job: dict, extra: str) -> str:
        """The job's message: script output, the agent's reply, or "" for nothing."""
        script = job.get("script")
        output = _run_script(self.app, job, self.stopping) if script else ""
        if not job.get("prompt"):
            return output.strip()
        if script and not _wake_agent(output):
            return ""
        sent = "saved, not sent" if job["deliver"] == "local" else "sent as a text message"
        prompt = f"[Arti: scheduled job '{job['name']}' ({job['id']}); your final reply is {sent}. "
        prompt += "Reply [SILENT] to send nothing."
        due = job.get("due_at")
        if due and self.now() - datetime.fromisoformat(due) > MIN_GRACE:
            prompt += f" It was due at {due} and runs late (Arti was asleep or stopped)."
        prompt += "]\n\n" + job["prompt"]
        if script:
            prompt += f"\n\nOutput of the job's script {script}:\n{output}"
        if extra:
            prompt += f"\n\nExtra context for this run:\n{extra}"
        trusted = self.app.is_owner(job.get("creator") or "")  # the owner's job runs trusted
        skills = (job.get("skills") or []) if trusted else []  # skills are the owner's
        preload = self.app.services.get("preload_skills")  # the skills plugin's, if it is loaded
        run = self.app.run_headless(
            prompt,
            scope=self.app.scope_for(where(self.app, job)),
            trusted=trusted,
            instructions=preload(self.app, skills) if preload and skills else "",
            tools=job.get("tools"),
            provider=job.get("provider"),
            model=job.get("model"),
            timeout=RUN_TIMEOUT,
        )
        return self._call(run, RUN_TIMEOUT + 60).strip()

    def _finish(self, job_id: str, error: str | None, pause: bool) -> None:
        """Record a run's outcome; tell the job's conversation on a first failure and when
        pausing."""
        message, key = "", None
        try:
            with edit(self.app) as jobs:
                job = next((j for j in jobs if j["id"] == job_id), {"name": job_id})
                key = where(self.app, job)
                failures = job.get("failures", 0) + 1 if error else 0
                job.update(running_since=None, last_run_at=self.now().isoformat())
                job.update(last_status="failed" if error else "ok", last_error=error)
                job["failures"] = failures
                if error and (pause or failures >= MAX_FAILURES):
                    job["enabled"] = False
                    message = f"Cron job '{job['name']}' is paused: {error}. Resume it when fixed."
                elif error and failures == 1:
                    message = f"Cron job '{job['name']}' failed: {error}. It pauses after "
                    message += f"{MAX_FAILURES} failures in a row."
        except CorruptJobs as e:  # the next tick tells the owner
            log.error("cron: could not record the run of %s: %s", job_id, e)
        finally:
            self.running.discard(job_id)
        if message:
            self._tell(message, key)

    def _save_output(self, job_id: str, text: str) -> None:
        folder = private_dir(output_dir(self.app, job_id))
        (folder / f"{self.now():%Y%m%d-%H%M%S}.txt").write_text(text, encoding="utf-8")
        for old in sorted(folder.glob("*.txt"))[:-OUTPUTS_KEPT]:
            old.unlink(missing_ok=True)

    def _call(self, coro, timeout: float):
        """Run `coro` on the App loop and wait; cancelled by a timeout or stop()."""
        future = asyncio.run_coroutine_threadsafe(coro, self.app.loop)
        self.futures.add(future)
        try:
            return future.result(timeout)
        finally:
            future.cancel()  # a no-op once it is done
            self.futures.discard(future)

    def _tell(self, text: str, key: str | None = None) -> None:
        """Text the owner, or a job's conversation (best effort: a failure is only logged)."""
        try:
            deliver = self.app.deliver(key or self.app.home, f"[cron] {text}")
            self._call(deliver, DELIVER_TIMEOUT)
        except Exception as e:
            log.warning("cron: could not tell the owner %r: %s", text, e)


# --- the cronjob tool --------------------------------------------------------------------------


def _approve_script(ctx, job: dict, name: str) -> dict:
    """Show the owner the script's full source; the fields binding the job to its sha256."""
    path = script_path(ctx.app, name)
    source = path.read_bytes()  # the sha256 is of exactly what the owner saw
    detail = (
        f"Cron job '{job['name']}' ({job['schedule']}) wants to run {path} unattended. "
        f"Helper files it imports are not covered.\n\n{source.decode('utf-8', 'replace')}"
    )
    if not approvals.ask_owner(ctx, "Cron script", detail):
        raise ToolError("the owner did not approve the script; nothing was changed")
    return {"script": name, "script_sha256": hashlib.sha256(source).hexdigest()}


def _guest_check(ctx, new: dict, now: datetime) -> None:
    """Someone else's job: a prompt only, delivered to its group, not too often or too many."""
    if not new.get("prompt"):
        raise ToolError("a job here needs a prompt")
    gap = schedule.shortest_gap(new["sched"], now)
    if gap is not None and gap < GUEST_MIN_GAP:
        raise ToolError(f"a repeating job here may run at most every {GUEST_MIN_GAP}")
    others = [j for j in load(ctx.app) if j.get("key") == ctx.key and j["enabled"]]
    others = [j for j in others if j["id"] != new["id"]]
    if new["enabled"] and len(others) >= MAX_GUEST_JOBS:
        raise ToolError(f"you already have {MAX_GUEST_JOBS} active jobs here; remove one first")


def _changes(ctx, job: dict, args: dict, now: datetime) -> dict:
    """The fields create/update sets on `job`, checked against the job as it would become."""
    if not ctx.trusted and (fields := [k for k in OWNER_FIELDS if k in args]):
        raise ToolError(f"only {ctx.app.owner_name} can set {', '.join(fields)}")
    changes = {key: args[key] or None for key in EDITABLE if key in args}
    if "schedule" in args:
        try:
            sched = schedule.parse(args["schedule"], now)
            # A one-shot a few minutes in the past (a model computing "now") runs at once.
            once = sched["kind"] == "once"
            first = schedule.next_run(sched, now - timedelta(minutes=10) if once else now)
        except ValueError as e:
            raise ToolError(str(e)) from None
        if first is None:
            raise ToolError(f"{args['schedule']!r} is in the past")
        changes.update(schedule=args["schedule"], sched=sched, next_run_at=first.isoformat())
    new = {**job, **changes}
    name = new.get("name") or (new.get("prompt") or args.get("script") or "")[:40]
    new["name"] = changes["name"] = name
    changes["deliver"] = new.get("deliver") or "chat"
    if unknown := [t for t in new.get("tools") or [] if t not in ctx.app.tools]:
        raise ToolError(f"unknown tools: {', '.join(unknown)}")
    if not new.get("prompt") and not new.get("script") and not args.get("script"):
        raise ToolError("a job needs a prompt, a script, or both")
    if not ctx.trusted:
        _guest_check(ctx, new, now)
    if args.get("script"):
        changes.update(_approve_script(ctx, new, args["script"]))
    return changes


def _resume(ctx, job: dict, now: datetime) -> dict:
    changes = {"enabled": True, "failures": 0, "completed_at": None}
    if not ctx.trusted:
        _guest_check(ctx, {**job, **changes}, now)
    script = job.get("script")
    if script and script_sha(script_path(ctx.app, script)) != job["script_sha256"]:
        changes.update(_approve_script(ctx, job, script))
    # A one-shot whose time has passed runs now.
    changes["next_run_at"] = (schedule.next_run(job["sched"], now) or now).isoformat()
    return changes


def _describe(job: dict) -> str:
    state = "on" if job["enabled"] else ("done" if job.get("completed_at") else "paused")
    kind = "+".join(k for k in ("script", "prompt") if job.get(k))
    return (
        f"{job['id']} '{job['name']}' [{state}] {kind} schedule={job['schedule']!r} "
        f"skills={job.get('skills') or '-'} next={job.get('next_run_at') or '-'} "
        f"last={job.get('last_run_at') or '-'} ({job.get('last_status') or 'never run'}) "
        f"error={(job.get('last_error') or '-')[:200]} output=cron/output/{job['id']}/"
    )


def cronjob(args: dict, ctx) -> str:
    try:
        return _cronjob(args, ctx)
    except CorruptJobs as e:
        raise ToolError(f"cron is stopped: {e}") from None


def _cronjob(args: dict, ctx) -> str:
    app, action, job_id = ctx.app, args["action"], args.get("job_id", "")
    now = datetime.now(app.tz)
    jobs = [j for j in load(app) if ctx.trusted or j.get("key") == ctx.key]
    if action == "list":
        return "\n".join(_describe(j) for j in jobs) or "No cron jobs."
    if job_id:
        _find(jobs, job_id)  # someone else's lane: only its own jobs
    if action == "run":
        if not ctx.trusted:
            raise ToolError(f"only {app.owner_name} can run a job outside its schedule")
        app.state["cron"].run_now(job_id, args.get("context", ""))
        return f"Started job {job_id}; its result arrives as a message when it finishes."
    if action == "create":
        if "schedule" not in args:
            raise ToolError("create needs a schedule")
        if not ctx.trusted and not ctx.conv.person:  # e.g. someone else's job's own run
            raise ToolError("a background run for someone else can't create jobs")
        job = dict.fromkeys((*EDITABLE, "script", "script_sha256"))  # all unset
        job.update(id=uuid.uuid4().hex[:8], enabled=True, runs=0, failures=0)
        job["created_at"] = now.isoformat()
        job["key"] = app.home if args.get("private") else ctx.key or app.home
        job["creator"] = app.owner if ctx.trusted else ctx.conv.person
        job.update(_changes(ctx, job, args, now))  # may ask the owner: not under the lock
        with edit(app) as jobs:
            jobs.append(job)
        return "Created.\n" + _describe(job)
    if action == "remove":
        with edit(app) as jobs:
            jobs.remove(_find(jobs, job_id))
        shutil.rmtree(output_dir(app, job_id), ignore_errors=True)
        return f"Removed job {job_id}."
    current = _find(load(app), job_id)
    if action == "update":
        changes = _changes(ctx, current, args, now)
    elif action == "resume":
        changes = _resume(ctx, current, now)
    else:  # pause
        changes = {"enabled": False}
    # Save only what this call changed: the scheduler may have run the job meanwhile (an owner
    # approval can take minutes), and its runs, next_run_at and status must stand.
    with edit(app) as jobs:
        job = _find(jobs, job_id)
        job.update(changes)
    return "OK.\n" + _describe(job)


ACTIONS = ["create", "list", "update", "pause", "resume", "remove", "run"]
SCHEDULES = "'in 30m', 'once at 2026-10-24T09:00', 'every 2h' or cron '10 7 * * *' (agent.timezone)"
STRING = {"type": "string"}
SCHEMA = {
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "job_id": STRING,
        "schedule": {"type": "string", "description": SCHEDULES},
        "prompt": {"type": "string", "description": "what the agent should do each run"},
        "script": {"type": "string", "description": "a file in <data_dir>/scripts/"},
        "name": STRING,
        "skills": {"type": "array", "items": STRING, "maxItems": 10},
        "tools": {"type": "array", "items": STRING, "description": "limit the run's tools"},
        "model": STRING,
        "provider": STRING,
        "repeat": {"type": "integer", "minimum": 1, "description": "runs, then done"},
        "deliver": {"type": "string", "enum": ["chat", "local"]},
        "private": {"type": "boolean", "description": "create: results go to the owner's chat"},
        "context": {"type": "string", "description": "run: extra context for this run"},
    },
    "required": ["action"],
}


def register(api) -> None:
    settings = api.settings("cron", SETTINGS)
    scheduler = api.app.state["cron"] = Scheduler(api.app)
    api.tool(
        "cronjob",
        "Schedule work (a reminder, a regular check) that runs unattended and texts its result "
        "to this conversation; private=true sends it to the owner's own chat instead. A prompt "
        "job runs a fresh agent (no memory of this chat) whose final reply is sent; a script job "
        "sends the script's stdout (empty = nothing); with both, the script's output is added to "
        'the prompt and a last line {"wakeAgent": false} skips the agent. Scripts need the '
        "owner's approval of their exact contents. deliver='local' only saves the output. A "
        "reply starting with [SILENT] sends nothing. For anyone but the owner: prompt jobs only, "
        f"at most {MAX_GUEST_JOBS} active, repeating at most hourly, and only their own jobs.",
        SCHEMA,
        cronjob,
        review=lambda args: None if args["action"] in ("list", "pause", "remove") else "prompt",
        guest=True,
        available=lambda app: settings["enabled"],
    )
    if settings["enabled"]:
        api.on_start(scheduler.start)
        api.on_stop(scheduler.stop)
