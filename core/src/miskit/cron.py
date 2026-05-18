import asyncio
import json
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from miskit.message import log_message_to_dict


@dataclass
class CronJob:
    id: str
    name: str
    message: str
    at: datetime
    every_seconds: int | None = None

    @classmethod
    def from_data(cls, data):
        return cls(
            data["id"],
            data["name"],
            data["message"],
            parse_at(data["at"]),
            parse_every_seconds(data.get("everySeconds")),
        )

    def data(self):
        data = {
            "id": self.id,
            "name": self.name,
            "message": self.message,
            "at": self.at.isoformat(),
        }
        if self.every_seconds is not None:
            data["everySeconds"] = self.every_seconds
        return data


def parse_at(text):
    try:
        at = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError("at must be an ISO datetime, like 2026-05-01T15:30:00-07:00") from error

    if at.tzinfo is None:
        at = at.astimezone()

    return at


def parse_every_seconds(value):
    if value is None or value == "":
        return None

    try:
        every_seconds = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("everySeconds must be a positive integer") from error

    if every_seconds <= 0:
        raise ValueError("everySeconds must be a positive integer")

    return every_seconds


class CronService:
    def __init__(self, path, on_job=None):
        self.path = Path(path)
        self.on_job = on_job
        self._task = None
        self._changed = None
        self._running = False

    def setup(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save([])

    def add_job(self, name, at, message):
        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name or message[:30] or "scheduled task",
            message=message,
            at=parse_at(at),
        )

        jobs = self.list_jobs()
        jobs.append(job)
        self._save(jobs)
        self._wake()
        return job

    def add_recurring_job(self, name, every_seconds, message, at=None, job_id=None):
        every_seconds = parse_every_seconds(every_seconds)
        if every_seconds is None:
            raise ValueError("everySeconds must be a positive integer")

        if at is None:
            at = datetime.now().astimezone() + timedelta(seconds=every_seconds)
        else:
            at = parse_at(at)

        job = CronJob(
            id=job_id or str(uuid.uuid4())[:8],
            name=name or message[:30] or "scheduled task",
            message=message,
            at=at,
            every_seconds=every_seconds,
        )

        jobs = [old_job for old_job in self.list_jobs() if old_job.id != job.id]
        jobs.append(job)
        self._save(jobs)
        self._wake()
        return job

    def list_jobs(self):
        self.setup()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"jobs": []}

        jobs = []
        for item in data.get("jobs", []):
            jobs.append(CronJob.from_data(item))

        return sorted(jobs, key=lambda job: job.at)

    def remove_job(self, job_id):
        jobs = self.list_jobs()
        kept = [job for job in jobs if job.id != job_id]
        removed = len(kept) != len(jobs)

        if removed:
            self._save(kept)
            self._wake()

        return removed

    async def start(self):
        self.setup()
        self._running = True
        self._changed = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        self._wake()

        task = self._task
        if task is None:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._changed = None

    async def _run(self):
        while self._running:
            wait = self._seconds_until_next_job()
            if wait is None:
                await self._wait_for_change()
            elif wait > 0:
                await self._wait_for_change(timeout=wait)
            else:
                await self._run_due_jobs()

    def _seconds_until_next_job(self):
        jobs = self.list_jobs()
        if not jobs:
            return None

        return jobs[0].at.timestamp() - datetime.now().astimezone().timestamp()

    async def _wait_for_change(self, timeout=None):
        if self._changed is None:
            return

        self._changed.clear()
        try:
            await asyncio.wait_for(self._changed.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    async def _run_due_jobs(self):
        now = datetime.now().astimezone()
        due = [job for job in self.list_jobs() if job.at <= now]

        for job in due:
            # Remove one-time jobs or unschedule recurring jobs before running,
            # then reschedule recurring jobs after. This way a crash during
            # execution doesn't leave a job stuck in the past firing repeatedly.
            jobs = [j for j in self.list_jobs() if j.id != job.id]
            self._save(jobs)

            try:
                if self.on_job is not None:
                    await self.on_job(job)
            except Exception:
                pass

            if job.every_seconds is not None:
                job.at = datetime.now().astimezone() + timedelta(seconds=job.every_seconds)
                jobs = [j for j in self.list_jobs() if j.id != job.id]
                jobs.append(job)
                self._save(jobs)

        self._wake()

    def _save(self, jobs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "jobs": [job.data() for job in jobs],
        }
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def _wake(self):
        if self._changed is not None:
            self._changed.set()


class CronLog:
    def __init__(self, path):
        self.path = Path(path)

    def setup(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def log(self, timestamp, job_id, job_name, messages):
        self.setup()
        entry = {
            "timestamp": timestamp,
            "job_id": job_id,
            "job_name": job_name,
            "messages": [log_message_to_dict(message) for message in messages],
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")

    def read(self, limit=10):
        self.setup()
        recent = deque(maxlen=limit)
        with self.path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    recent.append(json.loads(line))
        return list(recent)

    def read_by_job_id(self, job_id):
        self.setup()
        result = None
        with self.path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("job_id") == job_id:
                        result = entry
        return result
