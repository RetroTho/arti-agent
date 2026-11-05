"""The cron scheduler and the cronjob tool, with a fake App and an inline thread pool."""

import asyncio
import json
import logging
import tempfile
import threading
import time
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from arti import app as app_module
from arti.plugin import Api
from arti.tools import ToolError
from plugins import cron as plugin
from plugins.safety import approvals
from tests.support import GROUP, HOME, OWNER, RecordingChannel, make_app, scope, text_reply

TZ = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 10, 23, 8, 0, tzinfo=TZ)


RILEY = f"{GROUP}|U2"  # someone else's lane of the group


class FakeApp:
    def __init__(self, data_dir: Path):
        self.data_dir, self.tz, self.owner_name = data_dir, TZ, "Sam"
        self.home, self.owner = HOME, OWNER
        self.channel = RecordingChannel(HOME, OWNER)  # keys like the link's
        self.config = {"cron": {"enabled": True}}
        self.tools = {"terminal": object()}
        self.plugin_sections = {}
        self.state = {}
        self.services = {
            "preload_skills": lambda app, names: "# Skill: brief\nDo it well."
            if "brief" in names
            else ""
        }
        self.sent, self.runs, self.places = [], [], []
        self.reply = "the report"
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()

    def is_owner(self, handle):
        return handle == OWNER

    def scope_for(self, key):
        return self.channel.scope_of(key)

    async def deliver(self, key, text):
        self.sent.append(text)
        self.places.append(key)
        return "sent"

    async def run_headless(self, prompt, **options):
        self.runs.append((prompt, options))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class InlinePool:
    """Runs each submitted job at once, after recording what jobs.json held at that moment."""

    def __init__(self, app):
        self.app, self.seen = app, []

    def submit(self, fn, *args):
        self.seen.append(plugin.load(self.app))
        fn(*args)

    def shutdown(self, **kwargs):
        pass


def make_job(**fields):
    job = {
        "id": "j1",
        "name": "briefing",
        "schedule": "10 7 * * *",
        "sched": {"kind": "cron", "expr": "10 7 * * *"},
        "prompt": "brief me",
        "script": None,
        "script_sha256": None,
        "skills": None,
        "tools": None,
        "model": None,
        "provider": None,
        "deliver": "chat",
        "repeat": None,
        "runs": 0,
        "enabled": True,
        "failures": 0,
        "key": HOME,
        "creator": OWNER,
        "next_run_at": (NOW - timedelta(minutes=1)).isoformat(),
        "running_since": None,
    }
    job.update(fields)
    return job


class CronTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.app = FakeApp(self.dir)
        self.addCleanup(self.app.close)
        self.scheduler = plugin.Scheduler(self.app)
        self.scheduler.pool = self.pool = InlinePool(self.app)
        self.scheduler.now = lambda: self.now
        self.now = NOW
        logging.disable(logging.CRITICAL)  # failures are logged on purpose here
        self.addCleanup(logging.disable, logging.NOTSET)

    def write_jobs(self, *jobs):
        (self.dir / "cron").mkdir(exist_ok=True)
        (self.dir / "cron" / "jobs.json").write_text(json.dumps({"jobs": list(jobs)}))

    def job(self, job_id="j1"):
        return next(j for j in plugin.load(self.app) if j["id"] == job_id)

    def write_script(self, name, text):
        path = plugin.scripts_dir(self.app) / name
        path.write_text(text)
        return plugin.script_sha(path)


class SchedulerTest(CronTestCase):
    def test_marks_before_running_and_delivers(self):
        self.write_jobs(make_job())
        self.scheduler.tick()
        at_fire = self.pool.seen[0][0]
        self.assertIsNotNone(at_fire["running_since"])
        self.assertEqual(at_fire["next_run_at"], "2026-10-24T07:10:00-07:00")
        self.assertEqual(at_fire["runs"], 1)
        self.assertEqual(self.app.sent, ["the report"])
        prompt, options = self.app.runs[0]
        self.assertIn("brief me", prompt)
        self.assertEqual((options["scope"], options["trusted"]), ("home", True))
        self.assertEqual(self.app.places, [HOME])

    def test_the_owners_job_preloads_its_skills(self):
        self.write_jobs(make_job(skills=["brief", "nope"]))
        self.scheduler.tick()
        _, options = self.app.runs[0]
        self.assertEqual(
            options["instructions"], "# Skill: brief\nDo it well."
        )  # (the skills plugin's text)

    def test_someone_elses_job_runs_as_their_lane_and_delivers_there(self):
        self.write_jobs(make_job(key=RILEY, creator="U2", skills=["x"]))
        self.scheduler.tick()
        _, options = self.app.runs[0]
        self.assertEqual(options["scope"], scope(GROUP))
        self.assertEqual((options["trusted"], options["instructions"]), (False, ""))
        self.assertEqual(self.app.places, [RILEY])
        job = self.job()
        self.assertIsNone(job["running_since"])
        self.assertEqual(job["last_status"], "ok")
        self.assertEqual(len(list((self.dir / "cron" / "output" / "j1").iterdir())), 1)

    def test_a_daily_job_keeps_its_local_time_across_dst(self):
        # Stored as 09:00-07:00 (PDT); after fall back, the next fire is 09:00 PST, not 08:00.
        self.now = datetime(2026, 10, 31, 9, 0, 30, tzinfo=TZ)
        self.write_jobs(
            make_job(
                sched={"kind": "cron", "expr": "0 9 * * *"}, next_run_at="2026-10-31T09:00:00-07:00"
            )
        )
        self.scheduler.tick()
        self.assertEqual(self.job()["next_run_at"], "2026-11-01T09:00:00-08:00")

    def test_a_job_that_cannot_be_scheduled_is_paused_alone(self):
        self.write_jobs(make_job(id="bad", sched={"kind": "cron", "expr": "nonsense"}), make_job())
        self.scheduler.tick()
        self.assertFalse(self.job("bad")["enabled"])
        self.assertEqual(len(self.app.runs), 1)  # the good one still ran
        self.assertIn("is paused", self.app.sent[0])

    def test_grace_is_half_the_period_clamped(self):
        self.assertEqual(plugin.grace(timedelta(days=1)), timedelta(hours=2))
        self.assertEqual(plugin.grace(timedelta(minutes=1)), timedelta(minutes=2))
        self.assertEqual(plugin.grace(timedelta(minutes=30)), timedelta(minutes=15))
        # The 07:10 briefing still runs when the Mac wakes at 08:00 ...
        self.write_jobs(make_job(next_run_at="2026-10-23T07:10:00-07:00"))
        self.scheduler.tick()
        self.assertEqual(len(self.app.runs), 1)
        # ... but not at 10:00: that slot is skipped to the next one.
        self.write_jobs(make_job(next_run_at="2026-10-23T07:10:00-07:00"))
        self.now = NOW.replace(hour=10)
        self.scheduler.tick()
        self.assertEqual(len(self.app.runs), 1)
        self.assertEqual(self.job()["next_run_at"], "2026-10-24T07:10:00-07:00")

    def test_one_shot_runs_late_once_and_completes(self):
        at = (NOW - timedelta(days=2)).isoformat()
        self.write_jobs(make_job(sched={"kind": "once", "at": at}, next_run_at=at))
        self.scheduler.tick()
        self.scheduler.tick()
        self.assertEqual(len(self.app.runs), 1)
        self.assertIn("runs late", self.app.runs[0][0])
        job = self.job()
        self.assertFalse(job["enabled"])
        self.assertIsNotNone(job["completed_at"])

    def test_repeat_completes_the_job(self):
        self.write_jobs(make_job(repeat=2, runs=1))
        self.scheduler.tick()
        self.assertFalse(self.job()["enabled"])

    def test_interrupted_run_is_reported_not_rerun(self):
        self.write_jobs(
            make_job(running_since=NOW.isoformat(), next_run_at="2027-01-01T00:00:00-08:00")
        )
        self.scheduler.tick()
        self.assertEqual(self.app.runs, [])
        self.assertEqual(self.job()["last_status"], "interrupted")
        self.assertIn("interrupted", self.app.sent[0])

    def test_three_failures_pause_with_texts(self):
        self.app.reply = RuntimeError("model down")
        self.write_jobs(make_job(sched={"kind": "every", "minutes": 5}))
        for minute in range(3):
            self.now = NOW + timedelta(minutes=5 * minute)
            self.scheduler.tick()
        job = self.job()
        self.assertEqual(job["failures"], 3)
        self.assertFalse(job["enabled"])
        self.assertEqual(len(self.app.sent), 2)  # first failure, then the pause
        self.assertIn("failed", self.app.sent[0])
        self.assertIn("paused", self.app.sent[1])

    def test_silent_and_local_are_not_sent(self):
        self.write_jobs(make_job(), make_job(id="j2", deliver="local"))
        self.app.reply = "[SILENT]"
        self.scheduler.tick()
        self.assertEqual(self.app.sent, [])
        self.assertEqual(len(self.app.runs), 2)

    def test_silent_prefix_and_slow_confirmations(self):
        self.write_jobs(make_job())
        self.app.reply = "[SILENT] nothing new today"
        self.scheduler.tick()
        self.assertEqual(self.app.sent, [])
        # The App's own send timeout decides "sent"/"unknown"; cron must not give up first.
        self.assertGreater(plugin.DELIVER_TIMEOUT, app_module.SEND_TIMEOUT)

    def test_stop_cancels_a_run_in_flight(self):
        started = threading.Event()

        async def slow(prompt, **options):
            started.set()
            await asyncio.sleep(30)

        self.app.run_headless = slow
        self.write_jobs(make_job())
        self.scheduler.pool = plugin.ThreadPoolExecutor(1)
        self.scheduler.tick()
        self.assertTrue(started.wait(5))
        begun = time.monotonic()
        self.scheduler.stop(self.app)
        self.scheduler.pool.shutdown(wait=True)
        self.assertLess(time.monotonic() - begun, 5)
        self.assertEqual(self.app.sent, [])
        self.assertIsNotNone(self.job()["running_since"])  # reported as interrupted next start

    def test_corrupt_file_runs_nothing_and_is_kept(self):
        (self.dir / "cron").mkdir()
        (self.dir / "cron" / "jobs.json").write_text("{oops")
        self.scheduler.tick()
        self.scheduler.tick()
        self.assertEqual((self.dir / "cron" / "jobs.json").read_text(), "{oops")
        self.assertEqual(len(self.app.sent), 1)
        self.assertIn("Cron is stopped", self.app.sent[0])


class ScriptTest(CronTestCase):
    def test_script_only_sends_stdout_and_empty_sends_nothing(self):
        sha = self.write_script("score.sh", "echo 'Score: 2-1'\n")
        empty = self.write_script("quiet.sh", "true\n")
        self.write_jobs(
            make_job(prompt=None, script="score.sh", script_sha256=sha),
            make_job(id="j2", prompt=None, script="quiet.sh", script_sha256=empty),
        )
        self.scheduler.tick()
        self.assertEqual(self.app.sent, ["Score: 2-1"])
        self.assertEqual(self.app.runs, [])

    def test_script_output_feeds_prompt_and_can_skip_agent(self):
        sha = self.write_script("pre.sh", "echo data-123\n")
        self.write_jobs(make_job(script="pre.sh", script_sha256=sha))
        self.scheduler.tick()
        self.assertIn("data-123", self.app.runs[0][0])
        sha = self.write_script("skip.sh", "echo '{\"wakeAgent\": false}'\n")
        self.write_jobs(make_job(id="j3", script="skip.sh", script_sha256=sha))
        self.scheduler.tick()
        self.assertEqual(len(self.app.runs), 1)

    def test_stderr_is_not_sent(self):
        sha = self.write_script("noisy.sh", "echo 'Score: 3-1'; echo 'warning: slow' >&2\n")
        self.write_jobs(make_job(prompt=None, script="noisy.sh", script_sha256=sha))
        self.scheduler.tick()
        self.assertEqual(self.app.sent, ["Score: 3-1"])

    def test_changed_script_pauses_job(self):
        sha = self.write_script("w.sh", "echo approved\n")
        self.write_script("w.sh", "echo swapped\n")
        self.write_jobs(make_job(prompt=None, script="w.sh", script_sha256=sha))
        self.scheduler.tick()
        job = self.job()
        self.assertFalse(job["enabled"])
        self.assertIn("changed", job["last_error"])
        self.assertNotIn("swapped", " ".join(self.app.sent))

    def test_failing_script_counts_as_failure(self):
        sha = self.write_script("bad.sh", "exit 3\n")
        self.write_jobs(make_job(prompt=None, script="bad.sh", script_sha256=sha))
        self.scheduler.tick()
        self.assertEqual(self.job()["failures"], 1)


class ToolTest(CronTestCase):
    def setUp(self):
        super().setUp()
        self.app.state["cron"] = self.scheduler
        self.ctx = self.context(HOME)
        self.approved = []
        patcher = mock.patch.object(approvals, "ask_owner", lambda *a: self.ask_owner(*a))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.answer = True

    def context(self, key):
        person = self.app.channel.person_of(key)
        return types.SimpleNamespace(
            app=self.app,
            key=key,
            scope=self.app.scope_for(key),
            trusted=not person or person == OWNER,
            conv=types.SimpleNamespace(person=person),
        )

    def ask_owner(self, ctx, title, detail):
        self.approved.append(detail)
        return self.answer

    def call(self, **args):
        return plugin.cronjob(args, self.ctx)

    def test_a_job_goes_back_where_it_was_set(self):
        self.ctx = self.context(f"{GROUP}|{OWNER}")
        self.call(action="create", schedule="in 5m", prompt="remind the group", name="here")
        self.call(action="create", schedule="in 5m", prompt="remind me", name="me", private=True)
        jobs = {j["name"]: j for j in plugin.load(self.app)}
        self.assertEqual(jobs["here"]["key"], f"{GROUP}|{OWNER}")
        self.assertEqual(jobs["me"]["key"], HOME)
        self.assertEqual(jobs["here"]["creator"], OWNER)

    def test_someone_elses_background_run_cannot_create_jobs(self):
        self.ctx = types.SimpleNamespace(
            app=self.app,
            key="",
            scope="home",
            trusted=False,
            conv=types.SimpleNamespace(person=None),
        )
        with self.assertRaisesRegex(ToolError, "can't create"):
            self.call(action="create", schedule="in 5m", prompt="x")
        self.assertEqual(plugin.load(self.app), [])

    def test_someone_elses_jobs_are_bounded_and_their_own(self):
        self.write_jobs(make_job(id="mine", name="owners"))
        self.ctx = self.context(RILEY)
        for field in ({"script": "w.py"}, {"private": True}, {"model": "x"}):
            with self.assertRaises(ToolError, msg=field):
                self.call(action="create", schedule="in 5m", prompt="x", **field)
        with self.assertRaisesRegex(ToolError, "every 1:00:00"):
            self.call(action="create", schedule="every 30m", prompt="x")
        for n in range(plugin.MAX_GUEST_JOBS):
            self.call(action="create", schedule="every 2h", prompt="x", name=f"riley{n}")
        with self.assertRaisesRegex(ToolError, "already have 5"):
            self.call(action="create", schedule="in 5m", prompt="x")
        listing = self.call(action="list")
        self.assertIn("riley0", listing)
        self.assertNotIn("owners", listing)  # the owner's jobs are not even listed
        for action in ("remove", "pause", "run"):
            with self.assertRaises(ToolError):
                self.call(action=action, job_id="mine")
        self.assertEqual(self.job("mine")["name"], "owners")
        self.assertEqual(plugin.load(self.app)[1]["key"], RILEY)
        self.ctx = self.context(HOME)  # the owner sees and manages all
        self.assertIn("riley0", self.call(action="list"))
        self.call(action="remove", job_id=plugin.load(self.app)[1]["id"])

    def test_create_list_pause_resume_remove(self):
        out = self.call(action="create", schedule="10 7 * * *", prompt="brief me", name="Brief")
        job_id = out.split("\n")[1].split()[0]
        self.assertIn("Brief", self.call(action="list"))
        self.call(action="pause", job_id=job_id)
        self.assertFalse(self.job(job_id)["enabled"])
        self.call(action="resume", job_id=job_id)
        self.assertTrue(self.job(job_id)["enabled"])
        self.call(action="remove", job_id=job_id)
        self.assertEqual(self.call(action="list"), "No cron jobs.")

    def test_script_needs_owner_approval_of_its_source(self):
        self.write_script("w.py", "print('hello')\n")
        self.answer = False
        with self.assertRaises(ToolError):
            self.call(action="create", schedule="every 5m", script="w.py")
        self.assertEqual(plugin.load(self.app), [])
        self.assertIn("print('hello')", self.approved[0])
        self.answer = True
        self.call(action="create", schedule="every 5m", script="w.py")
        self.assertEqual(len(plugin.load(self.app)[0]["script_sha256"]), 64)

    def test_script_must_stay_in_scripts_dir(self):
        (self.dir / "evil.sh").write_text("echo hi")
        with self.assertRaises(ToolError):
            self.call(action="create", schedule="every 5m", script="../evil.sh")

    def test_bad_requests(self):
        with self.assertRaises(ToolError):
            self.call(action="create", schedule="whenever", prompt="x")
        with self.assertRaises(ToolError):
            self.call(action="create", schedule="2020-01-01T00:00", prompt="x")
        with self.assertRaises(ToolError):
            self.call(action="create", schedule="in 5m")
        with self.assertRaises(ToolError):
            self.call(action="create", schedule="in 5m", prompt="x", tools=["nope"])
        with self.assertRaises(ToolError):
            self.call(action="pause", job_id="missing")

    def test_update_keeps_what_the_scheduler_did_meanwhile(self):
        sha = self.write_script("w.sh", "echo hi\n")
        self.write_jobs(make_job(repeat=1, script="w.sh", script_sha256=sha))

        def ask_owner(ctx, title, detail):  # the job fires while the owner reads the script
            self.scheduler.tick()
            return True

        self.ask_owner = ask_owner
        self.call(action="update", job_id="j1", script="w.sh", prompt="new prompt")
        job = self.job()
        self.assertEqual(job["prompt"], "new prompt")
        self.assertEqual(job["runs"], 1)
        self.assertFalse(job["enabled"])  # the one allowed run happened; it stays done
        self.assertIsNone(job["next_run_at"])

    def test_run_now_runs_without_moving_the_schedule(self):
        self.write_jobs(make_job(next_run_at="2027-01-01T00:00:00-08:00"))
        self.call(action="run", job_id="j1", context="extra bit")
        self.assertIn("extra bit", self.app.runs[0][0])
        self.assertEqual(self.job()["next_run_at"], "2027-01-01T00:00:00-08:00")


class RegisterTest(unittest.TestCase):
    def test_cronjob_is_a_guest_tool_and_hooks_scheduler(self):
        app = FakeApp(Path(tempfile.mkdtemp()))
        self.addCleanup(app.close)
        app.config = {}  # [cron] comes from the plugin, its defaults filled in
        api = Api(app, "cron")
        plugin.register(api)
        self.assertEqual(app.config["cron"], {"enabled": True})
        self.assertTrue(api.tools["cronjob"].guest)
        self.assertTrue(api.tools["cronjob"].available(app))
        self.assertEqual((len(api.starts), len(api.stops)), (1, 1))

    def test_cron_off_hides_the_tool_and_starts_no_scheduler(self):
        app = FakeApp(Path(tempfile.mkdtemp()))
        self.addCleanup(app.close)
        app.config = {"cron": {"enabled": False}}
        api = Api(app, "cron")
        plugin.register(api)
        self.assertFalse(api.tools["cronjob"].available(app))
        self.assertEqual((api.starts, api.stops), ([], []))


class RealAppTest(unittest.IsolatedAsyncioTestCase):
    """The scheduler inside a real App: runs from pool threads go through the App's loop."""

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        jobs = Path(self.tmp.name) / "cron"
        jobs.mkdir()
        due = (datetime.now(TZ) - timedelta(minutes=1)).isoformat()
        jobs.joinpath("jobs.json").write_text(json.dumps({"jobs": [make_job(next_run_at=due)]}))

    async def wait_for(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while not condition():
            self.assertLess(time.monotonic(), deadline, "timed out")
            await asyncio.sleep(0.02)

    def job(self, app):
        return plugin.load(app)[0]

    async def test_due_job_is_delivered_and_recorded_in_home_history(self):
        app = await make_app(
            self.tmp.name, plugins=[plugin.register], replies=[text_reply("Good morning")]
        )
        try:
            await self.wait_for(lambda: self.job(app).get("last_status") == "ok")
            self.assertEqual(app.channel.texts(HOME), ["Good morning"])
            notes = [i.text for i in app.conversation(HOME).pending]
            self.assertTrue(any("Good morning" in text for text in notes))
        finally:
            await app.shutdown()

    async def test_silent_reply_is_not_sent(self):
        app = await make_app(
            self.tmp.name, plugins=[plugin.register], replies=[text_reply("[SILENT]")]
        )
        try:
            await self.wait_for(lambda: self.job(app).get("last_status") == "ok")
            self.assertEqual(app.channel.texts(HOME), [])
        finally:
            await app.shutdown()

    async def test_shutdown_while_a_run_is_in_flight(self):
        app = await make_app(self.tmp.name, plugins=[plugin.register], start=False)
        started = asyncio.Event()

        async def hang(*args, **kwargs):
            started.set()
            await asyncio.sleep(60)

        app.fake.complete = hang
        with mock.patch.object(app_module.lock, "acquire"):
            await app.start()
        await asyncio.wait_for(started.wait(), 5)
        begun = time.monotonic()
        await app.shutdown()
        self.assertLess(time.monotonic() - begun, 5)
        await self.wait_for(lambda: not app.state["cron"].running)
        self.assertEqual(app.channel.texts(HOME), [])
        self.assertIsNotNone(self.job(app)["running_since"])  # told as interrupted next start

    async def test_once_mode_does_not_run_jobs(self):
        app = await make_app(self.tmp.name, plugins=[plugin.register], start=False)
        with mock.patch.object(app_module.lock, "acquire"):
            await app.start(serve=False)  # what --once does
        try:
            await asyncio.sleep(0.2)
            self.assertEqual(self.job(app)["runs"], 0)
        finally:
            await app.shutdown()
