"""terminal plugin: persistent cwd, timeouts, background jobs and the process tool."""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from plugins import terminal
from plugins.terminal import ToolError


def make_app(tmp: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config={"agent": {"tool_timeout": 3600}}, state={}, cache_dir=lambda name: tmp
    )


def make_ctx(app, key="chat-a") -> SimpleNamespace:
    ctx = SimpleNamespace(
        app=app, key=key, state={}, cancel=threading.Event(), notices=[], cache_dir=app.cache_dir
    )
    ctx.notify = ctx.notices.append

    def wait(timeout, until=None, on_message=True):
        return "done" if until.wait(timeout) else "timeout"

    ctx.wait = wait
    return ctx


def wait_for(predicate, seconds=5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class TerminalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.app = make_app(self.tmp)
        self.ctx = make_ctx(self.app)
        self.addCleanup(terminal.stop_jobs, self.app)

    def test_cd_persists_even_after_exit(self):
        out = terminal.terminal({"command": f"cd {self.tmp}; echo hi; exit 4"}, self.ctx)
        self.assertEqual(out, f"hi\n[exit 4; cwd {self.tmp}]")
        out = terminal.terminal({"command": "pwd"}, self.ctx)
        self.assertTrue(out.startswith(f"{self.tmp}\n"))

    def test_workdir_is_for_one_call(self):
        (self.tmp / "sub").mkdir()
        self.ctx.state["cwd"] = str(self.tmp)
        out = terminal.terminal({"command": "pwd", "workdir": "sub"}, self.ctx)
        self.assertTrue(out.startswith(f"{self.tmp / 'sub'}\n"))
        self.assertEqual(self.ctx.state["cwd"], str(self.tmp))
        with self.assertRaises(ToolError):
            terminal.terminal({"command": "pwd", "workdir": "missing"}, self.ctx)

    def test_a_vanished_cwd_is_refused_not_replaced(self):
        # The review judged the command from the stored cwd; running it in home instead would
        # read different files than the ones reviewed.
        gone = self.tmp / "gone"
        self.ctx.state["cwd"] = str(gone)
        with self.assertRaisesRegex(ToolError, "gone"):
            terminal.terminal({"command": "cat .aws/credentials"}, self.ctx)
        self.assertNotIn("cwd", self.ctx.state)

    def test_no_secrets_in_the_environment(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-value-123456"
        self.addCleanup(os.environ.pop, "ANTHROPIC_API_KEY")
        out = terminal.terminal({"command": "env"}, self.ctx)
        self.assertNotIn("sk-ant-test-value", out)
        self.assertIn("PAGER=cat", out)

    def test_timeout_kills_the_tree(self):
        pidfile = self.tmp / "pid"
        out = terminal.terminal(
            {"command": f"sleep 999 & echo $! > {pidfile}; wait", "timeout": 1}, self.ctx
        )
        self.assertIn("[timed out after 1s; process group killed", out)
        pid = int(pidfile.read_text())
        self.assertTrue(wait_for(lambda: not _alive(pid)))

    def test_background_job_notify_and_process(self):
        out = terminal.terminal(
            {
                "command": "echo started; sleep 0.3; echo finished",
                "background": True,
                "notify": True,
            },
            self.ctx,
        )
        job_id = out.split()[3]
        self.assertTrue(job_id.startswith("bg_"))
        done = terminal.process({"action": "wait", "id": job_id, "timeout": 10}, self.ctx)
        self.assertIn("exited with code 0", done)
        self.assertIn("finished", done)
        self.assertTrue(wait_for(lambda: self.ctx.notices))
        self.assertIn(job_id, self.ctx.notices[0])
        log = terminal.process({"action": "log", "id": job_id, "offset": 2}, self.ctx)
        self.assertIn("finished\n[lines 2-2 of 2]", log)
        # Other chats do not see it.
        with self.assertRaises(ToolError):
            terminal.process({"action": "poll", "id": job_id}, make_ctx(self.app, "chat-b"))

    def test_kill_and_stop_jobs(self):
        first = terminal.terminal({"command": "sleep 999", "background": True}, self.ctx)
        other = make_ctx(self.app, "chat-b")
        second = terminal.terminal({"command": "sleep 999", "background": True}, other)
        first_id, second_id = first.split()[3], second.split()[3]
        status = terminal.process({"action": "kill", "id": first_id}, self.ctx)
        self.assertIn("killed", status)
        self.assertEqual(self.ctx.notices, [])
        terminal.stop_jobs(self.app, "chat-b")  # what /new in chat-b does
        self.assertIn("killed", terminal.process({"action": "poll", "id": second_id}, other))

    def test_at_most_four_jobs(self):
        for _ in range(terminal.MAX_JOBS):
            terminal.terminal({"command": "sleep 999", "background": True}, self.ctx)
        with self.assertRaisesRegex(ToolError, "already running"):
            terminal.terminal({"command": "sleep 999", "background": True}, self.ctx)
        terminal.stop_jobs(self.app)
        listing = terminal.process({"action": "list"}, self.ctx)
        self.assertNotIn("running", listing)

    def test_process_review(self):
        self.assertIsNone(terminal._process_review({"action": "poll"}))
        self.assertEqual(terminal._process_review({"action": "kill"}), "shell")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


if __name__ == "__main__":
    unittest.main()
