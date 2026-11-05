"""proc: whole-tree kills on timeout and cancel, bounded output, records and the startup sweep."""

import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from arti.secret import remember
from plugins.safety import proc, redact

ENV = {"PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8"}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); ps tells them apart.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(state.stdout.strip()) and not state.stdout.strip().startswith("Z")


def wait_gone(pid: int, seconds: float = 3) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_exit_code_and_merged_output(self):
        result = proc.run(["/bin/sh", "-c", "echo out; echo err >&2; exit 3"], env=ENV, timeout=10)
        self.assertEqual(result.code, 3)
        self.assertIn("out", result.out)
        self.assertIn("err", result.out)
        self.assertFalse(result.timed_out)

    def test_separate_stderr_and_input(self):
        result = proc.run(
            ["/bin/sh", "-c", "cat; echo oops >&2"],
            env=ENV,
            input=b"hello",
            timeout=10,
            merge_stderr=False,
        )
        self.assertEqual(result.out, "hello")
        self.assertEqual(result.err.strip(), "oops")

    def test_timeout_kills_the_whole_tree(self):
        pidfile = self.tmp / "pid"
        start = time.monotonic()
        result = proc.run(
            ["/bin/bash", "-c", f"sleep 999 & echo $! > {pidfile}; sleep 999"],
            env=ENV,
            timeout=1,
        )
        self.assertLess(time.monotonic() - start, 6)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.code)
        self.assertTrue(wait_gone(int(pidfile.read_text())))

    def test_background_child_is_killed_after_a_normal_exit(self):
        pidfile = self.tmp / "pid"
        result = proc.run(
            ["/bin/bash", "-c", f"sleep 999 >/dev/null 2>&1 & echo $! > {pidfile}"],
            env=ENV,
            timeout=10,
        )
        self.assertEqual(result.code, 0)
        self.assertTrue(wait_gone(int(pidfile.read_text())))

    def test_background_child_holding_the_pipe_does_not_stall(self):
        start = time.monotonic()
        result = proc.run(["/bin/bash", "-c", "sleep 999 & echo done"], env=ENV, timeout=20)
        self.assertLess(time.monotonic() - start, 5)
        self.assertEqual((result.code, result.timed_out), (0, False))
        self.assertIn("done", result.out)

    def test_a_secret_across_the_cut_is_redacted(self):
        secret = "Zq7" + "k" * 37
        remember([secret])
        for pad in (980, 1000, 1020):  # 1000 chars of the head are shown: the cut is in it
            script = f"head -c {pad} /dev/zero | tr '\\0' x; echo -n {secret}; seq 1 2000"
            result = proc.run(["/bin/sh", "-c", script], env=ENV, timeout=10, max_chars=2000)
            self.assertIn("chars omitted", result.out)
            self.assertNotIn("Zq7k", result.out)

    def test_cancel(self):
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        start = time.monotonic()
        result = proc.run(["/bin/sleep", "999"], env=ENV, timeout=60, cancel=cancel)
        self.assertLess(time.monotonic() - start, 5)
        self.assertTrue(result.cancelled)
        self.assertIsNone(result.code)

    def test_output_cap_and_spill(self):
        result = proc.run(
            ["/bin/sh", "-c", "seq 1 100000"],
            env=ENV,
            timeout=20,
            max_chars=2000,
            spill_dir=self.tmp,
        )
        self.assertLess(len(result.out), 2300)
        self.assertTrue(result.out.startswith("1\n2\n"))
        self.assertTrue(result.out.endswith("100000\n"))
        self.assertIn("chars omitted", result.out)
        self.assertEqual(result.spill.read_text(), "".join(f"{n}\n" for n in range(1, 100001)))

    def test_spill_file_is_redacted(self):
        secret = "p3X-spill-Secret/value+17"
        remember([secret])
        # Secrets at the start, across where a 64 KiB read would cut, and at the end.
        pad = "head -c 65530 /dev/zero | tr '\\0' x"
        script = f"echo {secret}; {pad}; echo {secret}; seq 1 9999; echo {secret}"
        result = proc.run(
            ["/bin/sh", "-c", script], env=ENV, timeout=20, max_chars=2000, spill_dir=self.tmp
        )
        spilled = result.spill.read_text()
        self.assertNotIn(secret, spilled)
        self.assertEqual(spilled.count(redact.REDACTED), 3)
        self.assertIn("9999\n", spilled)
        self.assertNotIn(secret, result.out)

    def test_spill_cap_kills_the_writer(self):
        with mock.patch.object(proc, "MAX_OUTPUT_BYTES", 100_000):
            result = proc.run(
                ["/usr/bin/yes"], env=ENV, timeout=20, max_chars=100, spill_dir=self.tmp
            )
        self.assertIsNone(result.code)
        self.assertFalse(result.timed_out)
        self.assertIn("process group killed", result.out)
        self.assertLess(result.spill.stat().st_size, 200_000)


class LinuxProcessTableTest(unittest.TestCase):
    def test_groups_and_start_times_come_from_proc(self):
        root = Path(tempfile.mkdtemp())
        (root / "stat").write_text("cpu 1 2 3\nbtime 1700000000\nprocesses 9\n")
        rest = " ".join(["0"] * 15)  # fields 7 to 21
        for pid, name, pgid, started in ((10, "sh", 10, 500), (11, "my (odd) name", 10, 700)):
            (root / str(pid)).mkdir()
            line = f"{pid} ({name}) S 1 {pgid} {pgid} {rest} {started} 0 0\n"
            (root / str(pid) / "stat").write_text(line)
        (root / "12").mkdir()  # ended while being read: no stat file
        (root / "self").mkdir()
        ticks = os.sysconf("SC_CLK_TCK")
        self.assertEqual(
            proc._process_groups(root),
            {10: {10: 1700000000 + 500 / ticks, 11: 1700000000 + 700 / ticks}},
        )


class RecordsAndSweepTest(unittest.TestCase):
    def setUp(self):
        self.data = Path(tempfile.mkdtemp())
        self.addCleanup(proc.keep_records, None)

    def leftover(self) -> subprocess.Popen:
        child = subprocess.Popen(["/bin/sleep", "999"], start_new_session=True)
        # A real leftover's parent is launchd, which reaps it; here this thread does.
        threading.Thread(target=child.wait, daemon=True).start()
        self.addCleanup(child.wait)
        self.addCleanup(lambda: proc._signal(child.pid, 9))
        return child

    def test_sweep_kills_a_recorded_leftover_group(self):
        child = self.leftover()
        records = self.data / proc.CHILDREN_DIR
        records.mkdir(parents=True)
        (records / str(child.pid)).touch()
        self.assertEqual(proc.sweep(self.data), 1)
        child.wait(timeout=5)
        self.assertFalse((records / str(child.pid)).exists())

    def test_sweep_ignores_a_reused_pid(self):
        child = self.leftover()
        records = self.data / proc.CHILDREN_DIR
        records.mkdir(parents=True)
        record = records / str(child.pid)
        record.touch()
        old = time.time() - 3600  # the record predates this process: its pid was reused
        os.utime(record, (old, old))
        self.assertEqual(proc.sweep(self.data), 0)
        self.assertIsNone(child.poll())
        self.assertFalse(record.exists())

    def test_spawn_records_and_kill_forgets(self):
        proc.keep_records(self.data)
        child = proc.spawn(["/bin/sleep", "999"], env=ENV, stdout=proc.DEVNULL)
        record = self.data / proc.CHILDREN_DIR / str(child.pgid)
        self.assertTrue(record.exists())
        self.assertTrue(child.kill())
        self.assertFalse(record.exists())
        self.assertIsNotNone(child.proc.returncode)

    def test_kill_all(self):
        children = [
            proc.spawn(["/bin/sleep", "999"], env=ENV, stdout=proc.DEVNULL) for _ in range(3)
        ]
        start = time.monotonic()
        proc.kill_all()
        self.assertLess(time.monotonic() - start, 3)
        for child in children:
            self.assertIsNotNone(child.proc.poll())


if __name__ == "__main__":
    unittest.main()
