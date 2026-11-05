"""Child processes: running and killing children, capturing output.

Every child runs in its own process group (`start_new_session=True`), so a timeout or a kill
reaches the whole tree, including anything it started with `&`. While Arti runs (from the safety
plugin's start hook, `keep_records`) a live group's id is recorded as an empty file
`run/children/<pgid>`; if Arti dies hard, the next start (`sweep`) kills the recorded groups
that are still around, and shutdown kills every live one (`kill_all`). A kill is SIGTERM, 2 s,
SIGKILL, 2 s, then give up and log: an always-on agent must never wedge on one unkillable
process. Children get an environment without secrets (redact.child_env) unless the caller
passes its own.
"""

import codecs
import logging
import os
import selectors
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import redact

log = logging.getLogger(__name__)

CHILDREN_DIR = "run/children"
MAX_OUTPUT_BYTES = 64 << 20  # a command writing more than this is killed
EXIT_GRACE = 1.0  # after the leader exits, how long its background children may hold the pipe
OVERLAP = 4096  # chars kept past each cut, so a secret across it is redacted whole

DEVNULL, PIPE, STDOUT = subprocess.DEVNULL, subprocess.PIPE, subprocess.STDOUT

_records: Path | None = None  # run/children of the running Arti's data dir; None = keep none
_live: dict[int, subprocess.Popen | None] = {}  # pgid -> leader (None: tracked asyncio child)
_live_lock = threading.Lock()


def keep_records(data_dir: Path | None) -> None:
    """Record every live child group under data_dir/run/children from now on, for `sweep`
    after a crash (None: keep no records). Only the Arti holding the data dir's lock does."""
    global _records
    _records = None if data_dir is None else Path(data_dir) / CHILDREN_DIR
    if _records is not None:
        _records.mkdir(parents=True, exist_ok=True)


def sweep(data_dir: Path) -> int:
    """Kill the process groups a dead Arti left behind; return how many were killed."""
    records = data_dir / CHILDREN_DIR
    if not records.is_dir():
        return 0
    try:
        groups, killed = _process_groups(), 0
    except (OSError, ValueError, StopIteration) as error:  # never a reason not to start
        log.warning("could not list processes (%s); leftover children not swept", error)
        return 0
    for record in records.iterdir():
        try:
            pgid, written = int(record.name), record.stat().st_mtime
        except (ValueError, OSError):
            record.unlink(missing_ok=True)
            continue
        members = groups.get(pgid)
        # A pid is never reused while its process group exists, so members without the leader
        # are surely ours. A live leader must have started before the record was written (the
        # boot time is in whole seconds, hence the 1 s slack), or its pid was reused by a stranger.
        if members and (pgid not in members or members[pgid] <= written + 1):
            killed += _kill_groups({pgid: None}, grace=2.0)
        record.unlink(missing_ok=True)
    if killed:
        log.warning("killed %d leftover child process group(s)", killed)
    return killed


def _process_groups(root: Path = Path("/proc")) -> dict[int, dict[int, float]]:
    """{pgid: {pid: start time}} of every process, from Linux's process table: each
    /proc/<pid>/stat holds the process group (field 5) and the start time in clock ticks after
    boot (field 22); /proc/stat has the boot time. A process that ends while it is read is left
    out."""
    boot = next(
        int(line.split()[1]) for line in (root / "stat").read_text().splitlines()
        if line.startswith("btime ")
    )  # fmt: skip
    ticks = os.sysconf("SC_CLK_TCK")
    groups: dict[int, dict[int, float]] = {}
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
            pgid, started = int(fields[2]), int(fields[19])  # after the name: field 3 onwards
        except (OSError, ValueError, IndexError):
            continue
        groups.setdefault(pgid, {})[int(entry.name)] = boot + started / ticks
    return groups


def track(pgid: int, leader: subprocess.Popen | None = None) -> None:
    """Record a live group so kill_all and a later sweep see it (asyncio children use this)."""
    with _live_lock:
        _live[pgid] = leader
    if _records is not None:
        (_records / str(pgid)).touch()


def untrack(pgid: int) -> None:
    with _live_lock:
        _live.pop(pgid, None)
    if _records is not None:
        (_records / str(pgid)).unlink(missing_ok=True)


def _signal(pgid: int, sig: int) -> bool:
    """Signal a group; False when it is gone. EPERM counts as gone: macOS answers it when only
    zombies are left."""
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _kill_groups(groups: dict[int, subprocess.Popen | None], grace: float) -> bool:
    """TERM every group, wait, KILL the survivors, wait. True if all are gone."""
    alive = dict(groups)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = {pgid: p for pgid, p in alive.items() if _signal(pgid, sig)}
        deadline = time.monotonic() + grace
        while alive and time.monotonic() < deadline:
            time.sleep(0.05)
            for leader in alive.values():
                if leader is not None:
                    leader.poll()  # reap the leader, or the probe keeps seeing its zombie
            alive = {pgid: p for pgid, p in alive.items() if _signal(pgid, 0)}
        if not alive:
            return True
    log.warning("process group(s) %s survived SIGKILL; giving up", sorted(alive))
    return False


class Child:
    """A spawned process group (Chrome, background jobs). Call kill() when done with it; that is
    cheap when the group is already gone, and it removes the record."""

    def __init__(self, proc: subprocess.Popen):
        self.proc, self.pgid = proc, proc.pid

    def kill(self, grace: float = 2.0) -> bool:
        gone = _kill_groups({self.pgid: self.proc}, grace)
        if gone:
            self.proc.poll()
            untrack(self.pgid)
        return gone


def spawn(argv, *, cwd=None, env=None, stdin=DEVNULL, stdout=PIPE, stderr=STDOUT) -> Child:
    """Start argv in a new process group; env=None means redact.child_env()."""
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=redact.child_env() if env is None else env,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        close_fds=True,
    )
    try:
        track(proc.pid, proc)
    except OSError:  # no record (e.g. a full disk): kill it rather than leave it untracked
        _kill_groups({proc.pid: proc}, grace=0)
        raise
    return Child(proc)


def kill_all() -> None:
    """Shutdown: kill every live child group, all at once."""
    with _live_lock:
        groups = dict(_live)
    if groups:
        _kill_groups(groups, grace=2.0)
        for pgid in groups:
            untrack(pgid)


@dataclass
class Result:
    code: int | None  # None when timed out, cancelled or killed for too much output
    out: str
    timed_out: bool
    cancelled: bool
    spill: Path | None  # the full output, when `out` had to be cut
    err: str = ""  # stderr, when merge_stderr=False


class _Capture:
    """The head and tail of a stream; once anything would be dropped, the full text also goes
    to a spill file (when there is a spill_dir), redacted. The file is written only when the
    stream ends, from the whole text: redacting it in pieces would miss a secret split between
    two of them, and the raw output never reaches the disk."""

    def __init__(self, max_chars: int, spill_dir: Path | None):
        self.keep = max(max_chars // 2, 1)  # chars shown from each end
        self.room = self.keep + OVERLAP  # chars kept at each end
        self.spill_dir, self.spill = spill_dir, None
        self.parts: list[str] = []  # everything, while there is a spill_dir
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.head = self.tail = ""
        self.chars = self.bytes = 0

    @property
    def full(self) -> bool:
        return self.bytes >= MAX_OUTPUT_BYTES

    def feed(self, data: bytes, final: bool = False) -> None:
        self.bytes += len(data)
        text = self.decoder.decode(data, final)
        self.chars += len(text)
        if self.spill_dir:
            self.parts.append(text)
        take = self.room - len(self.head)
        self.head += text[:take]
        self.tail = (self.tail + text[take:])[-self.room :]

    def close(self) -> None:
        self.feed(b"", final=True)
        if self.spill_dir and self.chars > 2 * self.keep:
            fd, name = tempfile.mkstemp(prefix="out-", suffix=".txt", dir=self.spill_dir)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(redact.redact("".join(self.parts)))
            self.spill = Path(name)
        self.parts = []

    def text(self) -> str:
        if self.chars <= 2 * self.keep:
            return self.head + self.tail
        # Redact before cutting: a cut through a secret would leave a part redact can't see.
        head = redact.redact(self.head)[: self.keep]
        tail = redact.redact(self.tail)[-self.keep :]
        if "\n" in head[len(head) // 2 :]:  # cut at a line end nearby, so no line shows half
            head = head[: head.rindex("\n") + 1]
        if "\n" in tail[: len(tail) // 2]:
            tail = tail[tail.index("\n") + 1 :]
        omitted = self.chars - len(head) - len(tail)
        where = f"full output: {self.spill}" if self.spill else "not saved"
        return f"{head}\n... [about {omitted} chars omitted; {where}] ...\n{tail}"


def run(
    argv: list[str],
    *,
    cwd=None,
    env=None,
    input: bytes | None = None,
    timeout: float,
    cancel: threading.Event | None = None,
    max_chars: int = 30_000,
    spill_dir: Path | None = None,
    merge_stderr: bool = True,
) -> Result:
    """Run argv to completion (or timeout/cancel) and return its captured output.

    The whole process group is killed at the end in every case, so nothing it started in the
    background outlives the call. Once the leader has exited, output is read for EXIT_GRACE
    more, not until background children close the pipe.
    """
    stderr = STDOUT if merge_stderr else PIPE
    child = spawn(argv, cwd=cwd, env=env, stdin=DEVNULL if input is None else PIPE, stderr=stderr)
    proc = child.proc
    if input is not None:
        threading.Thread(target=_feed_stdin, args=(proc.stdin, input), daemon=True).start()
    out, err = _Capture(max_chars, spill_dir), _Capture(max_chars, None)
    streams = {proc.stdout: out} if merge_stderr else {proc.stdout: out, proc.stderr: err}
    deadline, exited = time.monotonic() + timeout, None
    stop = ""  # why reading stopped early: "timeout", "cancelled" or "full"
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
            while True:
                now = time.monotonic()
                if exited is None and proc.poll() is not None:
                    exited = now
                if cancel is not None and cancel.is_set():
                    stop = "cancelled"
                elif now >= deadline:
                    stop = "timeout"
                elif out.full or err.full:
                    stop = "full"
                done = exited is not None and (not selector.get_map() or now > exited + EXIT_GRACE)
                if stop or done:
                    break
                for key, _ in selector.select(timeout=0.05):  # with no pipes left: a sleep
                    data = key.fileobj.read(65536)
                    if data:
                        streams[key.fileobj].feed(data)
                    elif data is not None:  # EOF
                        selector.unregister(key.fileobj)
    finally:
        child.kill()
        for stream, capture in streams.items():
            _drain(stream, capture)
    text = out.text()
    if stop == "full":
        text += f"\n[output passed {MAX_OUTPUT_BYTES >> 20} MiB; process group killed]"
    return Result(
        code=None if stop else proc.returncode,
        out=text,
        timed_out=stop == "timeout",
        cancelled=stop == "cancelled",
        spill=out.spill,
        err=err.text(),
    )


def _feed_stdin(pipe, data: bytes) -> None:
    try:
        pipe.write(data)
        pipe.close()
    except OSError:  # the child exited without reading it all
        pass


def _drain(stream, capture: _Capture) -> None:
    """Read what is left in a pipe after the group is gone (a process that left the group may
    still hold it: the pipe is non-blocking, so that ends the read), then close both."""
    try:
        while (data := stream.read(65536)) and not capture.full:
            capture.feed(data)
    except (OSError, ValueError):
        pass
    stream.close()
    capture.close()
