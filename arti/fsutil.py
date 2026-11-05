"""Small file helpers: private directories, crash-safe writes, JSON files and pruning.

Every file Arti writes that matters (grants, and plugins' notes and jobs) goes through
`atomic_write`, so a crash leaves either the old file or the new one, never half of one.
"""

import json
import logging
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)


def private_dir(path: Path) -> Path:
    """Create `path` (and its parents) if missing and make it owner-only (0700)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    return path


def atomic_write(path: Path, data: str | bytes, mode: int = 0o600) -> None:
    """Replace `path` with `data`: write a temp file beside it, fsync, rename, fsync the dir."""
    path = Path(path)
    if isinstance(data, str):
        data = data.encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    # The rename is only durable once the directory entry itself is on disk.
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_json(path: Path, value) -> None:
    """Atomically write `value` as indented JSON."""
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path, default=None):
    """The JSON in `path`, or `default` when the file is missing; ValueError when corrupt."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    try:
        return json.loads(text)
    except ValueError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from None


def delete_older_than(paths, seconds: float) -> int:
    """Delete each of `paths` last changed more than `seconds` ago (a folder with everything in
    it); return how many went. A symlink is never followed: an old one is deleted itself."""
    cutoff = time.time() - seconds
    removed = 0
    for path in paths:
        try:
            info = path.lstat()
            if info.st_mtime >= cutoff:
                continue
            if stat.S_ISDIR(info.st_mode):
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        except FileNotFoundError:
            pass  # already gone
        except OSError as error:
            log.warning("could not delete %s: %s", path, error)
    return removed
