import subprocess
from pathlib import Path


def read_text(path, run_as_user=None):
    if run_as_user:
        return _sudo_read(path, run_as_user)
    return Path(path).read_text(encoding="utf-8")


def write_text(path, content, run_as_user=None):
    if run_as_user:
        _sudo_write(path, content, run_as_user)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def is_file(path, run_as_user=None):
    if run_as_user:
        return _sudo_test_file(path, run_as_user)
    return Path(path).is_file()


def list_dir(path, run_as_user=None):
    if run_as_user:
        return _sudo_list_dir(path, run_as_user)
    entries = []
    for item in sorted(Path(path).iterdir()):
        entries.append((item.name, item.is_dir()))
    return entries


# --- Subprocess helpers for running commands as another OS user (sudo) ---


def _sudo_command(command, run_as_user):
    return ["sudo", "-u", run_as_user, "--"] + command


def _sudo_read(path, run_as_user):
    argv = _sudo_command(["cat", "--", str(path)], run_as_user)
    result = subprocess.run(argv, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(message or f"failed to read {path}")
    return result.stdout.decode("utf-8")


def _sudo_write(path, content, run_as_user):
    parent = str(Path(path).parent)
    mkdir_argv = _sudo_command(["mkdir", "-p", "--", parent], run_as_user)
    subprocess.run(mkdir_argv, capture_output=True)
    tee_argv = _sudo_command(["tee", "--", str(path)], run_as_user)
    result = subprocess.run(tee_argv, input=content.encode("utf-8"), capture_output=True)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(message or f"failed to write {path}")


def _sudo_test_file(path, run_as_user):
    argv = _sudo_command(["test", "-f", str(path)], run_as_user)
    result = subprocess.run(argv, capture_output=True)
    return result.returncode == 0


def _sudo_list_dir(path, run_as_user):
    argv = _sudo_command(["ls", "-1", "-p", "--", str(path)], run_as_user)
    result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        message = result.stderr.strip()
        raise OSError(message or f"failed to list {path}")
    entries = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith("/"):
            entries.append((line[:-1], True))
        else:
            entries.append((line, False))
    return entries
