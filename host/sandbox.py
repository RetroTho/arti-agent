"""Each user's sandbox: a Linux container of the sandbox image (sandbox/Containerfile), run with
Podman, rootless (on macOS, inside Podman's own Linux VM: `podman machine`).

A user's sandbox, `arti-<user>`, runs their agent (`python -m arti`) as the host's own user
(`--userns keep-id`: what it writes in its home is the operator's), with:
- their home folder mounted at /home/arti (the agent's data in /home/arti/.arti), the one place
  it can write besides a scratch /tmp;
- Arti's code (arti/, plugins/) mounted read only at /opt/arti, so a new version needs only a
  restart, and the image holding just the system and the child environments;
- no Linux capabilities, no privilege gain, a read-only system, and limits on memory, CPU and
  processes.
The host starts every sandbox afresh when it starts (current code, current image; on macOS it
first starts Podman's VM if it is not running, as after a reboot) and stops them when it stops;
in between the runtime restarts an agent that exits (the agent's hang watchdog, say).
"""

import shutil
import subprocess
import sys
from pathlib import Path

from host.users import SANDBOX_HOME, User

IMAGE = "arti-sandbox"
CODE = "/opt/arti"
STOP_SECONDS = 30  # an agent's own shutdown takes up to about 20 s
TIMEOUT = 120  # seconds for one runtime command (not `build`)
MACHINE_START = 300  # seconds for Podman's VM to start (macOS)


class SandboxError(RuntimeError):
    pass


def podman() -> str:
    """Podman's executable, on PATH."""
    if found := shutil.which("podman"):
        return found
    raise SandboxError("Podman is needed to run the sandboxes, and it is not installed")


class Sandboxes:
    def __init__(self, settings: dict, repo: Path, data_dir: Path):
        self.settings, self.repo, self.data_dir = settings, Path(repo), Path(data_dir)
        self.tool = podman()

    def name(self, user: User) -> str:
        return f"arti-{user.id}"

    def _run(self, *args: str, timeout: float = TIMEOUT, check: bool = True) -> str:
        try:
            result = subprocess.run(
                [self.tool, *args], capture_output=True, text=True, timeout=timeout
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SandboxError(f"podman {args[0]} failed: {error}") from None
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-300:]
            raise SandboxError(f"podman {args[0]} failed: {detail}")
        return result.stdout.strip()

    def ready(self) -> None:
        """Make sure Podman can run containers: on macOS start its VM when it is down."""
        if self._run("info", "--format", "{{.Host.OS}}", check=False):
            return
        if sys.platform != "darwin":
            raise SandboxError("Podman does not answer (see: podman info)")
        self._run("machine", "start", timeout=MACHINE_START, check=False)
        if not self._run("info", "--format", "{{.Host.OS}}", check=False):
            raise SandboxError("Podman's VM did not start (see: podman machine list)")

    def run_args(self, user: User) -> list[str]:
        """`run` arguments for a user's sandbox (detached, restarted by the runtime if it exits)."""
        s = self.settings
        home = user.home(self.data_dir).resolve()
        mounts = [f"{home}:{SANDBOX_HOME}"]
        mounts += [f"{self.repo / part}:{CODE}/{part}:ro" for part in ("arti", "plugins")]
        args = ["run", "--detach", "--name", self.name(user), "--hostname", "arti"]
        args += ["--label", f"arti.user={user.id}", "--restart", "unless-stopped", "--init"]
        args += ["--userns", "keep-id", "--workdir", CODE]
        for mount in mounts:
            args += ["--volume", mount]
        args += ["--read-only", "--tmpfs", "/tmp:rw,size=512m"]
        args += ["--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
        args += ["--memory", s["memory"], "--cpus", str(s["cpus"]), "--pids-limit", "512"]
        args += ["--env", f"HOME={SANDBOX_HOME}", "--env", "USER=arti"]
        args += ["--env", "PYTHONDONTWRITEBYTECODE=1"]
        return [*args, f"{IMAGE}:latest", "python", "-m", "arti"]

    def start(self, user: User) -> None:
        """(Re)create and start a user's sandbox."""
        self._run("rm", "--force", self.name(user), check=False)
        self._run(*self.run_args(user))

    def stop(self, user: User) -> None:
        self._run("stop", "--time", str(STOP_SECONDS), self.name(user), timeout=STOP_SECONDS + 30)

    def remove(self, user: User) -> None:
        self._run("rm", "--force", self.name(user), check=False)

    def running(self, user: User) -> bool:
        state = self._run("inspect", "--format", "{{.State.Running}}", self.name(user), check=False)
        return state == "true"

    def exec_argv(self, user: User, command: list[str]) -> list[str]:
        """The command line that runs `command` in a user's sandbox, at a terminal."""
        return [self.tool, "exec", "--interactive", "--tty", self.name(user), *command]

    def build(self) -> None:
        """Build the sandbox image from the repository (output shown as it goes)."""
        argv = [self.tool, "build", "--tag", f"{IMAGE}:latest", "--file", "sandbox/Containerfile"]
        argv.append(".")
        if subprocess.run(argv, cwd=self.repo).returncode != 0:
            raise SandboxError("building the sandbox image failed")
