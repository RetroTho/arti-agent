"""The users the host runs agents for: <data>/users.json, which only the host writes.

{"sam": {"name": "Sam", "handles": {"<channel>": "..."}}}: a user's name (lower case, the
folder and sandbox name), how the agent calls them, and their handle on each channel (the
stable id the channel names a sender by). The registry lives outside every sandbox, so no agent
can change whom it answers. Each user's home folder, <data>/homes/<user>/, is their sandbox's
/home/arti: the agent's data directory (`.arti/`) and everything it keeps.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from arti.fsutil import private_dir, read_json, write_json

NAME = re.compile(r"[a-z][a-z0-9-]{0,31}")
SANDBOX_HOME = "/home/arti"  # where a user's home folder is mounted in their sandbox


@dataclass
class User:
    id: str  # "sam": lower case, the folder and sandbox name
    name: str  # "Sam": how the agent calls them
    handles: dict[str, str] = field(default_factory=dict)  # channel -> handle

    def home(self, data_dir: Path) -> Path:
        return Path(data_dir) / "homes" / self.id

    def agent_data(self, data_dir: Path) -> Path:
        """The agent's data directory, as the host sees it."""
        return self.home(data_dir) / ".arti"


class Users:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "users.json"

    def all(self) -> list[User]:
        raw = read_json(self.path, {})
        return [User(uid, v.get("name", uid), dict(v.get("handles", {}))) for uid, v in raw.items()]

    def get(self, uid: str) -> User | None:
        return next((u for u in self.all() if u.id == uid), None)

    def by_handle(self, channel: str, handle: str) -> User | None:
        if not handle:
            return None
        return next((u for u in self.all() if u.handles.get(channel) == handle), None)

    def add(
        self, uid: str, name: str, channel: str, handle: str, timezone: str, provider: str
    ) -> User:
        """Register a user and make their home folder, with the agent's first config.toml (their
        name, time zone and model provider; they change it there later)."""
        if not NAME.fullmatch(uid):
            raise ValueError("a user's id is lower-case letters, digits and -, starting a letter")
        if self.get(uid):
            raise ValueError(f"there is already a user {uid}")
        if other := self.by_handle(channel, handle):
            raise ValueError(f"{other.id} already has that {channel} handle")
        user = User(uid, name, {channel: handle})
        data = private_dir(user.agent_data(self.data_dir))
        config = data / "config.toml"
        if not config.exists():
            lines = [f"user_name = {json.dumps(name)}", f"timezone = {json.dumps(timezone)}"]
            lines.append(f"provider = {json.dumps(provider)}")
            config.write_text("[agent]\n" + "\n".join(lines) + "\n")
        self._save([*self.all(), user])
        return user

    def remove(self, uid: str) -> None:
        """Forget a user (their home folder stays, for the operator to keep or delete)."""
        self._save([u for u in self.all() if u.id != uid])

    def _save(self, users: list[User]) -> None:
        private_dir(self.data_dir)
        write_json(self.path, {u.id: {"name": u.name, "handles": u.handles} for u in users})
