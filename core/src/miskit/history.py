import json
from datetime import datetime
from pathlib import Path

from miskit.message import message_to_dict
from miskit.message import message_from_dict


class History:
    session_file = "session.jsonl"

    def __init__(self, folder):
        self.folder = Path(folder)

    def path(self):
        return self.folder / self.session_file

    def setup(self):
        self.folder.mkdir(parents=True, exist_ok=True)

        path = self.path()
        if not path.exists():
            path.write_text("", encoding="utf-8")

    def log(self, message):
        self.setup()
        with self.path().open("a", encoding="utf-8") as file:
            self.write_message(file, message)

    def read(self):
        self.setup()
        messages = []

        for line in self.path().read_text(encoding="utf-8").splitlines():
            if line:
                messages.append(message_from_dict(json.loads(line)))

        return messages

    def archive_path(self):
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        return self.folder / "archive" / f"session-{timestamp}.jsonl"

    def archive(self):
        self.setup()
        archive_path = self.archive_path()
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        self.path().replace(archive_path)
        self.path().write_text("", encoding="utf-8")
        return archive_path

    def write(self, messages):
        self.setup()
        with self.path().open("w", encoding="utf-8") as file:
            for message in messages:
                self.write_message(file, message)

    def write_message(self, file, message):
        file.write(json.dumps(message_to_dict(message)) + "\n")
