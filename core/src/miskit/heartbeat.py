import json
from collections import deque
from pathlib import Path

from miskit.message import log_message_to_dict


HEARTBEAT_JOB_ID = "heartbeat"
HEARTBEAT_QUIET_REPLY = "HEARTBEAT_OK"
DEFAULT_HEARTBEAT_TEXT = """# Heartbeat

Review the active tasks below.
Only act on unchecked tasks, like "- [ ] Check mail".
Ignore checked tasks, like "- [x] Done".

Use the conversation, memory, and tools if they help.
Notify the user only when there is a useful result, status change, warning, or question.

If there are no active tasks or nothing useful to report, reply exactly HEARTBEAT_OK.

# Tasks

- [ ] Check whether there is anything useful to report.
"""


class HeartbeatTasks:
    def __init__(self, path):
        self.path = Path(path)

    def setup(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(DEFAULT_HEARTBEAT_TEXT, encoding="utf-8")

    def read(self):
        self.setup()
        return self.path.read_text(encoding="utf-8")

    def add(self, text):
        text = str(text).strip()
        if not text:
            raise ValueError("heartbeat add requires text")

        content = self.read().rstrip()
        if content:
            content += "\n\n"
        content += f"- [ ] {text}\n"
        self.path.write_text(content, encoding="utf-8")

    def list(self):
        tasks = []
        for line_number, line in enumerate(self.read().splitlines()):
            parsed = parse_task_line(line)
            if parsed is not None:
                done, text = parsed
                tasks.append({
                    "line_number": line_number,
                    "done": done,
                    "text": text,
                })
        return tasks

    def complete(self, task_id):
        self._replace_task(task_id, done=True)

    def remove(self, task_id):
        task = self._task_by_id(task_id)
        lines = self.read().splitlines()
        del lines[task["line_number"]]
        self.path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _replace_task(self, task_id, done):
        task = self._task_by_id(task_id)
        lines = self.read().splitlines()
        marker = "x" if done else " "
        lines[task["line_number"]] = f"- [{marker}] {task['text']}"
        self.path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _task_by_id(self, task_id):
        try:
            index = int(task_id) - 1
        except (TypeError, ValueError) as error:
            raise ValueError("task_id must be a task number") from error

        tasks = self.list()
        if index < 0 or index >= len(tasks):
            raise ValueError("task_id was not found")

        return tasks[index]


def parse_task_line(line):
    text = line.strip()
    if text.startswith("- [ ] "):
        return False, text[6:].strip()
    if text.lower().startswith("- [x] "):
        return True, text[6:].strip()
    return None


class HeartbeatLog:
    """Saves heartbeat turns to a separate JSONL file so the agent can review them later."""

    def __init__(self, path):
        self.path = Path(path)

    def setup(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def log(self, timestamp, messages, quiet):
        self.setup()
        entry = {
            "timestamp": timestamp,
            "quiet": quiet,
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

    def read_by_timestamp(self, timestamp):
        self.setup()
        result = None
        with self.path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("timestamp") == timestamp:
                        result = entry
        return result
