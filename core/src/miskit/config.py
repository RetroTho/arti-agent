import json
from pathlib import Path


class Config:
    instance = Path.home() / ".miskit"
    path = instance / "config.json"

    default = {
        "channel": {
            "name": "terminal",
        },
        "provider": {
            "name": "generic",
            "model": "llama",
            "apiKey": None,
            "baseUrl": None,
        },
        "context": {
            "tokens": 8000,
            "outputTokens": 2000,
            "maxInputChars": 20000,
            "compactAt": 0.5,
            "keepRecent": 10,
            "maxToolRounds": 20,
        },
        "heartbeat": {
            "enabled": False,
            "intervalSeconds": 1800,
        },
        "security": {
            "runAsUser": None,
        },
        "workspace": {
            "restrictToWorkspace": True,
        },
        "tools": [
            {
                "name": "read_file",
            },
            {
                "name": "write_file",
            },
            {
                "name": "edit_file",
            },
            {
                "name": "list_files",
            },
            {
                "name": "memory",
            },
            {
                "name": "recall",
            },
            {
                "name": "cron",
            },
            {
                "name": "heartbeat",
            },
            {
                "name": "read_more",
            },
        ],
    }

    def __init__(self, data):
        self.data = data

    @classmethod
    def load(cls, path):
        text = Path(path).read_text(encoding="utf-8")
        return cls(json.loads(text))

    @classmethod
    def write_default(cls, path):
        path = Path(path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(cls.default, indent=2)
            path.write_text(text + "\n", encoding="utf-8")

    def section(self, name):
        value = self.data.get(name, {})
        if isinstance(value, dict):
            return value
        return {}

    def entries(self, name):
        value = self.data.get(name, [])
        if isinstance(value, list):
            return value
        return []
