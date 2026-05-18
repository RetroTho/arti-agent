import json
from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

    def __post_init__(self):
        if not isinstance(self.arguments, dict):
            raise TypeError("tool arguments must be a dictionary")
        self.id = str(self.id or "")
        self.name = str(self.name or "")


def text_part(text):
    return {"type": "text", "text": str(text or "")}


def image_part(path, mime_type):
    return {"type": "image", "path": str(path), "mime_type": str(mime_type or "")}


def copy_content(content):
    if isinstance(content, list):
        return [dict(part) for part in content]
    return content


@dataclass
class Message:
    role: str
    content: str | list = ""
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None
    usage: dict | None = None
    stored_content: str | list | None = None

    def __post_init__(self):
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError("role must be system, user, assistant, or tool")

        if not isinstance(self.content, (str, list)):
            raise TypeError("content must be text or a list of content parts")

        if self.stored_content is not None and not isinstance(self.stored_content, (str, list)):
            raise TypeError("stored_content must be text or a list of content parts")

        if self.usage is not None and not isinstance(self.usage, dict):
            raise TypeError("usage must be a dictionary")

        self.tool_calls = list(self.tool_calls or [])
        self.usage = dict(self.usage or {})

    @property
    def text(self):
        if isinstance(self.content, str):
            return self.content

        lines = []
        for part in self.content:
            if part["type"] == "text":
                text = part["text"].strip()
                if text:
                    lines.append(text)
            elif part["type"] == "image":
                lines.append("[Image]")
        return "\n".join(lines)

    @property
    def parts(self):
        if isinstance(self.content, str):
            return [text_part(self.content)] if self.content else []
        return [dict(part) for part in self.content]

    def with_prepended_text(self, text):
        if not text:
            return self.copy()
        if isinstance(self.content, str):
            content = f"{text}{self.content}" if self.content else text
        else:
            content = [text_part(text)] + [dict(p) for p in self.content]
        return self.copy(content=content)

    def storage_message(self):
        if self.stored_content is None:
            return self

        return Message(
            self.role,
            self.stored_content,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
            name=self.name,
            usage=self.usage,
        )

    def copy(self, **changes):
        values = {
            "role": self.role,
            "content": copy_content(self.content),
            "tool_calls": list(self.tool_calls),
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "usage": dict(self.usage),
            "stored_content": copy_content(self.stored_content),
        }
        values.update(changes)
        return Message(**values)


# --- Serialization (converting messages to/from JSON-ready dicts) ---


def message_to_dict(message):
    data = {
        "role": message.role,
        "content": message.content,
        "tool_calls": [tool_call_to_dict(tc) for tc in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "name": message.name,
    }
    if message.usage:
        data["usage"] = message.usage
    return data


def message_from_dict(data):
    return Message(
        data.get("role"),
        data.get("content", ""),
        tool_calls=[tool_call_from_dict(item) for item in data.get("tool_calls", [])],
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
        usage=data.get("usage"),
    )


def tool_call_to_dict(tool_call):
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    }


def tool_call_from_dict(data):
    return ToolCall(
        data.get("id"),
        data.get("name"),
        data.get("arguments", {}),
    )


def format_message(message):
    label = message.role
    if message.name:
        label = f"{label} {message.name}"

    content = message.content or "(no text)"
    if message.tool_calls:
        calls = [f"{tc.name}({json.dumps(tc.arguments)})" for tc in message.tool_calls]
        content = f"{content}\nTool calls: {', '.join(calls)}"

    return f"{label}: {content}"


def log_message_to_dict(message):
    data = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_calls:
        data["tool_calls"] = [
            {"name": tc.name, "arguments": tc.arguments}
            for tc in message.tool_calls
        ]
    if message.name:
        data["name"] = message.name
    return data
