import asyncio
import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from miskit.message import Message
from miskit.message import ToolCall
from miskit.provider import Model
from miskit.provider import Provider


class GenericModel(Model):
    def __init__(self, model, api_key=None, base_url=None, max_tokens=None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens

    @classmethod
    def from_config(cls, config):
        return cls(
            config.get("model"),
            api_key=config.get("apiKey"),
            base_url=config.get("baseUrl"),
            max_tokens=config.get("maxTokens"),
        )

    async def complete(self, messages, tools=None):
        data = await asyncio.to_thread(self._send, messages, tools)
        return self._message(data)

    def _send(self, messages, tools=None):
        request = Request(
            self._url(),
            data=self._body(messages, tools),
            headers=self._headers(),
            method="POST",
        )

        try:
            with urlopen(request) as response:
                text = response.read().decode("utf-8")
        except HTTPError as error:
            body = error.read().decode("utf-8")
            raise ValueError(f"API error {error.code}: {body}") from error
        except URLError as error:
            raise ValueError(f"API connection error: {error.reason}") from error

        return json.loads(text)

    def _url(self):
        if not self.base_url:
            raise ValueError("provider.baseUrl is required")

        return self.base_url.rstrip("/") + "/chat/completions"

    def _body(self, messages, tools=None):
        body = {
            "model": self.model,
            "messages": self._messages(messages),
        }

        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens

        if tools:
            body["tools"] = self._tools(tools)

        text = json.dumps(body)
        return text.encode("utf-8")

    def _headers(self):
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _messages(self, messages):
        result = []

        for message in messages:
            content = self._content(message)
            if message.role == "assistant" and message.tool_calls and not content:
                content = None

            item = {
                "role": message.role,
                "content": content,
            }

            if message.role == "tool":
                item["tool_call_id"] = message.tool_call_id

            if message.tool_calls:
                item["tool_calls"] = self._tool_calls(message.tool_calls)

            result.append(item)

        return result

    def _content(self, message):
        if isinstance(message.content, str):
            return message.content

        parts = []
        for part in message.parts:
            if part["type"] == "text":
                parts.append({
                    "type": "text",
                    "text": part["text"],
                })
            elif part["type"] == "image":
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_data_url(part["path"], part["mime_type"]),
                    },
                })
        return parts

    def _tools(self, tools):
        result = []

        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            })

        return result

    def _tool_calls(self, tool_calls):
        result = []

        for tool_call in tool_calls:
            result.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            })

        return result

    def _message(self, data):
        message = data["choices"][0]["message"]
        content = message.get("content")
        if content is None:
            content = ""

        return Message("assistant", content, tool_calls=self._read_tool_calls(message), usage=self._usage(data))

    def _usage(self, data):
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            return {}

        result = {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if isinstance(value, int):
                result[name] = value

        return result

    def _read_tool_calls(self, message):
        result = []

        for item in message.get("tool_calls", []):
            function = item.get("function", {})
            arguments = self._read_arguments(function.get("arguments", "{}"))
            result.append(ToolCall(
                item.get("id", ""),
                function.get("name", ""),
                arguments,
            ))

        return result

    def _read_arguments(self, text):
        try:
            arguments = json.loads(text)
        except json.JSONDecodeError:
            return {}

        if isinstance(arguments, dict):
            return arguments

        return {}

    def _image_data_url(self, path, mime_type):
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{data}"


class GenericProvider(Provider):
    def create_model(self, config):
        return GenericModel.from_config(config)


provider = GenericProvider()
