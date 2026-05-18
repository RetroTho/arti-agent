import asyncio
import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from miskit.message import Message, ToolCall
from miskit.provider import Model
from miskit_codex.auth import get_codex_token


CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"


class CodexModel(Model):
    def __init__(self, model="gpt-5.1-codex", access_token=None, account_id=None, reasoning_effort=None):
        self.model = _strip_model_prefix(model)
        self.access_token = access_token
        self.account_id = account_id
        self.reasoning_effort = reasoning_effort

    @classmethod
    def from_config(cls, config):
        return cls(
            config.get("model", "gpt-5.1-codex"),
            access_token=config.get("accessToken"),
            account_id=config.get("accountId"),
            reasoning_effort=config.get("reasoningEffort"),
        )

    async def complete(self, messages, tools=None):
        content, tool_calls, usage = await asyncio.to_thread(self._send, messages, tools)
        return Message("assistant", content, tool_calls=tool_calls, usage=usage)

    def _send(self, messages, tools=None):
        request = Request(
            CODEX_URL,
            data=self._body(messages, tools),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                text = response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            raise RuntimeError(_codex_error_message(error)) from error
        return _parse_sse(text)

    def _body(self, messages, tools=None):
        instructions, input_items = self._input(messages)
        body = {
            "model": self.model,
            "store": False,
            "stream": True,
            "instructions": instructions,
            "input": input_items,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if tools:
            body["tools"] = self._tools(tools)
        return json.dumps(body).encode("utf-8")

    def _headers(self):
        access_token, account_id = self._auth()
        return {
            "Authorization": f"Bearer {access_token}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "miskit",
            "User-Agent": "miskit",
            "accept": "text/event-stream",
            "content-type": "application/json",
        }

    def _auth(self):
        if self.access_token and self.account_id:
            return self.access_token, self.account_id
        try:
            token = get_codex_token()
        except ValueError as error:
            raise ValueError(
                "codex requires provider.accessToken and provider.accountId, "
                "or run: miskit plugin setup codex"
            ) from error
        return token.access, str(token.account_id)

    def _input(self, messages):
        system_parts = []
        input_items = []
        for index, message in enumerate(messages):
            if message.role == "system":
                if isinstance(message.content, str) and message.content:
                    system_parts.append(message.content)

            elif message.role == "user":
                input_items.append({
                    "role": "user",
                    "content": self._user_content(message),
                })
            elif message.role == "assistant":
                input_items += self._assistant_input(message, index)
            elif message.role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": _split_tool_call_id(message.tool_call_id)[0],
                    "output": message.content,
                })
        return "\n\n".join(system_parts), input_items

    def _assistant_input(self, message, index):
        items = []
        if message.content:
            items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": message.content}],
                "status": "completed",
                "id": f"msg_{index}",
            })
        for tool_call in message.tool_calls:
            call_id, item_id = _split_tool_call_id(tool_call.id)
            items.append({
                "type": "function_call",
                "id": item_id or f"fc_{index}",
                "call_id": call_id,
                "name": tool_call.name,
                "arguments": json.dumps(tool_call.arguments),
            })
        return items

    def _user_content(self, message):
        if isinstance(message.content, str):
            return [{"type": "input_text", "text": message.content}]

        items = []
        for part in message.parts:
            if part["type"] == "text":
                items.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image":
                data = base64.b64encode(Path(part["path"]).read_bytes()).decode("ascii")
                items.append({
                    "type": "input_image",
                    "image_url": f"data:{part['mime_type']};base64,{data}",
                })
        return items

    def _tools(self, tools):
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in tools
        ]


def _parse_sse(text):
    content = ""
    tool_calls = []
    usage = {}
    buffers = {}

    for event in _sse_events(text):
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            content += event.get("delta") or ""
        elif event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call" and item.get("call_id"):
                buffers[item["call_id"]] = {
                    "item_id": item.get("id") or "fc_0",
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or "",
                }
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            if call_id in buffers:
                buffers[call_id]["arguments"] += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id in buffers:
                buffers[call_id]["arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            tool_call = _tool_call_from_event(event, buffers)
            if tool_call is not None:
                tool_calls.append(tool_call)
        elif event_type == "response.completed":
            usage = _parse_usage(event)
        elif event_type in ("error", "response.failed"):
            raise RuntimeError(f"Response failed: {event}")

    return content, tool_calls, usage


def _parse_usage(event):
    response = event.get("response") or {}
    raw = response.get("usage") or {}
    if not isinstance(raw, dict):
        return {}
    result = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = raw.get(key)
        if isinstance(value, int):
            result[key] = value
    if result.get("input_tokens") is not None:
        result["prompt_tokens"] = result["input_tokens"]
    return result


def _sse_events(text):
    lines = []
    for line in text.splitlines():
        if line:
            lines.append(line)
            continue
        event = _event_from_lines(lines)
        lines = []
        if event is not None:
            yield event
    event = _event_from_lines(lines)
    if event is not None:
        yield event


def _event_from_lines(lines):
    data = "\n".join(line[5:].strip() for line in lines if line.startswith("data:")).strip()
    if not data or data == "[DONE]":
        return None
    return json.loads(data)


def _tool_call_from_event(event, buffers):
    item = event.get("item") or {}
    if item.get("type") != "function_call":
        return None
    call_id = item.get("call_id")
    if not call_id:
        return None
    buffer = buffers.get(call_id, {})
    raw_arguments = buffer.get("arguments") or item.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        arguments = {}
    item_id = buffer.get("item_id") or item.get("id") or "fc_0"
    name = buffer.get("name") or item.get("name") or ""
    return ToolCall(f"{call_id}|{item_id}", name, arguments)


def _split_tool_call_id(tool_call_id):
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id
        return tool_call_id, None
    return "call_0", None


def _strip_model_prefix(model):
    for prefix in ("codex/", "openai-codex/", "openai_codex/"):
        if model.startswith(prefix):
            return model.split("/", 1)[1]
    return model


def _codex_error_message(error):
    text = error.read().decode("utf-8", errors="replace")
    if error.code == 429:
        return "ChatGPT usage quota exceeded or rate limit triggered. Please try again later."
    return f"HTTP {error.code}: {text}"
