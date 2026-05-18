import asyncio
from datetime import datetime

from miskit.message import Message
from miskit.message import format_message
from miskit.tool_runner import DEFAULT_MAX_TOOL_OUTPUT_CHARS
from miskit.tool_runner import ToolRunner

DEFAULT_MAX_TOOL_ROUNDS = 20
DEFAULT_MAX_INPUT_CHARS = 20_000
_CONTEXT_USAGE_LIMIT = 0.8


def runtime_metadata():
    now = datetime.now().astimezone()
    offset = now.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    timezone = now.tzname() or "local time"
    current_time = f"{now.strftime('%Y-%m-%d %H:%M (%A)')} {timezone}, UTC{offset}"
    return f"[Runtime Metadata]\nCurrent Time: {current_time}\n[/Runtime Metadata]"


class Conversation:
    def __init__(self):
        self.messages = []

    def add(self, message):
        if not isinstance(message, Message):
            raise TypeError("message must be a Message")
        self.messages.append(message)
        return message


class Runner:
    def __init__(
        self,
        model,
        conversation=None,
        memory=None,
        history=None,
        tools=None,
        compactor=None,
        dream=None,
        max_tool_rounds=DEFAULT_MAX_TOOL_ROUNDS,
        max_input_chars=DEFAULT_MAX_INPUT_CHARS,
        truncation_store=None,
        max_tool_output_chars=DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    ):
        self.model = model
        self.memory = memory
        self.history = history
        self.tools = tools or []
        self.compactor = compactor
        self.dream = dream
        self.max_tool_rounds = max_tool_rounds
        self.max_input_chars = max_input_chars
        self.truncation_store = truncation_store
        self.max_tool_output_chars = max_tool_output_chars
        self.tool_runner = ToolRunner(
            self.tools,
            truncation_store=truncation_store,
            max_output_chars=max_tool_output_chars,
        )
        self._last_prompt_tokens = None
        self._dream_requested = False
        self._dream_task = None

        if conversation is None:
            conversation = Conversation()
            if history is not None:
                for message in history.read():
                    conversation.add(message)

        self.conversation = conversation
        self._logged_count = len(self.conversation.messages)
        self._chat_lock = asyncio.Lock()

    async def chat(self, content):
        async with self._chat_lock:
            user_message = self._user_message(content)

            if self._message_too_long(user_message):
                return Message(
                    "assistant",
                    f"Your message is too long ({len(user_message.text):,} characters). "
                    f"The maximum is {self.max_input_chars:,} characters.",
                )

            stored_message = user_message.storage_message()
            self.conversation.add(stored_message)
            archived = await self.compact_if_needed()
            snapshot = len(self.conversation.messages)
            messages = self._messages_with_active_user_message(stored_message, user_message)
            try:
                message = await self.run_turn(messages, log_tools=True)
            except Exception:
                self.rollback_conversation(snapshot)
                raise
            self._last_prompt_tokens = message.usage.get("prompt_tokens")
            self.conversation.add(message)
            self.log_conversation()
            if archived:
                self.request_dream()
            return message

    async def run_turn(self, messages, log_tools=False):
        messages = self._messages_for_model(messages)
        turn_start = len(self.conversation.messages) - 1 if log_tools else 0

        if log_tools and self._context_nearly_full(self._last_prompt_tokens):
            if self.compactor is not None and self.history is not None:
                await self._compact_for_overflow(turn_start)
                messages = self._messages_for_model(list(self.conversation.messages))
            else:
                return Message(
                    "assistant",
                    "I need to stop here because the conversation context is nearly full. "
                    "Please start a new conversation to continue.",
                )

        for _ in range(self.max_tool_rounds):
            message = await self.complete(messages)
            prompt_tokens = message.usage.get("prompt_tokens")

            if message.role != "assistant":
                raise ValueError("model must return an assistant message")

            if not message.tool_calls:
                return message

            if log_tools:
                self.conversation.add(message)

            messages = list(messages) + [message]
            for tool_call in message.tool_calls:
                tool_result = self.run_tool(tool_call)
                messages.append(tool_result)
                if log_tools:
                    self.conversation.add(tool_result)

            if log_tools and self._context_nearly_full(prompt_tokens):
                if self.compactor is not None and self.history is not None:
                    await self._compact_for_overflow(turn_start)
                    messages = self._messages_for_model(list(self.conversation.messages))
                else:
                    return Message(
                        "assistant",
                        "I need to stop here because the conversation context is nearly full. "
                        "Please start a new conversation to continue.",
                    )

        raise ValueError(
            f"model kept asking for tools after {self.max_tool_rounds} rounds "
            "(increase context.maxToolRounds in config.json if you need more)"
        )

    def _messages_for_model(self, messages):
        messages = list(messages)

        if self.memory is not None:
            system_prompt = self.memory.system_prompt()
            if system_prompt:
                messages = [Message("system", system_prompt)] + messages

        metadata = runtime_metadata()
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role == "user":
                messages[index] = message.with_prepended_text(f"{metadata}\n\n")
                return messages

        messages.append(Message("user", metadata))
        return messages

    async def compact_if_needed(self):
        if self.compactor is None or self.history is None:
            return False
        if self._last_prompt_tokens is None:
            return False

        message_count = len(self.conversation.messages)
        if not self.compactor.should_compact(self._last_prompt_tokens, message_count):
            return False

        old_messages, recent_messages = self.compactor.split(self.conversation.messages)
        if not old_messages:
            return False

        await self._replace_with_compacted_history(old_messages, recent_messages)
        return True

    def request_dream(self):
        if self.dream is None:
            return

        self._dream_requested = True
        if self._dream_task is None or self._dream_task.done():
            self._dream_task = asyncio.create_task(self.run_dream_requests())

    async def run_dream_requests(self):
        while self._dream_requested:
            self._dream_requested = False
            try:
                await self.dream.run_once()
            except Exception:
                pass

    async def summarize(self, messages):
        summary = ""
        remaining = list(messages)
        while remaining:
            chunks = self.compactor.split_for_summary(remaining, reserved_chars=len(summary))
            chunk = chunks[0]
            remaining = remaining[len(chunk):]
            prompt = self.compactor.summary_prompt(chunk, prior_summary=summary)
            reply = await self.model.complete([Message("user", prompt)])
            if reply.role != "assistant":
                raise ValueError("model must return an assistant message")
            summary = reply.content
        return summary

    def log_conversation(self):
        if self.history is not None:
            for message in self.conversation.messages[self._logged_count:]:
                self.history.log(message)
            self._logged_count = len(self.conversation.messages)

    def rollback_conversation(self, message_count):
        self.conversation.messages = self.conversation.messages[:message_count]
        self._logged_count = message_count
        if self.history is not None:
            self.history.write(self.conversation.messages)

    async def complete(self, messages):
        if not self.tools:
            return await self.model.complete(messages)

        return await self.model.complete(messages, self.tools)

    def run_tool(self, tool_call):
        return self.tool_runner.run(tool_call)

    def _user_message(self, content):
        if isinstance(content, Message):
            if content.role != "user":
                raise ValueError("chat messages must have role='user'")
            return content
        return Message("user", content)

    def _message_too_long(self, message):
        if self.max_input_chars is None or self.max_input_chars <= 0:
            return False
        return len(message.text) > self.max_input_chars

    def _context_nearly_full(self, prompt_tokens):
        if self.compactor is None or prompt_tokens is None:
            return False
        return prompt_tokens >= self.compactor.context_tokens * _CONTEXT_USAGE_LIMIT

    def _messages_with_active_user_message(self, stored_message, active_message):
        messages = list(self.conversation.messages)
        for index in range(len(messages) - 1, -1, -1):
            if messages[index] is stored_message:
                messages[index] = active_message
                break
        return messages

    async def _compact_for_overflow(self, turn_start):
        prior = self.conversation.messages[:turn_start]
        current_turn = self.conversation.messages[turn_start:]
        if not prior:
            return
        old_messages, recent = self.compactor.split(prior)
        if not old_messages:
            return
        await self._replace_with_compacted_history(old_messages, recent + current_turn)

    async def _replace_with_compacted_history(self, old_messages, tail_messages):
        tail_messages = list(tail_messages)
        compacted_old = self.compactor.compact_messages(old_messages, store=self.truncation_store)
        store_id = self._save_transcript(old_messages)
        summary = await self.summarize(compacted_old)
        self._replace_history([
            self.compactor.summary_message(summary, store_id=store_id),
            *tail_messages,
        ])

    def _save_transcript(self, messages):
        if self.truncation_store is None:
            return None
        transcript = "\n\n".join(format_message(message) for message in messages)
        return self.truncation_store.save(transcript)

    def _replace_history(self, messages):
        self.history.archive()
        self.conversation.messages = list(messages)
        self.history.write(self.conversation.messages)
        self._logged_count = len(self.conversation.messages)
        self._last_prompt_tokens = None
