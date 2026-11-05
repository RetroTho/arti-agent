import json
import os
import unittest
from unittest import mock

from arti.providers import ProviderError, carrier
from plugins.anthropic import OPTIONS as API_OPTIONS
from plugins.anthropic.messages import AnthropicProvider, Assembler, rate_info, wire_history
from tests.providers.fakes import FakeServer, messages_stream, response, sse, stream_response

IMAGE = {"type": "image", "media_type": "image/png", "data": "AAAA"}
CALL = {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}}


def history():
    return [
        {"role": "user", "content": [{"type": "text", "text": "hi", "unknown": True}, IMAGE]},
        {"role": "assistant", "content": [{"type": "text", "text": "  "}, CALL]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "steer"},
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "out"}, IMAGE],
                    "is_error": False,
                },
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": []}]},
    ]


class RateInfoTest(unittest.TestCase):
    def test_unified_windows_status_and_api_limits(self):
        headers = {
            "anthropic-ratelimit-unified-5h-utilization": "0.25",
            "anthropic-ratelimit-unified-5h-reset": "1800000000",
            "anthropic-ratelimit-unified-7d-utilization": "nan",
            "anthropic-ratelimit-unified-status": "allowed_warning",
            "anthropic-ratelimit-tokens-limit": "1000",
            "anthropic-ratelimit-tokens-remaining": "250",
            "retry-after": "5",  # the standard header is the core's (providers.retry_info)
        }
        self.assertEqual(
            rate_info(headers),
            {
                "windows": {
                    "5h": {"used": 0.25, "resets_at": 1800000000.0},
                    "tokens": {"used": 0.75},
                },
                "status": "allowed_warning",
            },
        )
        self.assertEqual(rate_info({"anthropic-ratelimit-unified-status": "Bad Value"}), {})


class WireHistoryTest(unittest.TestCase):
    def test_api_fields_only_merged_and_results_first(self):
        turns = wire_history(history(), "anthropic", "m")
        self.assertEqual([t["role"] for t in turns], ["user", "assistant", "user"])
        self.assertEqual(turns[0]["content"][0], {"type": "text", "text": "hi"})
        self.assertEqual(turns[0]["content"][1]["source"]["data"], "AAAA")
        self.assertEqual(turns[1]["content"], [CALL])  # blank text dropped
        kinds = [b["type"] for b in turns[2]["content"]]
        self.assertEqual(kinds, ["tool_result", "tool_result", "text"])
        result = turns[2]["content"][0]
        self.assertEqual(result["content"][1]["type"], "image")
        self.assertEqual(
            turns[2]["content"][1]["content"], [{"type": "text", "text": "(no output)"}]
        )

    def test_carrier_replays_only_while_unedited(self):
        content = [{"type": "text", "text": "answer"}]
        thinking = {"type": "thinking", "thinking": "hm", "signature": "sig"}
        native = carrier("anthropic", "m", content, [thinking, *content])
        message = {"role": "assistant", "content": content, "native": native}
        turns = wire_history(
            [{"role": "user", "content": [{"type": "text", "text": "q"}]}, message],
            "anthropic",
            "m",
        )
        self.assertEqual(turns[1]["content"][0], thinking)
        self.assertEqual(wire_history([message], "anthropic", "other")[0]["content"], content)
        edited = {**message, "content": [{"type": "text", "text": "edited"}]}
        self.assertEqual(
            wire_history([edited], "anthropic", "m")[0]["content"][0]["text"], "edited"
        )


def feed_all(raw: bytes):
    assembler = Assembler()
    for block in raw.split(b"\n\n"):
        for line in block.split(b"\n"):
            if line.startswith(b"data: "):
                if (reply := assembler.feed(json.loads(line[6:]))) is not None:
                    return reply, assembler
    return None, assembler


class AssemblerTest(unittest.TestCase):
    def test_text_thinking_and_tool_call(self):
        thinking = {"type": "thinking", "thinking": "plan", "signature": "s"}
        reply, assembler = feed_all(
            messages_stream([thinking, {"type": "text", "text": "ok"}, CALL], "tool_use")
        )
        self.assertEqual(reply.content, [{"type": "text", "text": "ok"}, CALL])
        self.assertEqual(reply.stop_reason, "tool_use")
        self.assertEqual(reply.usage, {"input_tokens": 13, "output_tokens": 5, "cached_tokens": 3})
        self.assertEqual(assembler.blocks[0]["signature"], "s")

    def test_refusal_uses_the_explanation_and_drops_calls(self):
        details = {"explanation": "Not able to help with that."}
        reply, _ = feed_all(messages_stream([CALL], "refusal", stop_details=details))
        self.assertEqual(reply.stop_reason, "refusal")
        self.assertEqual(reply.content, [{"type": "text", "text": "Not able to help with that."}])

    def test_refusal_without_explanation_names_its_category(self):
        reply, _ = feed_all(messages_stream([], "refusal", stop_details={"category": "cyber"}))
        self.assertIn("category: cyber", reply.text)

    def test_error_event_is_classified(self):
        with self.assertRaises(ProviderError) as caught:
            Assembler().feed(
                {"type": "error", "error": {"type": "overloaded_error", "message": "x"}}
            )
        self.assertEqual(caught.exception.code, "server")
        self.assertTrue(caught.exception.retryable)


class ProviderTest(unittest.IsolatedAsyncioTestCase):
    async def complete(self, answers, **kw):
        async with FakeServer(answers) as server:
            options = {**API_OPTIONS, "base_url": server.url}
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
                reply = await AnthropicProvider(options).complete(
                    history(),
                    system="sys",
                    tools=[
                        {
                            "name": "read_file",
                            "description": "r",
                            "input_schema": {"type": "object"},
                        }
                    ],
                    model="m",
                    effort="high",
                    session_id="s",
                    **kw,
                )
            return reply, server.requests

    async def test_request_and_reply(self):
        raw = stream_response(messages_stream([{"type": "text", "text": "done"}]))
        reply, requests = await self.complete([raw])
        self.assertEqual(reply.text, "done")
        self.assertEqual(reply.native["provider"], "anthropic")
        sent = json.loads(requests[0]["body"])
        self.assertEqual(requests[0]["headers"]["x-api-key"], "sk-test-key")
        self.assertEqual(requests[0]["path"], "/messages")
        self.assertEqual(sent["output_config"], {"effort": "high"})
        self.assertEqual(sent["system"][0]["cache_control"], {"type": "ephemeral"})
        marked = [b for t in sent["messages"] for b in t["content"] if "cache_control" in b]
        self.assertEqual(len(marked), 2)  # end of history + previous user turn

    async def test_truncated_stream_is_a_retryable_error(self):
        whole = messages_stream([CALL], "tool_use")
        cut = whole[: whole.index(b"message_delta") - 20]
        with self.assertRaises(ProviderError) as caught:
            await self.complete([stream_response(cut)])
        self.assertEqual((caught.exception.code, caught.exception.retryable), ("server", True))

    async def test_cut_off_reply_runs_no_tools_and_keeps_no_carrier(self):
        raw = stream_response(messages_stream([CALL], "max_tokens"))
        reply, _ = await self.complete([raw])
        self.assertEqual(reply.stop_reason, "length")
        self.assertEqual(reply.tool_calls, [])
        self.assertEqual(reply.native, {})
        self.assertTrue(reply.text)

    async def test_context_length_error(self):
        error = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "prompt is too long: 1 > 0"},
        }
        with self.assertRaises(ProviderError) as caught:
            await self.complete([response(400, json.dumps(error).encode())])
        self.assertEqual(caught.exception.code, "context_length")
        self.assertIn("prompt is too long", caught.exception.message)

    async def test_missing_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaises(ProviderError) as caught:
                await AnthropicProvider(API_OPTIONS).complete(
                    [], system="", tools=[], model="m", effort="low", session_id=""
                )
        self.assertEqual(caught.exception.code, "auth")

    async def test_mid_stream_error_event(self):
        raw = stream_response(
            sse(
                {"type": "message_start", "message": {"usage": {}}},
                {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
            )
        )
        with self.assertRaises(ProviderError) as caught:
            await self.complete([raw])
        self.assertTrue(caught.exception.retryable)


if __name__ == "__main__":
    unittest.main()
