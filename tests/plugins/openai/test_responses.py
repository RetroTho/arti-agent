import json
import os
import unittest
from unittest import mock

from arti.providers import ProviderError, Reply, carrier, finish
from plugins.openai import API_OPTIONS
from plugins.openai.responses import OpenAIProvider, ResponsesStream, rate_info, wire_input
from tests.providers.fakes import FakeServer, sse, stream_response

IMAGE = {"type": "image", "media_type": "image/png", "data": "AAAA"}
CALL = {"type": "tool_use", "id": "c1", "name": "terminal", "input": {"command": "ls"}}
TOOLS = [{"name": "terminal", "description": "run", "input_schema": {"type": "object"}}]


def history():
    return [
        {"role": "user", "content": [{"type": "text", "text": "hi", "unknown": True}, IMAGE]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}, CALL]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": [], "is_error": True},
                {"type": "text", "text": "steer"},
            ],
        },
    ]


def completed(output, status="completed", **extra):
    usage = {"input_tokens": 20, "output_tokens": 4, "input_tokens_details": {"cached_tokens": 8}}
    return {
        "type": f"response.{status}",
        "response": {"status": status, "output": output, "usage": usage, **extra},
    }


MESSAGE = {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "hello"}],
}
FUNCTION = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "c9",
    "name": "terminal",
    "arguments": '{"command":"pwd"}',
}
REASONING = {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"}


class RateInfoTest(unittest.TestCase):
    def test_api_limits_and_chatgpt_windows(self):
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "75",
            "x-codex-primary-used-percent": "40",
            "x-codex-primary-reset-after-seconds": "60",
            "x-codex-secondary-used-percent": "0",
        }
        self.assertEqual(
            rate_info(headers, now=1000.0),
            {
                "windows": {
                    "requests": {"used": 0.25},
                    "primary": {"used": 0.4, "resets_at": 1060.0},
                    "secondary": {"used": 0.0},
                }
            },
        )
        self.assertEqual(rate_info({}), {})


class WireInputTest(unittest.TestCase):
    def test_items(self):
        items = wire_input(history(), "openai", "m")
        self.assertEqual(items[0]["content"][1]["image_url"], "data:image/png;base64,AAAA")
        self.assertEqual(items[1]["content"][0]["type"], "output_text")
        self.assertEqual(
            items[2],
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "terminal",
                "arguments": '{"command": "ls"}',
            },
        )
        output = items[3]["output"]
        self.assertEqual([p["text"] for p in output], ["Tool error:", "(no output)"])
        self.assertEqual(items[4]["content"], [{"type": "input_text", "text": "steer"}])

    def test_carrier_replay(self):
        native = carrier("openai", "m", [{"type": "text", "text": "x"}], [REASONING])
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "x"}],
            "native": native,
        }
        self.assertEqual(wire_input([message], "openai", "m"), [REASONING])
        self.assertEqual(wire_input([message], "openai-codex", "m")[0]["type"], "message")


class StreamTest(unittest.TestCase):
    def test_completed_with_call(self):
        stream = ResponsesStream()
        reply = stream.feed(
            completed(
                [
                    REASONING,
                    {"type": "reasoning", "id": "rs_tmp_2", "encrypted_content": "e"},
                    MESSAGE,
                    FUNCTION,
                ]
            )
        )
        self.assertEqual(reply.stop_reason, "tool_use")
        self.assertEqual(
            reply.tool_calls,
            [{"type": "tool_use", "id": "c9", "name": "terminal", "input": {"command": "pwd"}}],
        )
        self.assertEqual(reply.usage, {"input_tokens": 20, "output_tokens": 4, "cached_tokens": 8})
        self.assertEqual(
            [i["type"] for i in stream.items], ["reasoning", "message", "function_call"]
        )
        self.assertTrue(all("id" not in i for i in stream.items))

    def test_empty_snapshot_uses_streamed_items(self):
        stream = ResponsesStream()
        stream.feed({"type": "response.output_item.done", "output_index": 0, "item": MESSAGE})
        self.assertEqual(stream.feed(completed([])).text, "hello")

    def test_incomplete_and_refusal(self):
        reply = ResponsesStream().feed(
            completed([FUNCTION], "incomplete", incomplete_details={"reason": "content_filter"})
        )
        self.assertEqual(reply.stop_reason, "refusal")
        refusal = {**MESSAGE, "content": [{"type": "refusal", "refusal": "no"}]}
        self.assertEqual(ResponsesStream().feed(completed([refusal])).stop_reason, "refusal")

    def test_failed_event_is_classified_from_its_error_only(self):
        event = {
            "type": "response.failed",
            "response": {
                "error": {"code": "rate_limit_exceeded", "message": "slow"},
                "instructions": "secret",
            },
        }
        with self.assertRaises(ProviderError) as caught:
            ResponsesStream().feed(event)
        self.assertEqual(caught.exception.code, "rate_limit")
        self.assertNotIn("secret", caught.exception.message)

    def test_mid_stream_server_errors_are_retryable(self):
        events = [
            {"type": "response.failed", "response": {"error": {"code": "server_error"}}},
            {"type": "error", "code": "internal_error", "message": "oops"},
            {"type": "error", "error": {"type": "timeout", "message": "slow"}},
        ]
        for event in events:
            with self.assertRaises(ProviderError) as caught:
                ResponsesStream().feed(event)
            self.assertIn(caught.exception.code, ("server", "timeout"), event)
            self.assertTrue(caught.exception.retryable, event)

    def test_a_cut_off_reply_never_keeps_a_carrier(self):
        for stop in ("length", "refusal"):
            reply = Reply([{"type": "text", "text": "half"}], stop)
            self.assertEqual(finish(reply, "openai", "m", [MESSAGE]).native, {})
        reply = finish(Reply([{"type": "text", "text": "whole"}]), "openai", "m", [MESSAGE])
        self.assertEqual(reply.native["items"], [MESSAGE])


class ProviderTest(unittest.IsolatedAsyncioTestCase):
    async def test_end_to_end(self):
        raw = stream_response(sse({"type": "response.created"}, completed([MESSAGE])))
        async with FakeServer([raw]) as server:
            options = {**API_OPTIONS, "base_url": server.url, "max_output_tokens": 99}
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-test"}):
                reply = await OpenAIProvider(options).complete(
                    history(),
                    system="sys",
                    tools=TOOLS,
                    model="gpt-x",
                    effort="low",
                    session_id="s1",
                )
        self.assertEqual(reply.text, "hello")
        request = server.requests[0]
        self.assertEqual(request["path"], "/responses")
        self.assertEqual(request["headers"]["authorization"], "Bearer sk-openai-test")
        body = json.loads(request["body"])
        self.assertEqual(
            (body["store"], body["reasoning"]["effort"], body["max_output_tokens"]),
            (False, "low", 99),
        )
        self.assertEqual(len(body["prompt_cache_key"]), 64)
        self.assertEqual(body["tools"][0]["parameters"], {"type": "object"})
        self.assertEqual((body["tool_choice"], body["parallel_tool_calls"]), ("auto", True))

    async def test_stream_without_terminal_event_is_dropped(self):
        raw = stream_response(sse({"type": "response.created"}))
        async with FakeServer([raw]) as server:
            options = {**API_OPTIONS, "base_url": server.url}
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
                with self.assertRaises(ProviderError) as caught:
                    await OpenAIProvider(options).complete(
                        history(), system="", tools=[], model="m", effort="low", session_id=""
                    )
        self.assertTrue(caught.exception.retryable)
        body = json.loads(server.requests[0]["body"])
        self.assertFalse({"tools", "tool_choice", "parallel_tool_calls"} & set(body))  # no tools


if __name__ == "__main__":
    unittest.main()
