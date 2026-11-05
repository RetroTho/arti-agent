import asyncio
import http.client
import json
import unittest
from urllib.parse import urlsplit

from plugins.claude.relay import Relay
from tests.providers.fakes import FakeServer, messages_stream, stream_response

MODEL = "claude-opus-5-5[1m]"


def post(url, path_suffix="/v1/messages", headers=None, model=MODEL):
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=5)
    body = json.dumps({"model": model, "messages": []})
    conn.request("POST", parts.path + path_suffix, body, headers or {})
    response = conn.getresponse()
    return response.status, response.read()


class RelayTest(unittest.IsolatedAsyncioTestCase):
    async def test_admits_one_request_then_answers_400(self):
        upstream = stream_response(messages_stream([{"type": "text", "text": "hi"}]))
        async with FakeServer([upstream, upstream]) as server:
            relay = Relay([], server.url)
            await relay.start()
            try:
                first = await asyncio.to_thread(post, relay.url)
                second = await asyncio.to_thread(post, relay.url)
                wrong = await asyncio.to_thread(post, relay.url, "/v1/other")
                origin = await asyncio.to_thread(post, relay.url, headers={"Origin": "https://web"})
            finally:
                await relay.close()
        self.assertEqual(first[0], 200)
        self.assertEqual(relay.reply.text, "hi")
        self.assertEqual(second[0], 400)
        self.assertIn(b"ARTI_MODEL_ADMISSION_CONSUMED", second[1])
        self.assertEqual((wrong[0], origin[0]), (404, 404))
        self.assertEqual((relay.denied, len(server.requests)), (1, 1))


if __name__ == "__main__":
    unittest.main()
