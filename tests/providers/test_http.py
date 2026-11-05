import asyncio
import socket
import unittest
from unittest import mock

from arti.providers import ProviderError, http
from tests.providers.fakes import FakeServer, response, sse, stream_response


async def collect(chunks):
    return b"".join([c async for c in chunks])


async def events_of(pieces):
    async def chunks():
        for piece in pieces:
            yield piece

    return [e async for e in http.sse(chunks())]


class BodyTest(unittest.IsolatedAsyncioTestCase):
    async def fetch(self, answer):
        async with FakeServer([answer]) as server:
            resp = await http.request(server.url + "/x?q=1", headers={"A": "b"}, body=b"hi")
            try:
                return resp, await collect(resp.chunks()), server.requests[0]
            finally:
                resp.close()

    async def test_keepalive_is_on(self):
        async with FakeServer([response(200, b"")]) as server:
            resp = await http.request(server.url, headers={}, body=b"")
            try:
                sock = resp.writer.get_extra_info("socket")
                self.assertTrue(sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE))
            finally:
                resp.close()

    async def test_content_length_body_and_request_shape(self):
        resp, body, request = await self.fetch(response(201, b"hello", {"X-Y": "z"}))
        self.assertEqual((resp.status, body, resp.headers["x-y"]), (201, b"hello", "z"))
        self.assertEqual((request["path"], request["body"]), ("/x?q=1", b"hi"))
        self.assertEqual(request["headers"]["accept-encoding"], "identity")
        self.assertEqual(request["headers"]["connection"], "close")

    async def test_chunked_body(self):
        chunked = b"5;ext=1\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
        raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + chunked
        _, body, _ = await self.fetch(raw)
        self.assertEqual(body, b"hello world")

    async def test_close_delimited_body(self):
        async def answer(request, writer):
            writer.write(b"HTTP/1.1 200 OK\r\n\r\npart one, ")
            await writer.drain()
            writer.write(b"part two")
            writer.close()

        _, body, _ = await self.fetch(answer)
        self.assertEqual(body, b"part one, part two")

    async def test_eof_mid_body_is_a_retryable_drop(self):
        async def answer(request, writer):
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort")
            writer.close()

        with self.assertRaises(ProviderError) as caught:
            await self.fetch(answer)
        self.assertEqual((caught.exception.code, caught.exception.retryable), ("server", True))

    async def test_idle_stream_times_out(self):
        async def answer(request, writer):
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nstart")
            await writer.drain()  # then stalls: the server waits for the client to hang up

        with mock.patch.object(http, "IDLE", 0.2), self.assertRaises(ProviderError) as caught:
            await self.fetch(answer)
        self.assertTrue(caught.exception.retryable)
        self.assertIn("silent", caught.exception.message)

    async def test_cancellation_closes_the_connection(self):
        async def answer(request, writer):
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nstart")
            await writer.drain()  # then stalls: the server waits for the client to hang up

        async with FakeServer([answer]) as server:
            resp = await http.request(server.url, headers={})
            task = asyncio.create_task(collect(resp.chunks()))
            await asyncio.sleep(0.1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            resp.close()
            await asyncio.wait_for(server.closed.wait(), 2)


class SSETest(unittest.IsolatedAsyncioTestCase):
    async def test_events_split_across_chunks_and_crlf(self):
        raw = b'event: ping\r\ndata: {}\r\n\r\n: comment\n\ndata: {"type": "a",\ndata: "n": 1}\n\n'
        raw += b"data: [DONE]\n\n"
        pieces = [raw[i : i + 3] for i in range(0, len(raw), 3)]
        self.assertEqual(await events_of(pieces), [{"type": "ping"}, {"type": "a", "n": 1}])

    async def test_oversized_and_invalid_events(self):
        with self.assertRaises(ProviderError):
            await events_of([b"data: not json\n\n"])

        async def chunks():
            yield b"data: " + b"x" * 100

        with mock.patch.object(http, "EVENT_LIMIT", 50), self.assertRaises(ProviderError):
            [e async for e in http.sse(chunks())]


class StreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_stream_yields_events_and_rate_limits(self):
        body = sse({"type": "a"}, {"type": "b"})
        raw = stream_response(body).replace(b"\r\n\r\n", b"\r\nretry-after: 3\r\n\r\n", 1)
        async with FakeServer([raw]) as server:
            async with http.stream(server.url, {}, {"k": 1}) as (events, rate):
                got = [e async for e in events]
        self.assertEqual([e["type"] for e in got], ["a", "b"])
        self.assertEqual(rate["retry_after"], 3)

    async def test_the_callers_rate_parser_adds_its_own_headers(self):
        raw = stream_response(sse({"type": "a"})).replace(
            b"\r\n\r\n", b"\r\nretry-after-ms: 1500\r\nx-used: 0.5\r\n\r\n", 1
        )

        def rate(headers):
            return {"windows": {"day": {"used": float(headers["x-used"])}}}

        async with FakeServer([raw]) as server:
            async with http.stream(server.url, {}, {}, rate) as (events, info):
                [e async for e in events]
        self.assertEqual(info, {"retry_after": 1.5, "windows": {"day": {"used": 0.5}}})

    async def test_error_status_is_classified_with_the_api_message(self):
        error = b'{"type":"error","error":{"type":"overloaded_error","message":"busy"}}'
        async with FakeServer([response(529, error)]) as server:
            with self.assertRaises(ProviderError) as caught:
                async with http.stream(server.url, {}, {}):
                    pass
        self.assertEqual(caught.exception.code, "server")
        self.assertTrue(caught.exception.retryable)
        self.assertIn("busy", caught.exception.message)

    async def test_plain_http_only_to_loopback(self):
        for bad in ("http://example.com/x", "ftp://127.0.0.1/x", "https:///x"):
            with self.assertRaises(ProviderError):
                await http.request(bad, headers={})

    async def test_userinfo_never_reaches_the_host_header(self):
        async with FakeServer([response(200)]) as server:
            url = server.url.replace("://", "://user:pw@")
            (await http.request(url, headers={})).close()
        self.assertNotIn("pw", server.requests[0]["headers"]["host"])

    async def test_header_injection_is_refused(self):
        async with FakeServer([response(200)]) as server:
            with self.assertRaises(ProviderError):
                await http.request(server.url, headers={"X": "a\r\nEvil: 1"})


if __name__ == "__main__":
    unittest.main()
