import asyncio
import base64
import contextlib
import fcntl
import hashlib
import io
import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from arti.providers import ProviderError
from plugins.openai import CHATGPT_OPTIONS, codex
from plugins.openai.responses import CodexProvider
from plugins.safety import redact
from tests.providers.fakes import FakeServer, json_response, response, sse, stream_response


def jwt(**claims) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"h.{payload}.s"


class AuthFiles:
    """A temporary data dir with an optional saved login; tokens seen go to `self.secrets`."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.secrets = []

    def auth(self, url="http://127.0.0.1:9"):
        return codex.CodexAuth(self.dir, self.secrets.extend, issuer=url, base_url=url)

    def save(self, access="old-access", refresh="old-refresh", expires_in=3600):
        grant = {
            "access_token": access,
            "refresh_token": refresh,
            "expires_at": time.time() + expires_in,
        }
        (self.dir / "codex_auth.json").write_text(json.dumps(grant))

    def stored(self):
        return json.loads((self.dir / "codex_auth.json").read_text())


class CodexAuthTest(AuthFiles, unittest.IsolatedAsyncioTestCase):
    async def test_fresh_token_is_used_without_a_request(self):
        self.save()
        self.assertEqual(await self.auth().token(), "old-access")
        self.assertIn("old-refresh", self.secrets)

    async def test_expiring_token_is_refreshed_and_saved_privately(self):
        self.save(expires_in=30)
        new = jwt(exp=time.time() + 600)
        async with FakeServer(
            [json_response(200, {"access_token": new, "refresh_token": "r2"})]
        ) as server:
            token = await self.auth(server.url).token()
        self.assertEqual(token, new)
        self.assertEqual(self.stored()["refresh_token"], "r2")
        self.assertIn(b"refresh_token=old-refresh", server.requests[0]["body"])
        mode = stat.S_IMODE(os.stat(self.dir / "codex_auth.json").st_mode)
        self.assertEqual(mode, 0o600)
        self.assertIn("r2", self.secrets)

    async def test_a_token_a_peer_already_rotated_is_adopted(self):
        self.save(access="peer-rotated")  # on disk now; the caller was refused the old one
        async with FakeServer([]) as server:
            token = await self.auth(server.url).token(rejected="old-access")
        self.assertEqual(token, "peer-rotated")
        self.assertEqual(server.requests, [])

    async def test_rejected_token_is_refreshed(self):
        self.save()
        async with FakeServer([json_response(200, {"access_token": "a2"})]) as server:
            token = await self.auth(server.url).token(rejected="old-access")
        self.assertEqual(token, "a2")
        self.assertEqual(self.stored()["refresh_token"], "old-refresh")  # kept when not rotated

    async def test_invalid_grant_asks_for_a_new_login_and_keeps_the_file(self):
        self.save(expires_in=0)
        before = self.stored()
        async with FakeServer([json_response(400, {"error": "invalid_grant"})]) as server:
            with self.assertRaises(ProviderError) as caught:
                await self.auth(server.url).token()
        self.assertEqual(caught.exception.code, "auth")
        self.assertIn("python -m arti login", caught.exception.message)
        self.assertEqual(self.stored(), before)

    async def test_server_failure_is_retryable_and_keeps_the_file(self):
        self.save(expires_in=0)
        before = self.stored()
        async with FakeServer([response(503)]) as server:
            with self.assertRaises(ProviderError) as caught:
                await self.auth(server.url).token()
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(self.stored(), before)

    async def test_shared_auth_registers_tokens_for_redaction_by_default(self):
        # image_generate may create the shared login before the model route
        self.save(access="access-token-to-hide", refresh="refresh-token-to-hide")
        await codex.shared_auth(self.dir).token()
        text = redact.redact("access-token-to-hide refresh-token-to-hide")
        self.assertNotIn("to-hide", text)

    async def test_no_login(self):
        with self.assertRaises(ProviderError) as caught:
            await self.auth().token()
        self.assertEqual(caught.exception.code, "auth")

    def test_headers_from_claims(self):
        claims = {
            codex._AUTH_CLAIM: {
                "chatgpt_account_id": "acc-1",
                "chatgpt_data_residency": "bad value\r\n",
            }
        }
        headers = codex.headers(jwt(**claims))
        self.assertEqual(headers["ChatGPT-Account-ID"], "acc-1")
        self.assertNotIn("x-openai-internal-codex-residency", headers)
        self.assertEqual(headers["originator"], "arti")

    async def test_status_and_logout(self):
        auth = self.auth()
        self.assertFalse(auth.status()["logged_in"])
        self.save()
        self.assertTrue(auth.status()["logged_in"])
        auth.logout()
        self.assertFalse(auth.status()["logged_in"])
        (self.dir / "codex_auth.json").write_text('{"access_token": "a"}')
        status = auth.status()
        self.assertFalse(status["logged_in"])
        self.assertIn("damaged", status["error"])

    async def test_provider_refreshes_once_on_401(self):
        self.save()
        done = {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi"}],
                    }
                ],
            },
        }
        answers = [
            response(401, b'{"error":{"code":"invalid_token"}}'),
            json_response(200, {"access_token": "fresh"}),
            stream_response(sse(done)),
        ]
        async with FakeServer(answers) as server:
            options = {**CHATGPT_OPTIONS, "base_url": server.url}
            provider = CodexProvider(options, self.auth(server.url))
            reply = await provider.complete(
                [{"role": "user", "content": [{"type": "text", "text": "q"}]}],
                system="s",
                tools=[],
                model="m",
                effort="low",
                session_id="x",
            )
        self.assertEqual(reply.text, "hi")
        auths = [r["headers"].get("authorization") for r in server.requests]
        self.assertEqual(auths, ["Bearer old-access", None, "Bearer fresh"])
        self.assertEqual(server.requests[2]["path"], "/responses")

    async def test_provider_does_not_refresh_on_403(self):
        self.save()
        async with FakeServer([response(403, b'{"error":{"code":"forbidden"}}')]) as server:
            options = {**CHATGPT_OPTIONS, "base_url": server.url}
            provider = CodexProvider(options, self.auth(server.url))
            with self.assertRaises(ProviderError) as caught:
                await provider.complete(
                    [{"role": "user", "content": [{"type": "text", "text": "q"}]}],
                    system="s",
                    tools=[],
                    model="m",
                    effort="low",
                    session_id="x",
                )
        self.assertEqual(caught.exception.code, "auth")
        self.assertEqual(len(server.requests), 1)  # no refresh, no second send


class ImageTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dir = Path(tempfile.mkdtemp())
        grant = {"access_token": "tok", "refresh_token": "r", "expires_at": time.time() + 3600}
        (self.dir / "codex_auth.json").write_text(json.dumps(grant))

    async def test_image_is_returned(self):
        data = {"data": [{"b64_json": base64.b64encode(b"PNGDATA").decode()}]}
        async with FakeServer([json_response(200, data)]) as server:
            auth = codex.CodexAuth(self.dir, base_url=server.url)
            image = await codex.generate_image(auth, "a cat", "1024x1024", "low", [b"\x89PNG...."])
        self.assertEqual(image, b"PNGDATA")
        self.assertEqual(server.requests[0]["path"], "/images/edits")
        body = json.loads(server.requests[0]["body"])
        self.assertTrue(body["images"][0]["image_url"].startswith("data:image/png;base64,"))

    async def test_failed_image_request_is_never_retried(self):
        async with FakeServer(
            [response(500, b"echo of the prompt"), json_response(200, {})]
        ) as server:
            auth = codex.CodexAuth(self.dir, base_url=server.url)
            with self.assertRaises(codex.ImageError) as caught:
                await codex.generate_image(auth, "a cat", "1024x1024", "low")
        self.assertEqual(len(server.requests), 1)
        self.assertTrue(caught.exception.sent)
        self.assertNotIn("echo", str(caught.exception))


USER = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
DONE = {
    "type": "response.completed",
    "response": {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            }
        ],
    },
}


@contextlib.asynccontextmanager
async def fast_sleeps():
    """asyncio.sleep without the wait (the device login polls every few seconds)."""
    delays, real = [], asyncio.sleep

    async def sleep(delay, *args):
        delays.append(delay)
        await real(0)

    with mock.patch.object(codex.asyncio, "sleep", sleep):
        yield delays


class RefreshTest(AuthFiles, unittest.IsolatedAsyncioTestCase):
    """Refresh outcomes, the locks, and the provider's handling of refused requests."""

    async def refused(self, answers, url=None):
        self.save(expires_in=0)
        before = self.stored()
        async with FakeServer(answers) as server:
            with self.assertRaises(ProviderError) as caught:
                await self.auth(url or server.url).token()
        self.assertEqual(self.stored(), before)  # the file is left alone
        return caught.exception, server

    async def test_rate_limited_refresh_is_retryable(self):
        error, _ = await self.refused([response(429)])
        self.assertEqual((error.code, error.retryable), ("server", True))
        self.assertIn("HTTP 429", error.message)

    async def test_reused_or_revoked_refresh_token_asks_for_a_login(self):
        for status in (401, 403):
            with self.subTest(status):
                error, _ = await self.refused([json_response(status, {"error": "x"})])
                self.assertEqual((error.code, error.message), ("auth", codex.RELOGIN))
                self.assertFalse(error.retryable)

    async def test_an_odd_refresh_status_is_a_bad_response(self):
        error, _ = await self.refused([response(302, b"", {"Location": "/elsewhere"})])
        self.assertEqual(error.code, "bad_response")
        self.assertIn("HTTP 302", error.message)

    async def test_a_refresh_reply_without_tokens_is_a_bad_response(self):
        for body in (b"not json", b"[1, 2]", b'{"refresh_token": "r2"}', b'{"access_token": 5}'):
            with self.subTest(body):
                error, _ = await self.refused([response(200, body)])
                self.assertEqual(error.code, "bad_response")

    async def test_a_refresh_that_times_out_is_retryable(self):
        self.save(expires_in=0)
        with mock.patch.object(codex.http, "post", side_effect=TimeoutError):
            with self.assertRaises(ProviderError) as caught:
                await self.auth().token()
        self.assertEqual((caught.exception.code, caught.exception.retryable), ("timeout", True))

    async def test_an_unreachable_login_server_keeps_the_file(self):
        error, _ = await self.refused([], url="http://127.0.0.1:9")  # nothing listens there
        self.assertTrue(error.retryable)

    async def test_the_expiry_is_the_earlier_of_the_jwt_and_expires_in(self):
        self.save(expires_in=0)
        late = jwt(exp=time.time() + 7200)
        answer = json_response(200, {"access_token": late, "expires_in": 600})
        async with FakeServer([answer]) as server:
            self.assertEqual(await self.auth(server.url).token(), late)
        self.assertAlmostEqual(self.stored()["expires_at"], time.time() + 600, delta=5)
        self.save(expires_in=0)
        async with FakeServer([json_response(200, {"access_token": "opaque"})]) as server:
            await self.auth(server.url).token()  # no exp claim, no expires_in: an hour
        self.assertAlmostEqual(self.stored()["expires_at"], time.time() + 3600, delta=5)

    async def test_concurrent_callers_share_one_refresh(self):
        self.save(expires_in=0)
        answer = json_response(200, {"access_token": "a2", "refresh_token": "r2"})
        async with FakeServer([answer]) as server:
            auth = self.auth(server.url)
            tokens = await asyncio.gather(auth.token(), auth.token(), auth.token())
        self.assertEqual(tokens, ["a2", "a2", "a2"])
        self.assertEqual(len(server.requests), 1)  # the others re-read the rotated token

    async def test_another_process_rotating_first_is_adopted_under_the_flock(self):
        # Two CodexAuth objects on one data dir stand for two processes: no shared asyncio
        # lock, only the file lock. The second waits, re-reads, and never sends "old-refresh".
        self.save(expires_in=0)
        release = asyncio.Event()

        async def slow(request, writer):
            await release.wait()
            writer.write(json_response(200, {"access_token": "a2", "refresh_token": "r2"}))
            await writer.drain()

        async with FakeServer([slow]) as server:
            first = asyncio.create_task(self.auth(server.url).token())
            while not server.requests:
                await asyncio.sleep(0.01)
            second = asyncio.create_task(self.auth(server.url).token())
            await asyncio.sleep(0.05)
            self.assertFalse(second.done())  # held at the flock
            release.set()
            self.assertEqual(await asyncio.gather(first, second), ["a2", "a2"])
        self.assertEqual(len(server.requests), 1)
        self.assertEqual(self.stored()["refresh_token"], "r2")

    async def test_a_cancelled_caller_still_saves_the_rotated_token(self):
        self.save(expires_in=0)
        release = asyncio.Event()

        async def slow(request, writer):
            await release.wait()
            writer.write(json_response(200, {"access_token": "a2", "refresh_token": "r2"}))
            await writer.drain()

        async with FakeServer([slow]) as server:
            auth = self.auth(server.url)
            caller = asyncio.create_task(auth.token())
            while not server.requests:
                await asyncio.sleep(0.01)
            caller.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await caller
            self.assertEqual(await auth.token(), "a2")  # saved by the shielded refresh
        self.assertEqual(len(server.requests), 1)
        self.assertEqual(self.stored()["refresh_token"], "r2")

    async def test_a_lock_held_too_long_is_a_server_error(self):
        fd = os.open(self.dir / "codex_auth.lock", os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self.save()
        with mock.patch.object(codex, "LOCK_WAIT", 0.15):
            with self.assertRaises(ProviderError) as caught:
                await self.auth().token()
            self.assertEqual(caught.exception.code, "server")
            self.assertIn("locked by another process", caught.exception.message)
            with self.assertRaises(ProviderError):
                self.auth().logout()
        self.assertTrue((self.dir / "codex_auth.json").exists())  # not logged out

    async def test_damaged_logins(self):
        path = self.dir / "codex_auth.json"
        for text in (
            "{not json",
            "[]",
            '{"access_token": "a", "refresh_token": "r"}',
            '{"access_token": "a", "refresh_token": "r", "expires_at": "soon"}',
        ):
            with self.subTest(text):
                path.write_text(text)
                status = self.auth().status()
                self.assertFalse(status["logged_in"])
                self.assertIn("damaged", status["error"])
                with self.assertRaises(ProviderError) as caught:
                    await self.auth().token()
                self.assertEqual(caught.exception.code, "auth")

    async def test_status_reports_the_expiry_and_logout_twice_is_fine(self):
        self.save()
        status = self.auth().status()
        self.assertEqual(status, {"logged_in": True, "expires_at": self.stored()["expires_at"]})
        self.auth().logout()
        self.auth().logout()  # nothing to delete
        self.assertEqual(self.auth().status(), {"logged_in": False, "expires_at": None})

    async def complete(self, answers):
        async with FakeServer(answers) as server:
            options = {**CHATGPT_OPTIONS, "base_url": server.url}
            provider = CodexProvider(options, self.auth(server.url))
            try:
                return await provider.complete(
                    USER, system="s", tools=[], model="m", effort="low", session_id="x"
                ), server
            except ProviderError as exc:
                return exc, server

    async def test_provider_classifies_rate_limits_and_server_errors_without_refreshing(self):
        self.save()
        for status, code in ((429, "rate_limit"), (500, "server"), (503, "server")):
            with self.subTest(status):
                error, server = await self.complete([response(status, b"{}")])
                self.assertIsInstance(error, ProviderError)
                self.assertEqual((error.code, error.retryable, error.status), (code, True, status))
                self.assertEqual(len(server.requests), 1)  # one send, no refresh
                self.assertEqual(self.stored()["access_token"], "old-access")

    async def test_provider_refreshes_only_once(self):
        self.save()
        answers = [
            response(401, b"{}"),
            json_response(200, {"access_token": "fresh"}),
            response(401, b"{}"),  # refused again: given up, not refreshed again
        ]
        error, server = await self.complete(answers)
        self.assertEqual((error.code, error.status), ("auth", 401))
        self.assertEqual(len(server.requests), 3)

    async def test_provider_sends_the_session_hash_and_account_hint(self):
        token = jwt(**{codex._AUTH_CLAIM: {"chatgpt_account_id": "acc-9"}})
        self.save(access=token)
        reply, server = await self.complete([stream_response(sse(DONE))])
        self.assertEqual(reply.text, "hi")
        sent = server.requests[0]["headers"]
        self.assertEqual(sent["chatgpt-account-id"], "acc-9")
        self.assertEqual(sent["session_id"], hashlib.sha256(b"x").hexdigest())  # not the raw id


class LoginTest(AuthFiles, unittest.IsolatedAsyncioTestCase):
    """The device login against a fake auth server."""

    def device(self, **extra):
        return json_response(200, {"device_auth_id": "dev-1", "user_code": "ABCD-1234", **extra})

    async def login(self, answers):
        codes = []
        async with FakeServer(answers) as server, fast_sleeps() as delays:
            auth = self.auth(server.url)
            try:
                await auth._login(lambda url, code: codes.append((url, code)))
                error = None
            except ProviderError as exc:
                error = exc
        return error, server, codes, delays

    async def test_device_login_waits_for_approval_and_saves_the_login(self):
        access = jwt(exp=time.time() + 600)
        answers = [
            self.device(interval="2"),
            response(403),  # not approved yet
            response(404),
            json_response(200, {"authorization_code": "code-1", "code_verifier": "ver-1"}),
            json_response(200, {"access_token": access, "refresh_token": "r1"}),
        ]
        error, server, codes, delays = await self.login(answers)
        self.assertIsNone(error)
        self.assertEqual(codes, [(server.url + "/codex/device", "ABCD-1234")])
        self.assertEqual(delays, [2.0, 2.0, 2.0])
        paths = [r["path"] for r in server.requests]
        self.assertEqual(
            paths,
            [
                "/api/accounts/deviceauth/usercode",
                "/api/accounts/deviceauth/token",
                "/api/accounts/deviceauth/token",
                "/api/accounts/deviceauth/token",
                "/oauth/token",
            ],
        )
        self.assertEqual(json.loads(server.requests[0]["body"]), {"client_id": codex.CLIENT_ID})
        self.assertEqual(
            json.loads(server.requests[1]["body"]),
            {"device_auth_id": "dev-1", "user_code": "ABCD-1234"},
        )
        exchange = server.requests[4]["body"].decode()
        self.assertIn("grant_type=authorization_code", exchange)
        self.assertIn("code=code-1", exchange)
        self.assertIn("code_verifier=ver-1", exchange)
        self.assertEqual(self.stored()["refresh_token"], "r1")
        self.assertEqual(self.stored()["access_token"], access)
        mode = stat.S_IMODE(os.stat(self.dir / "codex_auth.json").st_mode)
        self.assertEqual(mode, 0o600)
        self.assertIn("r1", self.secrets)
        self.assertTrue(self.auth().status()["logged_in"])

    async def test_the_poll_interval_is_bounded(self):
        for interval, expected in ((None, 5.0), (0.01, 1.0), (3600, 60.0)):
            with self.subTest(interval):
                answers = [self.device(interval=interval), response(500)]
                error, _, _, delays = await self.login(answers)
                self.assertEqual(delays, [expected])
                self.assertEqual(error.code, "auth")
                self.assertIn("refused (HTTP 500)", error.message)

    async def test_a_login_that_cannot_start(self):
        error, server, codes, _ = await self.login([response(503)])
        self.assertEqual(error.code, "server")
        self.assertIn("could not start (HTTP 503)", error.message)
        self.assertEqual(codes, [])
        for body in ({"user_code": "X"}, {"device_auth_id": 1, "user_code": "X"}):
            with self.subTest(body):
                error, _, codes, _ = await self.login([json_response(200, body)])
                self.assertEqual(error.code, "bad_response")
                self.assertEqual(codes, [])

    async def test_a_login_that_cannot_finish_saves_nothing(self):
        approval = json_response(200, {"authorization_code": "c", "code_verifier": "v"})
        error, _, _, _ = await self.login([self.device(), approval, response(400)])
        self.assertEqual(error.code, "auth")
        self.assertIn("could not finish (HTTP 400)", error.message)
        error, _, _, _ = await self.login([self.device(), approval, json_response(200, {})])
        self.assertEqual(error.code, "bad_response")
        self.assertFalse((self.dir / "codex_auth.json").exists())

    async def test_an_unapproved_code_expires(self):
        answers = [self.device(), response(403), response(403), response(403)]
        ticks = iter([0.0, 1.0, 2.0, 1000.0])  # the deadline, then each loop check
        clock = mock.Mock(time=time.time, sleep=time.sleep, monotonic=lambda: next(ticks))
        with mock.patch.object(codex, "time", clock):
            error, server, _, delays = await self.login(answers)
        self.assertEqual(error.code, "auth")
        self.assertIn("code expired", error.message)
        self.assertEqual(len(delays), 2)
        self.assertEqual(len(server.requests), 3)
        self.assertFalse((self.dir / "codex_auth.json").exists())

    def test_login_runs_its_own_loop_and_prints_the_code(self):
        auth = self.auth()

        async def fake_login(on_code):
            on_code("https://auth.example/codex/device", "WXYZ-0000")

        out = io.StringIO()
        with mock.patch.object(auth, "_login", fake_login), contextlib.redirect_stdout(out):
            auth.login()
        self.assertIn("https://auth.example/codex/device", out.getvalue())
        self.assertIn("WXYZ-0000", out.getvalue())
        seen = []
        with mock.patch.object(auth, "_login", fake_login):
            auth.login(lambda url, code: seen.append(code))
        self.assertEqual(seen, ["WXYZ-0000"])


class ImageFailureTest(unittest.IsolatedAsyncioTestCase):
    """generate_image's refusals: before sending (sent=False) and after (sent=True)."""

    async def asyncSetUp(self):
        self.dir = Path(tempfile.mkdtemp())
        grant = {"access_token": "tok", "refresh_token": "r", "expires_at": time.time() + 3600}
        (self.dir / "codex_auth.json").write_text(json.dumps(grant))

    async def generate(self, answers, references=(), **kwargs):
        async with FakeServer(answers) as server:
            auth = codex.CodexAuth(self.dir, base_url=server.url)
            with self.assertRaises(codex.ImageError) as caught:
                await codex.generate_image(auth, "a cat", "1024x1024", "low", references, **kwargs)
        return caught.exception, server

    async def test_reference_types(self):
        webp = b"RIFF\x00\x00\x00\x00WEBPVP8 "
        for raw, kind in ((webp, "webp"), (b"\xff\xd8\xff", "jpeg"), (b"GIF89a", "gif")):
            with self.subTest(kind):
                data = {"data": [{"b64_json": base64.b64encode(b"img").decode()}]}
                async with FakeServer([json_response(200, data)]) as server:
                    auth = codex.CodexAuth(self.dir, base_url=server.url)
                    await codex.generate_image(auth, "a cat", "1024x1024", "low", [raw])
                body = json.loads(server.requests[0]["body"])
                self.assertTrue(body["images"][0]["image_url"].startswith(f"data:image/{kind};"))

    async def test_an_unknown_reference_type_sends_nothing(self):
        error, server = await self.generate([], [b"%PDF-1.7"])
        self.assertFalse(error.sent)
        self.assertIn("not PNG, JPEG, GIF or WebP", str(error))
        self.assertEqual(server.requests, [])

    async def test_no_login_sends_nothing(self):
        (self.dir / "codex_auth.json").unlink()
        error, server = await self.generate([])
        self.assertFalse(error.sent)
        self.assertIn("python -m arti login", str(error))
        self.assertEqual(server.requests, [])

    async def test_a_timeout_may_have_been_billed(self):
        async def stall(request, writer):
            pass  # never answers; the client gives up

        error, server = await self.generate([stall], timeout=0.1)
        self.assertTrue(error.sent)
        self.assertEqual(str(error), "The image request failed: timed out")
        self.assertEqual(len(server.requests), 1)

    async def test_an_unreachable_service_is_reported(self):
        auth = codex.CodexAuth(self.dir, base_url="http://127.0.0.1:9")
        with self.assertRaises(codex.ImageError) as caught:
            await codex.generate_image(auth, "a cat", "1024x1024", "low")
        self.assertTrue(caught.exception.sent)  # as far as the caller can tell
        self.assertIn("Could not connect", str(caught.exception))

    async def test_a_reply_without_a_single_valid_image(self):
        image = base64.b64encode(b"img").decode()
        for value in (
            {"data": []},
            {"data": [{"b64_json": image}, {"b64_json": image}]},
            {"data": "nope"},
            {"data": [{"url": "https://x"}]},
            {"data": [{"b64_json": "not base64!"}]},
            {"nothing": 1},
        ):
            with self.subTest(value):
                error, _ = await self.generate([json_response(200, value)])
                self.assertTrue(error.sent)
                self.assertIn("no single valid image", str(error))
        error, _ = await self.generate([response(200, b"<html>")])
        self.assertIn("no single valid image", str(error))


if __name__ == "__main__":
    unittest.main()
