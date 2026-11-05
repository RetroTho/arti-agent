"""Arti's own ChatGPT login (OAuth device flow) and the ChatGPT image client.

The login lives in data_dir/codex_auth.json (0600, replaced atomically):
`{"access_token", "refresh_token", "expires_at"}`. A refresh token is single use, so a
refresh happens under a cross-process flock and re-reads the file under it: a token another
process already rotated is adopted, never sent again. A refusal (`invalid_grant`,
`refresh_token_reused`) asks the owner to log in again; a network failure leaves the file
alone. Arti never reads another application's login (no ~/.codex import).
"""

from __future__ import annotations

import asyncio
import base64
import fcntl
import json
import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

from arti import fsutil, secret
from arti.providers import ProviderError, bad_response, http

ISSUER = "https://auth.openai.com"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
BASE_URL = "https://chatgpt.com/backend-api/codex"
REFRESH_SKEW = 120  # refresh this many seconds before expiry
LOGIN_LIFETIME = 900  # how long the owner has to enter the device code
LOCK_WAIT = 30
IMAGE_MODEL = "gpt-image-2"
IMAGE_LIMIT = 25_000_000
RELOGIN = "The ChatGPT login expired or was revoked: run `python -m arti login`."
_HINT = re.compile(r"[A-Za-z0-9._-]{1,256}")
_AUTH_CLAIM = "https://api.openai.com/auth"


class ImageError(Exception):
    """An image request failed; the text is safe to show. `sent`: it may have been billed."""

    def __init__(self, message: str, sent: bool):
        super().__init__(message)
        self.sent = sent


def claims(token: str) -> dict:
    """The JWT's claims, decoded WITHOUT checking the signature: hints only, never identity."""
    try:
        payload = token.split(".")[1]
        value = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return value if isinstance(value, dict) else {}
    except (IndexError, ValueError):
        return {}


def headers(token: str) -> dict:
    """The ChatGPT backend headers for an access token."""
    result = {
        "Authorization": "Bearer " + token,
        "originator": "arti",
        "User-Agent": http.USER_AGENT,
    }
    auth = claims(token).get(_AUTH_CLAIM)
    auth = auth if isinstance(auth, dict) else {}
    hints = {
        "ChatGPT-Account-ID": auth.get("chatgpt_account_id"),
        "x-openai-internal-codex-residency": auth.get("chatgpt_data_residency")
        or auth.get("chatgpt_compute_residency"),
    }
    for name, value in hints.items():
        if isinstance(value, str) and _HINT.fullmatch(value):  # headers: no control characters
            result[name] = value
    return result


def _flock(path: Path) -> int:
    """Take the cross-process lock (blocking, at most LOCK_WAIT seconds); returns its fd."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    end = time.monotonic() + LOCK_WAIT
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() > end:
                os.close(fd)
                raise ProviderError("The ChatGPT login is locked by another process", "server")
            time.sleep(0.1)


_SHARED: dict[Path, CodexAuth] = {}


def shared_auth(data_dir: Path) -> CodexAuth:
    """The one CodexAuth of a data dir, so the model route and image_generate share a lock;
    its tokens are registered as secrets (redacted everywhere)."""
    key = Path(data_dir).resolve()
    if key not in _SHARED:
        _SHARED[key] = CodexAuth(key)
    return _SHARED[key]


class CodexAuth:
    """Arti's ChatGPT login in a data dir: token() for requests, login/logout/status for the CLI."""

    def __init__(self, data_dir: Path, on_secrets=None, issuer=ISSUER, base_url=BASE_URL):
        self.path = Path(data_dir) / "codex_auth.json"
        self.lock_path = Path(data_dir) / "codex_auth.lock"
        self.on_secrets = on_secrets or secret.remember  # every token is redacted (by a redactor)
        self.issuer, self.base_url = issuer, base_url  # injectable for tests
        self._lock = asyncio.Lock()

    def _read(self) -> dict | None:
        try:
            grant = fsutil.read_json(self.path)
            if grant is not None:
                self.on_secrets([grant["access_token"], grant["refresh_token"]])
                grant["expires_at"] = float(grant["expires_at"])
        except (ValueError, KeyError, TypeError):
            raise ProviderError(f"{self.path} is damaged: run `python -m arti login`.", "auth")
        return grant

    def _save(self, grant: dict) -> None:
        self.on_secrets([grant["access_token"], grant["refresh_token"]])
        fsutil.atomic_write(self.path, json.dumps(grant), mode=0o600)

    async def token(self, rejected: str | None = None) -> str:
        """A fresh access token; refreshes when it expires soon or equals `rejected` (the one
        a request was just refused with). Shielded: once a refresh is sent, the rotated
        token is saved even if the caller is cancelled."""
        return await asyncio.shield(self._token(rejected))

    async def _token(self, rejected: str | None) -> str:
        async with self._lock:
            fd = await asyncio.to_thread(_flock, self.lock_path)
            try:
                grant = self._read()
                if grant is None:
                    raise ProviderError("No ChatGPT login: run `python -m arti login`.", "auth")
                fresh = grant["expires_at"] > time.time() + REFRESH_SKEW
                if fresh and grant["access_token"] != rejected:
                    return grant["access_token"]  # possibly one a peer just rotated
                grant = self._grant(await self._refresh(grant), grant["refresh_token"])
                self._save(grant)
                return grant["access_token"]
            finally:
                os.close(fd)

    async def _refresh(self, grant: dict) -> dict:
        form = {
            "grant_type": "refresh_token",
            "refresh_token": grant["refresh_token"],
            "client_id": CLIENT_ID,
        }
        status, _, body = await self._post("/oauth/token", form=form)
        if status == 200:
            return _json(body)
        if status == 429 or status >= 500:
            raise ProviderError(
                f"ChatGPT login refresh failed (HTTP {status}); try again later",
                "server",
                retryable=True,
            )
        if status in (400, 401, 403):  # invalid_grant, refresh_token_reused, revoked
            raise ProviderError(RELOGIN, "auth")
        raise bad_response(f"HTTP {status} from the ChatGPT login")

    async def _post(self, path: str, *, form=None, payload=None, timeout=15):
        if form is not None:
            data, kind = urlencode(form).encode(), "application/x-www-form-urlencoded"
        else:
            data, kind = json.dumps(payload).encode(), "application/json"
        try:
            return await http.post(
                self.issuer + path,
                headers={"Content-Type": kind, "Accept": "application/json"},
                data=data,
                timeout=timeout,
            )
        except TimeoutError:
            raise ProviderError(
                "The ChatGPT login did not answer in time", "timeout", retryable=True
            ) from None

    @staticmethod
    def _grant(reply: dict, previous_refresh: str = "") -> dict:
        access = reply.get("access_token")
        refresh = reply.get("refresh_token") or previous_refresh
        if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
            raise bad_response("no token in the ChatGPT login reply")
        exp = claims(access).get("exp")
        expires = [exp if isinstance(exp, (int, float)) else time.time() + 3600]
        if isinstance(reply.get("expires_in"), (int, float)):
            expires.append(time.time() + reply["expires_in"])
        return {"access_token": access, "refresh_token": refresh, "expires_at": min(expires)}

    def login(self, on_code=None) -> None:
        """The device login, for `python -m arti login` (runs its own event loop)."""
        on_code = on_code or (lambda url, code: print(f"Open {url} and enter the code {code}"))
        asyncio.run(self._login(on_code))

    async def _login(self, on_code) -> None:
        status, _, body = await self._post(
            "/api/accounts/deviceauth/usercode", payload={"client_id": CLIENT_ID}
        )
        if status != 200:
            raise ProviderError(f"ChatGPT login could not start (HTTP {status})", "server")
        device = _json(body)
        device_id, user_code = device.get("device_auth_id"), device.get("user_code")
        if not isinstance(device_id, str) or not isinstance(user_code, str):
            raise bad_response("no device code")
        interval = min(60.0, max(1.0, float(device.get("interval") or 5)))
        on_code(self.issuer + "/codex/device", user_code)
        end = time.monotonic() + LOGIN_LIFETIME
        while time.monotonic() < end:
            await asyncio.sleep(interval)
            poll = {"device_auth_id": device_id, "user_code": user_code}
            status, _, body = await self._post("/api/accounts/deviceauth/token", payload=poll)
            if status in (403, 404):
                continue  # not approved yet
            if status != 200:
                raise ProviderError(f"ChatGPT login was refused (HTTP {status})", "auth")
            approval = _json(body)
            form = {
                "grant_type": "authorization_code",
                "code": approval.get("authorization_code"),
                "code_verifier": approval.get("code_verifier"),
                "redirect_uri": self.issuer + "/deviceauth/callback",
                "client_id": CLIENT_ID,
            }
            status, _, body = await self._post("/oauth/token", form=form)
            if status != 200:
                raise ProviderError(f"ChatGPT login could not finish (HTTP {status})", "auth")
            grant = self._grant(_json(body))
            fd = await asyncio.to_thread(_flock, self.lock_path)
            try:
                self._save(grant)
            finally:
                os.close(fd)
            return
        raise ProviderError(
            "The ChatGPT login code expired; run `python -m arti login` again", "auth"
        )

    def logout(self) -> None:
        fd = _flock(self.lock_path)
        try:
            self.path.unlink(missing_ok=True)
        finally:
            os.close(fd)

    def status(self) -> dict:
        fd = _flock(self.lock_path)  # as token() does: a refresh in progress finishes first
        try:
            grant = self._read()
        except ProviderError as exc:
            return {"logged_in": False, "error": exc.message}
        finally:
            os.close(fd)
        return {"logged_in": grant is not None, "expires_at": grant and grant["expires_at"]}


def _json(body: bytes) -> dict:
    try:
        value = json.loads(body)
    except ValueError:
        value = None
    if not isinstance(value, dict):
        raise bad_response("a reply that is not a JSON object")
    return value


def _image_type(data: bytes) -> str:
    for magic, kind in (
        (b"\x89PNG", "image/png"),
        (b"\xff\xd8", "image/jpeg"),
        (b"GIF8", "image/gif"),
    ):
        if data.startswith(magic):
            return kind
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ImageError("A reference image is not PNG, JPEG, GIF or WebP", sent=False)


async def generate_image(
    auth: CodexAuth, prompt: str, size: str, quality: str, references=(), timeout: float = 300
) -> bytes:
    """Generate (or, with reference image bytes, edit) one image on the ChatGPT login.

    Sent exactly once, never retried: a retry could bill twice. A refused response's body
    is never read (it may echo the prompt). Raises ImageError.
    """
    payload = {
        "prompt": prompt,
        "model": IMAGE_MODEL,
        "n": 1,
        "size": size,
        "quality": quality,
        "background": "opaque",
    }
    if references:
        payload["images"] = [
            {"image_url": f"data:{_image_type(raw)};base64,{base64.b64encode(raw).decode()}"}
            for raw in references
        ]
    try:
        token = await auth.token()
    except ProviderError as exc:
        raise ImageError(exc.message, sent=False) from None
    url = auth.base_url + ("/images/edits" if references else "/images/generations")
    request_headers = {
        **headers(token),
        "x-codex-image-turn-id": str(uuid.uuid4()),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        status, _, body = await http.post(
            url,
            headers=request_headers,
            data=json.dumps(payload).encode(),
            timeout=timeout,
            limit=IMAGE_LIMIT * 4 // 3 + 65536,
            read_errors=False,
        )
    except (ProviderError, TimeoutError) as exc:
        raise ImageError(
            f"The image request failed: {str(exc) or 'timed out'}", sent=True
        ) from None
    if status != 200:
        raise ImageError(f"The image service answered HTTP {status}; not retried", sent=True)
    try:
        data = _json(body)["data"]
        if not isinstance(data, list) or len(data) != 1:
            raise ValueError
        return base64.b64decode(data[0]["b64_json"], validate=True)
    except (ProviderError, ValueError, KeyError, TypeError):
        raise ImageError("The image service returned no single valid image", sent=True) from None
