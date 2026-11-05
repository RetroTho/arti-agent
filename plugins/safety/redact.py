"""Keeps secrets out of text: redacting (the safety plugin's redactor), and clean environments
for children.

Every value from `.env` (the core loads it, arti/secret.py) and every value `register`ed later,
such as refreshed OAuth tokens, is replaced with `[REDACTED]` wherever it shows up: raw,
JSON-escaped, URL-encoded or base64-encoded at any offset. Common secret shapes (private keys,
Authorization headers, Bearer tokens, URL passwords, vendor API keys, `PASSWORD=...` pairs) are
redacted too. This guards against
accidents, not attackers. Values shorter than 8 characters are never redacted: replacing every
"test1" in ordinary text would garble it.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path
from urllib.parse import quote

from arti import secret

REDACTED = "[REDACTED]"
MIN_LEN = 8

_lock = threading.Lock()
_matcher: tuple[tuple, re.Pattern | None] = ((), None)

# --- names -------------------------------------------------------------------------------------

# A secret word as a name segment (SERVICE_API_KEY, client_secret), or a weaker one at the end.
_SECRET_WORD = re.compile(
    r"(?:^|_)(?:passwd|password|passphrase|token|secret|apikey|api_key|private_key|credentials?)"
    r"(?:_|$)|(?:^|_)(?:authorization|auth|cookie)$|password$|_(?:key|pass|pwd|pat|dsn)$"
    r"|^(?:op_session_|bw_session$)"
)
# Names of a place, a count or an id, not a credential (TOKEN_PATH, input_tokens, api_key_env).
_NOT_SECRET = re.compile(
    r"_(?:env|url|path|file|dir|tokens|count|id|type|expires_at)$"
    r"|(?:^|_)(?:primary|sort|cache|foreign|public|row|idempotency)_key$"
)


def is_secret_name(name: str) -> bool:
    """True when a variable or field name says its value is a secret.

    Matches GITHUB_TOKEN, apiKey, client_secret, SMTP_PASS, PGPASSWORD, AWS_ACCESS_KEY_ID;
    not PWD, api_key_env, TOKEN_PATH, input_tokens or public_key."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower().replace("-", "_")
    if name.endswith("access_key_id"):
        return True
    return not _NOT_SECRET.search(name) and bool(_SECRET_WORD.search(name))


# --- the matcher of known values ---------------------------------------------------------------


def _base64_fragments(raw: bytes) -> set[str]:
    """The base64 characters that encode only `raw`, at each of the three byte alignments.

    Encoding k zero bytes before the value shifts it to alignment k; the characters before
    ceil(4k/3) also carry the zero bytes and the ones after floor(8n/6) the next bytes, so the
    middle is what any base64 stream holding the value contains, wherever it starts.

    Example: "secretvalue1" alone encodes as c2VjcmV0dmFsdWUx; after two other bytes ("xy")
    the stream is eHlzZWNyZXR2YWx1ZTF6, which holds this function's k=2 fragment
    zZWNyZXR2YWx1ZT, so the secret is found there too."""
    found = set()
    for encode in (base64.b64encode, base64.urlsafe_b64encode):
        for k in range(3):
            text = encode(b"\0" * k + raw).decode().rstrip("=")
            start, end = -(-4 * k // 3), (8 * (k + len(raw))) // 6
            if end - start >= MIN_LEN:
                found.add(text[start:end])
    return found


def _encodings(value: str) -> set[str]:
    """`value` as it may appear in tool output, logs or JSON."""
    forms = {value, quote(value, safe="")}
    forms |= {json.dumps(value, ensure_ascii=ascii)[1:-1] for ascii in (True, False)}
    forms |= _base64_fragments(value.encode())
    return {form for form in forms if len(form) >= MIN_LEN}


def _dotenv() -> tuple[set[str], set[str]]:
    """The names and the values (long enough to redact) of the core's secrets: its .env and what
    plugins asked it to remember (a login's tokens)."""
    entries = secret.env()
    values = {*entries.values(), *secret.learned()}
    return set(entries), {v for v in values if len(v) >= MIN_LEN}


def _known_matcher() -> re.Pattern | None:
    """One regex matching every known secret in every encoding, built only when the set changes:
    registered values, .env values and environment variables secret by name (or named in .env).
    The key also holds the secret variables and .env values themselves, so one set or changed
    after start is noticed."""
    global _matcher
    names, dotenv = _dotenv()
    with _lock:
        env = {v for n, v in os.environ.items() if n in names or is_secret_name(n)}
        key = (frozenset(env), frozenset(dotenv))
        if _matcher[0] == key:
            return _matcher[1]
        forms = set()
        for value in dotenv | {v for v in env if len(v) >= MIN_LEN}:
            forms |= _encodings(value)
        # Longest first, so a value that contains another is replaced whole.
        ordered = sorted(forms, key=len, reverse=True)
        pattern = re.compile("|".join(map(re.escape, ordered))) if ordered else None
        _matcher = (key, pattern)
        return pattern


# --- secret shapes -----------------------------------------------------------------------------

_PEM = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----"
    r"(?:.*?-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----|.*)",
    re.S,
)
_AUTH_HEADER = re.compile(r"(?i)\b((?:proxy-)?authorization[\"']?[ \t]*[:=][ \t]*)[^\r\n]+")
_BEARER = re.compile(r"\bBearer\s+[\w.~+/=-]{16,}")
_URL_PASSWORD = re.compile(r"(?i)\b([a-z][a-z0-9+.-]{0,31}://[^/?#@\s:]*:)[^/?#@\s]+@")
_VENDOR = re.compile(
    r"\b(?:sk-ant-[\w-]{16,}|sk-[\w-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}"
    r"|glpat-[\w-]{16,}|xox[abprs]-[\w-]{10,}|AIza[\w-]{30,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20,}"
    r"|fc-[A-Za-z0-9]{16,}|npm_[A-Za-z0-9]{30,}|pypi-[\w-]{30,}|[sr]k_live_[A-Za-z0-9]{16,}"
    r"|eyJ[\w-]{8,}\.eyJ[\w-]{8,}\.[\w-]{8,})"
)
# NAME=value, NAME: value or "name": "value"; the value is quoted or runs to a separator.
_PAIR = re.compile(
    r"""(?<![\w-])(["']?)([A-Za-z_][\w-]*)\1([ \t]*(?:=|:)(?![=:])[ \t]*)"""
    r"""(\[REDACTED\]|"(?:\\.|[^"\\\n])*"|'[^'\n]*'|[^\s,;&'"}\])]+)"""
)
_LITERALS = {"", "none", "null", "true", "false", REDACTED.lower()}


def _pair(match: re.Match) -> str:
    """A `_PAIR` match with its value redacted when the name is a secret one."""
    quote_, name, separator, value = match.groups()
    mark = value[0] if value[0] in "\"'" else ""
    inner = value[1:-1] if mark else value
    if (
        not is_secret_name(name)
        or inner.lower() in _LITERALS
        or inner.startswith("$")
        or inner == name
    ):
        return match[0]
    return f"{quote_}{name}{quote_}{separator}{mark}{REDACTED}{mark}"


def redact(text: str) -> str:
    """`text` with known secret values and secret-looking forms replaced by [REDACTED]."""
    if not text:
        return text
    if known := _known_matcher():
        text = known.sub(REDACTED, text)
    text = _PEM.sub(REDACTED, text)
    text = _AUTH_HEADER.sub(lambda m: m[1] + REDACTED, text)
    text = _BEARER.sub("Bearer " + REDACTED, text)
    text = _URL_PASSWORD.sub(lambda m: m[1] + REDACTED + "@", text)
    text = _VENDOR.sub(REDACTED, text)
    return _PAIR.sub(_pair, text)


def redact_obj(value):
    """A copy of a message or other JSON-like value with every string redacted.

    Image data and signed thinking are left alone (changing them would break them); a field
    whose name is a secret one is replaced whole. A provider's `native` replay data is kept
    whole only when nothing in the message or in it needed redacting, and dropped otherwise:
    a redacted copy would be broken, and one that no longer matches the visible content is
    never replayed anyway."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, (list, tuple)):
        return type(value)(redact_obj(item) for item in value)
    if not isinstance(value, dict):
        return value
    kind = value.get("type")
    if kind in ("thinking", "redacted_thinking") and ("signature" in value or "data" in value):
        return value
    out = {}
    for key, item in value.items():
        if key == "native":
            continue
        if kind == "image" and key in ("data", "source"):
            out[key] = item
        elif isinstance(key, str) and is_secret_name(key) and isinstance(item, str):
            out[key] = REDACTED
        else:
            out[key] = redact_obj(item)
    if "native" in value:
        rest = {key: item for key, item in value.items() if key != "native"}
        native = value["native"]
        if out == rest and redact_obj(native) == native:
            out["native"] = native
    return out


# --- child environments ------------------------------------------------------------------------

_PROVIDER_PREFIXES = ("ANTHROPIC_", "OPENAI_", "CLAUDE_", "CODEX_", "AZURE_OPENAI_")
# Variables a shell runs code from at start-up, or that redirect `cd`.
_SHELL_HOOKS = {"BASH_ENV", "ENV", "CDPATH"}


def _keep(name: str, value: str, dotenv: tuple[set[str], set[str]]) -> bool:
    upper = name.upper()
    return not (
        name in dotenv[0]
        or value in dotenv[1]
        or is_secret_name(name)
        or upper.startswith(_PROVIDER_PREFIXES)
        or upper in _SHELL_HOOKS
        or _URL_PASSWORD.search(value)
    )


def child_env(extra: dict | None = None) -> dict:
    """os.environ without secrets, model-provider settings or shell hooks, plus `extra`.

    `extra` is added as given: that is how a secret reaches a child on purpose."""
    dotenv = _dotenv()
    with _lock:
        env = {name: value for name, value in os.environ.items() if _keep(name, value, dotenv)}
    return {**env, **(extra or {})}


def bare_env(**extra: str) -> dict:
    """A minimal environment for helpers that need nothing from the user's shell."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
    }
    return {**env, **extra}
