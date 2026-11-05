"""Secrets: the values in <data>/.env, and the redactor that keeps secrets out of what Arti writes.

`load_env` reads `<data>/.env` once at startup. Its values stay out of os.environ, where any
process of this user could read them (`ps -E`); the code that needs a key asks for it by name
with `get`, and `env()` is the whole file as a read-only mapping (a redactor hides its values).

The core passes everything it writes through `redact` (text) or `redact_obj` (a message or
other JSON-like value): history, inputs, sends, tool results, error messages and the log. Both
are the identity until a plugin registers a redactor (api.redactor, plugin.py). There is one per
process, because the log is: the one of the App that loaded its plugins last (App.load_plugins).
"""

import os
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

Redactor = tuple[Callable[[str], str], Callable[[Any], Any]]  # (text, obj), as api.redactor

_dotenv: dict[str, str] = {}
_redactor: Redactor | None = None

# --- .env --------------------------------------------------------------------------------------

_ENV_LINE = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)")


def _parse_value(raw: str) -> str | None:
    """A `.env` value: quoted (double quotes allow \\" and \\\\) or bare up to a ` #` comment."""
    if raw[:1] == '"':
        match = re.fullmatch(r'"((?:\\.|[^"\\])*)"\s*(?:#.*)?', raw)
        return re.sub(r"\\([\"\\])", r"\1", match[1]) if match else None
    if raw[:1] == "'":
        match = re.fullmatch(r"'([^']*)'\s*(?:#.*)?", raw)
        return match[1] if match else None
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def load_env(path: Path) -> None:
    """Load `KEY=VALUE` lines for `get` and `env` (a missing file loads nothing).

    Blank lines and `#` comments are skipped; `export` may lead a line. A bad line raises
    ValueError naming the file and line, never the value, and then nothing is loaded."""
    if not path.is_file():
        return
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE.fullmatch(line)
        value = _parse_value(match[2]) if match else None
        if value is None:
            raise ValueError(f"{path} line {number}: expected KEY=VALUE")
        entries[match[1]] = value
    _dotenv.update(entries)


def get(name: str) -> str:
    """A secret by variable name: the process environment first, then .env ("" if none)."""
    return os.environ.get(name) or _dotenv.get(name, "")


def env() -> MappingProxyType:
    """What .env holds, read-only: every value in it is a secret."""
    return MappingProxyType(_dotenv)


_learned: set[str] = set()  # secrets that arrive while Arti runs (a login's tokens)


def remember(values) -> None:
    """Treat `values` (a login's fresh tokens, say) as secrets too, like .env values."""
    _learned.update(v for v in values if isinstance(v, str) and v)


def learned() -> frozenset:
    """What `remember` was given."""
    return frozenset(_learned)


# --- redaction ---------------------------------------------------------------------------------


MIN_LEN = 8  # a shorter value would match ordinary text


def set_redactor(redactor: Redactor | None) -> None:
    """Redact through `redactor` from now on (None: only the known values, literally)."""
    global _redactor
    _redactor = redactor


def redact(text: str) -> str:
    """`text` with secrets hidden: by the registered redactor, or, before one is registered or
    with none (no plugin loaded yet; a failing one's traceback), by masking the .env values and
    remembered secrets as written."""
    if not text:
        return text
    if _redactor:
        return _redactor[0](text)
    for value in sorted({*_dotenv.values(), *_learned}, key=len, reverse=True):
        if len(value) >= MIN_LEN:
            text = text.replace(value, "[REDACTED]")
    return text


def redact_obj(value):
    """A copy of a JSON-like value with secrets hidden (see `redact`)."""
    if _redactor:
        return _redactor[1](value)
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [redact_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_obj(item) for key, item in value.items()}
    return value
