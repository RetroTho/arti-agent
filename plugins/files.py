"""read_file, write_file, patch and search_files.

Paths: `~` is expanded, a relative path is read from the terminal's working directory (home by
default, the same rule the approval review uses), and symlinks are followed, so the review and
the tool act on the same file. Overwriting an existing file needs a "read stamp": this session
read (or wrote) it, and its mtime and size have not changed since. That stops the one real
accident, clobbering a file the model never saw or that changed under it.
"""

import difflib
import re
import stat
import threading
from pathlib import Path

from arti.fsutil import atomic_write
from arti.tools import ToolError
from plugins.safety import proc, redact, risk

REQUIRES = ("safety",)  # imports from it, and its gate reviews this plugin's tools

PAGE_CHARS = 12_000  # one read_file page
MAX_LINES = 2000
LINE_CHARS = 2000
DIFF_CHARS = 8000
COUNT_LIMIT = 50 << 20  # past this many bytes read_file stops counting lines
MAX_STAMPS = 500
SEARCH_TIMEOUT = 10
SEARCH_CHARS = 400_000
DEFAULT_ROWS, MAX_ROWS = 50, 500
SKIP_DIRS = ["node_modules", ".venv*", "venv", "__pycache__"]  # hidden ones are skipped anyway

PEM_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
PEM_END = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY-----")

WRITE_LOCK = threading.Lock()  # one read-check-write at a time


# Read stamps: {path: [mtime_ns, size]} in the session state.


def _path(path: str, ctx) -> Path:
    """A path argument resolved from the session's cwd; a ToolError (not an unknown outcome) for
    one that names nothing, such as ~nobody."""
    try:
        return risk.resolve_path(path, ctx.state)
    except risk.PathError as e:
        raise ToolError(str(e)) from None


def _stamp(path: Path) -> list[int]:
    st = path.stat()
    return [st.st_mtime_ns, st.st_size]


def is_seen(ctx, path: Path) -> bool:
    return ctx.state.get("seen_files", {}).get(str(path)) == _stamp(path)


def mark_seen(ctx, path: Path, stamp: list[int] | None = None) -> None:
    stamps = ctx.state.setdefault("seen_files", {})
    stamps.pop(str(path), None)  # re-insert so the newest entries are kept
    stamps[str(path)] = stamp or _stamp(path)
    while len(stamps) > MAX_STAMPS:
        del stamps[next(iter(stamps))]


def read_file(args: dict, ctx) -> str:
    path = _path(args["path"], ctx)
    offset, limit = args.get("offset", 1), args.get("limit", MAX_LINES)
    if path.is_dir():
        raise ToolError(f"{path} is a directory; list it with search_files target='files'")
    if not path.is_file():
        raise ToolError(f"{path} does not exist or is not a regular file")
    with open(path, "rb") as f:
        if b"\0" in f.read(8192):
            raise ToolError(f"{path} is a binary file; for images use vision_analyze")
    stamp = _stamp(path)
    rows, chars, total, scanned, in_key, counted = [], 0, 0, 0, False, True
    with open(path, encoding="utf-8", errors="replace") as f:
        for total, line in enumerate(f, 1):
            scanned += len(line)
            line = line.rstrip("\r\n")
            # A private key stays hidden even when a page starts in the middle of it.
            begins, ends = PEM_BEGIN.search(line), PEM_END.search(line)
            hidden = in_key and not begins and not ends
            in_key = (in_key or bool(begins)) and not ends
            if total < offset:
                continue
            if len(rows) < limit and chars < PAGE_CHARS:
                if hidden:
                    line = redact.REDACTED
                elif len(line) > LINE_CHARS:
                    line = line[:LINE_CHARS] + "… [line cut]"
                row = f"{total}|{line}"
                if rows and chars + len(row) > PAGE_CHARS:
                    chars = PAGE_CHARS  # page full; keep counting lines
                else:
                    rows.append(row)
                    chars += len(row) + 1
            elif scanned > COUNT_LIMIT:
                counted = False
                break
    mark_seen(ctx, path, stamp)
    if not rows:
        return f"({path} is empty)" if total == 0 else f"[{path} has only {total} lines]"
    last = offset + len(rows) - 1
    if last == total and counted:
        return "\n".join(rows) + f"\n[end of file, {total} lines]"
    of = total if counted else "unknown (large file)"
    return "\n".join(rows) + f"\n[lines {offset}-{last} of {of}; continue with offset={last + 1}]"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600


def write_file(args: dict, ctx) -> str:
    path, content = _path(args["path"], ctx), args["content"]
    with WRITE_LOCK:
        if path.exists():
            if not path.is_file():
                raise ToolError(f"{path} is not a regular file")
            if not is_seen(ctx, path):
                raise ToolError(
                    f"Read {path} first: it is unread in this session or changed on disk since."
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, content, _mode(path))
        mark_seen(ctx, path)
    return f"Wrote {len(content)} chars ({content.count(chr(10))} newlines) to {path}"


def patch(args: dict, ctx) -> str:
    path = _path(args["path"], ctx)
    old, new = args["old_string"], args["new_string"]
    with WRITE_LOCK:
        if not path.is_file():
            raise ToolError(f"{path} is not an existing regular file")
        was_seen = is_seen(ctx, path)
        # surrogateescape keeps any invalid UTF-8 bytes exactly as they were.
        text = path.read_bytes().decode("utf-8", "surrogateescape")
        crlf = "\r\n" in text and text.count("\r\n") == text.count("\n")
        if crlf:
            text, old, new = (s.replace("\r\n", "\n") for s in (text, old, new))
        new_text, how = replace_text(text, old, new, bool(args.get("replace_all")))
        if new_text == text:
            raise ToolError("the patch changes nothing")
        data = new_text.replace("\n", "\r\n") if crlf else new_text
        atomic_write(path, data.encode("utf-8", "surrogateescape"), _mode(path))
        if was_seen:  # a patch alone does not grant the right to overwrite the whole file
            mark_seen(ctx, path)
    diff = "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True), new_text.splitlines(keepends=True), str(path), str(path)
        )
    )
    if len(diff) > DIFF_CHARS:
        diff = diff[:DIFF_CHARS] + f"\n... [diff cut; {len(diff) - DIFF_CHARS} more chars]"
    note = "" if how == "exact" else " (matched ignoring indentation)"
    return f"Patched {path}{note}.\n{diff}"


def _search_argv(args: dict, root: Path, mode: str) -> list[str]:
    pattern, glob, context = args["pattern"], args.get("file_glob"), args.get("context", 0)
    if args.get("target") == "files":
        skip = ["-name", ".*"]
        for d in SKIP_DIRS:
            skip += ["-o", "-name", d]
        test = ["-path", f"*/{pattern}"] if "/" in pattern else ["-name", pattern]
        find = ["/usr/bin/find", str(root), "-mindepth", "1", "(", *skip, ")", "-prune", "-o"]
        return find + ["-type", "f", *test, "-print"]
    argv = ["/usr/bin/grep", "-rnIE"] + [f"--exclude-dir={d}" for d in [".*", *SKIP_DIRS]]
    argv += [f"--include={glob}"] if glob else []
    flags = {"files_only": ["-l"], "count": ["-c"]}
    argv += flags.get(mode) or ["--null"] + ([f"-C{context}"] if context else [])
    return argv + ["-e", pattern, "--", str(root)]


def _row_path(row: str, mode: str) -> str:
    if mode == "content" and "\0" in row:
        return row.split("\0", 1)[0]
    return row.rsplit(":", 1)[0] if mode == "count" else row


def search_files(args: dict, ctx) -> str:
    root = _path(args.get("path") or ".", ctx)
    if not root.exists():
        raise ToolError(f"{root} does not exist")
    mode = "files_only" if args.get("target") == "files" else args.get("output_mode", "content")
    offset, limit = args.get("offset", 0), args.get("limit", DEFAULT_ROWS)
    result = proc.run(
        _search_argv(args, root, mode),
        env=redact.bare_env(),
        timeout=SEARCH_TIMEOUT,
        cancel=ctx.cancel,
        max_chars=SEARCH_CHARS,
        merge_stderr=False,
    )
    # proc.run cut the middle out of a huge output; only the head is used.
    head, marker, rest = result.out.partition("\n... [")
    cut = bool(marker) and "chars omitted" in rest[:80]
    rows = []
    for row in (head if cut else result.out).splitlines():
        if not row or (mode == "count" and row.endswith(":0")):
            continue
        # Rows from credential stores and key files never reach the model.
        if row != "--" and risk.is_secret_path(_row_path(row, mode)):
            continue
        rows.append(row.replace("\0", ":"))
    if not rows and result.code not in (0, 1, None):
        raise ToolError(f"search failed: {result.err.strip()[:500] or f'exit {result.code}'}")
    notes = []
    if offset + limit < len(rows):
        notes.append(f"[{len(rows)} rows; more with offset={offset + limit}]")
    if result.timed_out:
        notes.append(f"[stopped after {SEARCH_TIMEOUT}s; results are partial]")
    elif cut:
        notes.append("[output too large; results are partial; narrow the search]")
    return "\n".join((rows[offset : offset + limit] or ["No matches."]) + notes)


PATH = {"type": "string", "pattern": r"\S"}
INT = {"type": "integer", "minimum": 1}


def forget_reads(conv, span) -> None:
    """After a compaction the model no longer sees what it read: read files again to edit."""
    conv.state.pop("seen_files", None)


# --- the patch rule (the skills plugin keeps the same copy; test_replace_text checks both) ---


def replace_text(text: str, old: str, new: str, replace_all: bool = False) -> tuple[str, str]:
    """Replace old with new in text: an exact match, else whole lines compared without their
    surrounding whitespace (new is re-indented to match). Returns (new_text, "exact"|"trimmed")."""
    count = text.count(old)
    if count == 1 or (count > 1 and replace_all):
        return text.replace(old, new), "exact"
    if count > 1:
        raise ToolError(f"old_string matches {count} times; add context or set replace_all")
    lines = text.split("\n")
    old_lines = old.strip("\n").split("\n")
    wanted = [line.strip() for line in old_lines]
    n, matches, i = len(wanted), [], 0
    while i + n <= len(lines):
        if [line.strip() for line in lines[i : i + n]] == wanted:
            matches.append(i)
            i += n
        else:
            i += 1
    if not matches:
        raise ToolError("old_string not found (exactly, or as whole lines ignoring indentation)")
    if len(matches) > 1 and not replace_all:
        raise ToolError(f"old_string matches {len(matches)} places; add context or set replace_all")
    first = next(j for j, w in enumerate(wanted) if w)  # the first non-blank line sets the indent
    old_indent = _indent(old_lines[first])
    new_lines = new.strip("\n").split("\n") if new.strip("\n") else []
    for start in reversed(matches):
        file_indent = _indent(lines[start + first])
        lines[start : start + n] = [
            file_indent + line[len(old_indent) :]
            if line.strip() and line.startswith(old_indent)
            else line
            for line in new_lines
        ]
    return "\n".join(lines), "trimmed"


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def register(api) -> None:
    api.on_compact(forget_reads)
    api.tool(
        "read_file",
        "Read a text file as numbered lines (`N|line`), a page at a time. Use instead of cat.",
        {
            "properties": {"path": PATH, "offset": INT, "limit": {**INT, "maximum": MAX_LINES}},
            "required": ["path"],
        },
        read_file,
        review="read",
    )
    api.tool(
        "write_file",
        "Write a whole file, creating folders. To overwrite a file, read it first. For small "
        "edits use patch.",
        {
            "properties": {"path": PATH, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        write_file,
        review="write",
    )
    api.tool(
        "patch",
        "Replace old_string with new_string in a file; it must match once unless replace_all. "
        "Falls back to matching whole lines ignoring indentation. Returns the diff.",
        {
            "properties": {
                "path": PATH,
                "old_string": {"type": "string", "pattern": r"\S"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        patch,
        review="write",
    )
    api.tool(
        "search_files",
        "Regex search of file contents under path (default: the cwd), or with target='files' "
        "find files by glob ('*.md'; 'src/*.py' matches paths). Skips hidden files, "
        "node_modules and virtualenvs. Use instead of grep, find and ls.",
        {
            "properties": {
                "pattern": {"type": "string", "pattern": r"\S"},
                "path": PATH,
                "target": {"type": "string", "enum": ["content", "files"]},
                "file_glob": {"type": "string"},
                "output_mode": {"type": "string", "enum": ["content", "files_only", "count"]},
                "context": {"type": "integer", "minimum": 0, "maximum": 10},
                "limit": {**INT, "maximum": MAX_ROWS},
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": ["pattern"],
        },
        search_files,
        review="read",
        timeout=SEARCH_TIMEOUT + 20,
    )
