"""skills: procedural memory, one directory per skill under <data>/skills/, shared by all chats.

    skills/[<category>/]<name>/SKILL.md      frontmatter `name` and `description`, then the body
    skills/.../<name>/references|templates|scripts|assets/...   supporting files

An index of "- name: description" goes into the system prompt (capped at 16k chars), so the
model knows what exists and loads a skill with skill_view. The "preload_skills" service lets other
plugins preload skills into a headless run. A compaction keeps the names of the skills loaded
before it (`on_compact`). Every path the model names is confined to the skill's own directory,
and every write is atomic.

A skill is instructions Arti follows with all its tools, so they are the owner's: only their own
conversations (and their background runs) see the index and the tools. Nobody else's lane of a
group gets skill_view or skill_manage, nor the index in its prompt.
"""

import os
import re
import shutil
import threading
from pathlib import Path

from arti.fsutil import atomic_write, private_dir
from arti.tools import ToolError

REQUIRES = ("safety",)  # its gate reviews the skill_manage tool's writes (kind "prompt")

NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
SUBDIRS = ("references", "templates", "scripts", "assets")
MAX_DESCRIPTION = 400  # each description is in every prompt
MAX_INDEX = 16_000
MAX_VIEW = 50_000
_lock = threading.Lock()  # one skill_manage at a time
_cache: dict[Path, tuple[tuple, dict]] = {}  # skills root -> (SKILL.md mtimes, index)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split SKILL.md into (meta, body). Only top-level `key: value` lines are read."""
    match = re.match(r"---\n(.*?)\n---[ \t]*(?:\n|\Z)", text, re.DOTALL)
    if not match:
        return {}, text
    meta = {}
    for line in match[1].splitlines():
        key, colon, value = line.partition(":")
        if colon and not line[:1].isspace() and not line.startswith("-"):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            meta[key.strip()] = value
    return meta, text[match.end() :]


def _skill_files(root: Path) -> list[Path]:
    """Every SKILL.md under `root`, skipping hidden directories (".archive") and not looking
    inside a skill for more skills."""
    found = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
        if "SKILL.md" in files and directory != str(root):  # skills/SKILL.md is not a skill
            found.append(Path(directory) / "SKILL.md")
            subdirs[:] = []
    return found


def index(root: Path) -> dict[str, tuple[Path, str]]:
    """name -> (skill directory, description). Re-read only when a SKILL.md changed."""
    files = _skill_files(root)
    signature = tuple((str(f), f.stat().st_mtime_ns) for f in files)
    cached = _cache.get(root)
    if cached and cached[0] == signature:
        return cached[1]
    skills = {}
    for document in files:
        meta, _ = parse_frontmatter(document.read_text(encoding="utf-8", errors="replace"))
        name = meta.get("name") or document.parent.name
        skills.setdefault(name, (document.parent, meta.get("description", "")))
    _cache[root] = (signature, skills)
    return skills


def _root(app) -> Path:
    return Path(app.data_dir) / "skills"


def load_skill(app, name: str) -> str | None:
    """The full SKILL.md of skill `name`, or None when there is none."""
    found = index(_root(app)).get(name)
    return (found[0] / "SKILL.md").read_text(encoding="utf-8") if found else None


def preload(app, names) -> str:
    """The skills `names` as system prompt text, for a run that should start with them (the
    "preload_skills" service: scheduled jobs, delegated tasks). A name with no skill says so."""
    parts = []
    for name in names:
        text = load_skill(app, name)
        parts.append(f"# Skill: {name}\n{text.strip()}" if text else f"(skill {name!r} not found)")
    return "\n\n".join(parts)


def prompt_section(info) -> str | None:
    """The skills index; none in someone else's lane (skills are the owner's)."""
    skills = index(_root(info.app)) if info.trusted else {}
    if not skills:
        return None
    lines, size = [], 0
    for name, (_, description) in sorted(skills.items()):
        line = f"- {name}: {description[:MAX_DESCRIPTION]}"
        if size + len(line) > MAX_INDEX:
            lines.append(f"- ({len(skills) - len(lines)} more skills not listed: index full)")
            break
        lines.append(line)
        size += len(line) + 1
    return (
        "## Skills\nBefore replying, scan these. If one matches the task, load it with "
        "skill_view(name) and follow it.\n" + "\n".join(lines)
    )


def on_compact(conv, span) -> str | None:
    """The skills loaded so far (kept in the session state across compactions), so the model
    knows to load one again before following it: its text is gone with the summarised span."""
    calls = [b for m in span for b in m["content"] if b.get("type") == "tool_use"]
    viewed = {b["input"].get("name") or "" for b in calls if b["name"] == "skill_view"} - {""}
    seen = conv.state["skills_seen"] = sorted(set(conv.state.get("skills_seen", [])) | viewed)
    if not seen:
        return None
    return "Skills loaded earlier (call skill_view again before following one): " + ", ".join(seen)


def _target(skill_dir: Path, file_path: str | None) -> Path:
    """SKILL.md, or a supporting file `<subdir>/...` that stays inside the skill directory."""
    if not file_path:
        return skill_dir / "SKILL.md"
    parts = file_path.split("/")
    if len(parts) < 2 or parts[0] not in SUBDIRS or any(p in ("", ".", "..") for p in parts):
        raise ToolError(f"file_path must look like <{'|'.join(SUBDIRS)}>/<name>")
    path = skill_dir / file_path
    if not path.resolve().is_relative_to(skill_dir.resolve()):
        raise ToolError("file_path escapes the skill directory")
    return path


def _check_skill_md(text: str, name: str) -> None:
    meta, body = parse_frontmatter(text)
    if meta.get("name") != name:
        raise ToolError(f"SKILL.md frontmatter needs `name: {name}` (the skill's name)")
    if not meta.get("description"):
        raise ToolError("SKILL.md frontmatter needs a `description:` saying when to use it")
    if len(meta["description"]) > MAX_DESCRIPTION:
        raise ToolError(f"description is over {MAX_DESCRIPTION} characters")
    if not body.strip():
        raise ToolError("SKILL.md body is empty")


def skill_view(args: dict, ctx) -> str:
    skills = index(_root(ctx.app))
    name = args["name"]
    if name not in skills:
        raise ToolError(f"no skill {name!r}. Available: {', '.join(sorted(skills)) or 'none'}")
    skill_dir = skills[name][0]
    path = _target(skill_dir, args.get("file_path"))
    if not path.is_file():
        raise ToolError(f"{args.get('file_path') or 'SKILL.md'} not found in skill {name}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_VIEW:
        text = text[:MAX_VIEW] + f"\n[cut at {MAX_VIEW} chars; read_file {path} for the rest]"
    if args.get("file_path"):
        return text
    linked = sorted(
        str(p.relative_to(skill_dir))
        for sub in SUBDIRS
        for p in (skill_dir / sub).rglob("*")
        if p.is_file()
    )
    if linked:
        text += "\n\n[linked files: view with skill_view(name, file_path=...)]\n"
        text += "\n".join(f"- {p}" for p in linked[:200])
    return text


def skill_manage(args: dict, ctx) -> str:
    action, name = args["action"], args["name"]
    if not NAME.fullmatch(name):
        raise ToolError("name must be lowercase letters, digits, - and _ (at most 64)")
    root = private_dir(_root(ctx.app))
    with _lock:
        existing = index(root).get(name)
        if action == "create":
            return _create(root, name, args, existing)
        if not existing:
            raise ToolError(f"no skill {name!r}")
        skill_dir = existing[0]
        if action == "delete":
            shutil.rmtree(skill_dir)
            return f"Deleted skill {name}."
        path = _target(skill_dir, args.get("file_path"))
        label = f"{name}/{path.relative_to(skill_dir)}"
        if action in ("remove_file", "patch") and not path.is_file():
            raise ToolError(f"{label} not found")
        if action == "remove_file":
            if path == skill_dir / "SKILL.md":
                raise ToolError("remove_file needs a file_path; use delete to remove the skill")
            path.unlink()
            return f"Removed {label}."
        if action == "patch":
            text, _ = replace_text(
                path.read_text(encoding="utf-8"),
                args.get("old_string", ""),
                args.get("new_string", ""),
                bool(args.get("replace_all")),
            )
        else:  # write_file
            if "content" not in args:
                raise ToolError("write_file needs content")
            text = args["content"]
        if path == skill_dir / "SKILL.md":
            _check_skill_md(text, name)
        private_dir(path.parent)
        atomic_write(path, text)
        return f"{'Patched' if action == 'patch' else 'Wrote'} {label}."


def _create(root: Path, name: str, args: dict, existing) -> str:
    if existing:
        raise ToolError(f"skill {name} already exists; use patch or write_file")
    category = args.get("category", "")
    if category and not NAME.fullmatch(category):
        raise ToolError("category must be lowercase letters, digits, - and _")
    content = args.get("content", "")
    _check_skill_md(content, name)
    skill_dir = root / category / name
    if skill_dir.exists():
        raise ToolError(f"{skill_dir.relative_to(root)} already exists")
    private_dir(skill_dir)
    atomic_write(skill_dir / "SKILL.md", content)
    return f"Created skill {name}."


# --- the patch rule (the files plugin keeps the same copy; test_replace_text checks both) ----


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
    string = {"type": "string"}
    api.tool(
        "skill_view",
        "Load a skill's SKILL.md (with a list of its linked files), or one linked file via "
        "file_path (e.g. references/api.md).",
        {"properties": {"name": string, "file_path": string}, "required": ["name"]},
        skill_view,
        review=None,
    )
    actions = ["create", "patch", "write_file", "remove_file", "delete"]
    texts = ("name", "content", "category", "old_string", "new_string", "file_path")
    api.tool(
        "skill_manage",
        "Create or change a skill (procedural memory), one action per call:\n"
        "- create {content, category?}: SKILL.md starting with frontmatter `---\\nname: <name>\\n"
        'description: "Use when <trigger>. <what it does>."\\n---` and then the body\n'
        "- patch {old_string, new_string, replace_all?, file_path?}: replace a unique match\n"
        "- write_file {content, file_path?}: write a file (no file_path = replace SKILL.md)\n"
        "- remove_file {file_path}; delete: remove the whole skill\n"
        f"Names are lowercase letters, digits, - and _. Supporting files live under "
        f"{', '.join(SUBDIRS)}/. Only for {api.config['agent']['user_name']}: a skill is followed "
        "with all your tools, so nobody else can add or change one.",
        {
            "properties": {
                "action": {"type": "string", "enum": actions},
                **dict.fromkeys(texts, string),
                "replace_all": {"type": "boolean"},
            },
            "required": ["action", "name"],
        },
        skill_manage,
        review="prompt",
    )
    api.prompt_section(prompt_section)
    api.service("preload_skills", preload)
    api.on_compact(on_compact)
