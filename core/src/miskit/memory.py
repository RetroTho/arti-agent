from pathlib import Path

_MAX_ENTRY_LENGTH = 500
_MAX_PROMPT_ENTRIES_PER_CATEGORY = 20


class Memory:
    identity = """# Miskit

You are Miskit, a small and helpful AI assistant.
Use the memory below when it is relevant to the conversation.
When the user tells you something durable or asks you to remember it, save it with the memory tool.
When a question depends on older details, search memory and recall before answering.
Do not mention the memory unless it helps answer the user."""

    files = {
        "user": "user.md",
        "bot": "bot.md",
        "notes": "notes.md",
    }

    titles = {
        "user": "User",
        "bot": "Bot",
        "notes": "Notes",
    }

    def __init__(self, folder):
        self.folder = Path(folder)

    def setup(self):
        self.folder.mkdir(parents=True, exist_ok=True)

        for category in self.files:
            path = self._path(category)
            if not path.exists():
                title = self.titles[category]
                path.write_text(f"# {title}\n\n", encoding="utf-8")

    def read(self):
        parts = []

        for category in self.files:
            path = self._path(category)
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    parts.append(text)

        return "\n\n".join(parts)

    def system_prompt(self):
        memory_text = self.prompt_text()
        if not memory_text:
            return self.identity

        return f"{self.identity}\n\n# Memory\n\n{memory_text}"

    def prompt_text(self):
        parts = []
        skipped = 0

        for category in self.files:
            entries = self.entries(category)
            if not entries:
                continue

            title = self.titles[category]
            parts.append(f"# {title}")
            for entry in entries[:_MAX_PROMPT_ENTRIES_PER_CATEGORY]:
                parts.append(f"- {entry['text']}")
            skipped += max(0, len(entries) - _MAX_PROMPT_ENTRIES_PER_CATEGORY)

        if skipped:
            parts.append(
                f"\n[{skipped} more memories are saved. "
                "Use memory(action=\"search\") or memory(action=\"list\") when needed.]"
            )

        return "\n".join(parts)

    def add(self, category, text):
        self._check_category(category)

        if len(text) > _MAX_ENTRY_LENGTH:
            raise ValueError(
                f"memory entry is too long ({len(text)} characters). "
                f"Entries must be {_MAX_ENTRY_LENGTH} characters or fewer."
            )

        self.setup()
        with self._path(category).open("a", encoding="utf-8") as file:
            file.write(f"- {text}\n")

    def entries(self, category=None):
        self.setup()
        categories = list(self.files)
        if category is not None:
            self._check_category(category)
            categories = [category]

        entries = []
        for current_category in categories:
            count = 0
            lines = self._path(current_category).read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines):
                text = parse_memory_line(line)
                if text is None:
                    continue

                count += 1
                entries.append({
                    "id": f"{current_category}:{count}",
                    "category": current_category,
                    "line_number": line_number,
                    "text": text,
                })

        return entries

    def search(self, query, limit=10):
        text = str(query or "").strip().lower()
        if not text:
            return []

        terms = [part for part in text.split() if part]
        results = []
        for entry in self.entries():
            haystack = entry["text"].lower()
            if text in haystack:
                score = 10
            else:
                score = sum(1 for term in terms if term in haystack)

            if score:
                item = dict(entry)
                item["score"] = score
                results.append(item)

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def update(self, memory_id, text):
        text = str(text).strip()
        if not text:
            raise ValueError("memory update requires text")

        if len(text) > _MAX_ENTRY_LENGTH:
            raise ValueError(
                f"memory entry is too long ({len(text)} characters). "
                f"Entries must be {_MAX_ENTRY_LENGTH} characters or fewer."
            )

        entry = self._entry(memory_id)
        lines = self._path(entry["category"]).read_text(encoding="utf-8").splitlines()
        lines[entry["line_number"]] = f"- {text}"
        self._write_lines(entry["category"], lines)

    def remove(self, memory_id):
        entry = self._entry(memory_id)
        lines = self._path(entry["category"]).read_text(encoding="utf-8").splitlines()
        del lines[entry["line_number"]]
        self._write_lines(entry["category"], lines)

    def _path(self, category):
        return self.folder / self.files[category]

    def _entry(self, memory_id):
        text = str(memory_id).strip()
        parts = text.split(":", 1)
        if len(parts) != 2:
            raise ValueError("memory_id must look like user:1")

        category, number = parts
        self._check_category(category)

        try:
            index = int(number) - 1
        except ValueError as error:
            raise ValueError("memory_id must look like user:1") from error

        entries = self.entries(category)
        if index < 0 or index >= len(entries):
            raise ValueError("memory_id was not found")

        return entries[index]

    def _write_lines(self, category, lines):
        self._path(category).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _check_category(self, category):
        if category not in self.files:
            raise ValueError("category must be user, bot, or notes")


def parse_memory_line(line):
    text = line.strip()
    if text.startswith("- "):
        return text[2:].strip()
    return None
