import json
import re
from dataclasses import dataclass
from pathlib import Path

from miskit.message import format_message
from miskit.message import message_from_dict


_SNIPPET_CHARS = 320


@dataclass
class RecallHit:
    id: str
    source: str
    index: int
    score: int
    text: str


class RecallStore:
    def __init__(self, history):
        self.history = history

    def search(self, query, limit=10):
        query = str(query or "").strip()
        if not query:
            return []

        terms = search_terms(query)
        results = []
        for source, path in self.sources():
            for index, message in enumerate(self._read_path(path), start=1):
                text = format_message(message)
                score = score_text(text, query, terms)
                if score:
                    results.append(RecallHit(
                        id=f"{source}:{index}",
                        source=source,
                        index=index,
                        score=score,
                        text=snippet(text, query, terms),
                    ))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def read(self, recall_id, before=3, after=3):
        source, index = parse_recall_id(recall_id)
        path = self._source_path(source)
        messages = self._read_path(path)

        if index < 1 or index > len(messages):
            raise ValueError("recall id was not found")

        start = max(1, index - before)
        end = min(len(messages), index + after)
        result = []
        for current_index in range(start, end + 1):
            message = messages[current_index - 1]
            result.append({
                "id": f"{source}:{current_index}",
                "message": message,
            })
        return result

    def sources(self):
        current = self.history.path()
        if current.exists():
            yield "session", current

        archive = self.history.folder / "archive"
        if not archive.exists():
            return

        for path in sorted(archive.glob("session-*.jsonl"), reverse=True):
            yield path.stem, path

    def _source_path(self, source):
        source = str(source or "").strip()
        if source == "session":
            return self.history.path()

        path = self.history.folder / "archive" / f"{source}.jsonl"
        if not path.exists():
            raise ValueError("recall id was not found")
        return path

    def _read_path(self, path):
        messages = []
        path = Path(path)
        if not path.exists():
            return messages

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages.append(message_from_dict(data))
        return messages


def parse_recall_id(recall_id):
    text = str(recall_id or "").strip()
    source, separator, number = text.rpartition(":")
    if not separator or not source:
        raise ValueError("recall id must look like session:1")

    try:
        index = int(number)
    except ValueError as error:
        raise ValueError("recall id must look like session:1") from error

    return source, index


def search_terms(query):
    return re.findall(r"[A-Za-z0-9_'-]+", query.lower())


def score_text(text, query, terms):
    haystack = text.lower()
    needle = query.lower()
    score = haystack.count(needle) * 10
    for term in terms:
        score += haystack.count(term)
    return score


def snippet(text, query, terms):
    lower = text.lower()
    index = lower.find(query.lower())
    if index < 0:
        index = first_term_index(lower, terms)
    if index < 0:
        index = 0

    start = max(0, index - _SNIPPET_CHARS // 3)
    end = min(len(text), start + _SNIPPET_CHARS)
    value = text[start:end].replace("\n", " ").strip()
    if start > 0:
        value = "..." + value
    if end < len(text):
        value += "..."
    return value


def first_term_index(text, terms):
    positions = [text.find(term) for term in terms if term and text.find(term) >= 0]
    if not positions:
        return -1
    return min(positions)
