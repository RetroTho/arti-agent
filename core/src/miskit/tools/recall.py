from miskit.message import format_message
from miskit.recall import RecallStore
from miskit.tool import Tool


class RecallTool(Tool):
    name = "recall"
    description = "Search and read Miskit's full conversation history, including archived transcripts."
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "read"],
                "description": "Use search to find older conversation details, or read to view messages around a recall id.",
            },
            "query": {
                "type": "string",
                "description": "Text to search for in the conversation history.",
            },
            "recall_id": {
                "type": "string",
                "description": "Recall id from search results, like session:1 or session-2026-05-18T10-00-00-000000:12.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of search results to return.",
            },
            "before": {
                "type": "integer",
                "description": "For read: number of messages before the recall id to include.",
            },
            "after": {
                "type": "integer",
                "description": "For read: number of messages after the recall id to include.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, recall_store):
        self.recall_store = recall_store

    def run(self, arguments):
        action = str(arguments.get("action", "")).strip()
        if action == "search":
            return self._search(arguments)
        if action == "read":
            return self._read(arguments)

        return "Unknown recall action. Use search or read."

    def _search(self, arguments):
        query = str(arguments.get("query", "")).strip()
        limit = self._count(arguments.get("limit"), default=10, maximum=25, minimum=1)

        if not query:
            return "Error: recall search requires query."

        hits = self.recall_store.search(query, limit=limit)
        if not hits:
            return "No matching conversation history."

        lines = ["Matching conversation history:"]
        for hit in hits:
            lines.append(f"- {hit.id} [{hit.source}] {hit.text}")
        return "\n".join(lines)

    def _read(self, arguments):
        recall_id = str(arguments.get("recall_id", "")).strip()
        before = self._count(arguments.get("before"), default=3, maximum=20)
        after = self._count(arguments.get("after"), default=3, maximum=20)

        if not recall_id:
            return "Error: recall read requires recall_id."

        try:
            entries = self.recall_store.read(recall_id, before=before, after=after)
        except ValueError as error:
            return f"Error: {error}"

        lines = [f"Conversation around {recall_id}:"]
        for entry in entries:
            lines.append(f"\n[{entry['id']}]\n{format_message(entry['message'])}")
        return "\n".join(lines)

    def _count(self, value, default=10, maximum=25, minimum=0):
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = default

        if count < minimum:
            count = default
        return min(count, maximum)


def create_tool(config, services=None):
    services = services or {}
    history = services.get("history")
    if history is None:
        raise ValueError("recall tool requires a history service")

    return RecallTool(RecallStore(history))
