from miskit.tool import Tool


class MemoryTool(Tool):
    name = "memory"
    description = "Add, search, list, update, and remove long-term memories in Miskit's markdown memory files."
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "list", "update", "remove"],
                "description": "Use add to save a memory, search to find saved memories, list to show memories, update to edit one, or remove to delete one.",
            },
            "category": {
                "type": "string",
                "enum": ["user", "bot", "notes"],
                "description": "Use user for user facts/preferences, bot for Miskit facts, notes for project notes.",
            },
            "query": {
                "type": "string",
                "description": "Search text for saved memories.",
            },
            "text": {
                "type": "string",
                "description": "One concise memory bullet to save or use as replacement text.",
            },
            "memory_id": {
                "type": "string",
                "description": "Memory id from the list action, like user:1.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return for list or search.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, memory):
        self.memory = memory
        self.added = []

    def run(self, arguments):
        action = str(arguments.get("action", "add")).strip() or "add"
        if action == "add":
            return self._add(arguments)
        if action == "search":
            return self._search(arguments)
        if action == "list":
            return self._list(arguments)
        if action == "update":
            return self._update(arguments)
        if action == "remove":
            return self._remove(arguments)

        return "Unknown memory action. Use add, search, list, update, or remove."

    def _add(self, arguments):
        category = str(arguments.get("category", "")).strip()
        text = str(arguments.get("text", "")).strip()

        if not category:
            return "Error: memory requires category."
        if not text:
            return "Error: memory requires text."

        try:
            self.memory.add(category, text)
        except ValueError as error:
            return f"Error: {error}"

        self.added.append((category, text))
        return f"Saved {category} memory: {text}"

    def _list(self, arguments):
        category = str(arguments.get("category", "")).strip() or None
        limit = self._limit(arguments.get("limit"), default=50, maximum=200)

        try:
            entries = self.memory.entries(category)
        except ValueError as error:
            return f"Error: {error}"

        total = len(entries)
        entries = entries[:limit]
        if not entries:
            if category is None:
                return "No memories."
            return f"No {category} memories."

        lines = ["Memories:"]
        for entry in entries:
            lines.append(f"- {entry['id']} [{entry['category']}] {entry['text']}")
        if total > limit:
            lines.append(f"\n[{total - limit} more memories not shown]")
        return "\n".join(lines)

    def _search(self, arguments):
        query = str(arguments.get("query", "")).strip()
        limit = self._limit(arguments.get("limit"), default=10, maximum=50)

        if not query:
            return "Error: memory search requires query."

        results = self.memory.search(query, limit=limit)
        if not results:
            return "No matching memories."

        lines = ["Matching memories:"]
        for entry in results:
            lines.append(f"- {entry['id']} [{entry['category']}] {entry['text']}")
        return "\n".join(lines)

    def _update(self, arguments):
        memory_id = str(arguments.get("memory_id", "")).strip()
        text = str(arguments.get("text", "")).strip()

        if not memory_id:
            return "Error: memory update requires memory_id."
        if not text:
            return "Error: memory update requires text."

        try:
            self.memory.update(memory_id, text)
        except ValueError as error:
            return f"Error: {error}"

        return f"Updated memory {memory_id}: {text}"

    def _remove(self, arguments):
        memory_id = str(arguments.get("memory_id", "")).strip()
        if not memory_id:
            return "Error: memory remove requires memory_id."

        try:
            self.memory.remove(memory_id)
        except ValueError as error:
            return f"Error: {error}"

        return f"Removed memory {memory_id}."

    def _limit(self, value, default=10, maximum=50):
        try:
            limit = int(value)
        except (TypeError, ValueError):
            limit = default

        if limit < 1:
            limit = default
        return min(limit, maximum)


def create_tool(config, services=None):
    services = services or {}
    memory = services.get("memory")
    if memory is None:
        raise ValueError("memory tool requires a memory service")

    return MemoryTool(memory)
