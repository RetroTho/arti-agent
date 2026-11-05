"""An inert stdio MCP server for the Claude CLI: it lists Arti's tools and denies every call.

The CLI runs it as `python -I mcp_stub.py tools.json`, so the model sees Arti's tool
inventory while only Arti ever executes (and approves) a tool. Never imported by Arti.
"""

import json
import sys
from pathlib import Path


def respond(request: dict, manifest: list) -> dict | None:
    if "id" not in request:
        return None  # a notification: no reply
    method = request.get("method")
    if method == "initialize":
        version = (request.get("params") or {}).get("protocolVersion", "2024-11-05")
        result = {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "arti-inventory", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": manifest}
    elif method == "tools/call":
        denied = [{"type": "text", "text": "Denied: Arti runs its tools itself."}]
        result = {"isError": True, "content": denied}
    elif method == "ping":
        result = {}
    else:
        error = {"code": -32601, "message": "Method not found"}
        return {"jsonrpc": "2.0", "id": request["id"], "error": error}
    return {"jsonrpc": "2.0", "id": request["id"], "result": result}


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for line in sys.stdin:
        try:
            reply = respond(json.loads(line), manifest)
        except (ValueError, AttributeError, TypeError):
            reply = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
        if reply is not None:
            print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
