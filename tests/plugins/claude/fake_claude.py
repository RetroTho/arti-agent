"""A stand-in for the `claude` CLI that speaks stream-json and calls the relay like the real one.

Run as: python -I fake_claude.py CONFIG.json <cli arguments>. CONFIG holds `mode` and the
file paths where it records what it saw.
"""

import http.client
import json
import os
import subprocess
import sys
import time
from urllib.parse import urlsplit

config = json.load(open(sys.argv[1]))
argv = sys.argv[2:]
REMINDER = "\n<system-reminder>today is Thursday</system-reminder>"


def emit(event):
    print(json.dumps(event), flush=True)


def post(body):
    url = urlsplit(os.environ["ANTHROPIC_BASE_URL"])
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    headers = {"Content-Type": "application/json", "Authorization": "Bearer cli-oauth-token"}
    conn.request("POST", url.path + "/v1/messages?beta=true", json.dumps(body), headers)
    response = conn.getresponse()
    return response.status, response.read()


with open(config["record"], "w") as f:
    json.dump({"argv": argv, "env": dict(os.environ), "cwd": os.getcwd()}, f)
mode = config.get("mode", "reply")
emit({"type": "system", "subtype": "init"})
if mode == "hang":
    child = subprocess.Popen(["sleep", "60"])
    with open(config["pid_file"], "w") as f:
        f.write(str(child.pid))
    time.sleep(60)
if mode in ("logged_out", "limited"):  # the CLI refuses on its own, before any request
    error = "authentication_failed" if mode == "logged_out" else "rate_limit"
    emit({"type": "assistant", "error": error, "message": {"content": []}})
    emit({"type": "result", "subtype": "success", "is_error": True, "result": "refused"})
    sys.exit(1)

frames = []
for line in sys.stdin:
    frame = json.loads(line)
    frames.append(frame)
    if frame.get("shouldQuery") is False:
        time.sleep(config.get("ack_delay", 0))
        emit({"type": "result", "subtype": "success", "is_error": False, "num_turns": 0})

messages = [frame["message"] for frame in frames]
newest = messages[-1]["content"]
results = [b for b in newest if b["type"] == "tool_result"]
if results:  # like the CLI: its context rides on the last tool result
    results[-1]["content"].append({"type": "text", "text": REMINDER})
else:  # on the first turn it goes ahead of the frame
    newest.insert(0, {"type": "text", "text": REMINDER})
newest[-1]["cache_control"] = {"type": "ephemeral"}
messages.append(
    {
        "role": "system",
        "content": [{"type": "text", "text": "date", "cache_control": {"type": "ephemeral"}}],
    }
)
model = config.get("model") or argv[argv.index("--model") + 1]
status, data = post({"model": model, "messages": messages, "stream": True})
if mode == "stall":  # the reply arrived, but the CLI never reports it
    time.sleep(60)
if config.get("retry"):
    with open(config["record"] + ".retry", "w") as f:
        f.write(str(post({"model": model, "messages": messages})[0]))
emit(
    {
        "type": "rate_limit_event",
        "rate_limit_info": {  # the shape the real CLI sends (seen live)
            "status": "allowed",
            "rateLimitType": "five_hour",
            "resetsAt": 1760000000,
            "unifiedWindows": {
                "five_hour": {"utilization": 0.25, "resetsAt": 1760000000},
                "seven_day": {"utilization": 0.1, "resetsAt": 1760500000},
            },
        },
    }
)
if status != 200:
    emit({"type": "assistant", "error": "rate_limit", "message": {"content": []}})
    emit({"type": "result", "subtype": "success", "is_error": True, "num_turns": 1})
    sys.exit(1)
if b'"tool_use"' in data:
    emit({"type": "result", "subtype": "error_max_turns", "is_error": True, "num_turns": 1})
    sys.exit(1)
emit({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1})
