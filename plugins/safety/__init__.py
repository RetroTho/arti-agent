"""safety: the approval gate, redaction, network rules and child processes.

Without this plugin Arti runs unguarded: tool calls run with no review and nothing is redacted.
So every plugin that can do something risky declares `REQUIRES = ("safety",)` and is skipped when
this one is disabled or fails to load (plugin.py); a plugin with a reviewed tool must
(tests/test_plugin.py checks). Those plugins import what they need from here:

- `approvals.py`: the gate (`check`, registered with api.gate) and asking the owner
  (`ask_owner`, `/approve`, `/deny`);
- `risk.py`: what a tool call is (never allowed, harmless, or in need of review), and the paths
  tools may act on (`resolve_path`, `is_secret_path`);
- `redact.py`: the redactor (api.redactor) and secret-free environments for children;
- `net.py`: URL safety and the one HTTP request function every host-side fetch uses;
- `proc.py`: running and killing child processes, and the records that let the next start kill
  what a crashed Arti left behind.

Its two config sections are `[approvals]` (how calls are reviewed) and `[network]`.

Order: plugins start in load order and stop in reverse, and every plugin that may start a child
requires this one, so it loads before them. Its start hook therefore runs before theirs (the
leftovers of a crash are killed before anything new is started) and its stop hook after theirs
(it kills whatever their stop hooks left running).
"""

import logging
from pathlib import Path

from . import approvals, proc, redact, risk

log = logging.getLogger(__name__)

APPROVALS = {
    "mode": "smart",  # off | manual | smart (a judge model decides routine calls)
    "unattended": "deny",  # runs with no chat: deny, or let the judge decide (smart)
    "timeout": 300.0,  # seconds to wait for the owner's answer
    "provider": "",  # the judge's provider; "" = agent.provider
    "model": "",  # the judge's model; "" = that provider's judge model
    "policy": "",  # the judge's policy; "" = approvals.DEFAULT_POLICY
    "allow": [],  # category grants the owner trusts ("shell:git", "gui:browser")
    "deny": [],  # regexes refused in every mode and lane
}
CHOICES = {"mode": ("off", "manual", "smart"), "unattended": ("deny", "smart")}
NETWORK = {"allow_private_urls": False}  # let the owner's tools reach local addresses


def start(app) -> None:
    """Kill what a crashed Arti left running, then record this run's children."""
    if killed := proc.sweep(app.data_dir):
        log.warning("killed %d process groups left by a previous run", killed)
    proc.keep_records(app.data_dir)


def stop(app) -> None:
    """Kill every child still running (the other plugins have stopped by now). Records are
    still kept: a child something starts after this is swept at the next start."""
    proc.kill_all()


def secret_file(path: str) -> str | None:
    """No secret file leaves Arti, whoever asks (App.file_problem)."""
    return f"{Path(path).name} is a secret file" if risk.is_secret_path(path) else None


def register(api) -> None:
    api.settings("approvals", APPROVALS, CHOICES, provider_keys=("provider",))
    api.settings("network", NETWORK)
    api.gate(approvals.check)
    api.redactor(redact.redact, redact.redact_obj)
    api.file_check(secret_file)
    api.chat_command(
        "approve",
        lambda app, inp, arg: approvals.handle_command("approve", arg),
        "[id] [once|session|always] - answer an approval question",
    )
    api.chat_command(
        "deny",
        lambda app, inp, arg: approvals.handle_command("deny", arg),
        "[id] - refuse what an approval question asks",
    )
    api.on_start(start, once=True)
    api.on_stop(stop, timeout=15)  # TERM, a grace, then KILL every child group: give it time
