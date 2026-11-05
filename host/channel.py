"""A chat platform, as the host sees it. A channel package (channels/<name>/) implements one.

The channel brings messages in: for each it calls `host.receive(message)`, naming the sender by a
stable handle on the platform. The host finds the user with that handle and passes the message
to that user's agent; a message from anyone else is dropped. The channel sends what agents ask
(`send`, `typing`) to a chat, the channel's own string for a conversation, opaque to the host.

A channel package has `SETTINGS` (its config section's defaults, in host.toml under its name)
and `make(host, settings) -> Channel`.
"""

from dataclasses import dataclass, field
from pathlib import Path

from arti.channel import Part


@dataclass
class Message:
    """One message for an agent."""

    event_id: str  # unique and stable: a message delivered twice keeps it
    chat: str  # where it came from, and where an answer to it goes
    sender: str  # the sender's stable handle on the platform
    sender_name: str = ""
    text: str = ""
    files: list[Path] = field(default_factory=list)  # downloaded attachments (the host moves them)
    place: str = ""  # a group chat's stable id; "" for a one-on-one with Arti
    place_name: str = ""
    ts: float = 0.0  # when it was sent (unix time)


class Channel:
    """The base does nothing; a channel overrides what its platform has."""

    name = ""
    notes = ""  # how to write for this platform, for the agents' system prompts

    async def start(self, host) -> None:
        pass

    async def stop(self) -> None:
        pass

    def alive(self) -> bool:
        """Whether it still takes in new messages (the host's hang watchdog asks)."""
        return True

    def home_chat(self, handle: str) -> str:
        """The chat of a one-on-one with the person whose handle this is."""
        return handle

    async def send(self, chat: str, parts: list[Part]) -> str:
        """Deliver parts (text, or files: host paths) in order: "sent", "failed" (provably
        nothing sent) or "unknown". Stop at the first that failed or may have."""
        return "failed"

    async def typing(self, chat: str, on: bool) -> None:
        pass
