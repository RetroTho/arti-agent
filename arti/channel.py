"""The messaging platform, as the App sees it.

A channel brings inputs in (it calls `app.admit`) and delivers what Arti sends. Every method takes
a conversation key, the channel's own string for a conversation. The agent's channel is always
the link to the host (link.py), which owns the real chat platform; `FakeChannel` prints instead
(--once, tests).

The channel is the only part that knows the platform's spellings, so core never parses a key. It
tells core who the owner is (`home`, `owner`), and for each key where the conversation lives
(`scope_of`: "home", or a place such as a group chat, with its own context folder) and whose it is
(`person_of`). It hands every input over in one spelling: the owner's own chat always keyed
`home`, every sender a normalised handle (one person, one string), so core only ever compares
strings.

What Arti sends reaches a channel as parts (`send`): text, as written, and files. Core has
already turned every `MEDIA:/path` line of a reply into a file part, and checked that each file
may leave Arti (`App.file_problem`); the channel only formats text for its platform and sends.

"""

from dataclasses import dataclass


@dataclass
class Part:
    """One piece of what Arti sends: text, or a file (an absolute path)."""

    kind: str  # "text" | "file"
    value: str


class Channel:
    """The messaging platform. A channel plugin implements it; this base does nothing.

    Made by its factory before Arti starts (also just to read `home` and `owner`, for --once),
    so the constructor only reads settings: everything else waits for `start`."""

    home = ""  # the owner's one-on-one conversation key; "" when there is none
    owner = ""  # the owner's normalised handle, as senders are spelled

    async def start(self, app) -> None:
        pass

    async def stop(self) -> None:
        pass

    def scope_of(self, key: str) -> str:
        """The place conversation `key` lives in: "home" for the owner's own chat, or a stable
        id of another place (letters, digits and "-"; its files go in contexts/<scope>/), shared
        by every conversation there. This base has only home."""
        return "home"

    def person_of(self, key: str) -> str | None:
        """Whose conversation `key` is: "" for the owner's own one-on-one, else the normalised
        handle of the one person it talks with (the owner's own, for the owner's conversation in
        another place), or None for a key that is no conversation of this channel. Only "" and
        the owner's handle are the owner's (App.owns): trusted."""
        return ""

    def place_name(self, scope: str) -> str:
        """The name of a place for tags and questions ("Dinner crew"); "" for home."""
        return ""

    async def send(self, key: str, parts: list[Part]) -> str:
        """Deliver parts to a conversation, in order: "sent", "failed" (provably nothing sent)
        or "unknown". A file part is a file core has checked; a platform that re-reads it later
        (a copy for sending, say) checks the copy again with `App.file_problem`."""
        return "failed"

    def alive(self) -> bool:
        """Whether the channel still takes in new messages (the hang watchdog asks)."""
        return True

    async def typing(self, key: str, on: bool) -> None:
        pass

    def prompt_notes(self, key: str) -> str:
        """Platform hints for the system prompt."""
        return ""


class FakeChannel(Channel):
    """Prints what would be sent: --once and manual testing."""

    def __init__(self, home: str = "", owner: str = "", like: Channel | None = None):
        """`like`: a real channel (never started) whose keys this one reads the same way, so a
        --once task sees its places and people as the daemon would."""
        self.home, self.owner, self.like = home, owner, like

    def scope_of(self, key: str) -> str:
        return self.like.scope_of(key) if self.like else super().scope_of(key)

    def person_of(self, key: str) -> str | None:
        return self.like.person_of(key) if self.like else super().person_of(key)

    def place_name(self, scope: str) -> str:
        return self.like.place_name(scope) if self.like else super().place_name(scope)

    async def send(self, key: str, parts: list[Part]) -> str:
        for part in parts:
            what = "file to" if part.kind == "file" else "to"
            print(f"[{what} {key or 'console'}] {part.value}", flush=True)
        return "sent"
