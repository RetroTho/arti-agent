"""Reply text as the channel gets it: text parts and file parts, in order (channel.Part).

A `MEDIA:/absolute/path` line on its own sends that file, the one way a reply sends a file; inside
a fenced code block it is only text. Everything else stays text as written, fences included:
the channel formats it for its platform (one without markdown drops the markers, say). Which
files may go out is the App's to check (`App.file_problem`).
"""

import re

from arti.channel import Part

_FENCE = re.compile(r"\s*(```|~~~)")
_MEDIA = re.compile(r"\s*MEDIA:(/\S.*?)\s*")


def parts(text: str) -> list[Part]:
    out: list[Part] = []
    pending: list[str] = []

    def flush() -> None:
        body = "\n".join(pending).strip("\n")
        if body.strip():
            out.append(Part("text", body))
        pending.clear()

    fence = ""
    for line in text.splitlines():
        marker = _FENCE.match(line)
        if fence:
            if marker and marker[1] == fence and not line.strip().strip(fence):
                fence = ""
        elif marker:
            fence = marker[1]
        elif media := _MEDIA.fullmatch(line):
            flush()
            out.append(Part("file", media[1]))
            continue
        pending.append(line)
    flush()
    return out
