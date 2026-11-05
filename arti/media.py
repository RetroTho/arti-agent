"""Images and attachments: inbound inputs to content blocks, and images the model can read.

An image a model can take as it is (a PNG, JPEG, GIF or WebP of at most 2048 px and 3.75 MB,
measured from its own bytes) is used as is; any other is converted to a JPEG that is, with
ImageMagick (in the sandbox image). Either way it is kept under the chat's own context
folder (<data>/media for home, contexts/<scope>/media for a group) and stored in history as a
path. Only a request reads the file as base64 (`encode`), so the database never holds image
bytes. Any other file is named by its path (in a group, a copy in its folder).
"""

import base64
import logging
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from arti.fsutil import private_dir

log = logging.getLogger(__name__)

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}
API_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_EDGE = 2048
MAX_BYTES = 3_750_000
CONVERT_TIMEOUT = 30


# Leading bytes of the image formats known here (WebP and HEIC are checked further in).
_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
}


def image_kind(head: bytes) -> str:
    """The image format bytes start with: png, jpeg, gif, webp, tiff, heic, or "" if none.
    Extensions are never trusted."""
    for magic, kind in _MAGIC.items():
        if head.startswith(magic):
            return kind
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"mif1", b"avif"):
        return "heic"
    return ""


def sniff(path: Path) -> str:
    """The media type of an image file a model can take as is ("" for any other file)."""
    with open(path, "rb") as f:
        kind = image_kind(f.read(12))
    return f"image/{kind}" if kind in ("png", "jpeg", "gif", "webp") else ""


def dimensions(path: Path) -> tuple[int, int] | None:
    """(width, height) of a PNG, JPEG, GIF or WebP read from its own bytes, or None when they
    can't be told. Only for files of at most MAX_BYTES, which are read whole."""
    data = Path(path).read_bytes()
    kind = image_kind(data[:12])
    if kind == "png" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if kind == "gif" and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if kind == "webp" and len(data) >= 30:
        chunk = data[12:16]
        if chunk == b"VP8 ":
            return int.from_bytes(data[26:28], "little") & 0x3FFF, int.from_bytes(
                data[28:30], "little"
            ) & 0x3FFF
        if chunk == b"VP8L":
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8X":
            return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(
                data[27:30], "little"
            )
    if kind == "jpeg":
        at = 2
        while at + 9 < len(data) and data[at] == 0xFF:
            marker, length = data[at + 1], int.from_bytes(data[at + 2 : at + 4], "big")
            if marker in _SOF:
                return int.from_bytes(data[at + 7 : at + 9], "big"), int.from_bytes(
                    data[at + 5 : at + 7], "big"
                )
            at += 2 + length
    return None


_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}  # JPEG frames


def _converter(source: Path, out: Path, quality: int) -> list[str]:
    """The ImageMagick command that writes `source` as a JPEG of at most MAX_EDGE px (ValueError
    when ImageMagick is not installed)."""
    magick = shutil.which("magick") or shutil.which("convert")
    if not magick:
        raise ValueError("converting this image needs ImageMagick, which is not installed")
    size = f"{MAX_EDGE}x{MAX_EDGE}>"  # only ever shrunk
    return [magick, f"{source}[0]", "-auto-orient", "-resize", size, "-quality", str(quality),
            f"jpeg:{out}"]  # fmt: skip


def _convert(source: Path, out: Path, quality: int) -> None:
    """Run the converter with a bare environment; ValueError when it fails."""
    try:
        result = subprocess.run(
            _converter(source, out, quality),
            capture_output=True,
            env={"PATH": "/usr/bin:/bin"},
            timeout=CONVERT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("the image could not be read") from None
    if result.returncode != 0:
        raise ValueError("the image could not be read")


def image_block(path: Path, context_dir: Path) -> dict:
    """A history image block for `path`, converting it if a model could not take it as is.

    The image ends up in `context_dir` (the chat's own folder): one from elsewhere is copied into
    its media/, so history outlives the original and a chat's images stay in that chat's folder.
    """
    path = Path(path).expanduser()
    media_type = sniff(path)
    if media_type in API_TYPES and path.stat().st_size <= MAX_BYTES:
        size = dimensions(path)
        if size and max(size) <= MAX_EDGE:
            copy = own_copy(path, context_dir)
            return {"type": "image", "path": str(copy), "media_type": media_type}
    out = private_dir(Path(context_dir) / "media") / f"{uuid.uuid4().hex}.jpg"
    for quality in (80, 60, 40):
        _convert(path, out, quality)
        if out.is_file() and out.stat().st_size <= MAX_BYTES:
            return {"type": "image", "path": str(out), "media_type": "image/jpeg"}
    out.unlink(missing_ok=True)
    raise ValueError("the image is too large even after compression")


def file_size(path: str) -> int | None:
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


def encode(path: Path) -> tuple[str, str]:
    """(media_type, base64) of a normalised image file, for a request."""
    data = Path(path).read_bytes()
    return sniff(Path(path)) or "image/jpeg", base64.b64encode(data).decode("ascii")


def stamp(app, ts: float) -> str:
    """ "Thu Oct 23 4:05 PM" in the owner's time zone; now when ts is 0."""
    when = datetime.fromtimestamp(ts, app.tz) if ts else datetime.now(app.tz)
    return when.strftime("%a %b %-d %-I:%M %p")


def own_copy(path: Path, context_dir: Path) -> Path:
    """`path` itself when it is inside `context_dir`, else a copy in its media/ folder (OSError
    when it cannot be copied): a place's files stay in that place's folder, the only place its
    conversations may read or send."""
    if path.resolve().is_relative_to(Path(context_dir).resolve()):
        return path
    name = path.stem[:60] + path.suffix[:10]  # the sender's file name, kept short
    copy = private_dir(Path(context_dir) / "media") / f"{uuid.uuid4().hex[:8]}-{name}"
    shutil.copyfile(path, copy)
    return copy


def input_blocks(app, inp, cancel, context_dir: Path) -> list[dict]:
    """An input as content blocks: stamped text, then its attachments (runs in a thread).
    Images are stored under `context_dir`, the folder of the chat the input belongs to; in a
    group, so is every other file."""
    if inp.kind == "note":
        blocks = [{"type": "text", "text": inp.text}] if inp.text else []
    else:
        who = f" {inp.sender_name}:" if inp.sender_name else ""
        blocks = [{"type": "text", "text": f"[{stamp(app, inp.ts)}]{who} {inp.text}".rstrip()}]
    in_group = Path(context_dir) != Path(app.data_dir)
    for name in inp.attachments:
        blocks.append(_attachment_block(app, Path(name), cancel, context_dir, in_group))
    return blocks


def _attachment_block(app, path: Path, cancel, context_dir: Path, in_group: bool) -> dict:
    """One attachment: an image block, or a text block naming its path."""
    if path.suffix.lower() in IMAGE_TYPES:
        try:
            return image_block(path, context_dir)
        except (OSError, ValueError) as error:
            log.warning("image attachment skipped: %s", error)
            return {"type": "text", "text": f"[an attached image could not be loaded: {error}]"}
    if in_group:
        try:
            path = own_copy(path, context_dir)
        except OSError as error:
            log.warning("attachment not copied: %s", error)
            return {"type": "text", "text": f"[an attached file could not be loaded: {path.name}]"}
    return {"type": "text", "text": f"[attachment: {path}]"}
