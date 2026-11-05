"""vision_analyze: look at an image and answer a question about it.

The model looks at images natively: the tool loads the image (`load_image`: a local file, an
http(s) URL or a data: URL, under the rules described there), saves a copy in this chat's
cache/vision and returns it as an image block with the question. Fetching a URL that could
carry data out (net.url_carries_data) is reviewed, as it is for every other fetch.
"""

import base64
import binascii
import re
import uuid

from arti import media
from arti.tools import ToolError
from plugins.safety import net, risk

REQUIRES = ("safety",)  # imports from it, and its gate reviews this plugin's tools


def vision_analyze(args: dict, ctx) -> list:
    source = args["image"].strip()
    data = load_image(source, ctx)
    path = ctx.cache_dir("vision") / f"{uuid.uuid4().hex}{image_suffix(data)}"
    path.write_bytes(data)
    shown = "a data: URL" if source[:5].lower() == "data:" else source[:300]
    question = args.get("question") or "Describe this image."
    text = f"Image loaded from {shown}. Look at it and answer: {question}"
    return [{"type": "image", "path": str(path)}, {"type": "text", "text": text}]


# --- loading the image the model names (the image tool keeps the same copy) -------------------

MAX_IMAGE_BYTES = 10_000_000


def load_image(source: str, ctx) -> bytes:
    """The bytes of an image named by a local path, an http(s) URL or a data: URL. A URL is
    fetched under the safety plugin's address rules; a local file must not be a protected one, and
    in someone else's lane must be in this chat's own folder. ToolError when it can't be loaded."""
    source = source.strip()
    scheme = source.split(":", 1)[0].lower()
    if scheme == "data":
        match = re.fullmatch(r"data:image/[\w.+-]+;base64,([A-Za-z0-9+/=\s]+)", source, re.S | re.I)
        if not match:
            raise ToolError("a data: URL must be data:image/...;base64,...")
        try:
            data = base64.b64decode(match[1])
        except binascii.Error:
            raise ToolError("the data: URL is not valid base64") from None
    elif scheme in ("http", "https"):
        try:
            data = net.fetch_body(source, ctx, MAX_IMAGE_BYTES)
        except ValueError as exc:
            raise ToolError(f"could not download the image: {exc}") from None
    else:
        try:
            path = risk.readable_path(source, ctx)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        if not ctx.trusted and not path.is_relative_to(ctx.context_dir.resolve()):
            raise ToolError("in this chat only images in this chat's own folder can be loaded")
        if not path.is_file():
            raise ToolError(f"no image file at {path}")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ToolError("the image is larger than 10 MB")
        data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ToolError("the image is larger than 10 MB")
    image_suffix(data)
    return data


def image_suffix(data: bytes) -> str:
    """The file suffix for image bytes; ToolError when they are not a known image format."""
    kind = media.image_kind(data)
    if not kind:
        raise ToolError("that is not a PNG, JPEG, GIF, WebP, TIFF or HEIC image")
    return ".jpg" if kind == "jpeg" else f".{kind}"


def review(args: dict) -> str | None:
    return "url" if net.fetch_carries_data(args.get("image") or "") else None


def register(api) -> None:
    api.tool(
        "vision_analyze",
        "Look at an image (a local path, an http(s) URL or a data: URL) and answer a question "
        "about it.",
        {
            "properties": {
                "image": {"type": "string", "minLength": 1},
                "question": {"type": "string", "maxLength": 4000},
            },
            "required": ["image"],
        },
        vision_analyze,
        review=review,
        guest=True,
        timeout=60,
    )
