"""image_generate: one ChatGPT image generation or edit through Arti's own ChatGPT login.

Before `python -m arti login` the tool is hidden. [image] model picks the quality.

The image is saved privately under this chat's artifacts/images/ (kept, unlike cache/) and shown
to the model as a preview; it is never sent to the chat by itself (MEDIA:/path in a reply sends
it). The request is never retried: once it may have reached the service, a failure is an unknown
outcome, because the image may have been made and billed. Someone else's lane of a group, which
spends the owner's account, may ask for at most GUEST_PER_HOUR images an hour (counted in memory).
A reference image fetched from a URL that could carry data out (net.url_carries_data) is reviewed.
"""

import base64
import binascii
import re
import threading
import time
import uuid

from arti import fsutil, media
from arti.tools import ToolError, UnknownOutcome
from plugins.openai import codex
from plugins.safety import net, risk

REQUIRES = ("safety", "openai")  # its gate reviews the tool; the login and client are openai's

SETTINGS = {"model": "gpt-image-2-medium"}
CHOICES = {"model": ("gpt-image-2-low", "gpt-image-2-medium", "gpt-image-2-high")}
SIZES = {"square": "1024x1024", "landscape": "1536x1024", "portrait": "1024x1536"}
MAX_REFERENCE_BYTES = 50_000_000
REQUEST_SECONDS = 300
BILLED = "it may have been made and billed; do not retry without asking"
GUEST_PER_HOUR = 10
_asked: dict[str, list[float]] = {}  # a guest lane's key -> when it asked for images
_asked_lock = threading.Lock()


def _count_guest_request(ctx) -> None:
    """ToolError once someone else's lane has asked for GUEST_PER_HOUR images in the last hour."""
    if ctx.trusted:
        return
    now = time.monotonic()
    with _asked_lock:
        recent = [t for t in _asked.get(ctx.key, []) if now - t < 3600]
        if len(recent) >= GUEST_PER_HOUR:
            raise ToolError(
                f"this chat has had {GUEST_PER_HOUR} images in the last hour, its limit; "
                "try again later"
            )
        _asked[ctx.key] = [*recent, now]


def image_generate(args: dict, ctx) -> list:
    references = [load_image(src, ctx) for src in args.get("reference_images", [])]
    if sum(map(len, references)) > MAX_REFERENCE_BYTES:
        raise ToolError("the reference images are larger than 50 MB together")
    ctx.check()
    _count_guest_request(ctx)
    size = SIZES[args.get("aspect_ratio", "square")]
    quality = ctx.app.config["image"]["model"].rsplit("-", 1)[-1]  # gpt-image-2-medium -> medium
    request = codex.generate_image(
        codex.shared_auth(ctx.app.data_dir),
        args["prompt"],
        size,
        quality,
        references,
        timeout=REQUEST_SECONDS,
    )
    try:  # codex refuses reference types it cannot send (TIFF, HEIC) before sending anything
        data = ctx.call(request, timeout=REQUEST_SECONDS + 30)
        suffix = image_suffix(data)
    except codex.ImageError as exc:
        if not exc.sent:
            raise ToolError(str(exc)) from None
        raise UnknownOutcome(f"{exc}. The image {BILLED}.") from None
    except ToolError as exc:  # interrupted in flight, or the answer is no image
        raise UnknownOutcome(f"the image request did not finish ({exc}); {BILLED}.") from None
    folder = fsutil.private_dir(ctx.context_dir / "artifacts" / "images")
    path = folder / f"{uuid.uuid4().hex}{suffix}"
    fsutil.atomic_write(path, data)
    text = (
        f"Generated image saved at {path} ({size}, {quality} quality). "
        f"To send it, put MEDIA:{path} on its own line in a reply."
    )
    return [{"type": "image", "path": str(path)}, {"type": "text", "text": text}]


def review(args: dict) -> str | None:
    sources = args.get("reference_images") or []
    return "url" if any(net.fetch_carries_data(s) for s in sources) else None


def available(app) -> bool:
    return codex.shared_auth(app.data_dir).path.exists()


# --- loading the image the model names (vision keeps the same copy) -------------------

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


def register(api) -> None:
    """The [image] settings and the image_generate tool."""
    api.settings("image", SETTINGS, CHOICES)
    api.tool(
        "image_generate",
        "Generate an image with ChatGPT's image model, or edit/combine reference_images (paths "
        "or URLs). The image is saved and shown to you, not sent.",
        {
            "properties": {
                "prompt": {"type": "string", "minLength": 1, "maxLength": 32_000},
                "aspect_ratio": {"type": "string", "enum": list(SIZES)},
                "reference_images": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 8,
                },
            },
            "required": ["prompt"],
        },
        image_generate,
        review=review,
        guest=True,
        timeout=REQUEST_SECONDS + 60,
        available=available,
    )
