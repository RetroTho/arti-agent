"""web_search and web_extract.

Search reads DuckDuckGo's HTML results page. Extraction fetches a page in the host under the safety
plugin's address rules (net.py) and parses it in child.py, run under `.venv-web` (trafilatura,
pypdf). A bot wall or rate limit is an error, never "no results". Each page is its own block, marked
as untrusted; text over the visible limit is cut to head and tail, with the full text saved under
cache/web/. Private and local addresses are reachable only in the owner's own conversations, and
only when network.allow_private_urls is on.
"""

import base64
import json
import os
import re
import urllib.parse
import uuid
from html.parser import HTMLParser
from pathlib import Path

from arti.tools import MAX_RESULT_CHARS, ToolError
from plugins.safety import net, proc, redact

REQUIRES = ("safety",)  # imports from it, and its gate reviews this plugin's tools

DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
DEFAULT_CHARS = 15_000
MAX_CHARS = MAX_RESULT_CHARS - 2_000  # what fits in one result block with its header
REQUEST_SECONDS = 30
PARSE_SECONDS = 40
MAX_PAGE_BYTES = 5_000_000
BLOCKED = "the search was blocked or rate-limited ({}); this is not an empty result. Retry later."
TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml+xml")
CHILD = Path(__file__).resolve().with_name("child.py")  # run by path, in .venv-web


def web_python(app) -> str:
    return str(app.code_dir / ".venv-web" / "bin" / "python")


# --- search ----------------------------------------------------------------------------------


def web_search(args: dict, ctx) -> str:
    query, limit = args["query"].strip(), args.get("limit", 5)
    try:
        rows = ddg_search(query, ctx.cancel)
    except net.WebError as exc:
        raise ToolError(f"search failed: {exc}") from None
    lines = []
    for row in rows[:limit]:
        title, snippet = " ".join(row["title"].split()), " ".join(row["description"].split())
        lines.append(f"{len(lines) + 1}. {title}\n{row['url']}\n{snippet[:1000]}")
    return "\n\n".join(lines) if lines else f"No results for {query!r}."


class _DuckDuckGoPage(HTMLParser):
    """Collects div.result blocks: a.result__a gives link and title, a.result__snippet the text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self.blocked = self.no_results = False
        self._row: dict | None = None
        self._field: str | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        if tag == "div" and "result" in classes:
            self.no_results |= "result--no-result" in classes
            self._row = (
                None if "result--ad" in classes else {"url": "", "title": "", "description": ""}
            )
            if self._row is not None:
                self.results.append(self._row)
        elif tag == "a" and self._row is not None and "result__a" in classes:
            self._row["url"], self._field = attrs.get("href") or "", "title"
        elif tag == "a" and self._row is not None and "result__snippet" in classes:
            self._field = "description"
        elif "no-results" in classes:
            self.no_results = True
        elif any(c.startswith("anomaly") for c in classes) or (
            tag == "form" and re.search(r"anomaly|challenge|captcha", str(attrs).lower())
        ):
            self.blocked = True

    def handle_endtag(self, tag):
        if tag == "a":
            self._field = None

    def handle_data(self, data):
        if self._row is not None and self._field:
            self._row[self._field] += data


def _destination(href: str) -> str:
    """The real target of a result link (DuckDuckGo wraps them in /l/?uddg=...), or ""."""
    link = urllib.parse.urljoin(DUCKDUCKGO_URL, href.strip())
    parts = urllib.parse.urlsplit(link)
    if parts.hostname and parts.hostname.endswith("duckduckgo.com"):
        if parts.path != "/l/":
            return ""  # ads (/y.js) and internal links
        link = urllib.parse.parse_qs(parts.query).get("uddg", [""])[0]
    return link if link.startswith(("http://", "https://")) else ""


def parse_duckduckgo(page: str) -> list[dict]:
    """Organic results; a challenge page or an unknown layout raises instead of returning []."""
    parser = _DuckDuckGoPage()
    parser.feed(page)
    rows = []
    for row in parser.results:
        row["url"] = _destination(row["url"])
        if row["url"] and row["title"].strip():
            rows.append(row)
    if rows or parser.no_results:
        return rows
    if parser.blocked:
        raise net.WebError(BLOCKED.format("challenge page"))
    raise net.WebError("DuckDuckGo returned a page Arti does not recognise; not an empty result")


def ddg_search(query: str, cancel) -> list[dict]:
    try:
        response = net.request(
            DUCKDUCKGO_URL,
            method="POST",
            body=urllib.parse.urlencode({"q": query}).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=REQUEST_SECONDS,
            cancel=cancel,
        )
    except net.WebError as exc:
        if exc.status in (403, 429):
            raise net.WebError(BLOCKED.format(f"HTTP {exc.status}")) from None
        raise
    if response.status != 200:  # DuckDuckGo serves its challenge with HTTP 202
        raise net.WebError(BLOCKED.format(f"HTTP {response.status}"))
    return parse_duckduckgo(response.body.decode("utf-8", "replace"))


# --- local extraction ------------------------------------------------------------------------


def _decode(content_type: str, body: bytes) -> str:
    """body as text, in the charset the Content-Type header or a <meta> tag names (else UTF-8)."""
    header = re.search(r"charset=[\"']?([\w-]+)", content_type)
    meta = re.search(rb"<meta[^>]+charset\s*=\s*[\"']?([\w-]+)", body[:4096], re.I)
    charset = header[1] if header else meta[1].decode() if meta else "utf-8"
    if charset.lower().startswith("utf-16"):  # a declaration readable as ASCII is not UTF-16
        charset = "utf-8"
    try:
        return body.decode(charset, "replace")
    except LookupError:  # an unknown charset name
        return body.decode("utf-8", "replace")


def local_page(ctx, url: str) -> dict:
    response = net.request(
        url,
        timeout=REQUEST_SECONDS,
        allow_private=net.private_ok(ctx),
        limit=MAX_PAGE_BYTES,
        cancel=ctx.cancel,
    )
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type or response.body.lstrip().startswith(b"%PDF-"):
        body = base64.b64encode(response.body).decode()
        return _parse_in_child(ctx, {"kind": "pdf", "url": response.url, "body_b64": body})
    if content_type and not content_type.startswith(TEXT_TYPES):
        raise net.WebError(f"unsupported content type {content_type.split(';')[0]}")
    text = _decode(content_type, response.body)
    if "\x00" in text:
        raise net.WebError("the page is binary, not text")
    looks_like_html = text.lstrip()[:15].lower().startswith(("<!doctype", "<html"))
    if "html" not in content_type and not looks_like_html:
        return {"title": "", "text": text, "url": response.url}  # plain text, JSON, XML
    reject_challenge(text)
    return _parse_in_child(ctx, {"kind": "html", "url": response.url, "text": text})


def _parse_in_child(ctx, request: dict) -> dict:
    result = proc.run(
        [web_python(ctx.app), "-I", "-B", str(CHILD)],
        env=redact.bare_env(),
        input=json.dumps(request).encode(),
        timeout=PARSE_SECONDS,
        cancel=ctx.cancel,
        max_chars=3_000_000,  # the child's text limit, JSON-escaped
        merge_stderr=False,
    )
    if result.code is None:
        raise net.WebError("parsing the page was cancelled or timed out")
    try:
        answer = json.loads(result.out)
    except ValueError:
        raise net.WebError(f"the page parser failed (exit {result.code})") from None
    if not answer.get("ok"):
        raise net.WebError(answer.get("error") or "the page parser failed")
    return {"title": answer["title"], "text": answer["text"], "url": request["url"]}


# --- bot walls and long pages ----------------------------------------------------------------

WALL = "the site answered with a bot wall / challenge page, not the content"
_WALL_TITLES = {
    "just a moment...",
    "attention required! | cloudflare",
    "access denied",
    "verify you are human",
    "robot or human?",
}
# Markup only a challenge page has (not, say, the bot-detection script on ordinary pages).
_CHALLENGE_MARKUP = re.compile(r"cf-chl-|/cdn-cgi/challenge-platform/h/|id=.?challenge-form")
_WALL_PHRASES = (
    "verify you are human",
    "please complete the captcha",
    "enable javascript and cookies to continue",
    "unusual traffic from your computer",
)


def reject_challenge(markup: str) -> None:
    """Raise for raw HTML that is a challenge page (they are short; a long page is content)."""
    if len(markup) < 50_000 and _CHALLENGE_MARKUP.search(markup.lower()):
        raise net.WebError(WALL)


def reject_wall(title: str, text: str) -> None:
    """Raise for extracted text of a bot wall, which must never pass as the page's content."""
    head = text[:12_000].lower()
    if title.strip().lower() in _WALL_TITLES or (
        # Only short pages: a long article may quote these phrases.
        len(text) < 12_000 and any(phrase in head for phrase in _WALL_PHRASES)
    ):
        raise net.WebError(WALL)


def excerpt(ctx, text: str, limit: int) -> str:
    """text if it fits; else its head and tail plus the path of the full text."""
    if len(text) <= limit:
        return text
    path = ctx.cache_dir("web") / f"{uuid.uuid4().hex}.md"
    path.write_text(text, encoding="utf-8")
    head, tail = limit * 2 // 3, limit // 3
    return (
        f"{text[:head]}\n\n[... {len(text) - head - tail} characters omitted ...]\n\n"
        f"{text[-tail:]}\n\n[Full text ({len(text)} chars): {path} - read it with read_file]"
    )


def web_extract(args: dict, ctx) -> list[str]:
    """One text block per page (each is capped on its own, so no page crowds out another)."""
    limit = min(args.get("char_limit", DEFAULT_CHARS), MAX_CHARS)
    blocks, failed = [], 0
    for url in args["urls"]:
        ctx.check()
        url = url.strip()
        try:
            page = local_page(ctx, url)
            reject_wall(page["title"], page["text"])
            if not page["text"].strip():
                raise net.WebError("no readable text found")
        except (net.WebError, KeyError, TypeError, ValueError) as exc:
            failed += 1
            blocks.append(f"URL: {url[:500]}\nError: {exc}")
            continue
        title = str(page["title"] or "(no title)")[:300]
        header = f"{net.untrusted(page['url'])}\n# {title}\nURL: {url[:500]}"
        if page["url"] != url:
            header += f"\nFinal URL: {page['url'][:500]}"
        blocks.append(f"{header}\n\n{excerpt(ctx, page['text'], limit)}")
    if failed == len(blocks):
        raise ToolError("\n\n".join(blocks))
    return [f"{failed} of {len(blocks)} pages could not be read.", *blocks] if failed else blocks


def register(api) -> None:
    api.tool(
        "web_search",
        "Search the web: numbered titles, URLs and snippets.",
        {
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "default 5",
                },
            },
            "required": ["query"],
        },
        web_search,
        review=None,
        guest=True,
        timeout=60,
    )
    api.tool(
        "web_extract",
        "Read 1-5 web pages (HTML, PDF or text) as text; no JavaScript or logins (use the "
        "browser). Long pages are cut, with the full text saved for read_file.",
        {
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "char_limit": {
                    "type": "integer",
                    "minimum": 2000,
                    "maximum": MAX_CHARS,
                    "description": f"per page (default {DEFAULT_CHARS})",
                },
            },
            "required": ["urls"],
        },
        web_extract,
        # A URL that could carry data out (a file read earlier) is shown to the judge or owner.
        review=lambda args: "url" if any(map(net.url_carries_data, args["urls"])) else None,
        guest=True,
        timeout=300,
        available=lambda app: os.path.exists(web_python(app)),
    )
