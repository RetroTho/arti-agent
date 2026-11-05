"""Parse one fetched page into readable text; runs under `.venv-web/bin/python -I`.

The host fetches the page (with its URL safety rules) and hands it over on stdin as JSON:
`{"kind": "html", "url", "text"}` or `{"kind": "pdf", "url", "body_b64"}`. This child only
parses, with trafilatura (HTML -> markdown) or pypdf (text layer), under CPU-time, memory and
file-size limits, because hostile pages can be made to blow up parsers (the input is at most
5 MB, and the plugin's timeout ends the child). It never touches the network.
stdout: `{"ok": true, "title", "text"}` or `{"ok": false, "error"}`.
"""

import base64
import io
import json
import logging
import resource
import sys

MAX_PDF_PAGES = 200
MAX_TEXT_CHARS = 1_000_000  # the host reads at most 3M characters of JSON


MAX_MEMORY = 1 << 30  # bytes of address space: lxml and pypdf need a few hundred MB at most


def limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY, MAX_MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def html_text(text: str, url: str) -> tuple[str, str]:
    from lxml import html
    from trafilatura import extract

    parser = html.HTMLParser(encoding="utf-8", no_network=True, recover=True, huge_tree=False)
    tree = html.fromstring(text.encode("utf-8"), parser=parser)
    title = " ".join(" ".join(tree.xpath("//title/text()")).split())
    content = extract(
        tree,
        url=url,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        include_comments=False,
        include_images=False,
        favor_recall=True,
    )
    return title, content or ""


def pdf_text(body: bytes) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(body))
    if reader.is_encrypted:
        raise ValueError("encrypted PDF is not supported")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF has more than {MAX_PDF_PAGES} pages")
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if not text.strip():
            # A scanned page would silently vanish; say so instead (no OCR here).
            raise ValueError(f"PDF page {number} has no text layer (scanned?); OCR is not provided")
        pages.append(f"## Page {number}\n\n{text}")
    title = str((reader.metadata or {}).get("/Title", "") or "")
    return title, "\n\n".join(pages)


def main() -> dict:
    request = json.loads(sys.stdin.buffer.read())
    if request["kind"] == "pdf":
        title, text = pdf_text(base64.b64decode(request["body_b64"]))
    else:
        title, text = html_text(request["text"], request["url"])
    if not text.strip():
        raise ValueError("no readable text found on the page")
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + f"\n\n[text cut at {MAX_TEXT_CHARS} characters]"
    return {"ok": True, "title": title[:500], "text": text}


if __name__ == "__main__":
    limit_resources()
    logging.disable(logging.CRITICAL)
    try:
        answer = main()
    except ImportError:
        answer = {"ok": False, "error": "web packages missing: install requirements/web.lock.txt"}
    except Exception as exc:  # hostile input makes parsers raise anything; report one line
        answer = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    sys.stdout.buffer.write(json.dumps(answer, ensure_ascii=False).encode())
