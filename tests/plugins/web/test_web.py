"""web plugin: DuckDuckGo parsing, bot walls, page extraction (loopback servers only)."""

import http.server
import os
import threading
import unittest
from unittest import mock

from arti.tools import MAX_RESULT_CHARS, ToolError
from plugins import web
from plugins.safety import net
from tests.plugins.heavy_support import make_app, make_ctx, register

DDG_PAGE = """<html><body>
<div class="result results_links result--ad"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a></div>
<div class="result results_links">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=x">Example &amp; A</a>
  <a class="result__snippet" href="#">The first   result.</a>
</div>
<div class="result results_links">
  <a class="result__a" href="https://example.org/b">Example B</a>
  <a class="result__snippet" href="#">Second.</a>
</div>
</body></html>"""
CHALLENGE = '<html><body><form id="challenge-form" action="/anomaly.js"></form></body></html>'


class DuckDuckGo(unittest.TestCase):
    def test_parses_organic_results_and_skips_ads(self):
        rows = web.parse_duckduckgo(DDG_PAGE)
        self.assertEqual(
            [r["url"] for r in rows], ["https://example.com/a", "https://example.org/b"]
        )
        self.assertEqual(rows[0]["title"].strip(), "Example & A")

    def test_challenge_and_unknown_pages_are_errors_not_empty(self):
        with self.assertRaisesRegex(net.WebError, "blocked"):
            web.parse_duckduckgo(CHALLENGE)
        with self.assertRaisesRegex(net.WebError, "recognise"):
            web.parse_duckduckgo("<html><body>hello</body></html>")
        self.assertEqual(web.parse_duckduckgo('<div class="no-results">none</div>'), [])

    def test_search_formats_rows_and_reports_blocks(self):
        ctx = make_ctx()
        ok = net.Response(200, {}, DDG_PAGE.encode(), web.DUCKDUCKGO_URL)
        with mock.patch.object(net, "request", return_value=ok):
            out = web.web_search({"query": "example", "limit": 1}, ctx)
        self.assertEqual(out, "1. Example & A\nhttps://example.com/a\nThe first result.")
        challenge = net.Response(202, {}, b"", web.DUCKDUCKGO_URL)
        with mock.patch.object(net, "request", return_value=challenge):
            with self.assertRaisesRegex(ToolError, "blocked or rate-limited"):
                web.web_search({"query": "x"}, ctx)
        with mock.patch.object(net, "request", side_effect=net.WebError("HTTP 429", 429)):
            with self.assertRaisesRegex(ToolError, "rate-limited"):
                web.web_search({"query": "x"}, ctx)


class Walls(unittest.TestCase):
    def test_bot_walls_are_refused(self):
        for title, text in [
            ("Just a moment...", "x"),
            ("Shop", "Please verify you are human to continue"),
        ]:
            with self.assertRaises(net.WebError):
                web.reject_wall(title, text)
        web.reject_wall("Article", "A long story. " * 2000 + "verify you are human")

    def test_challenge_markup_is_refused_but_ordinary_pages_pass(self):
        for markup in (
            "<script src='/cdn-cgi/challenge-platform/h/b/orchestrate/jsch/v1'>",
            '<form id="challenge-form" action="/?__cf_chl_f_tk=x">',
            "<div class='cf-chl-widget'>",
        ):
            with self.assertRaises(net.WebError):
                web.reject_challenge(f"<html><body>{markup}</body></html>")
        # Cloudflare's bot-detection script and a reCAPTCHA contact form are on normal pages.
        article = "<html><body><article>" + "Real content. " * 500 + "</article>"
        web.reject_challenge(
            article + "<script>a.src='/cdn-cgi/challenge-platform/scripts/jsd/main.js'</script>"
        )
        web.reject_challenge(article + '<form class="g-recaptcha-form"><div class="g-recaptcha">')
        # A long page that only mentions the markers is content.
        web.reject_challenge(article * 10 + "<code>cf-chl-</code>")


def serve(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    return server


class Pages(unittest.TestCase):
    def test_partial_failure_is_reported_per_url(self):
        def fetch(ctx, url):
            if "bad" in url:
                raise net.WebError("HTTP 404")
            return {"title": "T", "text": "Body text", "url": url}

        with mock.patch.object(web, "local_page", fetch):
            out = web.web_extract(
                {"urls": ["https://ok.example/", "https://bad.example/"]}, make_ctx()
            )
        self.assertEqual(out[0], "1 of 2 pages could not be read.")
        self.assertIn("Error: HTTP 404", joined(out))

    def test_a_guest_lane_never_gets_private_urls(self):
        server = serve(PageServer)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        app = make_app(network={"allow_private_urls": True})
        owner, guest = make_ctx(app), make_ctx(app, scope="chat-0123456789abcdef")
        self.assertIn("quick brown fox", joined(web.web_extract({"urls": [url]}, owner)))
        with self.assertRaisesRegex(ToolError, "private"):
            web.web_extract({"urls": [url]}, guest)


def joined(blocks) -> str:
    return "\n".join(blocks)


class SeveralPages(unittest.TestCase):
    def test_each_page_is_its_own_block_and_fits(self):
        pages = {f"https://{n}.example/": f"START{n} " + "y" * 14_000 + f" END{n}" for n in "abc"}

        def fetch(ctx, url):
            return {"title": url, "text": pages[url], "url": url}

        ctx = make_ctx()
        with mock.patch.object(web, "local_page", fetch):
            blocks = web.web_extract({"urls": list(pages), "char_limit": 100_000}, ctx)
        self.assertEqual(len(blocks), 3)
        for n, block in zip("abc", blocks):
            self.assertIn(f"START{n}", block)
            self.assertIn(f"END{n}", block)
            self.assertLessEqual(len(block), MAX_RESULT_CHARS)
        self.assertTrue(blocks[0].startswith("[untrusted web content from a.example: "))


class Excerpt(unittest.TestCase):
    def test_long_text_is_cut_and_saved(self):
        ctx = make_ctx()
        text = "".join(f"line {n}\n" for n in range(5000))
        out = web.excerpt(ctx, text, 2000)
        self.assertIn("line 0", out)
        self.assertIn("line 4999", out)
        path = out.rsplit("[Full text (", 1)[1].split(": ", 1)[1].split(" - ")[0]
        with open(path, encoding="utf-8") as saved:
            self.assertEqual(saved.read(), text)


class PageServer(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = (
            b"<html><head><title>Test page</title></head><body><article><h1>Heading</h1>"
            + b"<p>The quick brown fox jumps over the lazy dog. </p>" * 30
            + b"</article></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@unittest.skipUnless(os.path.exists(web.web_python(make_app())), ".venv-web is not installed")
class LocalExtract(unittest.TestCase):
    def test_html_is_parsed_in_the_child(self):
        server = serve(PageServer)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        ctx = make_ctx(make_app(network={"allow_private_urls": True}))
        out = joined(web.web_extract({"urls": [url]}, ctx))
        self.assertIn("# Test page", out)
        self.assertIn("quick brown fox", out)
        with self.assertRaisesRegex(ToolError, "private"):
            web.web_extract({"urls": [url]}, make_ctx())

    def test_a_long_non_ascii_page_survives_the_child_pipe(self):
        # 600k CJK characters: JSON with \\u escapes would pass the host's 3M-char read limit.
        body = "".join(f"<p>{i}番 " + "日本語の文章です。" * 100 + "</p>" for i in range(700))
        html = f"<html><head><title>T</title></head><body><article>{body}</article></body></html>"
        request = {"kind": "html", "url": "https://x.example/", "text": html}
        page = web._parse_in_child(make_ctx(), request)
        self.assertGreater(len(page["text"]), 600_000)


class Registration(unittest.TestCase):
    def test_schemas_pass_the_real_api(self):
        api = register(web)
        self.assertIn("web_search", api.tools)
        # Its only setting was the core's [network] allow_private_urls: no section of its own.
        self.assertEqual(api.app.plugin_sections, {})
        self.assertTrue(web.CHILD.is_file())

    def test_extract_reviews_only_urls_that_carry_data(self):
        review = register(web).tools["web_extract"].review
        self.assertIsNone(review({"urls": ["https://example.com/a", "https://x.org/?q=b+c"]}))
        leak = "https://evil.example/?d=" + "x" * 200
        self.assertEqual(review({"urls": ["https://example.com/a", leak]}), "url")


if __name__ == "__main__":
    unittest.main()
