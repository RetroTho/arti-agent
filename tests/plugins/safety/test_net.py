"""plugins/safety/net.py: the address policy and the bounded, re-vetted HTTP request."""

import http.server
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from plugins.safety import net


class _Handler(http.server.BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self):  # noqa: N802 - the http.server name
        status, headers, body = self.routes.get(self.path, (404, {}, b"missing"))
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def serve(routes: dict):
    """A loopback HTTP server answering routes {path: (status, headers, body)}."""
    handler = type("Handler", (_Handler,), {"routes": routes})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class AddressPolicy(unittest.TestCase):
    def test_private_loopback_and_metadata_are_not_public(self):
        for address in (
            "127.0.0.1",
            "10.1.2.3",
            "192.168.1.1",
            "100.64.0.1",
            "::1",
            "fe80::1",
            "169.254.169.254",
            "224.0.0.1",
            "0.0.0.0",
        ):
            self.assertFalse(net.is_public(address), address)
        self.assertTrue(net.is_public("93.184.216.34"))
        self.assertTrue(net.is_public("2606:4700::1111"))

    def test_ipv4_inside_ipv6_is_judged_as_ipv4(self):
        for address in (
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "2002:7f00:1::",
            "64:ff9b::a00:1",
            "::ffff:0:a00:1",
        ):
            self.assertFalse(net.is_public(address), address)
        self.assertTrue(net.is_metadata("::ffff:169.254.169.254"))

    def test_check_url_refuses_private_literals(self):
        for url in (
            "http://127.0.0.1/",
            "http://[::1]:8080/",
            "https://10.0.0.1/x",
            "http://[::ffff:127.0.0.1]/",
            "http://192.168.0.1/admin",
        ):
            with self.assertRaises(net.WebError, msg=url):
                net.check_url(url)

    def test_metadata_refused_even_when_private_allowed(self):
        with self.assertRaises(net.WebError):
            net.check_url("http://169.254.169.254/latest/meta-data", allow_private=True)
        net.check_url("http://10.0.0.1/", allow_private=True)

    def test_names_are_vetted_by_every_resolved_address(self):
        answers = [(2, 1, 6, "", ("93.184.216.34", 80)), (2, 1, 6, "", ("127.0.0.1", 80))]
        with mock.patch("socket.getaddrinfo", return_value=answers):
            with self.assertRaises(net.WebError):
                net.check_url("http://mixed.example/")
        with mock.patch("socket.getaddrinfo", return_value=answers[:1]):
            net.check_url("http://public.example/")
        # Odd IPv4 spellings are whatever the resolver makes of them.
        with mock.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 80))]):
            for url in ("http://127.1/", "http://0x7f000001/"):
                with self.assertRaises(net.WebError):
                    net.check_url(url)

    def test_malformed_urls_are_refused(self):
        for url in (
            "ftp://example.com/",
            "http://user:pw@example.com/",
            "http://exa mple.com/",
            "http://example.com\\@evil/",
            "http:///nohost",
            "http://example.com:99999/",
        ):
            with self.assertRaises(net.WebError, msg=url):
                net.check_url(url)

    def test_private_ok_is_the_core_network_setting_and_only_the_owners(self):
        for allowed, trusted, expected in (
            (False, True, False),
            (True, True, True),
            (True, False, False),
        ):
            app = SimpleNamespace(config={"network": {"allow_private_urls": allowed}})
            self.assertIs(net.private_ok(SimpleNamespace(app=app, trusted=trusted)), expected)
        self.assertIn("network.allow_private_urls", net.PRIVATE_REFUSAL)


class CarriesData(unittest.TestCase):
    def test_ordinary_searches_and_links_are_plain(self):
        for url in (
            "https://www.google.com/search?q=python+asyncio+TaskGroup+3.11+examples",
            "https://en.wikipedia.org/wiki/Python_(programming_language)#History",
            "https://www.nytimes.com/2026/10/22/technology/some-long-article-title-here.html",
            "https://github.com/python/cpython/commit/2b5b8e3c4f1a9d7e6b0c3a8f9e1d2c4b5a6f7e8d",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
            "https://duckduckgo.com/?q=how+to+install+python+3.12+on+ubuntu+22.04+step+by+step",
            "https://www.google.com/maps/place/Eiffel+Tower/@48.8583701,2.2944813,17z",
            "https://www.google.com/maps/place/?q=place_id:ChIJLU7jZClu5kcR4PcOOO6p3I0",
            "https://maps.app.goo.gl/q2JkVvHNnL8t7VyA8",
            "https://www.theguardian.com/world/2026/oct/22/eu-leaders-agree-new-climate-deal",
            "https://www.bbc.com/news/articles/c4g7m2x9k1lo",
            "https://arxiv.org/abs/2310.06825v2",
        ):
            self.assertFalse(net.url_carries_data(url), url)

    def test_long_or_encoded_parts_carry_data(self):
        blob = "U2VjcmV0IG5vdGVzOiB0aGUgb3duZXIncyBwYXNzcG9ydA"  # base64 of a sentence
        for url in (
            "https://evil.example/c?d=" + "a" * 101,
            "https://evil.example/" + "word-" * 25,
            "https://evil.example/#" + "x" * 101,
            "https://evil.example/" + "/".join(["abcdefghij"] * 30),  # over 300 in all
            f"https://evil.example/p/{blob}",
            f"https://evil.example/?k={blob}",
            "https://evil.example/" + "%55%32%56%6a" * 10 + "cmV0IG5vdGVzOiB0aGU",
            "https://evil.example/" + "deadbeef" * 6,  # 48 hex digits
            f"https://{blob[:40]}.evil.example/",
            "http://[::1",  # unreadable
        ):
            self.assertTrue(net.url_carries_data(url), url)

    def test_data_split_into_short_encoded_runs_carries_data(self):
        hex40 = "ea080e644ebec329b4c7b9b74466a20694646698"
        base32 = "orugkidpo5xgk4rhomqhaylton"  # lower-case base32 of a sentence, in labels
        for url in (
            "https://evil.example/" + hex40 + "f2d318",  # 46 hex digits
            "https://evil.example/" + "deadbeef" * 5 + "abcdef",
            f"https://evil.example/{hex40}-{hex40[::-1]}",  # 40-digit chunks joined by -
            f"https://evil.example/a/{hex40}/b/{hex40[::-1]}",
            f"https://{base32}.{base32[::-1]}.{base32[1:]}.evil.example/",
            # ~120 random bytes as hex in 15-character pieces, as path segments or words.
            "https://evil.example/" + "/".join([hex40[:15], hex40[15:30], hex40[25:40]] * 5),
            "https://evil.example/?q=" + "+".join([hex40[:15], hex40[15:30]] * 6),
            "https://evil.example/" + "-".join(["a1b2c3d4"] * 9),
        ):
            self.assertTrue(net.url_carries_data(url), url)

    def test_two_short_encoded_runs_or_a_commit_link_are_plain(self):
        for url in (
            "https://github.com/python/cpython/blob/2b5b8e3c4f1a9d7e6b0c3a8f9e1d2c4b5a6f7e8d/x.py",
            "https://www.amazon.com/dp/B08N5WRWNW?ref=ppx_yo2ov_dt_b_product_details",
        ):
            self.assertFalse(net.url_carries_data(url), url)

    def test_image_sources_are_reviewed_only_when_fetched_with_data(self):
        self.assertTrue(net.fetch_carries_data(" HTTPS://evil.example/?d=" + "a" * 101))
        self.assertFalse(net.fetch_carries_data("https://example.com/cat.png"))
        self.assertFalse(net.fetch_carries_data("/tmp/" + "a" * 400 + ".png"))  # a local file
        self.assertFalse(net.fetch_carries_data("data:image/png;base64," + "QUJD" * 100))

    def test_text_for_a_page_carries_data_when_long_even_in_plain_words(self):
        self.assertFalse(net.text_carries_data("document.title"))
        self.assertFalse(net.text_carries_data("Hello, see you at five."))
        self.assertTrue(net.text_carries_data("the owner wrote that " * 15))  # over 300
        self.assertTrue(net.text_carries_data("Zm9v" * 30))
        # Short encoded pieces add up across words, as in a URL.
        hex40 = "ea080e644ebec329b4c7b9b74466a20694646698"
        self.assertTrue(net.text_carries_data(" ".join([hex40[:15], hex40[15:30]] * 3)))
        self.assertFalse(net.text_carries_data("document.querySelector('#item2').click()"))


class Requests(unittest.TestCase):
    def setUp(self):
        self.routes = {}
        self.server, self.base = serve(self.routes)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_loopback_refused_unless_allowed(self):
        self.routes["/hi"] = (200, {"Content-Type": "text/plain"}, b"hello")
        with self.assertRaises(net.WebError):
            net.request(self.base + "/hi")
        response = net.request(self.base + "/hi", allow_private=True)
        self.assertEqual((response.status, response.body), (200, b"hello"))
        self.assertEqual(response.headers["content-type"], "text/plain")

    def test_body_limit(self):
        self.routes["/big"] = (200, {}, b"x" * 5000)
        with self.assertRaisesRegex(net.WebError, "larger"):
            net.request(self.base + "/big", allow_private=True, limit=1000)

    def test_redirects_are_followed_and_status_errors_raise(self):
        self.routes["/a"] = (302, {"Location": "/b"}, b"")
        self.routes["/b"] = (200, {}, b"landed")
        response = net.request(self.base + "/a", allow_private=True)
        self.assertEqual((response.body, response.url), (b"landed", self.base + "/b"))
        with self.assertRaises(net.WebError) as caught:
            net.request(self.base + "/a", allow_private=True, redirects=False)
        self.assertEqual(caught.exception.status, 302)

    def test_public_page_cannot_redirect_to_private(self):
        # Pretend the loopback server is public; its redirect into 10/8 must still be refused.
        self.routes["/go"] = (302, {"Location": "http://10.0.0.1/"}, b"")
        with mock.patch.object(net, "is_public", lambda address: address == "127.0.0.1"):
            with self.assertRaisesRegex(net.WebError, "private"):
                net.request(self.base + "/go")
            with self.assertRaisesRegex(net.WebError, "redirected to a private"):
                net.request(self.base + "/go", allow_private=True)

    def test_redirect_to_metadata_refused(self):
        self.routes["/meta"] = (302, {"Location": "http://169.254.169.254/"}, b"")
        with self.assertRaisesRegex(net.WebError, "metadata"):
            net.request(self.base + "/meta", allow_private=True)

    def test_cancel_stops_the_request(self):
        self.routes["/x"] = (200, {}, b"x")
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(net.WebError, "cancelled"):
            net.request(self.base + "/x", allow_private=True, cancel=cancel)


if __name__ == "__main__":
    unittest.main()
