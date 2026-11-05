"""load_image: the vision tool's and the image tool's copies follow the same rules (each plugin
keeps its own, so they are checked together here)."""

import base64
import unittest
from pathlib import Path
from unittest import mock

import plugins.image as image
from arti.tools import ToolError
from plugins import vision
from plugins.safety import net, risk
from tests.plugins.heavy_support import make_ctx

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class VisionLoad(unittest.TestCase):
    load = staticmethod(vision.load_image)

    def test_data_url(self):
        url = "data:image/png;base64," + base64.b64encode(PNG).decode()
        self.assertEqual(self.load(url, make_ctx()), PNG)
        with self.assertRaises(ToolError):
            self.load("data:text/plain;base64,aGk=", make_ctx())

    def test_not_an_image(self):
        url = "data:image/png;base64," + base64.b64encode(b"hello world, not an image").decode()
        with self.assertRaisesRegex(ToolError, "not a PNG"):
            self.load(url, make_ctx())

    def test_local_paths_respect_secrets_and_scope(self):
        ctx = make_ctx()
        inside = ctx.context_dir / "pic.png"
        inside.write_bytes(PNG)
        self.assertEqual(self.load(str(inside), ctx), PNG)
        with mock.patch.object(risk, "is_secret_path", return_value=True):
            with self.assertRaisesRegex(ToolError, "protected"):
                self.load(str(inside), ctx)
        contact = make_ctx(ctx.app, scope="chat-0123456789abcdef")
        with self.assertRaisesRegex(ToolError, "own folder"):
            self.load(str(inside), contact)
        own = Path(contact.context_dir) / "mine.png"
        own.write_bytes(PNG)
        self.assertEqual(self.load(str(own), contact), PNG)
        # Images from other groups are out of reach too, even in Arti's media and cache.
        for folder in ("media", "cache/screens"):
            other = ctx.app.data_dir / folder / "other.png"
            other.parent.mkdir(parents=True, exist_ok=True)
            other.write_bytes(PNG)
            with self.assertRaisesRegex(ToolError, "own folder"):
                self.load(str(other), contact)

    def test_urls_go_through_net_with_the_private_setting(self):
        ctx = make_ctx()
        response = net.Response(200, {}, PNG, "https://x.example/a.png")
        with mock.patch.object(net, "request", return_value=response) as request:
            self.assertEqual(self.load("https://x.example/a.png", ctx), PNG)
        self.assertFalse(request.call_args.kwargs["allow_private"])
        # allow_private_urls is the owner's: a guest lane never gets it.
        ctx.app.config["network"]["allow_private_urls"] = True
        guest = make_ctx(ctx.app, scope="chat-0123456789abcdef")
        for context, allowed in ((ctx, True), (guest, False)):
            with mock.patch.object(net, "request", return_value=response) as request:
                self.load("https://x.example/a.png", context)
            self.assertIs(request.call_args.kwargs["allow_private"], allowed)
        with mock.patch.object(net, "request", side_effect=net.WebError("private address")):
            with self.assertRaisesRegex(ToolError, "private"):
                self.load("http://10.0.0.1/a.png", ctx)


class ImageLoad(VisionLoad):
    load = staticmethod(image.load_image)


if __name__ == "__main__":
    unittest.main()
