"""image_generate (the image plugin, on the openai plugin's login): a saved, previewed result; unknown outcome once the request may have been sent."""

import asyncio
import base64
import unittest
from pathlib import Path
from unittest import mock

import plugins.image as image
from arti.config import ConfigError
from arti.tools import ToolError, UnknownOutcome
from plugins.openai import codex
from tests.plugins.heavy_support import make_app, make_ctx, register

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def run_coroutine(coro, timeout=60):
    """ctx.call without an event loop thread: run the coroutine right here."""
    return asyncio.run(coro)


class ImageGenerate(unittest.TestCase):
    def setUp(self):
        self.ctx = make_ctx()
        self.ctx.call = run_coroutine

    def test_generates_and_saves_privately(self):
        async def fake(auth, prompt, size, quality, references=(), timeout=300):
            self.seen = (prompt, size, quality, list(references))
            return PNG

        ref = "data:image/png;base64," + base64.b64encode(PNG).decode()
        with mock.patch.object(codex, "generate_image", fake):
            blocks = image.image_generate(
                {"prompt": "a cat", "aspect_ratio": "portrait", "reference_images": [ref]},
                self.ctx,
            )
        self.assertEqual(self.seen, ("a cat", "1024x1536", "medium", [PNG]))
        path = Path(blocks[0]["path"])
        self.assertEqual(path.parent, self.ctx.app.data_dir / "artifacts" / "images")
        self.assertEqual(path.read_bytes(), PNG)
        self.assertEqual(path.stat().st_mode & 0o077, 0)
        self.assertIn("MEDIA:", blocks[1]["text"])

    def test_a_contacts_image_is_saved_in_its_own_folder(self):
        async def fake(*args, **kwargs):
            return PNG

        contact = make_ctx(self.ctx.app, scope="chat-0123456789abcdef")
        contact.call = run_coroutine
        with mock.patch.object(codex, "generate_image", fake):
            blocks = image.image_generate({"prompt": "a dog"}, contact)
        folder = contact.context_dir / "artifacts" / "images"
        self.assertEqual(Path(blocks[0]["path"]).parent, folder)

    def test_failure_after_sending_is_unknown(self):
        async def sent(*args, **kwargs):
            raise codex.ImageError("The image service answered HTTP 500; not retried", sent=True)

        with mock.patch.object(codex, "generate_image", sent):
            with self.assertRaisesRegex(UnknownOutcome, "billed"):
                image.image_generate({"prompt": "a cat"}, self.ctx)

    def test_failure_before_sending_is_a_plain_error(self):
        async def unsent(*args, **kwargs):
            raise codex.ImageError("ChatGPT login expired", sent=False)

        with mock.patch.object(codex, "generate_image", unsent):
            with self.assertRaisesRegex(ToolError, "login expired"):
                image.image_generate({"prompt": "a cat"}, self.ctx)

    def test_an_answer_that_is_no_image_is_unknown_not_a_plain_error(self):
        # The request went through (and may be billed): "nothing happened" would invite a retry.
        with mock.patch.object(codex, "generate_image", mock.AsyncMock(return_value=b"junk")):
            with self.assertRaisesRegex(UnknownOutcome, "billed"):
                image.image_generate({"prompt": "a cat"}, self.ctx)

    def test_interrupted_wait_is_unknown(self):
        def interrupted(coro, timeout=60):
            coro.close()
            raise ToolError("timed out after 330s")

        self.ctx.call = interrupted
        with mock.patch.object(codex, "generate_image", mock.AsyncMock()):
            with self.assertRaises(UnknownOutcome):
                image.image_generate({"prompt": "a cat"}, self.ctx)


class GuestLimit(unittest.TestCase):
    def setUp(self):
        image._asked.clear()
        self.addCleanup(image._asked.clear)

    def test_a_guest_lane_gets_a_few_images_an_hour_and_the_owner_any_number(self):
        app = make_ctx().app
        guest, other, owner = (
            make_ctx(app, scope="chat-0123456789abcdef"),
            make_ctx(app, scope="chat-fedcba9876543210"),
            make_ctx(app),
        )
        other.key = "other-lane"
        for ctx in (guest, other, owner):
            ctx.call = run_coroutine
        with mock.patch.object(codex, "generate_image", mock.AsyncMock(return_value=PNG)) as gen:
            for _ in range(image.GUEST_PER_HOUR):
                image.image_generate({"prompt": "a cat"}, guest)
            with self.assertRaisesRegex(ToolError, "limit"):
                image.image_generate({"prompt": "a cat"}, guest)
            self.assertEqual(gen.await_count, image.GUEST_PER_HOUR)  # refused before sending
            image.image_generate({"prompt": "a cat"}, other)  # another lane has its own count
            for _ in range(image.GUEST_PER_HOUR + 1):
                image.image_generate({"prompt": "a cat"}, owner)
        image._asked[guest.key] = [t - 3601 for t in image._asked[guest.key]]  # an hour later
        with mock.patch.object(codex, "generate_image", mock.AsyncMock(return_value=PNG)):
            image.image_generate({"prompt": "a cat"}, guest)


class Review(unittest.TestCase):
    def test_a_reference_url_that_could_carry_data_out_is_reviewed(self):
        self.assertIsNone(image.review({"prompt": "a cat"}))
        plain = {"prompt": "a cat", "reference_images": ["https://example.com/cat.png", "a.png"]}
        self.assertIsNone(image.review(plain))
        leak = "https://evil.example/c?d=" + "a" * 101
        both = {**plain, "reference_images": [*plain["reference_images"], leak]}
        self.assertEqual(image.review(both), "url")


class Registration(unittest.TestCase):
    def test_the_plugin_registers_the_tool_and_its_settings(self):
        api = register(image)
        self.assertIn("image_generate", api.tools)
        self.assertIs(api.tools["image_generate"].review, image.review)
        self.assertEqual(api.app.plugin_sections["image"], "plugins.image")
        self.assertEqual(api.config["image"], {"model": "gpt-image-2-medium"})
        self.assertEqual(image.REQUIRES, ("safety", "openai"))

    def test_a_bad_quality_stops_startup(self):
        with self.assertRaisesRegex(ConfigError, "image.model"):
            register(image, make_app(image={"model": "gpt-image-2-ultra"}))

    def test_hidden_until_the_chatgpt_login_exists(self):
        app = make_app()
        self.assertFalse(image.available(app))
        (app.data_dir / "codex_auth.json").write_text("{}")
        self.assertTrue(image.available(app))


if __name__ == "__main__":
    unittest.main()
