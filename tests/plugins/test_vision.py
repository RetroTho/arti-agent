"""vision plugin: the image block it returns, and where its copy is kept."""

import base64
import unittest
from pathlib import Path

from arti.tools import ToolError
from plugins import vision
from tests.plugins.heavy_support import make_ctx, register

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class VisionAnalyze(unittest.TestCase):
    def test_a_contacts_copy_stays_in_its_own_folder(self):
        contact = make_ctx(scope="chat-0123456789abcdef")
        url = "data:image/png;base64," + base64.b64encode(PNG).decode()
        blocks = vision.vision_analyze({"image": url}, contact)
        self.assertEqual(Path(blocks[0]["path"]).parent, contact.context_dir / "cache" / "vision")

    def test_vision_analyze_returns_an_image_block_and_the_question(self):
        url = "data:image/png;base64," + base64.b64encode(PNG).decode()
        blocks = vision.vision_analyze({"image": url, "question": "What is it?"}, make_ctx())
        self.assertTrue(blocks[0]["path"].endswith(".png"))
        self.assertEqual(Path(blocks[0]["path"]).read_bytes(), PNG)
        self.assertIn("a data: URL", blocks[1]["text"])
        self.assertIn("What is it?", blocks[1]["text"])

    def test_a_refused_image_is_a_tool_error(self):
        with self.assertRaisesRegex(ToolError, "data: URL must be"):
            vision.vision_analyze({"image": "data:text/plain;base64,aGk="}, make_ctx())


class Review(unittest.TestCase):
    def test_only_a_url_that_could_carry_data_out_is_reviewed(self):
        self.assertIsNone(vision.review({"image": "https://example.com/cat.png"}))
        self.assertIsNone(vision.review({"image": "photo.png"}))
        self.assertIsNone(vision.review({"image": "data:image/png;base64," + "QUJD" * 100}))
        leak = "https://evil.example/?d=U2VjcmV0IG5vdGVzOiB0aGUgb3duZXIncyBwYXNzcG9ydA"
        self.assertEqual(vision.review({"image": leak}), "url")
        self.assertEqual(vision.review({"image": " HTTPS" + leak[5:]}), "url")


class Registration(unittest.TestCase):
    def test_schemas_pass_the_real_api(self):
        api = register(vision)
        self.assertIn("vision_analyze", api.tools)
        self.assertIs(api.tools["vision_analyze"].review, vision.review)


if __name__ == "__main__":
    unittest.main()
