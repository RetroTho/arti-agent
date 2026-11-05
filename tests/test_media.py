"""media: a group's attachments are kept in its own folder; image sizes and conversion."""

import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

from arti import media
from arti.store import Input

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class InputBlocksTest(unittest.TestCase):
    def setUp(self):
        self.data = Path(tempfile.mkdtemp())
        self.app = SimpleNamespace(data_dir=self.data, tz=ZoneInfo("UTC"))
        self.file = Path(tempfile.mkdtemp()) / "menu.pdf"  # like ~/Library/Messages/Attachments
        self.file.write_text("pdf")

    def blocks(self, context_dir):
        inp = Input("e", "c", "message", text="look", attachments=[str(self.file)], ts=1)
        return media.input_blocks(self.app, inp, None, context_dir)

    def test_home_names_the_file_where_it_is(self):
        self.assertEqual(self.blocks(self.data)[-1]["text"], f"[attachment: {self.file}]")

    def test_a_group_gets_a_copy_in_its_folder(self):
        group = self.data / "contexts" / "chat-1"
        text = self.blocks(group)[-1]["text"]
        copy = Path(text.removeprefix("[attachment: ").removesuffix("]"))
        self.assertTrue(copy.is_relative_to(group / "media"))
        self.assertEqual((copy.name[9:], copy.read_text()), ("menu.pdf", "pdf"))
        self.file.unlink()
        self.assertIn("could not be loaded: menu.pdf", self.blocks(group)[-1]["text"])


if __name__ == "__main__":
    unittest.main()


def png(width: int, height: int) -> bytes:
    header = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + header + b"\x00" * 4


class DimensionsTest(unittest.TestCase):
    """Sizes read from the images' own bytes, so no converter runs for an image that fits."""

    def size(self, data: bytes):
        path = Path(tempfile.mkdtemp()) / "image"
        path.write_bytes(data)
        return media.dimensions(path)

    def test_each_format_a_model_takes(self):
        jpeg = (
            b"\xff\xd8"
            + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9  # APP0, skipped
            + b"\xff\xc2" + struct.pack(">HBHH", 17, 8, 600, 800) + b"\x00" * 10  # progressive
        )  # fmt: skip
        vp8x = b"RIFF\x00\x00\x00\x00WEBPVP8X" + b"\x00" * 8 + (3999).to_bytes(3, "little")
        vp8x += (2999).to_bytes(3, "little")
        bits = (640 - 1) | ((480 - 1) << 14)
        vp8l = (
            b"RIFF\x00\x00\x00\x00WEBPVP8L" + b"\x00" * 5 + bits.to_bytes(4, "little") + b"\x00" * 6
        )
        for name, data, size in (
            ("png", png(3000, 20), (3000, 20)),
            ("gif", b"GIF89a" + struct.pack("<HH", 32, 16), (32, 16)),
            ("jpeg", jpeg, (800, 600)),
            ("webp extended", vp8x, (4000, 3000)),
            ("webp lossless", vp8l, (640, 480)),
        ):
            with self.subTest(name):
                self.assertEqual(self.size(data), size)

    def test_what_cannot_be_told_is_none(self):
        self.assertIsNone(self.size(b"\xff\xd8\xff\xe0 truncated"))
        self.assertIsNone(self.size(b"not an image at all"))

    def test_an_image_that_fits_is_used_as_is_without_a_converter(self):
        folder = Path(tempfile.mkdtemp())
        source = folder / "small.png"
        source.write_bytes(png(100, 100))
        with mock.patch.object(media, "_convert") as convert:
            block = media.image_block(source, folder)
        convert.assert_not_called()
        self.assertEqual((block["path"], block["media_type"]), (str(source), "image/png"))


class ConverterTest(unittest.TestCase):
    def test_imagemagick_converts_and_its_absence_is_said(self):
        src, out = Path("/x/a.heic"), Path("/y/b.jpg")
        with mock.patch.object(media.shutil, "which", lambda name: f"/usr/bin/{name}"):
            argv = media._converter(src, out, 60)
        self.assertEqual(argv[:2], ["/usr/bin/magick", "/x/a.heic[0]"])
        self.assertEqual(argv[-3:], ["-quality", "60", "jpeg:/y/b.jpg"])
        with mock.patch.object(media.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "needs ImageMagick"):
                media._converter(src, out, 60)
