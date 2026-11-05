"""The browser plugin's DevTools connection: WebSocket frames."""

import unittest

from plugins.browser import devtools


class Frames(unittest.TestCase):
    def test_round_trip_for_every_length_form(self):
        for size in (0, 5, 125, 126, 70_000):
            payload = bytes(range(256)) * (size // 256) + bytes(size % 256)
            frame = devtools.encode_frame(0x1, payload, mask=b"\x01\x02\x03\x04")
            self.assertEqual(frame[1] & 0x80, 0x80)  # clients mask
            final, opcode, decoded, used = devtools.decode_frame(frame + b"rest")
            self.assertEqual((final, opcode, decoded, used), (True, 0x1, payload, len(frame)))

    def test_unmasked_server_frame_and_partial_data(self):
        frame = bytes([0x81, 5]) + b"hello"
        self.assertEqual(devtools.decode_frame(frame), (True, 1, b"hello", 7))
        self.assertIsNone(devtools.decode_frame(frame[:4]))
        self.assertIsNone(devtools.decode_frame(bytes([0x81, 126, 1])))

    def test_a_huge_message_is_refused_from_its_header(self):
        header = bytes([0x81, 127]) + (21_000_000).to_bytes(8, "big")
        with self.assertRaisesRegex(devtools.BrowserError, "20 MB"):
            devtools.decode_frame(header)


if __name__ == "__main__":
    unittest.main()
