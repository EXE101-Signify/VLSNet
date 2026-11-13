import unittest

from mode.streaming_ws import FRAME_HEADER, parse_binary_frame


class BinaryProtocolTests(unittest.TestCase):
    def test_versioned_binary_frame(self):
        jpeg = b'\xff\xd8' + b'0' * 98
        message = FRAME_HEADER.pack(1, 7, 123456789, len(jpeg)) + jpeg
        self.assertEqual(parse_binary_frame(message), (7, 123456789, jpeg))

    def test_rejects_truncated_wrong_version_and_size(self):
        jpeg = b'\xff\xd8' + b'0' * 98
        for message in [b'bad', FRAME_HEADER.pack(2, 1, 1, len(jpeg)) + jpeg,
                        FRAME_HEADER.pack(1, 1, 1, len(jpeg) + 1) + jpeg]:
            with self.assertRaises(ValueError):
                parse_binary_frame(message)


if __name__ == '__main__':
    unittest.main()
