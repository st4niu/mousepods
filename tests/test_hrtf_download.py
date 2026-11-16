# SPDX-License-Identifier: Apache-2.0
"""The HRTF is fetched once, verified, and never replaced by a file that doesn't match.

Run:  python -m unittest tests.test_hrtf_download -v
"""
import hashlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hrtf_download as h  # noqa: E402

GOOD = b"\x89HDF\r\n\x1a\n" + b"x" * 100


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class HrtfDownloadTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        p = mock.patch.multiple(h, SHA256=hashlib.sha256(GOOD).hexdigest(), SIZE=len(GOOD))
        p.start()
        self.addCleanup(p.stop)

    def test_downloads_and_verifies(self):
        calls = []

        def opener(req, timeout):
            calls.append(req.full_url)
            return FakeResponse(GOOD)
        path = h.ensure(self.dir, opener=opener)
        self.assertEqual(calls, [h.URL])
        self.assertTrue(h.valid(path))
        self.assertEqual(h.ensure(self.dir, opener=opener), path)   # second time: no download
        self.assertEqual(len(calls), 1)

    def test_wrong_file_is_refused_and_not_kept(self):
        with self.assertRaises(ValueError):
            h.ensure(self.dir, opener=lambda req, timeout: FakeResponse(b"not the hrtf"))
        self.assertFalse(os.path.exists(h.path_in(self.dir)))
        self.assertEqual([f for f in os.listdir(os.path.join(self.dir, "hrtf"))], [])

    def test_https_only(self):
        self.assertTrue(h.URL.startswith("https://"))


if __name__ == "__main__":
    unittest.main()
