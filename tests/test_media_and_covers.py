# SPDX-License-Identifier: Apache-2.0
"""Overview pictures (/media/<name>, closed list) and Steam covers (/api/games/<appid>/cover).

Run:  python -m unittest tests.test_media_and_covers -v
"""
import http.client
import json
import os
import queue
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import games_service  # noqa: E402
import mousepods_web as web  # noqa: E402
from steam_games import SteamRoot  # noqa: E402
from test_steam_games import make_steam  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0fake-jpeg"
PNG = b"\x89PNG\r\n\x1a\nfake-png"


def put(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


class CoverPathTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.steam = os.path.join(self.home, "Steam")
        self.cache = os.path.join(self.steam, "appcache", "librarycache")
        os.makedirs(self.cache)
        self.roots = [SteamRoot(self.steam, False)]

    def tearDown(self):
        shutil.rmtree(self.home)

    def cover(self, appid):
        return games_service.cover_path(appid, self.roots)

    def test_old_layout_prefers_the_wide_header(self):
        put(os.path.join(self.cache, "244210_library_600x900.jpg"), JPEG)
        self.assertEqual(self.cover("244210"), os.path.join(self.cache, "244210_library_600x900.jpg"))
        put(os.path.join(self.cache, "244210_header.jpg"), JPEG)
        self.assertEqual(self.cover("244210"), os.path.join(self.cache, "244210_header.jpg"))

    def test_new_layout_and_hashed_subfolder(self):
        put(os.path.join(self.cache, "227300", "library_600x900.jpg"), JPEG)
        self.assertEqual(self.cover("227300"), os.path.join(self.cache, "227300", "library_600x900.jpg"))
        put(os.path.join(self.cache, "227300", "3f2a9c", "header.jpg"), JPEG)
        self.assertEqual(self.cover(227300), os.path.join(self.cache, "227300", "3f2a9c", "header.jpg"))

    def test_png_and_no_logo(self):
        put(os.path.join(self.cache, "10", "logo.png"), PNG)
        self.assertIsNone(self.cover("10"))
        put(os.path.join(self.cache, "10", "library_hero.png"), PNG)
        self.assertTrue(self.cover("10").endswith("library_hero.png"))

    def test_missing(self):
        self.assertIsNone(self.cover("999"))
        self.assertIsNone(games_service.cover_path("999", [SteamRoot(os.path.join(self.home, "none"), False)]))

    def test_bad_appids_rejected(self):
        put(os.path.join(self.home, "secret_header.jpg"), JPEG)
        put(os.path.join(self.cache, "header.jpg"), JPEG)
        for bad in ("../secret", "..", "", "%2e%2e", "12a", "²", "1/../..", "1" * 13, "-1", " 1"):
            self.assertIsNone(self.cover(bad), bad)

    def test_symlink_out_of_the_cache_ignored(self):
        outside = os.path.join(self.home, "outside.jpg")
        put(outside, JPEG)
        os.makedirs(os.path.join(self.cache, "42"))
        os.symlink(outside, os.path.join(self.cache, "42", "header.jpg"))
        os.symlink(outside, os.path.join(self.cache, "42_header.jpg"))
        os.symlink(self.home, os.path.join(self.cache, "42", "escape"))
        put(os.path.join(self.home, "header.jpg"), JPEG)          # reachable as 42/escape/header.jpg
        self.assertIsNone(self.cover("42"))
        put(os.path.join(self.cache, "42", "library_600x900.jpg"), JPEG)
        self.assertEqual(self.cover("42"), os.path.join(self.cache, "42", "library_600x900.jpg"))


class MediaAndCoverHttpTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.steam = make_steam(self.home)
        self.cache = os.path.join(self.steam, "appcache", "librarycache")
        put(os.path.join(self.cache, "244210_header.jpg"), JPEG)                    # old layout
        put(os.path.join(self.cache, "227300", "ab12cd", "library_600x900.png"), PNG)  # new, hashed
        put(os.path.join(self.steam, "appcache", "secret.jpg"), JPEG)
        self.cfg_path = os.path.join(self.home, "config.json")
        with open(self.cfg_path, "w") as f:
            f.write("{}")
        self.env = mock.patch.dict(os.environ, {"MOUSEPODS_STEAM_ROOT": self.steam,
                                                "MOUSEPODS_DLL_DIR": os.path.join(self.home, "dlls")})
        self.env.start()
        self.run_patch = mock.patch.object(games_service, "is_running", lambda appid, cfg_path=None: False)
        self.run_patch.start()
        self.ui = web.WebUI(types.SimpleNamespace(cfg={"ui_lang": "en", "games": {}}), self.cfg_path, port=0)
        self.ui.q = queue.Queue()
        self.ui.start()
        self.port = self.ui.server.server_address[1]

    def tearDown(self):
        self.ui.server.shutdown()
        self.run_patch.stop()
        self.env.stop()
        shutil.rmtree(self.home)

    def get(self, path):
        """Raw request: the path goes out exactly as written (no client-side normalization)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            c.putrequest("GET", path, skip_host=True)
            c.putheader("Host", f"127.0.0.1:{self.port}")
            c.endheaders()
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()

    def test_media_files_served(self):
        for name in ("cockpit.jpg", "vr.jpg"):
            status, headers, data = self.get("/media/" + name)
            self.assertEqual(status, 200, name)
            self.assertEqual(headers["Content-Type"], "image/jpeg")
            self.assertEqual(headers["Cache-Control"], "max-age=86400")
            self.assertTrue(data.startswith(b"\xff\xd8"), name)
            self.assertLess(len(data), 60 * 1024, name)

    def test_media_allow_list(self):
        for path in ("/media/", "/media/unknown.jpg", "/media/../index.html", "/media/%2e%2e/index.html",
                     "/media/cockpit.jpg/", "/media//cockpit.jpg", "/media/vendor/three.module.min.js"):
            status, _, _ = self.get(path)
            self.assertEqual(status, 404, path)

    def test_every_media_file_is_listed_and_shipped(self):
        self.assertEqual(set(os.listdir(os.path.join(ROOT, "ui", "media"))), set(web.MEDIA))
        manifest = os.path.join(ROOT, "packaging", "flatpak", "io.github.starman_tech.MousePods.json")
        with open(manifest, encoding="utf-8") as f:
            text = json.dumps(json.load(f))
        self.assertRegex(text, r"cp -r [^\"]*\bui\b")          # ui/ copied recursively: ui/media ships

    def test_cover_both_layouts(self):
        status, headers, data = self.get("/api/games/244210/cover")
        self.assertEqual((status, headers["Content-Type"], data), (200, "image/jpeg", JPEG))
        self.assertEqual(headers["Cache-Control"], "max-age=86400")
        status, headers, data = self.get("/api/games/227300/cover?v=1")
        self.assertEqual((status, headers["Content-Type"], data), (200, "image/png", PNG))

    def test_cover_missing_and_traversal(self):
        for path in ("/api/games/999/cover", "/api/games/../secret.jpg", "/api/games/..%2fsecret/cover",
                     "/api/games/%2e%2e/cover", "/api/games/abc/cover", "/api/games/12a/cover",
                     "/api/games/244210/../../appcache/secret.jpg", "/api/games/244210/cover/x",
                     "/api/games//cover", "/api/games/244210%2f..%2f/cover"):
            status, headers, _ = self.get(path)
            self.assertEqual(status, 404, path)
            self.assertTrue(headers["Content-Type"].startswith("application/json"), path)

    def test_cover_is_read_only(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request("POST", "/api/games/244210/cover", body=b"{}",
                  headers={"Content-Type": "application/json", "Host": f"127.0.0.1:{self.port}"})
        self.assertEqual(c.getresponse().status, 404)       # no POST route for covers: nothing changes
        c.close()
        with open(os.path.join(self.cache, "244210_header.jpg"), "rb") as f:
            self.assertEqual(f.read(), JPEG)

    def test_list_payload_says_which_games_have_a_cover(self):
        status, _, data = self.get("/api/games")
        self.assertEqual(status, 200)
        games = {g["appid"]: g for g in json.loads(data)["games"]}
        self.assertTrue(games["244210"]["cover"])
        self.assertTrue(games["227300"]["cover"])
        os.remove(os.path.join(self.cache, "244210_header.jpg"))
        games = {g["appid"]: g for g in json.loads(self.get("/api/games")[2])["games"]}
        self.assertFalse(games["244210"]["cover"])


if __name__ == "__main__":
    unittest.main()
