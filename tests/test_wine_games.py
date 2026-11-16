# SPDX-License-Identifier: Apache-2.0
"""Games outside Steam (any Wine prefix) and the TrackIR tables of games already switched on.

Run:  python -m unittest tests.test_wine_games -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import bridge_install as bi  # noqa: E402
import games_service  # noqa: E402
import game_bridge  # noqa: E402
import steam_games as sg  # noqa: E402
from test_bridge_install import DLLS, OPENTRACK_REG  # noqa: E402
import test_games_api  # noqa: E402

TABLES = {3450: bytes.fromhex("344ecef4a2b4c63e")}


class WineGamesApiTests(unittest.TestCase):
    setUp = test_games_api.GamesApiTests.setUp
    tearDown = test_games_api.GamesApiTests.tearDown
    get = test_games_api.GamesApiTests.get
    post = test_games_api.GamesApiTests.post
    saved = test_games_api.GamesApiTests.saved

    def make_prefix(self, rel="Games/star-citizen", reg=OPENTRACK_REG):
        prefix = os.path.join(self.home, rel)
        os.makedirs(prefix, exist_ok=True)
        with open(os.path.join(prefix, "user.reg"), "w") as f:
            f.write(reg)
        return prefix

    def post_error(self, path, body):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post(path, body)
        return cm.exception.code, json.load(cm.exception)

    def wine(self):
        return [g for g in self.get("/api/games")["games"] if g.get("wine")]

    def test_add_list_enable_remove(self):
        prefix = self.make_prefix()
        added = self.post("/api/games/wine", {"name": "  Star   Citizen ", "prefix": prefix})
        self.assertEqual(added["appid"], "wine-starcitizen")
        self.assertEqual(self.saved()["wine_games"]["wine-starcitizen"], {"name": "Star Citizen", "prefix": prefix})
        [g] = self.wine()
        self.assertEqual((g["name"], g["status"], g["has_prefix"], g["prefix"]), ("Star Citizen", "foreign", True, prefix))
        self.assertEqual(g["foreign_tracker"], "opentrack")

        self.post("/api/games/wine-starcitizen", {"enabled": True})
        with open(os.path.join(prefix, "user.reg")) as f:
            reg = f.read()
        self.assertIn(r'"Path"="C:\\mousepods\\"', reg)
        self.assertIn("pose-wine-starcitizen", reg)
        self.assertTrue(os.path.isfile(os.path.join(prefix, "drive_c", "mousepods", "NPClient64.dll")))
        self.assertEqual(self.wine()[0]["status"], "on")

        code, err = self.post_error("/api/games/wine-starcitizen", {"remove": True})
        self.assertEqual((code, err["code"]), (409, "switch_off_first"))
        self.post("/api/games/wine-starcitizen", {"enabled": False})
        with open(os.path.join(prefix, "user.reg")) as f:
            self.assertEqual(f.read().count("opentrack"), OPENTRACK_REG.count("opentrack"))   # given back
        self.post("/api/games/wine-starcitizen", {"remove": True})
        self.assertEqual(self.wine(), [])
        self.assertIsNone(self.saved()["wine_games"]["wine-starcitizen"])

    def test_proton_layout_and_duplicates(self):
        prefix = self.make_prefix("Games/lutris/pfx")
        added = self.post("/api/games/wine", {"name": "Élite", "prefix": os.path.dirname(prefix)})
        self.assertEqual((added["appid"], added["game"]["prefix"]), ("wine-elite", prefix))
        code, err = self.post_error("/api/games/wine", {"name": "Again", "prefix": prefix})
        self.assertEqual((code, err["code"]), (409, "already_added"))
        steam_pfx = os.path.join(self.steam, "steamapps", "compatdata", "244210", "pfx")
        self.assertEqual(self.post_error("/api/games/wine", {"name": "AC", "prefix": steam_pfx})[1]["code"],
                         "already_added")
        other = self.make_prefix("Games/other")
        self.assertEqual(self.post("/api/games/wine", {"name": "Elite", "prefix": other})["appid"], "wine-elite-2")

    def test_bad_input(self):
        self.assertEqual(self.post_error("/api/games/wine", {"name": " ", "prefix": self.make_prefix()})[1]["code"],
                         "bad_name")
        self.assertEqual(self.post_error("/api/games/wine", {"name": "X", "prefix": self.home})[1]["code"],
                         "bad_prefix")
        self.assertEqual(self.post_error("/api/games/wine", {"name": "X", "prefix": "/"})[1]["code"], "bad_prefix")
        self.assertEqual(self.post_error("/api/games/wine-nothere", {"enabled": True})[0], 404)

    def test_prefix_gone_while_on_can_be_removed(self):
        prefix = self.make_prefix()
        self.post("/api/games/wine", {"name": "Star Citizen", "prefix": prefix})
        self.post("/api/games/wine-starcitizen", {"enabled": True})
        shutil.rmtree(prefix)
        [g] = self.wine()
        self.assertEqual((g["enabled"], g["has_prefix"]), (True, False))
        self.post("/api/games/wine-starcitizen", {"remove": True})
        self.assertEqual(self.wine(), [])
        self.assertIsNone(self.saved()["games"]["wine-starcitizen"])

    def test_flatpak_asks_to_confirm_the_game_is_closed(self):
        prefix = self.make_prefix()
        self.post("/api/games/wine", {"name": "Star Citizen", "prefix": prefix})
        with mock.patch.object(games_service, "in_flatpak", lambda: True):
            self.assertEqual(self.post_error("/api/games/wine-starcitizen", {"enabled": True})[1]["code"],
                             "confirm_closed")
            self.assertEqual(self.wine()[0]["status"], "foreign")          # nothing changed
            self.post("/api/games/wine-starcitizen", {"enabled": True, "closed_confirmed": True})
            self.assertEqual(self.wine()[0]["status"], "on")
            self.assertEqual(games_service.refresh_tables(self.cfg_path), [])   # never on its own there

    def test_flatpak_hidden_folder_gives_its_path(self):
        missing = os.path.join(self.home, "elsewhere", "sc")
        with mock.patch.object(games_service, "in_flatpak", lambda: True):
            code, err = self.post_error("/api/games/wine", {"name": "SC", "prefix": missing})
        self.assertEqual((code, err["code"], err["path"]), (409, "prefix_hidden", missing))

    def test_pfx_link_is_resolved(self):
        real = self.make_prefix("Games/real")
        os.makedirs(os.path.join(self.home, "Games/lutris"))
        os.symlink(real, os.path.join(self.home, "Games/lutris/pfx"))
        added = self.post("/api/games/wine", {"name": "SC", "prefix": os.path.join(self.home, "Games/lutris")})
        self.assertEqual(added["game"]["prefix"], real)

    def test_running_game_is_not_touched(self):
        prefix = self.make_prefix()
        self.post("/api/games/wine", {"name": "Star Citizen", "prefix": prefix})
        self.running.add("wine-starcitizen")
        self.assertEqual(self.post_error("/api/games/wine-starcitizen", {"enabled": True})[1]["code"], "running")

    def test_steam_games_still_listed(self):
        self.post("/api/games/wine", {"name": "Star Citizen", "prefix": self.make_prefix()})
        names = [g["name"] for g in self.get("/api/games")["games"]]
        self.assertIn("Assetto Corsa", names)
        self.assertEqual(names[-1], "Star Citizen")


class RunningPrefixTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        bi._PREFIX_SEEN.clear()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def proc(self, pid, env):
        d = os.path.join(self.dir, "proc", str(pid))
        os.makedirs(d)
        with open(os.path.join(d, "environ"), "wb") as f:
            f.write(b"\0".join(env) + b"\0")

    def test_wineprefix_in_environment(self):
        real = os.path.join(self.dir, "pfx")
        os.makedirs(real)
        os.symlink(real, os.path.join(self.dir, "link"))
        self.proc(10, [b"HOME=/home/x", b"WINEPREFIX=" + os.path.join(self.dir, "link").encode()])
        self.proc(11, [b"WINEPREFIX="])
        proc = os.path.join(self.dir, "proc")
        self.assertEqual(bi.running_prefixes(proc, now=100.0), {real})
        shutil.rmtree(os.path.join(proc, "10"))
        self.assertEqual(bi.running_prefixes(proc, now=100.0 + bi.GRACE_S), {real})   # Wine may still write
        self.assertEqual(bi.running_prefixes(proc, now=101.0 + bi.GRACE_S), set())

    def test_pose_path_accepts_wine_ids_only(self):
        self.assertTrue(game_bridge.pose_path("wine-starcitizen", self.dir).endswith("pose-wine-starcitizen"))
        for bad in ("wine-", "wine-../x", "Wine-sc", "steam-1"):
            with self.assertRaises(ValueError):
                game_bridge.pose_path(bad, self.dir)


class RefreshTablesTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.prefix = os.path.join(self.dir, "pfx")
        os.makedirs(self.prefix)
        with open(os.path.join(self.prefix, "user.reg"), "w") as f:
            f.write(OPENTRACK_REG)
        self.dlls = os.path.join(self.dir, "dlls")
        os.makedirs(self.dlls)
        for n in DLLS:
            with open(os.path.join(self.dlls, n), "wb") as f:
                f.write(b"MZ")
        self.game = sg.Game("wine-sc", "SC", "", self.prefix, None, "", False)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def reg(self):
        with open(os.path.join(self.prefix, "user.reg")) as f:
            return f.read()

    def test_game_switched_on_without_tables_gets_them(self):
        bi.install(self.game, self.dlls, "/dev/shm/x/pose-wine-sc", lambda a: False, tables={})
        self.assertNotIn("Table", self.reg())
        self.assertFalse(bi.refresh_tables(self.game, lambda a: True, TABLES))       # running: wait
        self.assertTrue(bi.refresh_tables(self.game, lambda a: False, TABLES))
        self.assertIn('"3450"=hex:34,4e,ce,f4,a2,b4,c6,3e', self.reg())
        before = self.reg()
        self.assertFalse(bi.refresh_tables(self.game, lambda a: False, TABLES))      # already up to date
        self.assertEqual(self.reg(), before)
        bi.uninstall(self.game, lambda a: False)
        self.assertNotIn("Table", self.reg())

    def test_table_rewritten_by_wine_is_not_rewritten_again(self):
        bi.install(self.game, self.dlls, "/dev/shm/x/pose-wine-sc", lambda a: False, tables=TABLES | {13: bytes(8)})
        reg = self.reg()
        # what wine leaves after a session: #time= line, values sorted as text, upper-case hex
        wine = reg.replace('"13"=hex:00,00,00,00,00,00,00,00\n"3450"=hex:34,4e,ce,f4,a2,b4,c6,3e',
                           '#time=1dc5a0b2c3d4e5f\n"3450"=hex:34,4E,CE,F4,A2,B4,C6,3E\n"13"=hex:00,00,00,00,00,00,00,00')
        self.assertNotEqual(wine, reg)
        with open(os.path.join(self.prefix, "user.reg"), "w") as f:
            f.write(wine)
        self.assertFalse(bi.refresh_tables(self.game, lambda a: False, TABLES | {13: bytes(8)}))
        self.assertEqual(self.reg(), wine)

    def test_game_not_ours_is_left_alone(self):
        self.assertFalse(bi.refresh_tables(self.game, lambda a: False, TABLES))
        self.assertEqual(self.reg(), OPENTRACK_REG)


if __name__ == "__main__":
    unittest.main()
