# SPDX-License-Identifier: Apache-2.0
"""AAP status messages and DeviceConnect details shown on the Labo "AirPods data" card
(docs/DISCOVERIES.md §14). Packets are real AirPods 3 captures with the addresses replaced.

Run:  python -m unittest tests.test_airpods_info -v
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import airpods_mouse_controller as m  # noqa: E402

H = bytes.fromhex
MAC = "112233445566"          # stand-in address


def plist(service, loc, serial=None, prxc=False):
    extra = ""
    if serial:
        import base64
        extra += ("<key>SerialNumber</key><data>%s</data><key>CFG#</key><data>%s</data>"
                  % (base64.b64encode(serial + b"\0").decode(),
                     base64.b64encode(b"X2055/MP/TEST/*/*\0").decode()))
    if prxc:
        extra += "<key>PRXC</key><data>BAW6AdMB</data>"
    return ('<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict>'
            '<key>VendorID</key><integer>1452</integer><key>AccessoryService</key><string>%s</string>'
            '<key>LocationID</key><integer>%d</integer>%s</dict></plist>' % (service, loc, extra)).encode()


class StatusTests(unittest.TestCase):
    def test_bud_role(self):
        self.assertEqual(m.aap_status(H("04000400080001000100")), {"primary_bud": "left"})
        self.assertEqual(m.aap_status(H("04000400080002000100")), {"primary_bud": "right"})
        self.assertEqual(m.aap_status(H("04000400080003000100")), {"primary_bud": "headset"})

    def test_audio_source_and_connected_devices(self):
        self.assertEqual(m.aap_status(H("04000400 0e00" + MAC + "00")), {"audio_source": "none"})
        self.assertEqual(m.aap_status(H("04000400 0e00" + MAC + "02")), {"audio_source": "media"})
        self.assertEqual(m.aap_status(H("04000400 2e00 010001" + MAC + "0207")), {"connected_devices": 1})

    def test_other_packets_and_garbage_give_nothing(self):
        self.assertEqual(m.aap_status(H("0400040006000001")), {})   # ear detection: parsed elsewhere
        self.assertEqual(m.aap_status(b"\x04\x00"), {})
        self.assertEqual(m.aap_status(H("04000400 0e00 1122")), {})  # truncated

    def test_no_address_is_kept(self):
        aap = m.AAPClient(auto_connect=False)
        for pkt in ("04000400 0e00" + MAC + "00", "04000400 2e00 010001" + MAC + "0207",
                    "04000400 0c00" + MAC + "0101"):
            aap.process(H(pkt))
        self.assertNotIn("112233", repr(aap.status))


class DeviceInfoTests(unittest.TestCase):
    def test_serial_is_masked_and_config_read(self):
        d = m.device_info(plist("cma", 2, b"GX6ABCDEFGH1234", prxc=False))
        self.assertEqual(d["service"], "cma")
        self.assertEqual(d["location"], 2)
        self.assertEqual(d["serial"], "…1234")
        self.assertEqual(d["config"], "X2055/MP/TEST/*/*")
        self.assertFalse(d["proximity_calibration"])
        self.assertTrue(m.device_info(plist("kadabra", 1, prxc=True))["proximity_calibration"])
        self.assertEqual(m.device_info(b"not a plist"), {})

    def test_bud_switch_learns_side_and_motion_bud(self):
        aap = m.AAPClient(auto_connect=False)

        def announce(service, loc, serial=None):
            body = plist(service, loc, serial)
            aap.process(m.rhid_frame(0, [m.rhid_sub(7, m.RHID_CONNECT, body)]))

        announce("devmotion6", 1, b"SERIALRIGHT0001")
        self.assertEqual(aap.motion_loc, 1)
        aap.process(H("04000400080001000100"))            # left becomes primary
        announce("devmotion6", 2, b"SERIALLEFT00002")
        self.assertEqual(aap.motion_loc, 2)
        self.assertEqual(aap.loc_sides[2], "left")
        self.assertEqual(sorted(aap.buds), [1, 2])
        self.assertEqual(aap.buds[2]["serial"], "…0002")
        info = m.airpods_info(aap, object())
        self.assertEqual(info["motion_bud"], "left")
        self.assertEqual(info["primary_bud"], "left")
        self.assertEqual([s["name"] for s in info["services"]][:3], ["kadabra", "mando", "neorelay"])


class LinkTests(unittest.TestCase):
    def test_counter_gaps_count_lost_blocks(self):
        class Ctl:
            _link_feed = m.Controller._link_feed
        ctl = Ctl()
        metas = [{"counter": c, "sensor_clock": 1000 * i} for i, c in enumerate([0, 4, 8, 20, 24, 65532, 0])]
        prev = None
        for i, meta in enumerate(metas):
            ctl._link_feed(prev, meta, float(i))
            prev = meta
        self.assertEqual(ctl.link["samples"], 7)
        self.assertEqual(ctl.link["lost"], 2)              # 8 -> 20 = two blocks missing; the wrap is fine

    def test_info_without_client(self):
        self.assertEqual(m.airpods_info(None, object()), {"connected": False})


if __name__ == "__main__":
    unittest.main()
