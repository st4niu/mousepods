# SPDX-License-Identifier: Apache-2.0
"""Undocumented fields of the 58-byte motion block (docs/DISCOVERIES.md §12), checked on real
labelled AirPods 3 recordings.

Run:  python -m unittest tests.test_motion_meta -v
"""
import collections
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import airpods_mouse_controller as m  # noqa: E402


def blocks(name):
    """[(label, block)] of the devmotion6 input reports of a recording in captures/."""
    out = []
    with open(os.path.join(ROOT, "captures", name)) as f:
        for row in json.load(f):
            for kind, dev, payload in m.rhid_parse(bytes.fromhex(row[-1])):
                if dev == 7 and kind == 2 and len(payload) == 59:
                    out.append((row[1] if len(row) > 2 else "", payload[1:]))
    return out


def states_by_label(rows):
    c = collections.defaultdict(collections.Counter)
    for label, blk in rows:
        c[label][m.parse_meta(blk)["motion_state"]] += 1
    return c


class MotionMetaTests(unittest.TestCase):
    def test_walking_is_moving_and_standing_is_still(self):
        c = states_by_label(blocks("cma_walk.json"))
        walk = c["marche"]
        self.assertGreater(walk["moving"] / sum(walk.values()), 0.7)          # 75 %: the rest is getting up / stopping
        jog = c["trot"]
        self.assertGreater(jog["moving"] / sum(jog.values()), 0.9)
        stand = c["debout"]
        self.assertGreater(stand["still"] / sum(stand.values()), 0.45)
        self.assertLess(stand["moving"] / sum(stand.values()), 0.15)

    def test_silence_is_still_talking_and_chewing_are_fidget(self):
        c = states_by_label(blocks("gestures_labeled.json"))
        self.assertGreater(c["silence"]["still"] / sum(c["silence"].values()), 0.7)
        for label in ("parle", "mache"):
            self.assertGreater(c[label]["fidget"] / sum(c[label].values()), 0.7, label)

    def test_counter_steps_by_four_and_energy_follows_motion(self):
        rows = blocks("cma_walk.json")
        metas = [m.parse_meta(b) for _, b in rows]
        steps = collections.Counter((b["counter"] - a["counter"]) % 65536 for a, b in zip(metas, metas[1:]))
        self.assertEqual(steps.most_common(1)[0][0], 4)
        energy = collections.defaultdict(list)
        for (label, _), meta in zip(rows, metas):
            energy[meta["motion_state"]].append(meta["motion_energy"])
        med = {k: sorted(v)[len(v) // 2] for k, v in energy.items()}
        self.assertLess(med["still"], med["fidget"])
        self.assertLess(med["fidget"], med["moving"])

    def test_unknown_state_is_shown_in_hex(self):
        blk = bytearray(58)
        blk[11] = 0x21
        self.assertEqual(m.parse_meta(bytes(blk))["motion_state"], "0x21")


if __name__ == "__main__":
    unittest.main()
