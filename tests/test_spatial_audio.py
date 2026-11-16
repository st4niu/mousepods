# SPDX-License-Identifier: Apache-2.0
"""Tests of spatial_audio.py: directions, panning, head rotation, graph text, update diffs, WAV
files, HRIR extraction (libmysofa mocked). No audio hardware and no PipeWire needed.

Run:  python -m unittest tests.test_spatial_audio -v
"""
import math
import os
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatial_audio as sa  # noqa: E402


def idx(az, el):
    """Index of the fixed direction (az, el)."""
    for i, (a, e) in enumerate(sa.DIRECTIONS):
        if abs(e - el) < 1e-9 and abs(((a - az) + 180) % 360 - 180) < 1e-9:
            return i
    raise KeyError((az, el))


class Directions(unittest.TestCase):
    def test_rings(self):
        self.assertEqual(sa.N_DIRS, 60)
        self.assertEqual(len(sa.DIRECTIONS), 60)
        els = [e for _, e in sa.DIRECTIONS]
        self.assertEqual((els.count(-30.0), els.count(0.0), els.count(30.0)), (12, 36, 12))
        mid = sorted(a % 360 for a, e in sa.DIRECTIONS if e == 0.0)
        self.assertEqual(mid, [i * 10.0 for i in range(36)])
        for az in (0, 30, -30):                       # FC, FL, FR at rest sit on a direction
            idx(az, 0)
        for a, _ in sa.DIRECTIONS:
            self.assertTrue(-180 < a <= 180)


class Panning(unittest.TestCase):
    def energy(self, g):
        return sum(x * x for x in g)

    def test_energy_is_one_everywhere(self):
        for az in range(-360, 361, 7):
            for el in (-90, -40, -30, -17, 0, 5, 15, 29, 30, 60):
                self.assertAlmostEqual(self.energy(sa.gains_for(az, el)), 1.0, places=9)

    def test_front(self):
        g = sa.gains_for(0, 0)
        self.assertAlmostEqual(g[idx(0, 0)], 1.0)
        self.assertAlmostEqual(sum(abs(x) for x in g), 1.0)

    def test_between_two_neighbours(self):
        g = sa.gains_for(5, 0)
        a, b = g[idx(0, 0)], g[idx(10, 0)]
        self.assertAlmostEqual(a, math.cos(math.pi / 4))
        self.assertAlmostEqual(b, math.sin(math.pi / 4))
        self.assertAlmostEqual(sum(x for i, x in enumerate(g) if i not in (idx(0, 0), idx(10, 0))), 0.0)
        # closer to 10 -> more of it
        g = sa.gains_for(8, 0)
        self.assertGreater(g[idx(10, 0)], g[idx(0, 0)])

    def test_wraps_at_180(self):
        self.assertEqual(sa.gains_for(180, 0), sa.gains_for(-180, 0))
        self.assertAlmostEqual(sa.gains_for(180, 0)[idx(180, 0)], 1.0)
        g = sa.gains_for(-177, 0)          # between 180 and -170
        self.assertGreater(g[idx(180, 0)], 0.5)
        self.assertGreater(g[idx(-170, 0)], 0.1)
        for a, b in zip(sa.gains_for(183, 0), sa.gains_for(-177, 0)):
            self.assertAlmostEqual(a, b)

    def test_elevation_blend(self):
        g = sa.gains_for(0, 15)            # halfway between ring 0 and ring +30
        self.assertAlmostEqual(g[idx(0, 0)], math.cos(math.pi / 4))
        self.assertAlmostEqual(g[idx(0, 30)], math.sin(math.pi / 4))
        g = sa.gains_for(0, -30)
        self.assertAlmostEqual(g[idx(0, -30)], 1.0)
        self.assertEqual(sa.gains_for(0, 80), sa.gains_for(0, 30))     # clamped
        self.assertEqual(sa.gains_for(10, -80), sa.gains_for(10, -30))
        g = sa.gains_for(20, 10)           # on ring 0 at 20, only partly up
        self.assertGreater(g[idx(20, 0)], 0.5)
        self.assertGreater(g[idx(0, 30)] + g[idx(30, 30)], 0.1)


class HeadRotation(unittest.TestCase):
    def test_yaw_right_moves_sources_left(self):
        az, el = sa.relative_direction(0, 0, 30, 0)
        self.assertAlmostEqual(az, 30)
        self.assertAlmostEqual(el, 0)
        m = sa.gain_matrix(yaw=30)
        fc = sa.CHANNELS.index("FC")
        self.assertAlmostEqual(np.sum(m[fc] ** 2), 1.0)
        self.assertAlmostEqual(m[fc, idx(30, 0)], 1.0)                     # at +30 (left)

    def test_yaw_matches_simple_formula(self):
        for src in (0, 30, -30, 90, 150, -150):
            for yaw in (-120, -45, 10, 90):
                az, el = sa.relative_direction(src, 0, yaw, 0)
                self.assertAlmostEqual((az - (src + yaw) + 180) % 360 - 180, 0, places=9)
                self.assertAlmostEqual(el, 0, places=9)

    def test_pitch(self):
        self.assertAlmostEqual(sa.relative_direction(0, 0, 0, 20)[1], -20)   # look up: front goes down
        self.assertAlmostEqual(sa.relative_direction(180, 0, 0, 20)[1], 20)  # ... and behind goes up
        az, el = sa.relative_direction(90, 0, 0, 20)                         # side barely moves
        self.assertAlmostEqual(az, 90)
        self.assertAlmostEqual(el, 0)

    def test_layout(self):
        spk = sa.speakers(40)
        self.assertEqual(spk["FL"][:2], (40.0, 0.0))
        self.assertEqual(spk["FR"][:2], (-40.0, 0.0))
        self.assertEqual(spk["LFE"], (0.0, -30.0, 0.5))
        m = sa.gain_matrix()
        self.assertAlmostEqual(np.sum(m[sa.CHANNELS.index("LFE")] ** 2), 0.25)
        self.assertAlmostEqual(m[sa.CHANNELS.index("SL"), idx(90, 0)], 1.0)
        self.assertAlmostEqual(m[sa.CHANNELS.index("SR"), idx(-90, 0)], 1.0)


class Conf(unittest.TestCase):
    def conf(self, sink="bluez_output.40_DA_5C_BD_69_73.1"):
        files = [f"/x/dir{i:02d}.wav" for i in range(sa.N_DIRS)]
        return sa.build_conf(files, 1.5, sink, sa.gain_matrix(), 256)

    def test_structure(self):
        c = self.conf()
        nodes = re.findall(r"label = (\w+) name = (\S+)", c)
        names = [n for _, n in nodes]
        self.assertEqual(len(names), len(set(names)))                         # unique names
        self.assertEqual(sum(1 for lab, n in nodes if lab == "mixer" and re.fullmatch(r"m\d+", n)), 60)
        self.assertEqual(sum(1 for lab, _ in nodes if lab == "convolver"), 120)
        self.assertIn('outputs = [ "sl:Out" "sr:Out" ]', c)
        self.assertIn('inputs = [ "cFL:In" "cFR:In" "cFC:In" "cLFE:In" "cRL:In" "cRR:In" "cSL:In" "cSR:In" ]', c)
        self.assertIn('node.name = "mousepods_spatial"', c)
        self.assertIn('node.description = "MousePods Spatial"', c)
        self.assertIn("media.class = Audio/Sink", c)
        self.assertIn("audio.position = [ FL FR FC LFE RL RR SL SR ]", c)
        self.assertIn('node.name = "mousepods_spatial.output"', c)
        self.assertIn("node.passive = true", c)
        self.assertIn('target.object = "bluez_output.40_DA_5C_BD_69_73.1"', c)
        self.assertIn("blocksize = 256", c)
        self.assertIn("gain = 1.500000", c)

    def test_links(self):
        c = self.conf()
        links = re.findall(r'\{ output = "([^"]+)" input = "([^"]+)" \}', c)
        inputs = [i for _, i in links]
        self.assertEqual(len(inputs), len(set(inputs)))                      # an input has one link
        # every direction mixer gets all 8 channels, each ear convolver its mixer
        for d in range(sa.N_DIRS):
            for k in range(8):
                self.assertIn(f"m{d}:In {k + 1}", inputs)
            self.assertIn((f"m{d}:Out", f"l{d}:In"), links)
            self.assertIn((f"m{d}:Out", f"r{d}:In"), links)
        # no output fans out to more than 4 inputs (PipeWire 1.2.4 crashes with large fan-outs)
        fan = {}
        for o, _ in links:
            fan[o] = fan.get(o, 0) + 1
        self.assertLessEqual(max(fan.values()), 4)
        # the copy tree really brings channel k to "In k+1": follow it back to the source
        src = {i: o for o, i in links}
        for d in (0, 17, 59):
            for k, ch in enumerate(sa.CHANNELS):
                port = src[f"m{d}:In {k + 1}"]
                while not port.startswith("c"):
                    port = src[port.replace(":Out", ":In")]
                self.assertEqual(port, f"c{ch}:Out")
        # sum trees: 60 left and 60 right convolvers reach sl / sr (8 mixers of <= 8, then 1)
        self.assertEqual(sum(1 for o, i in links if re.fullmatch(r"l\d+:Out", o) and i.startswith("sl0_")), 60)
        self.assertEqual(sum(1 for o, i in links if re.fullmatch(r"r\d+:Out", o) and i.startswith("sr0_")), 60)
        self.assertEqual(sum(1 for o, i in links if re.fullmatch(r"sl0_\d:Out", o) and i.startswith("sl:In")), 8)
        self.assertEqual(sum(1 for o, i in links if re.fullmatch(r"sr0_\d:Out", o) and i.startswith("sr:In")), 8)
        self.assertLessEqual(max(int(i.rsplit(" ", 1)[1]) for _, i in links if ":In " in i), 8)

    def test_initial_gains_in_controls(self):
        c = self.conf()
        m_fc = re.search(r"name = m%d control = \{([^}]*)\}" % idx(0, 0), c).group(1)
        self.assertIn('"Gain 3" = 1.000000', m_fc)                       # FC straight ahead
        self.assertIn('"Gain 1" = 0.000000', m_fc)

    def test_never_targets_itself(self):
        for bad in (None, "", sa.OWN_SINK, sa.OWN_OUTPUT):
            with self.assertRaises(ValueError):
                self.conf(bad)


class Updates(unittest.TestCase):
    def test_threshold_and_stale_zeroes(self):
        last = np.zeros((8, 32))
        last[0, 1] = 0.5
        last[1, 2] = 0.3
        last[2, 3] = 0.01
        new = last.copy()
        new[0, 1] = 0.5015           # below threshold: not sent
        new[1, 2] = 0.31             # sent
        new[2, 3] = 0.0015           # fell under the threshold: sent as exact 0
        new[3, 4] = 0.001            # tiny new gain: stays 0, not sent
        ups = sa.diff_updates(last, new)
        self.assertEqual(sorted(ups), sorted([(2, 1, 0.31), (3, 2, 0.0)]))

    def test_no_change_no_line(self):
        m = sa.gain_matrix(10, 5)
        self.assertEqual(sa.diff_updates(m, m), [])

    def test_lines(self):
        lines = sa.set_param_lines(88, [(0, 0, 0.5), (31, 7, 0.0)])
        self.assertEqual(lines, ['set-param 88 Props { params = [ "m0:Gain 1" 0.5000 "m31:Gain 8" 0.0000 ] }'])
        # a full update (every gain) is split under pw-cli's 4 KiB input buffer
        ups = [(d, k, 0.123456) for d in range(sa.N_DIRS) for k in range(8)]
        lines = sa.set_param_lines(88, ups)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(x) <= sa.MAX_LINE for x in lines))
        self.assertEqual(sum(x.count(":Gain") for x in lines), 8 * sa.N_DIRS)

    def test_worker_send_tracks_state(self):
        sp = sa.SpatialAudio({}, tempfile.mkdtemp())
        sp.node_id = 7

        class Pipe:
            def __init__(self):
                self.data = b""

            def write(self, b):
                self.data += b

            def flush(self):
                pass

        class Cli:
            stdin = Pipe()

            def poll(self):
                return None

        sp._cli = Cli()
        sp._sent = sa.gain_matrix(0, 0)
        want = sa.gain_matrix(30, 0)
        prev, n = sp._sent.copy(), 0
        while sp._send(want):                                            # glides, MAX_STEP at a time
            self.assertLessEqual(np.max(np.abs(sp._sent - prev)), sa.MAX_STEP + 1e-9)
            prev, n = sp._sent.copy(), n + 1
            self.assertLess(n, 100)
        self.assertGreater(n, 5)
        text = sp._cli.stdin.data.decode()
        self.assertTrue(text.startswith("set-param 7 Props"))
        self.assertTrue(np.allclose(np.where(want < sa.THRESHOLD, 0, want), sp._sent, atol=sa.THRESHOLD))
        before, updates = len(sp._cli.stdin.data), sp._updates
        self.assertFalse(sp._send(want))                                 # reached: nothing more
        self.assertEqual(len(sp._cli.stdin.data), before)
        self.assertEqual(sp._updates, updates)

    def test_limit_step(self):
        last = np.zeros((2, 3))
        new = np.array([[1.0, 0.01, -1.0], [0.0, 0.5, 0.03]])
        out = sa.limit_step(last, new, 0.04)
        self.assertTrue(np.allclose(out, [[0.04, 0.01, -0.04], [0.0, 0.04, 0.03]]))

    def test_smooth_pose(self):
        self.assertEqual(sa.smooth_pose(None, (10, 5), 0.01), (10, 5))
        p = sa.smooth_pose((0.0, 0.0), (10.0, -10.0), 0.01, 0.05)
        a = 1 - math.exp(-0.01 / 0.05)
        self.assertAlmostEqual(p[0], 10 * a)
        self.assertAlmostEqual(p[1], -10 * a)
        p = sa.smooth_pose((175.0, 0.0), (-175.0, 0.0), 0.01, 0.05)      # short way through 180
        self.assertGreater(p[0], 175.0)
        p = (0.0, 0.0)
        for _ in range(100):                                             # 1 s: there
            p = sa.smooth_pose(p, (40.0, 10.0), 0.01, 0.05)
        self.assertEqual(p, (40.0, 10.0))

    def test_set_pose_ignores_garbage(self):
        sp = sa.SpatialAudio({}, tempfile.mkdtemp())
        sp.set_pose(float("nan"), 0)
        sp.set_pose(None, 0)
        self.assertEqual(sp._pose, (0.0, 0.0))
        sp.set_pose(12, -3)
        self.assertEqual(sp._pose, (12.0, -3.0))
        st = sp.status()
        self.assertFalse(st["active"])
        self.assertEqual(st["directions"], sa.N_DIRS)


class Wav(unittest.TestCase):
    def test_roundtrip(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "x.wav")
        x = np.vstack([np.linspace(-0.98, 0.98, 500), np.sin(np.arange(500) / 7) * 0.5])
        sa.write_wav(p, x, 48000)
        y, rate = sa.read_wav(p)
        self.assertEqual(rate, 48000)
        self.assertEqual(y.shape, (2, 500))
        self.assertLess(np.max(np.abs(x - y)), 1e-8)

    def test_normalise(self):
        irs = np.zeros((2, 2, 10))
        irs[0, 0, 3] = 2.0
        irs[1, 1, 4] = -0.5
        out, k = sa.normalise(irs)
        self.assertAlmostEqual(np.max(np.abs(out)), 0.98)
        self.assertAlmostEqual(k, 0.49)
        self.assertTrue(np.allclose(out / k, irs))


class FakeMysofa:
    """Stands in for libmysofa: a 'click' whose level and position depend on the direction."""
    N = 16

    def __init__(self):
        self.closed = False
        self.asked = []

    def open(self, filename, rate):
        return 1234, self.N

    def getfilter(self, h, n, x, y, z):
        self.asked.append((x, y, z))
        left, right = np.zeros(n), np.zeros(n)
        left[2] = 1.0 + 0.5 * y / sa.SOFA_RADIUS          # louder in the left ear for y > 0 (left)
        right[2] = 1.0 - 0.5 * y / sa.SOFA_RADIUS
        return left, right, (0.0 if y >= 0 else 3 / 48000), (3 / 48000 if y > 0 else 0.0)

    def close(self, h):
        self.closed = True


class Hrir(unittest.TestCase):
    def test_extract_with_fake_library(self):
        fake = FakeMysofa()
        irs = sa.extract_hrirs("/nope.sofa", sa.DIRECTIONS, 48000, fake)
        self.assertTrue(fake.closed)
        self.assertEqual(irs.shape[:2], (sa.N_DIRS, 2))
        self.assertEqual(irs.shape[2], 6)               # click at 2, delayed by up to 3, tail trimmed
        # coordinates: radius 1.2, x front, y left
        x, y, z = fake.asked[idx(90, 0)]
        self.assertAlmostEqual(y, 1.2)
        self.assertAlmostEqual(x, 0.0)
        x, y, z = fake.asked[idx(0, 30)]
        self.assertAlmostEqual(z, 0.6)
        left90 = irs[idx(90, 0)]
        self.assertAlmostEqual(left90[0, 2], 1.5)       # left ear: no delay
        self.assertAlmostEqual(left90[1, 5], 0.5)       # right ear: 3 samples later
        self.assertEqual(left90[1, 2], 0.0)

    def test_cache(self):
        d = tempfile.mkdtemp()
        sofa = os.path.join(d, "x.sofa")
        with open(sofa, "wb") as f:
            f.write(b"fake")
        fake = FakeMysofa()
        meta = sa.build_hrir_cache(sofa, os.path.join(d, "cache"), mysofa=fake)
        self.assertEqual(len(meta["files"]), sa.N_DIRS)
        self.assertTrue(all(os.path.exists(p) for p in meta["files"]))
        y, rate = sa.read_wav(meta["files"][idx(0, 0)])
        self.assertEqual(rate, 48000)
        peak = max(np.max(np.abs(sa.read_wav(p)[0])) for p in meta["files"])
        self.assertAlmostEqual(peak, 0.98, places=6)                               # one global peak
        self.assertGreater(meta["factor"], 0)
        n = len(fake.asked)
        meta2 = sa.build_hrir_cache(sofa, os.path.join(d, "cache"), mysofa=fake)      # cached
        self.assertEqual(len(fake.asked), n)
        self.assertEqual(meta2["key"], meta["key"])
        os.utime(sofa, ns=(1, 1))                                                    # new file
        meta3 = sa.build_hrir_cache(sofa, os.path.join(d, "cache"), mysofa=fake)
        self.assertNotEqual(meta3["key"], meta["key"])
        self.assertEqual(len(fake.asked), 2 * n)
        self.assertEqual(os.listdir(os.path.join(d, "cache")), [meta3["key"]])      # old cache gone

    @unittest.skipUnless(os.path.exists(os.path.expanduser("~/.local/share/mousepods/hrtf/mit_kemar_normal_pinna.sofa")),
                         "no MIT KEMAR file")
    def test_real_libmysofa(self):
        try:
            lib = sa.Mysofa()
        except OSError:
            self.skipTest("no libmysofa")
        path = os.path.expanduser("~/.local/share/mousepods/hrtf/mit_kemar_normal_pinna.sofa")
        irs = sa.extract_hrirs(path, [(90, 0), (-90, 0), (0, 0)], 48000, lib)
        e = np.sum(irs ** 2, axis=2)
        self.assertGreater(e[0, 0], 3 * e[0, 1])         # +90 = left: left ear louder
        self.assertGreater(e[1, 1], 3 * e[1, 0])
        self.assertAlmostEqual(e[2, 0] / e[2, 1], 1.0, delta=0.2)


class DiffuseField(unittest.TestCase):
    def colored(self):
        """60 directions x 2 ears of noisy IRs sharing a strong common resonance (~3 kHz)."""
        rng = np.random.default_rng(1)
        n = 400
        t = np.arange(n) / sa.RATE
        common = np.exp(-t * 900) * np.sin(2 * np.pi * 3000 * t)       # ringing resonance
        common[0] += 0.3
        irs = np.zeros((sa.N_DIRS, 2, n))
        for d in range(sa.N_DIRS):
            for e in (0, 1):
                irs[d, e] = np.convolve(common, rng.normal(0, 1, 20) * np.exp(-np.arange(20) / 4))[:n]
        return irs

    def band_spread(self, irs):
        f = np.fft.rfftfreq(4096, 1 / sa.RATE)
        p = np.mean(np.abs(np.fft.rfft(irs, 4096, axis=-1)) ** 2, axis=(0, 1))
        ps = 10 * np.log10(sa._third_octave_smooth(f, p))
        band = (f > 300) & (f < 12000)
        return ps[band].max() - ps[band].min()

    def test_flattens_and_keeps_loudness(self):
        irs = self.colored()
        out = sa.equalise(irs)
        self.assertLessEqual(out.shape[-1], sa.MAX_IR)
        self.assertLess(self.band_spread(out), self.band_spread(irs) - 6)
        self.assertAlmostEqual(sa.pink_energy(out) / sa.pink_energy(irs), 1.0, places=6)

    def test_filter_limits_and_minimum_phase(self):
        h = sa.diffuse_field_filter(self.colored())
        self.assertEqual(len(h), 256)
        H = 20 * np.log10(np.abs(np.fft.rfft(h, 4096)))
        f = np.fft.rfftfreq(4096, 1 / sa.RATE)
        band = (f > 200) & (f < 15000)
        self.assertLess(H[band].max() - H[band].min(), 24.5)            # +-12 dB around the centre
        e = np.cumsum(h ** 2) / np.sum(h ** 2)
        self.assertGreater(e[32], 0.8)                                    # energy at the start

    def test_flat_set_is_left_alone(self):
        irs = np.zeros((4, 2, 64))
        irs[:, :, 0] = 1.0                                                # already flat
        h = sa.diffuse_field_filter(irs)
        H = 20 * np.log10(np.abs(np.fft.rfft(h, 4096)))
        self.assertLess(np.max(np.abs(H[10:-10])), 0.5)


class Sinks(unittest.TestCase):
    def dump(self, default="alsa_output.speaker", bluez=True):
        def node(i, name, cls="Audio/Sink"):
            return {"id": i, "type": "PipeWire:Interface:Node",
                    "info": {"props": {"node.name": name, "media.class": cls}}}
        d = [node(1, "alsa_output.speaker"), node(2, sa.OWN_SINK),
             node(3, sa.OWN_OUTPUT, "Stream/Output/Audio")]
        if bluez:
            d.append(node(4, "bluez_output.40_DA_5C_BD_69_73.1"))
        d.append({"id": 0, "type": "PipeWire:Interface:Metadata", "props": {"metadata.name": "default"},
                  "metadata": [{"subject": 0, "key": "default.audio.sink", "type": "Spa:String:JSON",
                                "value": {"name": default}},
                               {"subject": 0, "key": "default.configured.audio.sink",
                                "type": "Spa:String:JSON", "value": '{"name": "alsa_output.speaker"}'}]})
        return d

    def test_prefers_bluez(self):
        self.assertEqual(sa.find_output_sink(True, self.dump()), "bluez_output.40_DA_5C_BD_69_73.1")

    def test_default_when_no_bluez(self):
        self.assertEqual(sa.find_output_sink(True, self.dump(bluez=False)), "alsa_output.speaker")
        self.assertEqual(sa.find_output_sink(False, self.dump()), "alsa_output.speaker")

    def test_never_itself(self):
        d = self.dump(default=sa.OWN_SINK, bluez=False)
        self.assertEqual(sa.find_output_sink(True, d), "alsa_output.speaker")

    def test_metadata(self):
        d = self.dump()
        self.assertEqual(sa.default_metadata(d, "default.configured.audio.sink"), "alsa_output.speaker")
        self.assertEqual(sa.find_node(d, sa.OWN_SINK), 2)
        self.assertIsNone(sa.find_node(d, "nothing"))

    def test_find_hrtf(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "hrtf"))
        open(os.path.join(d, "hrtf", "mit_kemar_normal_pinna.sofa"), "wb").close()
        self.assertEqual(sa.find_hrtf(d), os.path.join(d, "hrtf", "mit_kemar_normal_pinna.sofa"))


if __name__ == "__main__":
    unittest.main()
