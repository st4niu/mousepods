# SPDX-License-Identifier: Apache-2.0
"""Tests of spatial audio (spatial_audio.py + spatial_render.py): head rotation, binaural
renderer (overlap-add, old/new HRIR crossfade, loudness, downmix, intensity, silence skipping,
LFE, limiter), stream containers, diffuse-field EQ, libmysofa (mocked), command lines and the
pipe plumbing with fake processes. No audio hardware and no PipeWire needed.

Run:  python -m unittest tests.test_spatial_audio -v
"""
import math
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatial_audio as sa  # noqa: E402
import spatial_render as sr  # noqa: E402

B = sr.BLOCK
KEMAR = os.path.expanduser("~/.local/share/mousepods/hrtf/mit_kemar_normal_pinna.sofa")


class FakeHrtf:
    """Pure delay + gain per ear, both depending on the direction: big HRIR changes when the head
    moves, easy to predict."""
    L = 64

    def __init__(self):
        self.calls = 0

    def params(self, az, el):
        s = math.sin(math.radians(az))
        dl = 20 + int(round(8 * (1 - s)))         # source on the left: left ear earlier
        dr = 20 + int(round(8 * (1 + s)))
        return dl, dr, 0.6 + 0.3 * s, 0.6 - 0.3 * s

    def __call__(self, az, el):
        self.calls += 1
        dl, dr, gl, gr = self.params(az, el)
        h = np.zeros((2, self.L))
        h[0, dl], h[1, dr] = gl, gr
        return h


def renderer(**kw):
    args = dict(eq=None, intensity=1.0, binaural_gain=1.0, dry_delay=0)
    args.update(kw)
    fetch = args.pop("fetch", None) or FakeHrtf()
    return sr.Renderer(fetch, **args)


def run(r, x, poses=None):
    out = []
    for i in range(len(x) // B):
        yaw, pitch = poses[i] if poses else (0.0, 0.0)
        out.append(r.process(x[i * B:(i + 1) * B], yaw, pitch))
    return np.concatenate(out).astype(np.float64)


class HeadRotation(unittest.TestCase):
    def test_yaw_right_moves_sources_left(self):
        az, el = sr.relative_direction(0, 0, 30, 0)
        self.assertAlmostEqual(az, 30)
        self.assertAlmostEqual(el, 0)

    def test_yaw_matches_simple_formula(self):
        for src in (0, 30, -30, 90, 150, -150):
            for yaw in (-120, -45, 10, 90):
                az, el = sr.relative_direction(src, 0, yaw, 0)
                self.assertAlmostEqual((az - (src + yaw) + 180) % 360 - 180, 0, places=9)
                self.assertAlmostEqual(el, 0, places=9)

    def test_pitch(self):
        self.assertAlmostEqual(sr.relative_direction(0, 0, 0, 20)[1], -20)   # look up: front goes down
        self.assertAlmostEqual(sr.relative_direction(180, 0, 0, 20)[1], 20)  # ... and behind goes up
        az, el = sr.relative_direction(90, 0, 0, 20)                         # side barely moves
        self.assertAlmostEqual(az, 90)
        self.assertAlmostEqual(el, 0)

    def test_round_direction(self):
        self.assertEqual(sr.round_direction(29.6, 0.4), (30, 0))
        self.assertEqual(sr.round_direction(-180.2, 0), (180, 0))
        self.assertEqual(sr.round_direction(190, 0), (-170, 0))
        self.assertEqual(sr.round_direction(57, 95), (0, 90))

    def test_smooth_pose(self):
        self.assertEqual(sr.smooth_pose(None, (10, 5), 0.01), (10, 5))
        p = sr.smooth_pose((0.0, 0.0), (10.0, -10.0), 0.01, 0.05)
        a = 1 - math.exp(-0.01 / 0.05)
        self.assertAlmostEqual(p[0], 10 * a)
        self.assertAlmostEqual(p[1], -10 * a)
        p = sr.smooth_pose((175.0, 0.0), (-175.0, 0.0), 0.01, 0.05)      # short way through 180
        self.assertGreater(p[0], 175.0)
        p = (0.0, 0.0)
        for _ in range(100):
            p = sr.smooth_pose(p, (40.0, 10.0), 0.01, 0.05)
        self.assertEqual(p, (40.0, 10.0))

    def test_layout(self):
        spk = sr.speakers(40)
        self.assertEqual(spk["FL"][:2], (40.0, 0.0))
        self.assertEqual(spk["FR"][:2], (-40.0, 0.0))
        self.assertEqual(spk["SL"][:2], (90.0, 0.0))
        self.assertEqual(spk["RR"][:2], (-150.0, 0.0))


class Render(unittest.TestCase):
    def test_overlap_add_equals_direct_convolution(self):
        rng = np.random.default_rng(3)
        n = 12 * B
        x = np.zeros((n, 8))
        x[:, 2] = rng.normal(0, 0.05, n)                 # FC
        fetch = FakeHrtf()
        y = run(renderer(fetch=fetch), x)
        h = fetch(0, 0)
        for ear in (0, 1):
            ref = np.convolve(x[:, 2], h[ear])[:n]
            self.assertLess(np.max(np.abs(y[:, ear] - ref)), 1e-6)

    def test_long_filter_tail_carries_over_blocks(self):
        def fetch(az, el):                                # IR longer than two blocks
            h = np.zeros((2, 1400))
            h[:, 0] = 0.5
            h[:, 1300] = 0.25
            return h
        n = 8 * B
        x = np.zeros((n, 8))
        x[:B, 2] = np.sin(np.arange(B) / 5) * 0.1
        y = run(renderer(fetch=fetch), x)
        ref = np.convolve(x[:, 2], fetch(0, 0)[0])[:n]
        self.assertLess(np.max(np.abs(y[:, 0] - ref)), 1e-6)

    def test_crossfade_has_no_discontinuity(self):
        n = 40 * B
        t = np.arange(n) / sr.RATE
        x = np.zeros((n, 8))
        x[:, 2] = 0.1 * np.sin(2 * np.pi * 1000 * t)
        poses = [(7.0 * i, 0.0) for i in range(n // B)]   # the head jumps every block
        r = renderer()
        y = run(r, x, poses)
        bound = 2 * np.pi * 1000 / sr.RATE * 0.1 * 0.9 * 1.3     # steepest slope of the signal
        self.assertLess(np.max(np.abs(np.diff(y[B:], axis=0))), bound)
        self.assertGreater(r.stats["iffts"], n // B)              # it did crossfade
        # same run without the crossfade: the HRIR switch clicks (so the test can see a click)
        r = renderer()
        r._ramp = np.ones_like(r._ramp)
        y = run(r, x, poses)
        self.assertGreater(np.max(np.abs(np.diff(y[B:], axis=0))), 3 * bound)

    def test_loudness_matches_dry_stereo(self):
        rng = np.random.default_rng(5)
        n = 60 * B
        x = np.zeros((n, 8))
        x[:, 0] = rng.normal(0, 0.05, n)
        x[:, 1] = rng.normal(0, 0.05, n)
        r = sr.Renderer(FakeHrtf(), None, intensity=1.0, dry_delay=0)   # automatic gain
        y = run(r, x)
        ratio = np.mean(y[B:] ** 2) / np.mean(x[B:, :2] ** 2)
        self.assertAlmostEqual(ratio, 1.0, delta=0.08)

    def test_intensity_mix(self):
        rng = np.random.default_rng(7)
        n = 6 * B
        x = rng.normal(0, 0.02, (n, 8))
        dry = sr.itu_downmix(x)
        y0 = run(renderer(intensity=0.0, dry_delay=10), x)
        self.assertLess(np.max(np.abs(y0[10:] - dry[:-10])), 1e-6)       # plain delayed downmix
        self.assertLess(np.max(np.abs(y0[:10])), 1e-9)
        y1 = run(renderer(intensity=1.0, dry_delay=10), x)
        yh = run(renderer(intensity=0.5, dry_delay=10), x)
        self.assertLess(np.max(np.abs(yh - 0.5 * (y0 + y1))), 1e-6)

    def test_downmix(self):
        x = np.zeros((1, 8))
        x[0] = [1, 2, 3, 4, 5, 6, 7, 8]                   # FL FR FC LFE RL RR SL SR
        c = math.sqrt(0.5)
        left, right = sr.itu_downmix(x)[0]
        self.assertAlmostEqual(left, 1 + c * 3 + c * 7 + c * 5 + 0.5 * 4)
        self.assertAlmostEqual(right, 2 + c * 3 + c * 8 + c * 6 + 0.5 * 4)
        st = np.zeros((3, 8))
        st[:, :2] = [[0.1, -0.2], [0.3, 0.0], [0.0, 0.5]]
        self.assertTrue(np.allclose(sr.itu_downmix(st), st[:, :2]))  # stereo passes through

    def test_silence_is_skipped(self):
        r = renderer()
        x = np.zeros((B, 8))
        x[:, 0] = 0.01
        x[:, 1] = -0.01
        r.process(x)
        self.assertEqual(r.last_active, [0, 1])
        self.assertEqual(r.stats["ffts"], 2)
        for _ in range(5):                                 # tail rings out, then nothing at all
            r.process(np.zeros((B, 8)))
        ffts, iffts = r.stats["ffts"], r.stats["iffts"]
        for _ in range(5):
            y = r.process(np.zeros((B, 8)))
            self.assertFalse(y.any())
        self.assertEqual((r.stats["ffts"], r.stats["iffts"]), (ffts, iffts))

    def test_lfe_no_hrtf_low_passed(self):
        fetch = FakeHrtf()
        r = renderer(fetch=fetch, dry_delay=40)
        calls = fetch.calls
        n = 10 * B
        t = np.arange(n) / sr.RATE
        x = np.zeros((n, 8))
        x[:, 3] = 0.2 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 5000 * t)
        y = run(r, x)
        self.assertEqual(fetch.calls, calls)                       # no HRIR for the LFE
        self.assertTrue(np.allclose(y[:, 0], y[:, 1]))             # same in both ears
        spec = np.abs(np.fft.rfft(y[2 * B:, 0]))
        f = np.fft.rfftfreq(n - 2 * B, 1 / sr.RATE)
        lo, hi = spec[np.argmin(np.abs(f - 50))], spec[np.argmin(np.abs(f - 5000))]
        self.assertGreater(lo, 30 * hi)

    def test_limiter_never_clips(self):
        rng = np.random.default_rng(9)
        n = 20 * B
        x = np.zeros((n, 8))
        x[:, 0] = rng.normal(0, 0.05, n)
        x[5 * B:8 * B, 0] *= 60                            # sudden very loud part
        r = renderer(gain_db=6.0)
        y = run(r, x)
        self.assertLess(np.max(np.abs(y)), 1.0)
        self.assertGreater(r.stats["limited"], 0)
        quiet = renderer()                                 # quiet signal: untouched
        run(quiet, x[:4 * B] * 0.1)
        self.assertEqual(quiet.stats["limited"], 0)

    def test_hrir_cache(self):
        fetch = FakeHrtf()
        r = renderer(fetch=fetch)
        x = np.zeros((B, 8))
        x[:, 2] = 0.01
        r.process(x, 0, 0)
        calls = fetch.calls
        for _ in range(5):
            r.process(x, 0.3, 0.2)                          # same 1° cell
        self.assertEqual(fetch.calls, calls)
        r.process(x, 40, 0)
        self.assertGreater(fetch.calls, calls)

    def test_prepare_hrir(self):
        h = np.ones((2, 3000))
        out = sr.prepare_hrir(h, np.array([1.0, 1.0]), 1537)
        self.assertEqual(out.shape, (2, 1537))
        self.assertAlmostEqual(out[0, -1], 0.0)
        self.assertAlmostEqual(out[0, 10], 2.0)


class Containers(unittest.TestCase):
    def reader(self, data):
        pos = [0]

        def read(n):
            chunk = data[pos[0]:pos[0] + n]
            pos[0] += len(chunk)
            return chunk
        return read

    def test_raw_f32_roundtrip(self):
        frames = np.array([[0.5, -0.25], [1.0, 0.0]])
        data = sr.encode_f32(frames)
        self.assertEqual(len(data), 16)
        info, left = sr.read_stream_header(self.reader(data))
        self.assertEqual(info["container"], "raw")
        self.assertTrue(np.allclose(sr.decode_frames(left + data[4:], info["dtype"], 2), frames))

    def test_au_header(self):
        frames = np.array([[0.5, -0.25], [1.0, 0.0]])
        data = (struct.pack(">4sIIIII", b".snd", 24, 0xFFFFFFFF, 6, 48000, 2)
                + np.ascontiguousarray(frames, ">f4").tobytes())
        info, left = sr.read_stream_header(self.reader(data))
        self.assertEqual((info["container"], info["channels"], info["rate"]), ("au", 2, 48000))
        self.assertEqual(left, b"")
        self.assertTrue(np.allclose(sr.decode_frames(data[24:], info["dtype"], 2), frames))

    def wav_header(self, tag=3, bits=32, ch=8, extensible=False):
        if extensible:
            fmt = struct.pack("<HHIIHHHHIH14s", 0xFFFE, ch, 48000, 48000 * ch * bits // 8, ch * bits // 8,
                              bits, 22, bits, 0, tag, b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71")
        else:
            fmt = struct.pack("<HHIIHH", tag, ch, 48000, 48000 * ch * bits // 8, ch * bits // 8, bits)
        return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt))
                + fmt + b"LIST" + struct.pack("<I", 4) + b"abcd" + b"data" + struct.pack("<I", 0xFFFFFFFF))

    def test_wav_float_stream(self):
        frames = (np.arange(16, dtype=np.float32).reshape(2, 8) / 20)
        for ext in (False, True):
            data = self.wav_header(extensible=ext) + frames.astype("<f4").tobytes()
            read = self.reader(data)
            info, left = sr.read_stream_header(read)
            self.assertEqual((info["container"], info["channels"], info["dtype"]), ("wav", 8, "<f4"))
            self.assertTrue(np.allclose(sr.decode_frames(left + read(1000), info["dtype"], 8), frames))

    def test_wav_pcm16_and_raw(self):
        info, _ = sr.read_stream_header(self.reader(self.wav_header(tag=1, bits=16, ch=2)))
        self.assertEqual(info["dtype"], "<i2")
        self.assertAlmostEqual(sr.decode_frames(np.array([32767, -32767], "<i2").tobytes(), "<i2", 2)[0, 0], 1.0)
        info, left = sr.read_stream_header(self.reader(b"\x00\x00\x80\x3f" * 8))
        self.assertEqual(info["container"], "raw")
        self.assertEqual(left, b"\x00\x00\x80\x3f")             # first bytes handed back
        with self.assertRaises(EOFError):
            sr.read_stream_header(self.reader(b"RIFF\x00"))


class DiffuseField(unittest.TestCase):
    def colored(self):
        """60 directions x 2 ears of noisy IRs sharing a strong common resonance (~3 kHz)."""
        rng = np.random.default_rng(1)
        n = 400
        t = np.arange(n) / sa.RATE
        common = np.exp(-t * 900) * np.sin(2 * np.pi * 3000 * t)
        common[0] += 0.3
        irs = np.zeros((sa.N_DIRS, 2, n))
        for d in range(sa.N_DIRS):
            for e in (0, 1):
                irs[d, e] = np.convolve(common, rng.normal(0, 1, 20) * np.exp(-np.arange(20) / 4))[:n]
        return irs

    def test_flattens(self):
        irs = self.colored()
        h = sa.diffuse_field_filter(irs)
        out = np.apply_along_axis(lambda v: np.convolve(v, h), -1, irs)
        f = np.fft.rfftfreq(4096, 1 / sa.RATE)
        band = (f > 300) & (f < 12000)

        def spread(a):
            p = np.mean(np.abs(np.fft.rfft(a, 4096, axis=-1)) ** 2, axis=(0, 1))
            ps = 10 * np.log10(sa._third_octave_smooth(f, p))
            return ps[band].max() - ps[band].min()
        self.assertLess(spread(out), spread(irs) - 6)

    def test_filter_limits_and_minimum_phase(self):
        h = sa.diffuse_field_filter(self.colored())
        self.assertEqual(len(h), 256)
        H = 20 * np.log10(np.abs(np.fft.rfft(h, 4096)))
        f = np.fft.rfftfreq(4096, 1 / sa.RATE)
        band = (f > 200) & (f < 15000)
        self.assertLess(H[band].max() - H[band].min(), 24.5)
        e = np.cumsum(h ** 2) / np.sum(h ** 2)
        self.assertGreater(e[32], 0.8)

    def test_flat_set_is_left_alone(self):
        irs = np.zeros((4, 2, 64))
        irs[:, :, 0] = 1.0
        H = 20 * np.log10(np.abs(np.fft.rfft(sa.diffuse_field_filter(irs), 4096)))
        self.assertLess(np.max(np.abs(H[10:-10])), 0.5)


class FakeMysofa:
    """Stands in for libmysofa: a 'click' whose level and ear delay depend on the direction."""
    N = 16

    def __init__(self):
        self.closed = False
        self.asked = []

    def open(self, filename, rate):
        return 1234, self.N

    def getfilter(self, h, n, x, y, z):
        self.asked.append((x, y, z))
        left, right = np.zeros(n), np.zeros(n)
        left[2] = 1.0 + 0.5 * y / sa.SOFA_RADIUS
        right[2] = 1.0 - 0.5 * y / sa.SOFA_RADIUS
        return left, right, (0.0 if y >= 0 else 3 / 48000), (3 / 48000 if y > 0 else 0.0)

    def close(self, h):
        self.closed = True


class Hrir(unittest.TestCase):
    def test_sofa_hrtf_with_fake_library(self):
        fake = FakeMysofa()
        h = sa.SofaHrtf("/nope.sofa", 48000, fake)
        pair = h(90, 0)
        x, y, z = fake.asked[-1]
        self.assertAlmostEqual(y, 1.2)                   # radius 1.2, y = left
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(pair[0, 2], 1.5)          # left ear: no delay
        self.assertAlmostEqual(pair[1, 5], 0.5)          # right ear: 3 samples later
        h(0, 30)
        self.assertAlmostEqual(fake.asked[-1][2], 0.6)
        h.close()
        self.assertTrue(fake.closed)

    def test_extract_and_eq_cache(self):
        fake = FakeMysofa()
        irs = sa.extract_hrirs("/nope.sofa", sa.DIRECTIONS, 48000, fake)
        self.assertEqual(irs.shape[:2], (sa.N_DIRS, 2))
        d = tempfile.mkdtemp()
        sofa = os.path.join(d, "x.sofa")
        with open(sofa, "wb") as f:
            f.write(b"fake")
        h1 = sa.eq_for(sofa, d, mysofa=fake)
        n = len(fake.asked)
        h2 = sa.eq_for(sofa, d, mysofa=fake)             # cached: libmysofa not asked again
        self.assertEqual(len(fake.asked), n)
        self.assertTrue(np.allclose(h1, h2))
        os.utime(sofa, ns=(1, 1))                         # new file: rebuilt, old cache removed
        sa.eq_for(sofa, d, mysofa=fake)
        self.assertGreater(len(fake.asked), n)
        self.assertEqual(len([p for p in os.listdir(d) if p.startswith("eq-")]), 1)

    @unittest.skipUnless(os.path.exists(KEMAR), "no MIT KEMAR file")
    def test_real_libmysofa(self):
        try:
            lib = sa.Mysofa()
        except OSError:
            self.skipTest("no libmysofa")
        h = sa.SofaHrtf(KEMAR, 48000, lib)
        try:
            e = [np.sum(h(az, 0) ** 2, axis=1) for az in (90, -90, 0)]
        finally:
            h.close()
        self.assertGreater(e[0][0], 3 * e[0][1])          # +90 = left: left ear louder
        self.assertGreater(e[1][1], 3 * e[1][0])
        self.assertAlmostEqual(e[2][0] / e[2][1], 1.0, delta=0.2)


class Commands(unittest.TestCase):
    def test_sink(self):
        c = sa.sink_command()
        for part in ("create-node adapter {", "factory.name = support.null-audio-sink",
                     'node.name = "mousepods_spatial"', 'node.description = "MousePods Spatial"',
                     "media.class = Audio/Sink", "audio.channels = 8",
                     "audio.position = [ FL FR FC LFE RL RR SL SR ]", "object.linger = false"):
            self.assertIn(part, c)
        self.assertNotIn("\n", c)

    def test_capture(self):
        c = sa.capture_command("mousepods_spatial")
        self.assertEqual(c[0], "pw-record")
        self.assertEqual(c[c.index("--target") + 1], "mousepods_spatial")
        self.assertIn("stream.capture.sink = true", c[c.index("-P") + 1])
        self.assertEqual(c[c.index("--format") + 1], "f32")
        self.assertEqual(c[c.index("--channels") + 1], "8")
        self.assertEqual(c[c.index("--rate") + 1], "48000")
        self.assertEqual(c[-1], "-")

    def test_playback_never_targets_itself(self):
        c = sa.playback_command("bluez_output.40_DA_5C_BD_69_73.1")
        self.assertEqual(c[c.index("--target") + 1], "bluez_output.40_DA_5C_BD_69_73.1")
        props = c[c.index("-P") + 1]
        self.assertIn("node.dont-reconnect = true", props)
        self.assertEqual(c[c.index("--format") + 1], "f32")
        self.assertEqual(c[c.index("--channels") + 1], "2")
        self.assertNotIn("node.group", props)             # one group would stall the graph
        self.assertEqual(c[-1], "-")
        for bad in (None, "", "mousepods_spatial"):
            with self.assertRaises(ValueError):
                sa.playback_command(bad)


class RawFlagTests(unittest.TestCase):
    def setUp(self):
        sa._RAW_FLAG.clear()
        self.addCleanup(sa._RAW_FLAG.clear)

    def _help(self, text):
        from unittest import mock
        res = mock.Mock(stdout=text, stderr="")
        return mock.patch.object(sa.subprocess, "run", return_value=res)

    def test_new_pipewire_gets_raw(self):
        with self._help("  -a, --raw                             RAW mode\n"):
            c = sa.playback_command("bluez_output.x")
        self.assertIn("--raw", c)
        self.assertEqual(c[-1], "-")

    def test_old_pipewire_has_no_raw(self):
        with self._help("  -h, --help  Show this help\n"):
            self.assertNotIn("--raw", sa.playback_command("bluez_output.x"))

    def test_missing_tool_is_not_fatal(self):
        from unittest import mock
        with mock.patch.object(sa.subprocess, "run", side_effect=OSError):
            self.assertEqual(sa.raw_flag("pw-play"), [])


class FollowOutputTests(unittest.TestCase):
    """Spatial audio started before the AirPods connected must move onto them."""

    def engine(self, output, auto=True):
        from unittest import mock
        e = sa.SpatialAudio({}, tempfile.mkdtemp())
        e.output, e._auto_output, e._output_present = output, auto, True
        e._kill_play = mock.Mock()
        return e

    def follow(self, e, best, present=True):
        from unittest import mock
        with mock.patch.object(sa, "find_output_sink", return_value=best):
            return e._follow_output({}, present)

    def test_moves_from_speakers_to_airpods(self):
        e = self.engine("alsa_output.speaker")
        self.assertTrue(self.follow(e, "bluez_output.40_DA_5C_BD_69_73.1"))
        self.assertEqual(e.output, "bluez_output.40_DA_5C_BD_69_73.1")
        e._kill_play.assert_called_once()

    def test_stays_when_already_on_airpods_or_nothing_better(self):
        e = self.engine("bluez_output.a")
        self.assertFalse(self.follow(e, "bluez_output.b"))       # another headset: keep ours
        self.assertFalse(self.follow(e, "bluez_output.a"))
        e = self.engine("alsa_output.speaker")
        self.assertFalse(self.follow(e, "alsa_output.hdmi"))     # no AirPods: no reason to move

    def test_falls_back_when_the_output_goes(self):
        e = self.engine("bluez_output.a")
        self.assertTrue(self.follow(e, "alsa_output.speaker", present=False))
        self.assertEqual(e.output, "alsa_output.speaker")

    def test_an_output_chosen_in_the_settings_is_kept(self):
        e = self.engine("alsa_output.speaker", auto=False)
        self.assertFalse(self.follow(e, "bluez_output.a"))
        self.assertEqual(e.output, "alsa_output.speaker")


class FakeProc:
    def __init__(self, stdin=None, stdout=None):
        self.stdin, self.stdout, self.pid = stdin, stdout, 999999
        self.dead = False

    def poll(self):
        return 1 if self.dead else None


class Plumbing(unittest.TestCase):
    def service(self):
        sp = sa.SpatialAudio({"make_default": False}, tempfile.mkdtemp())
        sp.output = "alsa_output.speaker"
        return sp

    def test_set_pose_ignores_garbage(self):
        sp = self.service()
        sp.set_pose(float("nan"), 0)
        sp.set_pose(None, 0)
        self.assertEqual(sp._pose, (0.0, 0.0))
        sp.set_pose(12, -3)
        self.assertEqual(sp._pose, (12.0, -3.0))
        st = sp.status()
        self.assertFalse(st["active"])
        for k in ("active", "output", "error", "output_present"):
            self.assertIn(k, st)

    def test_read_block_from_pipe(self):
        sp = self.service()
        r, w = os.pipe()
        frames = np.random.default_rng(2).normal(0, 0.1, (2 * B, 8)).astype(np.float32)
        hdr = Containers().wav_header()
        os.write(w, hdr + frames[:B].astype("<f4").tobytes())
        out = os.fdopen(r, "rb", buffering=0)
        sp._rec = FakeProc(stdout=out)
        sp._rec_info, sp._rec_left = sr.read_stream_header(out.read)
        x = sp._read_block()
        self.assertTrue(np.allclose(x, frames[:B]))
        os.write(w, frames[B:].astype("<f4").tobytes())
        os.close(w)
        self.assertTrue(np.allclose(sp._read_block(), frames[B:]))
        self.assertIsNone(sp._read_block())               # writer gone: end of stream
        out.close()

    def test_read_block_drops_backlog(self):
        sp = self.service()
        r, w = os.pipe()
        sa._set_pipe_size(w, 1 << 20)
        os.write(w, Containers().wav_header())
        out = os.fdopen(r, "rb", buffering=0)
        sp._rec = FakeProc(stdout=out)
        sp._rec_info, sp._rec_left = sr.read_stream_header(out.read)
        block = np.zeros((B, 8), "<f4").tobytes()
        for _ in range(sa.MAX_PIPE_BLOCKS + 3):
            os.write(w, block)
        sp._read_block()
        self.assertGreater(sp._dropped, 0)
        self.assertLessEqual(sa.pipe_pending(r), sa.MAX_PIPE_BLOCKS * len(block))
        os.close(w)
        out.close()

    def test_write_block_and_respawn_rules(self):
        sp = self.service()
        spawned = []

        def fake_spawn():
            r, w = os.pipe()
            spawned.append(r)
            sp._play = FakeProc(stdin=os.fdopen(w, "wb", buffering=0))
            sp._last_play_spawn = 0.0
        sp._spawn_play = fake_spawn
        block = np.zeros((B, 2), np.float32)
        sp._write_block(block)
        self.assertEqual(len(spawned), 1)
        self.assertEqual(sa.pipe_pending(spawned[0]), B * 2 * 4)
        # output gone: no new pw-play
        sp._play.dead = True
        sp._output_present = False
        sp._write_block(block)
        self.assertEqual(len(spawned), 1)
        self.assertIsNone(sp._play)
        # back: a new one
        sp._output_present = True
        sp._write_block(block)
        self.assertEqual(len(spawned), 2)
        # too much queued: the block is dropped instead of adding latency
        for _ in range(sa.MAX_PIPE_BLOCKS + 1):
            sp._play.stdin.write(sr.encode_f32(block))
        before = sa.pipe_pending(spawned[1])
        sp._write_block(block)
        self.assertEqual(sa.pipe_pending(spawned[1]), before)
        self.assertEqual(sp._dropped, 1)
        sp._kill_play = lambda: None
        for fd in spawned:
            os.close(fd)

    def test_pipe_pending(self):
        r, w = os.pipe()
        os.write(w, b"x" * 100)
        self.assertEqual(sa.pipe_pending(r), 100)
        self.assertEqual(sa.pipe_pending(w), 100)          # both ends on Linux
        os.close(r)
        os.close(w)


class Sinks(unittest.TestCase):
    def dump(self, default="alsa_output.speaker", bluez=True):
        def node(i, name, cls="Audio/Sink"):
            return {"id": i, "type": "PipeWire:Interface:Node",
                    "info": {"props": {"node.name": name, "media.class": cls}}}
        d = [node(1, "alsa_output.speaker"), node(2, sa.OWN_SINK), node(5, "mousepods_spatial_test"),
             node(3, sa.OWN_OUTPUT, "Stream/Output/Audio")]
        if bluez:
            d.append(node(4, "bluez_output.40_DA_5C_BD_69_73.1"))
        d.append({"id": 9, "type": "PipeWire:Interface:Link",
                  "info": {"output-node-id": 3, "input-node-id": 1}})
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

    def test_metadata_nodes_links(self):
        d = self.dump()
        self.assertEqual(sa.default_metadata(d, "default.configured.audio.sink"), "alsa_output.speaker")
        self.assertEqual(sa.find_node(d, sa.OWN_SINK), 2)
        self.assertIsNone(sa.find_node(d, "nothing"))
        self.assertTrue(sa.links_into(d, sa.OWN_OUTPUT, "alsa_output.speaker"))
        self.assertFalse(sa.links_into(d, sa.OWN_OUTPUT, sa.OWN_SINK))

    def test_find_hrtf(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "hrtf"))
        open(os.path.join(d, "hrtf", "mit_kemar_normal_pinna.sofa"), "wb").close()
        self.assertEqual(sa.find_hrtf(d), os.path.join(d, "hrtf", "mit_kemar_normal_pinna.sofa"))


if __name__ == "__main__":
    unittest.main()
