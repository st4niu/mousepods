# SPDX-License-Identifier: Apache-2.0
"""Binaural renderer for spatial_audio.py: pure DSP (numpy), no PipeWire, no subprocess.

Block by block (1024 frames at 48 kHz, the usual Bluetooth quantum), every source channel of a 7.1 stream is placed at its
virtual speaker position relative to the head, and convolved with the HRIR pair of that exact
direction (from libmysofa, which interpolates between the measured directions). When the head
moves the HRIR changes: the block is convolved with the previous HRIR and with the new one, and
the two results are crossfaded linearly across the block, so there is no step and no click.

Details:
  * FFT overlap-add (FFT size 2048): HRIR + diffuse-field EQ up to 1025 samples (HRIR up to 770
    with the 256-tap EQ; MIT KEMAR is 558). The EQ is multiplied into every HRIR spectrum once,
    when it is fetched; spectra are cached by direction rounded to 1°. Spectra of all sources are summed before the inverse FFT (it is linear), so the cost is
    one forward FFT per active channel plus 2 (4 while the head moves) inverse FFTs per block.
  * Channels whose block is digital silence are skipped.
  * LFE: no HRTF, a gentle linear-phase low-pass into both ears, aligned with the dry path.
  * "intensity": out = intensity * binaural + (1 - intensity) * dry stereo (ITU downmix, delayed
    to line up with the HRIR onset so the mix does not comb).
  * Gain staging: the binaural path is scaled to the loudness of the dry stereo (pink-weighted,
    front pair at rest); a block limiter (gain ramps inside the block) and a soft knee keep the
    output under full scale.

Conventions (as in spatial_audio.py): azimuth + = LEFT, elevation + = up; yaw + = head turned
RIGHT, pitch + = looking UP. Roll is ignored.
"""
from __future__ import annotations

import math
import struct
from collections import OrderedDict

import numpy as np

RATE = 48000
BLOCK = 512
NFFT = 2048
MAX_FILTER = NFFT - BLOCK + 1                  # longest HRIR (+ EQ) one FFT block can take
CHANNELS = ("FL", "FR", "FC", "LFE", "RL", "RR", "SL", "SR")
LFE = CHANNELS.index("LFE")
SMOOTH_S = 0.05                                # pose low-pass time constant (s)
CACHE_SIZE = 4096                              # HRIR spectra kept (1° grid)
CEILING = 0.95                                 # limiter ceiling; soft knee from here to 1.0
RELEASE_DB_S = 6.0                             # limiter release speed (dB per second)
ATTACK = 64                                    # limiter attack ramp (samples)


# ------------------------------------------------------------------ geometry
def speakers(width=30.0):
    """Virtual speaker layout: {channel: (azimuth, elevation, gain)}. `width` is the azimuth of
    FL/FR (the other speakers do not move). LFE is not placed (no HRTF), gain only."""
    w = float(width)
    return {
        "FL": (w, 0.0, 1.0), "FR": (-w, 0.0, 1.0), "FC": (0.0, 0.0, 1.0),
        "LFE": (0.0, -30.0, 0.5),
        "RL": (150.0, 0.0, 1.0), "RR": (-150.0, 0.0, 1.0),
        "SL": (90.0, 0.0, 1.0), "SR": (-90.0, 0.0, 1.0),
    }


def relative_direction(src_az, src_el, yaw, pitch):
    """Where a room-fixed source at (src_az, src_el) is heard when the head has turned by
    (yaw, pitch). For yaw alone this is exactly (src_az + yaw, src_el); for a source in front,
    looking up by p puts it at elevation -p. A real rotation (yaw then pitch) is used so that a
    source behind or beside the head also moves the right way when looking up or down."""
    a, e = math.radians(src_az), math.radians(src_el)
    v = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))   # x front, y left, z up
    # head orientation = Rz(-yaw) . Ry(-pitch); the source in head coordinates is its transpose
    # applied to v: first undo the yaw (rotate by +yaw about z) ...
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    x, y, z = cy * v[0] - sy * v[1], sy * v[0] + cy * v[1], v[2]
    # ... then undo the pitch (rotate by +pitch about y, the left axis: front goes down)
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    x, z = cp * x + sp * z, -sp * x + cp * z
    az = math.degrees(math.atan2(y, x))
    el = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    return az, el


def round_direction(az, el):
    """Direction on the 1° grid used by the HRIR cache: az in (-180, 180], el in [-90, 90]."""
    el = int(round(max(-90.0, min(90.0, el))))
    az = int(round(az)) % 360
    if az > 180:
        az -= 360
    if abs(el) == 90:
        az = 0                                   # straight up / down: azimuth is meaningless
    return az, el


def smooth_pose(cur, target, dt, tau=SMOOTH_S):
    """One step of a one-pole low-pass on (yaw, pitch); yaw takes the short way round."""
    if cur is None:
        return tuple(target)
    a = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
    dy = (target[0] - cur[0] + 180.0) % 360.0 - 180.0
    dp = target[1] - cur[1]
    if abs(dy) < 0.02 and abs(dp) < 0.02:
        return tuple(target)
    return (cur[0] + a * dy, cur[1] + a * dp)


# ------------------------------------------------------------------ small DSP helpers
def itu_downmix(x):
    """(n, 8) 7.1 frames -> (n, 2) stereo, ITU-R BS.775 style: L = FL + .707 FC + .707 SL
    + .707 RL + .5 LFE (and the mirror for R). Plain stereo (FL/FR only) is unchanged."""
    x = np.asarray(x)
    c = math.sqrt(0.5)
    fl, fr, fc, lfe, rl, rr, sl, sr = (x[:, i] for i in range(8))
    left = fl + c * fc + c * sl + c * rl + 0.5 * lfe
    right = fr + c * fc + c * sr + c * rr + 0.5 * lfe
    return np.stack([left, right], axis=1)


def lowpass_fir(delay, cutoff=150.0, rate=RATE):
    """Linear-phase low-pass of length 2 * delay + 1 (its centre is `delay` samples), Blackman
    windowed sinc, unity gain at DC."""
    n = 2 * max(int(delay), 8) + 1
    t = np.arange(n) - (n - 1) / 2
    h = np.sinc(2 * cutoff / rate * t) * np.blackman(n)
    return h / np.sum(h)


def pink_energy(irs, rate=RATE, nfft=4096):
    """Loudness proxy: energy of IRs weighted by a pink spectrum (1/f), 20 Hz - 20 kHz."""
    f = np.fft.rfftfreq(nfft, 1.0 / rate)
    band = (f >= 20.0) & (f <= 20000.0)
    p = np.abs(np.fft.rfft(irs, nfft, axis=-1)) ** 2
    return float(np.sum(p[..., band] / f[band]))


def soft_knee(y, knee=CEILING, top=0.999):
    """Leave |y| <= knee untouched, bend what is above smoothly towards `top` (never reached)."""
    over = np.abs(y) > knee
    if np.any(over):
        a = np.abs(y[over])
        y = y.copy()
        y[over] = np.sign(y[over]) * (knee + (top - knee) * np.tanh((a - knee) / (top - knee)))
    return y


def prepare_hrir(pair, eq=None, max_len=MAX_FILTER):
    """(2, n) HRIR -> (2, m) filter actually used: convolved with the EQ FIR, at most max_len
    samples (half-Hann fade on the last 64 if it had to be cut)."""
    pair = np.asarray(pair, dtype=np.float64)
    if eq is not None:
        pair = np.stack([np.convolve(pair[0], eq), np.convolve(pair[1], eq)])
    if pair.shape[1] > max_len:
        pair = pair[:, :max_len].copy()
        fade = min(64, max_len // 4)
        pair[:, -fade:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade)))
    return pair


# ------------------------------------------------------------------ the renderer
class Renderer:
    """Turns 8-channel blocks into binaural stereo blocks.

    fetch(az, el) -> (2, n) HRIR (left, right) for that direction (integer az/el in degrees).
    """

    def __init__(self, fetch, eq=None, width=30.0, intensity=0.75, gain_db=0.0,
                 rate=RATE, block=BLOCK, nfft=NFFT, binaural_gain=None, dry_delay=None):
        if nfft < 2 * block:
            raise ValueError("nfft must be at least twice the block")
        self.fetch, self.eq = fetch, (None if eq is None else np.asarray(eq, dtype=np.float64))
        self.rate, self.block, self.nfft = rate, block, nfft
        self.max_filter = nfft - block + 1
        self.width = float(width)
        self.intensity = min(max(float(intensity), 0.0), 1.0)
        self.gain = 10 ** (float(gain_db) / 20.0)
        self.speakers = speakers(width)
        self._cache = OrderedDict()                  # (az, el) -> (2, nfft//2+1) spectra
        self._prev = [None] * len(CHANNELS)          # direction each channel used last block
        self._tail = np.zeros((2, nfft - block))     # overlap-add tail of the binaural path
        self._pose = None                            # smoothed (yaw, pitch)
        self._lim = 1.0                              # limiter gain
        self.stats = {"blocks": 0, "ffts": 0, "iffts": 0, "fetches": 0, "limited": 0}
        self.last_active = []
        w = np.ones(nfft)
        w[:block] = (np.arange(block) + 1) / block   # crossfade old -> new across the block
        self._ramp = w
        # front pair at rest: loudness match and dry alignment
        fl = self._filter(*round_direction(self.width, 0.0))
        fr = self._filter(*round_direction(-self.width, 0.0))
        if binaural_gain is None:
            imp = np.zeros(8)
            imp[0] = 1.0
            ref = pink_energy(imp, rate)
            e_l = pink_energy(fl[0], rate) + pink_energy(fr[0], rate)
            e_r = pink_energy(fl[1], rate) + pink_energy(fr[1], rate)
            binaural_gain = math.sqrt(2 * ref / (e_l + e_r)) if e_l + e_r > 0 else 1.0
        self.binaural_gain = float(binaural_gain)
        if dry_delay is None:
            dry_delay = int(round((np.argmax(np.abs(fl[0])) + np.argmax(np.abs(fr[1]))) / 2))
        self.dry_delay = max(0, min(int(dry_delay), block))
        self._dry_buf = np.zeros((self.dry_delay, 2))
        # EQ applied in the frequency domain: the HRIR is cut so that HRIR * EQ still fits
        self._eq_spec = None if self.eq is None else np.fft.rfft(self.eq, nfft)
        self._hrir_max = self.max_filter - (0 if self.eq is None else len(self.eq) - 1)
        if self._hrir_max < 16:
            raise ValueError("EQ too long for this FFT size")
        lp = lowpass_fir(self.dry_delay, rate=rate)[:self.max_filter]
        self._lfe_spec = np.fft.rfft(lp, nfft) * speakers(width)["LFE"][2] * self.binaural_gain ** -1

    # ---- HRIR spectra
    def _filter(self, az, el):
        h = prepare_hrir(self.fetch(az, el), self.eq, self.max_filter)
        self.stats["fetches"] += 1
        return h

    def spectrum(self, az, el):
        key = (az, el)
        s = self._cache.get(key)
        if s is None:
            # cut the raw HRIR, then EQ by a product of spectra (cheaper than convolving)
            h = prepare_hrir(self.fetch(az, el), None, self._hrir_max)
            self.stats["fetches"] += 1
            s = np.fft.rfft(h, self.nfft, axis=1)
            if self._eq_spec is not None:
                s = s * self._eq_spec
            self._cache[key] = s
            if len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return s

    # ---- one block
    def process(self, x, yaw=0.0, pitch=0.0):
        """x: (block, 8) float frames -> (block, 2) float32. The pose is low-passed here."""
        x = np.asarray(x, dtype=np.float64)
        b, nf = self.block, self.nfft
        self.stats["blocks"] += 1
        self._pose = smooth_pose(self._pose, (float(yaw), float(pitch)), b / self.rate)
        active = [k for k in range(len(CHANNELS)) if x[:, k].any()]
        self.last_active = active
        if not active and not self._tail.any() and not self._dry_buf.any():
            # digital silence with nothing left ringing: no FFT at all
            for k in range(len(CHANNELS)):
                self._prev[k] = None
            self._lim = min(1.0, self._lim * 10 ** (RELEASE_DB_S * b / self.rate / 20))
            return np.zeros((b, 2), dtype=np.float32)

        y_new = y_old = None
        moved = False
        for k in active:
            xs = np.fft.rfft(x[:, k], nf)
            self.stats["ffts"] += 1
            if k == LFE:
                lfe = xs * self._lfe_spec
                contrib_new = contrib_old = np.stack([lfe, lfe])
            else:
                az, el, g = self.speakers[CHANNELS[k]]
                d = round_direction(*relative_direction(az, el, *self._pose))
                prev = self._prev[k] if self._prev[k] is not None else d
                hn = self.spectrum(*d)
                contrib_new = xs * hn * g
                if prev != d:
                    moved = True
                    contrib_old = xs * self.spectrum(*prev) * g
                else:
                    contrib_old = contrib_new
                self._prev[k] = d
            y_new = contrib_new if y_new is None else y_new + contrib_new
            y_old = contrib_old if y_old is None else y_old + contrib_old
        # silent channels just follow the head (no output, so nothing to crossfade later)
        for k in range(len(CHANNELS)):
            if k != LFE and k not in active and self._prev[k] is not None:
                az, el, _ = self.speakers[CHANNELS[k]]
                self._prev[k] = round_direction(*relative_direction(az, el, *self._pose))

        if y_new is None:
            y = np.zeros((2, nf))
        elif moved:
            yn = np.fft.irfft(y_new, nf, axis=1)
            yo = np.fft.irfft(y_old, nf, axis=1)
            self.stats["iffts"] += 2
            y = yo + self._ramp * (yn - yo)
        else:
            y = np.fft.irfft(y_new, nf, axis=1)
            self.stats["iffts"] += 1
        # overlap-add
        tail = self._tail
        y[:, :tail.shape[1]] += tail
        binaural = y[:, :b].T * self.binaural_gain
        self._tail = y[:, b:].copy()

        # dry path, delayed to line up with the HRIR onset
        dry = x[:, :2] if all(k < 2 for k in active) else itu_downmix(x)
        if self.dry_delay:
            full = np.concatenate([self._dry_buf, dry])
            dry, self._dry_buf = full[:b], full[b:]
        out = (self.intensity * binaural + (1 - self.intensity) * dry) * self.gain
        return self._limit(out).astype(np.float32)

    def _limit(self, out):
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        target = min(1.0, CEILING / peak) if peak > 0 else 1.0
        g0 = self._lim
        n = len(out)
        if target >= 1.0 and g0 >= 1.0:
            return out                                           # nothing to do (the usual case)
        if target < g0:                                          # attack: fast ramp, then hold
            g1 = target
            a = min(ATTACK, n)
            ramp = np.full(n, g1)
            ramp[:a] = np.linspace(g0, g1, a, endpoint=False)
        else:                                                    # release: slow, across the block
            g1 = min(target, g0 * 10 ** (RELEASE_DB_S * n / self.rate / 20))
            ramp = np.linspace(g0, g1, n, endpoint=False)
        self._lim = g1
        self.stats["limited"] += 1
        return soft_knee(out * ramp[:, None])


# ------------------------------------------------------------------ stream containers
def encode_f32(frames):
    """(n, ch) float -> raw interleaved float32 little-endian bytes (what `pw-play --format f32 -`
    reads: with "-" pw-cat streams raw samples, no file header)."""
    return np.ascontiguousarray(frames, dtype="<f4").tobytes()


def read_stream_header(read):
    """Read the header, if any, in front of pw-record's samples (PipeWire 1.2.4 writes none to
    stdout: raw f32). `read(n)` returns up to n bytes. Returns (info, leftover bytes): info = {"container", "dtype", "channels", "rate"}.
    Knows RIFF/RF64 WAV (float or PCM) and Sun .au; anything else is taken as raw f32."""
    def exact(n):
        buf = b""
        while len(buf) < n:
            chunk = read(n - len(buf))
            if not chunk:
                raise EOFError("stream ended in its header")
            buf += chunk
        return buf

    head = exact(4)
    if head in (b"RIFF", b"RF64"):
        exact(8)                                              # size + "WAVE"
        info = {"container": "wav", "dtype": "<f4", "channels": None, "rate": None}
        while True:
            cid, size = struct.unpack("<4sI", exact(8))
            if cid == b"data":
                return info, b""
            body = exact(size + (size & 1))
            if cid == b"fmt ":
                tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
                if tag == 0xFFFE and len(body) >= 26:
                    tag = struct.unpack("<H", body[24:26])[0]   # WAVE_FORMAT_EXTENSIBLE subformat
                info["channels"], info["rate"] = ch, rate
                if tag == 3 and bits == 32:
                    info["dtype"] = "<f4"
                elif tag == 3 and bits == 64:
                    info["dtype"] = "<f8"
                elif tag == 1 and bits == 16:
                    info["dtype"] = "<i2"
                elif tag == 1 and bits == 32:
                    info["dtype"] = "<i4"
                else:
                    raise ValueError(f"unsupported WAV format {tag}/{bits}")
    if head == b".snd":
        off, _, enc, rate, ch = struct.unpack(">IIIII", exact(20))
        exact(max(0, off - 24))
        dtypes = {6: ">f4", 7: ">f8", 3: ">i2", 5: ">i4"}
        if enc not in dtypes:
            raise ValueError(f"unsupported .au encoding {enc}")
        return {"container": "au", "dtype": dtypes[enc], "channels": ch, "rate": rate}, b""
    return {"container": "raw", "dtype": "<f4", "channels": None, "rate": None}, head


def decode_frames(raw, dtype, channels):
    """bytes -> (n, channels) float64 in [-1, 1]."""
    a = np.frombuffer(raw, dtype=dtype)
    if a.dtype.kind == "i":
        a = a / float(np.iinfo(a.dtype).max)
    return a.astype(np.float64).reshape(-1, channels)
