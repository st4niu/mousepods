# SPDX-License-Identifier: Apache-2.0
"""Head-tracked spatial audio on Linux with PipeWire, like Apple's spatial audio.

A virtual 7.1 output, "MousePods Spatial", is added to PipeWire. Audio sent to it is rendered
binaurally with an HRTF onto the AirPods (or any stereo sink), and the virtual speakers stay fixed
in the room while the head turns.

How it works:
  * HRIRs: libmysofa (through ctypes) reads a SOFA file and gives the left/right impulse responses
    for 60 fixed directions (3 rings: elevation -30° x 12, 0° x 36, +30° x 12). A diffuse-field
    equaliser (the inverse of the average spectrum of all directions, 1/3-octave smoothed, ±12 dB,
    minimum phase) removes the coloration common to every direction. Each direction is written as
    a stereo WAV (32-bit PCM) into a cache, rebuilt only when the SOFA file, the sample rate or the
    direction set change.
  * Graph: a PipeWire filter-chain (run as its own `pipewire -c filter-chain.conf` process, with a
    private XDG_CONFIG_HOME so only our fragment is loaded). The 8 input channels go through `copy`
    nodes into 60 `mixer` nodes, one per direction; each direction mixer feeds two `convolver` nodes
    (left ear, right ear); two mixer trees sum the 60 left and the 60 right ears.
  * Head tracking: the gains of the 60 direction mixers ("m<d>:Gain <k>" = how much of input
    channel k goes to direction d) are the panning. Each virtual speaker is panned between its
    nearest directions (pairwise, energy preserving), at its position relative to the head. A
    long-lived `pw-cli` receives `set-param` lines with the gains that changed, at most 100 Hz.
    The builtin mixer has no gain ramp, so the pose is low-passed (50 ms) and every gain moves by
    at most MAX_STEP per update: each step stays tiny and inaudible. The HRIR convolutions never
    change, so there is no reload.

Conventions:
  * azimuth in degrees, positive = to the LEFT (counter-clockwise seen from above), 0 = front;
    elevation positive = up. SOFA coordinates: x front, y left, z up (verified with MIT KEMAR:
    azimuth +90 is louder in the left ear).
  * head pose: yaw positive = head turned RIGHT, pitch positive = looking UP (same as
    opentrack_out.py). Roll is ignored for now: tilting the head sideways does not move the
    speakers (a small error, much less audible than yaw).

Only the stdlib and numpy. PipeWire needs the builtin filter-chain plugins `convolver`, `mixer`
and `copy` (PipeWire >= 0.3.x; the `sofa` plugin is not needed).
"""
from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import glob
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import threading
import time
import wave
import weakref

import numpy as np

log = logging.getLogger(__name__)

RATE = 48000
OWN_SINK = "mousepods_spatial"                  # node.name of our virtual sink (capture side)
OWN_OUTPUT = "mousepods_spatial.output"        # node.name of our playback stream
DESCRIPTION = "MousePods Spatial"
CHANNELS = ("FL", "FR", "FC", "LFE", "RL", "RR", "SL", "SR")
# rings of fixed HRIR directions: (elevation in degrees, number of azimuths, evenly spaced from 0)
# The horizontal ring is dense (every 10°) so that neighbouring HRIRs have close interaural delays
# (panning between two HRIRs whose delays differ a lot gives comb filtering), and so that FC and
# FL/FR at ±30° sit exactly on a direction when the head is at rest.
RINGS = ((-30.0, 12), (0.0, 36), (30.0, 12))
EL_MIN, EL_MAX = RINGS[0][0], RINGS[-1][0]
SOFA_RADIUS = 1.2                               # m, distance asked to libmysofa
UPDATE_HZ = 100.0                               # most set-param lines per second
SMOOTH_S = 0.05                                 # time constant of the pose low-pass (s)
MAX_STEP = 0.04                                 # largest change of one gain per update
THRESHOLD = 0.002                               # smallest gain change worth sending
HEALTH_S = 5.0                                  # how often the worker checks the graph is still there
CACHE_VERSION = 2                               # bump when the HRIR processing changes
MAX_IR = 512                                    # HRIR length kept (samples, faded out at the end)
MAX_LINE = 3800                                 # pw-cli reads its input in 4 KiB chunks

DEFAULTS = {
    "width": 30.0,          # azimuth of the front left/right speakers (degrees)
    "make_default": True,   # make "MousePods Spatial" the default output while it runs
    "gain_db": -3.0,        # overall level; -3 dB of headroom (the near ear gets up to +4 dB)
    "blocksize": 256,       # convolver partition size (samples): larger = less CPU, same latency
    "output": None,         # node.name of the sink to render onto; None = AirPods, else default
    "hrtf": None,           # path of a .sofa file; None = the first one found (see find_hrtf)
}


# ------------------------------------------------------------------ directions and panning
def directions():
    """The 60 fixed directions as (azimuth, elevation) in degrees, ring by ring (low, middle,
    high), azimuths in (-180, 180]."""
    out = []
    for el, n in RINGS:
        step = 360.0 / n
        for i in range(n):
            az = i * step
            out.append((az - 360.0 if az > 180.0 else az, el))
    return out


DIRECTIONS = directions()
_RING_START = []                                # index of the first direction of each ring
_i = 0
for _el, _n in RINGS:
    _RING_START.append(_i)
    _i += _n
N_DIRS = _i


def _ring_gains(az, n):
    """Pairwise amplitude panning on a ring of n evenly spaced directions: list of (index in the
    ring, gain), sin/cos law on the angular fraction between the two neighbours."""
    step = 360.0 / n
    pos = (az % 360.0) / step
    i = int(math.floor(pos)) % n
    f = pos - math.floor(pos)
    j = (i + 1) % n
    return [(i, math.cos(f * math.pi / 2)), (j, math.sin(f * math.pi / 2))]


def gains_for(az, el):
    """Gains of the 60 directions for one source seen at (az, el), energy preserving (the sum of
    the squares is 1). Elevation is clamped to the rings [-30, 30]; between two rings the source
    is shared with cos/sin weights."""
    el = min(max(float(el), EL_MIN), EL_MAX)
    g = [0.0] * N_DIRS
    # the two rings around el
    for r in range(len(RINGS) - 1):
        lo, hi = RINGS[r][0], RINGS[r + 1][0]
        if el <= hi or r == len(RINGS) - 2:
            t = (el - lo) / (hi - lo)
            t = min(max(t, 0.0), 1.0)
            weights = ((r, math.cos(t * math.pi / 2)), (r + 1, math.sin(t * math.pi / 2)))
            break
    for ring, w in weights:
        if w < 1e-9:
            continue
        start, n = _RING_START[ring], RINGS[ring][1]
        for k, gk in _ring_gains(az, n):
            g[start + k] += w * gk
    return g


def speakers(width=30.0):
    """Virtual speaker layout: {channel: (azimuth, elevation, gain)}. `width` is the azimuth of
    FL/FR (the other speakers do not move)."""
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


def gain_matrix(yaw=0.0, pitch=0.0, width=30.0):
    """8 x N_DIRS numpy array: gain of input channel k (CHANNELS order) into direction d."""
    spk = speakers(width)
    m = np.zeros((len(CHANNELS), N_DIRS))
    for k, ch in enumerate(CHANNELS):
        az, el, g = spk[ch]
        raz, rel = relative_direction(az, el, yaw, pitch)
        m[k] = np.asarray(gains_for(raz, rel)) * g
    return m


def control_name(d, k):
    """Filter-chain control for "input channel k (0-based) into direction d"."""
    return f"m{d}:Gain {k + 1}"


def limit_step(last, new, max_step=MAX_STEP):
    """Move every gain from `last` towards `new` by at most `max_step`: the builtin mixer has no
    ramp, so a large gain jump in one quantum would click."""
    last = np.asarray(last)
    return last + np.clip(np.asarray(new) - last, -max_step, max_step)


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


def diff_updates(last, new, threshold=THRESHOLD):
    """Which gains to send. `last` and `new` are 8 x N_DIRS arrays (last = what PipeWire has now).
    A gain is sent when it moved by more than `threshold`; a gain that fell below the threshold
    while PipeWire still has something else is sent as an exact 0 (no stale leftovers).
    Returns a list of (d, k, value)."""
    last = np.asarray(last)
    new = np.where(np.asarray(new) < threshold, 0.0, np.asarray(new))
    changed = (np.abs(new - last) > threshold) | ((new == 0.0) & (last != 0.0))
    out = []
    for k, d in zip(*np.nonzero(changed)):
        out.append((int(d), int(k), float(new[k, d])))
    return out


def set_param_lines(node_id, updates, max_len=MAX_LINE):
    """pw-cli lines for a list of (d, k, value); normally one line, split only if it would be
    longer than pw-cli's input buffer."""
    head = f"set-param {node_id} Props {{ params = [ "
    tail = " ] }"
    lines, cur = [], []
    size = len(head) + len(tail)
    for d, k, v in updates:
        item = f'"{control_name(d, k)}" {v:.4f}'
        if cur and size + len(item) + 1 > max_len:
            lines.append(head + " ".join(cur) + tail)
            cur, size = [], len(head) + len(tail)
        cur.append(item)
        size += len(item) + 1
    if cur:
        lines.append(head + " ".join(cur) + tail)
    return lines


# ------------------------------------------------------------------ HRIRs (libmysofa) and WAV files
class Mysofa:
    """Thin ctypes binding of libmysofa's "easy" API (only what we need)."""

    def __init__(self, path=None):
        name = path or ctypes.util.find_library("mysofa") or "libmysofa.so.1"
        self.lib = lib = ctypes.CDLL(name)
        lib.mysofa_open.restype = ctypes.c_void_p
        lib.mysofa_open.argtypes = [ctypes.c_char_p, ctypes.c_float,
                                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        lib.mysofa_getfilter_float.restype = None
        lib.mysofa_getfilter_float.argtypes = ([ctypes.c_void_p] + [ctypes.c_float] * 3
                                               + [ctypes.POINTER(ctypes.c_float)] * 4)
        lib.mysofa_close.restype = None
        lib.mysofa_close.argtypes = [ctypes.c_void_p]

    def open(self, filename, rate):
        """-> (handle, filter length in samples at `rate`). libmysofa resamples to `rate`."""
        n, err = ctypes.c_int(0), ctypes.c_int(0)
        h = self.lib.mysofa_open(os.fsencode(filename), float(rate), ctypes.byref(n), ctypes.byref(err))
        if not h or err.value != 0:
            raise RuntimeError(f"libmysofa could not open {filename} (error {err.value})")
        return h, n.value

    def getfilter(self, h, n, x, y, z):
        """-> (left IR, right IR, left delay s, right delay s) for the direction (x, y, z)."""
        left, right = (ctypes.c_float * n)(), (ctypes.c_float * n)()
        dl, dr = ctypes.c_float(0.0), ctypes.c_float(0.0)
        self.lib.mysofa_getfilter_float(h, x, y, z, left, right, ctypes.byref(dl), ctypes.byref(dr))
        return (np.frombuffer(left, dtype=np.float32).astype(np.float64),
                np.frombuffer(right, dtype=np.float32).astype(np.float64), dl.value, dr.value)

    def close(self, h):
        if h:
            self.lib.mysofa_close(h)


def extract_hrirs(sofa_path, dirs=DIRECTIONS, rate=RATE, mysofa=None):
    """HRIRs for each direction: numpy array (n_dirs, 2, length), ear delays prepended as zeros,
    all padded to the same length and the common silent tail trimmed."""
    mysofa = mysofa or Mysofa()
    h, n = mysofa.open(sofa_path, rate)
    try:
        irs = []
        for az, el in dirs:
            a, e = math.radians(az), math.radians(el)
            x = SOFA_RADIUS * math.cos(e) * math.cos(a)
            y = SOFA_RADIUS * math.cos(e) * math.sin(a)
            z = SOFA_RADIUS * math.sin(e)
            left, right, dl, dr = mysofa.getfilter(h, n, x, y, z)
            pair = []
            for ir, delay in ((left, dl), (right, dr)):
                pad = max(0, int(round(float(delay) * rate)))
                pair.append(np.concatenate([np.zeros(pad), np.asarray(ir, dtype=np.float64)]))
            irs.append(pair)
    finally:
        mysofa.close(h)
    length = max(len(ir) for pair in irs for ir in pair)
    out = np.zeros((len(irs), 2, length))
    for i, pair in enumerate(irs):
        for ear in (0, 1):
            out[i, ear, :len(pair[ear])] = pair[ear]
    # trim the tail that is silent in every IR (keeps the convolvers short)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        loud = np.nonzero(np.max(np.abs(out), axis=(0, 1)) > peak * 1e-5)[0]
        out = out[:, :, :int(loud[-1]) + 1]
    return out


def normalise(irs, peak=0.98):
    """One global factor for all IRs so the loudest sample is `peak` (32-bit PCM cannot go past
    full scale). Returns (scaled IRs, factor); the convolver gain must be 1 / factor to keep the
    original level."""
    m = float(np.max(np.abs(irs))) if np.size(irs) else 0.0
    if m <= 0:
        raise ValueError("the HRIRs are silent")
    k = peak / m
    return irs * k, k


def _third_octave_smooth(f, p):
    """Average a power spectrum over a 1/3 octave around every frequency."""
    cs = np.concatenate([[0.0], np.cumsum(p)])
    lo = np.searchsorted(f, f * 2 ** (-1 / 6), side="left")
    hi = np.searchsorted(f, f * 2 ** (1 / 6), side="right")
    hi = np.maximum(hi, lo + 1)
    return (cs[hi] - cs[lo]) / (hi - lo)


def _minimum_phase(mag, nfft):
    """Minimum-phase impulse response (length nfft) of a magnitude response on the rfft grid,
    by the real cepstrum (homomorphic) method."""
    c = np.fft.irfft(np.log(np.maximum(mag, 1e-9)), nfft)
    w = np.zeros(nfft)
    w[0] = 1.0
    w[1:nfft // 2] = 2.0
    w[nfft // 2] = 1.0
    return np.fft.irfft(np.exp(np.fft.rfft(c * w)), nfft)


def pink_energy(irs, rate=RATE, nfft=4096):
    """Loudness proxy: energy of all IRs weighted by a pink spectrum (1/f), 20 Hz - 20 kHz."""
    f = np.fft.rfftfreq(nfft, 1.0 / rate)
    band = (f >= 20.0) & (f <= 20000.0)
    p = np.abs(np.fft.rfft(irs, nfft, axis=-1)) ** 2
    return float(np.sum(p[..., band] / f[band]))


def diffuse_field_filter(irs, rate=RATE, nfft=4096, taps=256, limit_db=12.0, band=(150.0, 16000.0)):
    """Diffuse-field equaliser for a set of HRIRs (n, 2, length): minimum-phase FIR (`taps`
    samples) whose response is the inverse of the average power spectrum over every direction
    and both ears, 1/3-octave smoothed, held constant outside `band` (the measurement is not
    reliable in the deep bass), centred on its mean level in the band and limited to ±limit_db."""
    f = np.fft.rfftfreq(nfft, 1.0 / rate)
    p = np.mean(np.abs(np.fft.rfft(irs, nfft, axis=-1)) ** 2, axis=(0, 1))
    p = _third_octave_smooth(f, p)
    inv_db = -10.0 * np.log10(np.maximum(p, 1e-20))
    i0, i1 = np.searchsorted(f, band[0]), np.searchsorted(f, band[1]) - 1
    inv_db[:i0] = inv_db[i0]
    inv_db[i1 + 1:] = inv_db[i1]
    # centre on the mean over the band, every octave weighted the same (log-frequency mean)
    sel = slice(i0, i1 + 1)
    wts = 1.0 / f[sel]
    inv_db -= np.sum(inv_db[sel] * wts) / np.sum(wts)
    inv_db = np.clip(inv_db, -limit_db, limit_db)
    h = _minimum_phase(10 ** (inv_db / 20.0), nfft)[:taps]
    fade = taps // 4                                     # half-Hann fade of the last quarter
    h[-fade:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade)))
    return h


def equalise(irs, rate=RATE, max_len=MAX_IR):
    """Apply the diffuse-field equaliser to every HRIR, keep `max_len` samples (half-Hann fade
    on the last 64) and keep the pink-weighted loudness of the set unchanged."""
    h = diffuse_field_filter(irs, rate)
    out = np.apply_along_axis(lambda x: np.convolve(x, h), -1, irs)
    if max_len and out.shape[-1] > max_len:
        out = out[..., :max_len].copy()
        fade = min(64, max_len // 4)
        out[..., -fade:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade)))
    before, after = pink_energy(irs, rate), pink_energy(out, rate)
    if after > 0:
        out *= math.sqrt(before / after)
    return out


def write_wav(path, stereo, rate=RATE):
    """Write a (2, n) float array in [-1, 1] as a stereo 32-bit PCM WAV (stdlib wave)."""
    data = np.clip(np.asarray(stereo, dtype=np.float64).T, -1.0, 1.0)
    ints = np.round(data * 2147483647.0).astype("<i4")
    tmp = path + ".tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(4)
        w.setframerate(int(rate))
        w.writeframes(ints.tobytes())
    os.replace(tmp, path)


def read_wav(path):
    """Read back a WAV written by write_wav: (array (channels, n) in [-1, 1], rate)."""
    with wave.open(path, "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width != 4:
        raise ValueError("only 32-bit PCM is supported")
    ints = np.frombuffer(raw, dtype="<i4").reshape(-1, ch)
    return (ints.astype(np.float64) / 2147483647.0).T, rate


def cache_key(sofa_path, rate=RATE, dirs=DIRECTIONS):
    """Key of the HRIR cache: SOFA file identity (path, size, mtime) + rate + direction set."""
    st = os.stat(sofa_path)
    blob = json.dumps({"v": CACHE_VERSION, "sofa": os.path.abspath(sofa_path), "size": st.st_size,
                       "mtime": st.st_mtime_ns, "rate": int(rate), "max_ir": MAX_IR,
                       "dirs": [[round(a, 4), round(e, 4)] for a, e in dirs]}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def build_hrir_cache(sofa_path, cache_root, rate=RATE, dirs=DIRECTIONS, mysofa=None, eq=True):
    """Make (or reuse) the per-direction WAVs. Returns {"files": [...], "factor": k, "length": n}.
    Old caches (other keys) in cache_root are removed."""
    key = cache_key(sofa_path, rate, dirs)
    d = os.path.join(cache_root, key)
    meta_path = os.path.join(d, "meta.json")
    files = [os.path.join(d, f"dir{i:02d}.wav") for i in range(len(dirs))]
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("key") == key and all(os.path.exists(p) for p in files):
            meta["files"] = files
            return meta
    except (OSError, ValueError):
        pass
    irs = extract_hrirs(sofa_path, dirs, rate, mysofa)
    if eq:
        irs = equalise(irs, rate)
    scaled, k = normalise(irs)
    os.makedirs(d, exist_ok=True)
    for i, p in enumerate(files):
        write_wav(p, scaled[i], rate)
    meta = {"key": key, "factor": k, "length": int(irs.shape[2]), "rate": int(rate),
            "sofa": os.path.abspath(sofa_path)}
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(meta_path + ".tmp", meta_path)
    for other in glob.glob(os.path.join(cache_root, "*")):          # drop stale caches
        if os.path.isdir(other) and os.path.basename(other) != key:
            shutil.rmtree(other, ignore_errors=True)
    meta["files"] = files
    return meta


def find_hrtf(data_dir=None):
    """First .sofa file in <data_dir>/hrtf/ or ~/.local/share/mousepods/hrtf/ (MIT KEMAR
    preferred), or None."""
    dirs = []
    if data_dir:
        dirs.append(os.path.join(data_dir, "hrtf"))
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    dirs.append(os.path.join(xdg, "mousepods", "hrtf"))
    found = []
    for d in dirs:
        found += sorted(glob.glob(os.path.join(d, "*.sofa")))
    found.sort(key=lambda p: (0 if "kemar" in os.path.basename(p).lower() else 1))
    return found[0] if found else None


# ------------------------------------------------------------------ filter-chain configuration
def _fmt(v):
    return f"{float(v):.6f}"


def build_conf(ir_files, ir_gain, output_sink, initial, blocksize=256, fanout=4):
    """Text of the filter-chain fragment. `ir_files`: one stereo WAV per direction;
    `ir_gain`: convolver gain (undoes the normalisation, plus the level); `output_sink`: node.name
    of the sink to play onto; `initial`: 8 x N_DIRS starting gains."""
    if not output_sink or output_sink in (OWN_SINK, OWN_OUTPUT):
        raise ValueError("the spatial output needs a real sink, never itself")
    if len(ir_files) != N_DIRS:
        raise ValueError(f"{N_DIRS} HRIR files expected, got {len(ir_files)}")
    initial = np.asarray(initial)
    nodes, links = [], []
    for ch in CHANNELS:
        nodes.append(f'{{ type = builtin label = copy name = c{ch} }}')
    for d in range(N_DIRS):
        ctl = " ".join(f'"Gain {k + 1}" = {_fmt(initial[k, d])}' for k in range(len(CHANNELS)))
        nodes.append(f'{{ type = builtin label = mixer name = m{d} control = {{ {ctl} }} }}')
        for ear, side in ((0, "l"), (1, "r")):
            fn = json.dumps(os.path.abspath(ir_files[d]))
            nodes.append(f'{{ type = builtin label = convolver name = {side}{d} config = {{ '
                         f'filename = {fn} channel = {ear} gain = {_fmt(ir_gain)} '
                         f'blocksize = {int(blocksize)} }} }}')
    # source channel k -> "In k+1" of every direction mixer, through a tree of copy nodes:
    # PipeWire 1.2.4's filter-chain corrupts its memory (malloc abort) when outputs fan out to
    # many inputs (one output to 32 inputs, or 8 outputs to 24 each), so no output feeds more
    # than `fanout` inputs.
    for k, ch in enumerate(CHANNELS):
        targets = [f"m{d}:In {k + 1}" for d in range(N_DIRS)]
        level = 0
        while len(targets) > fanout:
            nxt = []
            for g in range(0, len(targets), fanout):
                name = f"t{ch}_{level}_{g // fanout}"
                nodes.append(f'{{ type = builtin label = copy name = {name} }}')
                for tgt in targets[g:g + fanout]:
                    links.append(f'{{ output = "{name}:Out" input = "{tgt}" }}')
                nxt.append(f"{name}:In")
            targets, level = nxt, level + 1
        for tgt in targets:
            links.append(f'{{ output = "c{ch}:Out" input = "{tgt}" }}')
    for d in range(N_DIRS):
        for side in ("l", "r"):
            links.append(f'{{ output = "m{d}:Out" input = "{side}{d}:In" }}')
    # sum tree per ear: mixers of (up to) 8 inputs, level after level, down to one "s<l|r>"
    for side in ("l", "r"):
        outs = [f"{side}{d}:Out" for d in range(N_DIRS)]
        level = 0
        while True:
            groups = [outs[g:g + 8] for g in range(0, len(outs), 8)]
            last = len(groups) == 1
            nxt = []
            for g, grp in enumerate(groups):
                name = f"s{side}" if last else f"s{side}{level}_{g}"
                nodes.append(f'{{ type = builtin label = mixer name = {name} }}')
                for i, o in enumerate(grp):
                    links.append(f'{{ output = "{o}" input = "{name}:In {i + 1}" }}')
                nxt.append(f"{name}:Out")
            if last:
                break
            outs, level = nxt, level + 1
    inputs = " ".join(f'"c{ch}:In"' for ch in CHANNELS)
    pos = " ".join(CHANNELS)
    ind = "                    "
    return f"""# Generated by MousePods (spatial_audio.py). Do not edit: rewritten at every start.
context.modules = [
    {{ name = libpipewire-module-filter-chain
        args = {{
            node.description = "{DESCRIPTION}"
            media.name = "{DESCRIPTION}"
            filter.graph = {{
                nodes = [
{ind}{(chr(10) + ind).join(nodes)}
                ]
                links = [
{ind}{(chr(10) + ind).join(links)}
                ]
                inputs = [ {inputs} ]
                outputs = [ "sl:Out" "sr:Out" ]
            }}
            capture.props = {{
                node.name = "{OWN_SINK}"
                node.description = "{DESCRIPTION}"
                media.class = Audio/Sink
                audio.channels = 8
                audio.position = [ {pos} ]
            }}
            playback.props = {{
                node.name = "{OWN_OUTPUT}"
                node.description = "{DESCRIPTION} (output)"
                node.passive = true
                audio.channels = 2
                audio.position = [ FL FR ]
                target.object = "{output_sink}"
                node.dont-reconnect = true
                stream.dont-remix = true
            }}
        }}
    }}
]
"""


# ------------------------------------------------------------------ PipeWire helpers
def _run(args, timeout=3.0):
    """Run a command, return stdout text ('' on any failure). Never raises."""
    try:
        r = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=timeout, text=True)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def pw_dump():
    """pw-dump as a list of objects ([] when PipeWire cannot be reached)."""
    out = _run(["pw-dump"], timeout=4.0)
    try:
        return json.loads(out) if out else []
    except ValueError:
        return []


def _node_props(obj):
    if not str(obj.get("type", "")).endswith(":Node"):
        return None
    return (obj.get("info") or {}).get("props") or {}


def find_node(dump, name):
    """id of the node whose node.name is `name`, or None."""
    for o in dump:
        p = _node_props(o)
        if p is not None and p.get("node.name") == name:
            return o.get("id")
    return None


def sinks(dump):
    """[(id, node.name, description)] of the audio sinks."""
    out = []
    for o in dump:
        p = _node_props(o)
        if p and p.get("media.class") == "Audio/Sink":
            out.append((o.get("id"), p.get("node.name"), p.get("node.description")))
    return out


def default_metadata(dump, key):
    """Value name of a key of the "default" metadata (e.g. "default.audio.sink"), or None."""
    for o in dump:
        if str(o.get("type", "")).endswith(":Metadata") and \
                (o.get("props") or {}).get("metadata.name") == "default":
            for m in o.get("metadata") or []:
                if m.get("key") == key:
                    v = m.get("value")
                    if isinstance(v, str):
                        try:
                            v = json.loads(v)
                        except ValueError:
                            return v
                    return v.get("name") if isinstance(v, dict) else None
    return None


def find_output_sink(prefer_bluez=True, dump=None):
    """node.name of the sink to render onto: a connected Bluetooth (AirPods) sink if any, else
    the current default sink. Never our own sink."""
    dump = pw_dump() if dump is None else dump
    own = (OWN_SINK, OWN_OUTPUT)
    names = [n for _, n, _ in sinks(dump) if n and n not in own]
    default = default_metadata(dump, "default.audio.sink")
    if not default and shutil.which("pactl"):
        default = _run(["pactl", "get-default-sink"]).strip() or None
    if prefer_bluez:
        bt = [n for n in names if n.startswith("bluez_output.")]
        if bt:
            return default if default in bt else bt[0]     # several headsets: the default one
    if default and default not in own and (not names or default in names):
        return default
    return names[0] if names else None


def _filter_chain_module():
    for d in ("/usr/lib/x86_64-linux-gnu/pipewire-0.3", "/usr/lib64/pipewire-0.3", "/usr/lib/pipewire-0.3",
              "/usr/lib/aarch64-linux-gnu/pipewire-0.3"):
        p = os.path.join(d, "libpipewire-module-filter-chain.so")
        if os.path.exists(p):
            return p
    found = glob.glob("/usr/lib*/**/pipewire-0.3/libpipewire-module-filter-chain.so", recursive=True)
    return found[0] if found else None


def available(data_dir=None, hrtf_path=None):
    """(True, "") when everything spatial audio needs is there, else (False, why)."""
    for tool in ("pipewire", "pw-cli", "pw-dump"):
        if not shutil.which(tool):
            return False, f"{tool} not found (PipeWire is needed)"
    mod = _filter_chain_module()
    if not mod:
        return False, "PipeWire filter-chain module not found"
    try:
        # the builtin plugins live inside the module; their labels are plain strings in it
        with open(mod, "rb") as f:
            blob = f.read()
        if b"convolver\0" not in blob:
            return False, "PipeWire filter-chain has no convolver"
    except OSError as e:
        return False, f"cannot read {mod}: {e}"
    try:
        Mysofa()
    except OSError:
        return False, "libmysofa not found (install libmysofa1)"
    path = hrtf_path or find_hrtf(data_dir)
    if not path or not os.path.exists(path):
        return False, "no HRTF (.sofa) file in ~/.local/share/mousepods/hrtf/"
    return True, ""


# ------------------------------------------------------------------ the running service
class SpatialAudio:
    """Owns the filter-chain process, the pw-cli control pipe and the panning worker.

    cfg keys: see DEFAULTS. Typical use:
        sp = SpatialAudio(cfg, data_dir); sp.start(); sp.set_pose(yaw, pitch) ...; sp.stop()
    """

    def __init__(self, cfg: dict, data_dir: str, hrtf_path: str | None = None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update({k: v for k, v in (cfg or {}).items() if v is not None})
        self.width = float(self.cfg.get("width") or 30.0)
        self.data_dir = data_dir
        self.dir = os.path.join(data_dir, "spatial")
        self.hrtf_path = hrtf_path or self.cfg.get("hrtf") or find_hrtf(data_dir)
        self.output = None
        self.error = None
        self.node_id = None
        self._proc = None           # pipewire -c filter-chain.conf
        self._cli = None            # pw-cli (stdin = our set-param lines)
        self._thread = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pose = (0.0, 0.0)
        self._sent = np.zeros((len(CHANNELS), N_DIRS))   # what PipeWire has
        self._prev_default = None   # ("configured", name or None) before we changed it
        self._changed_default = False
        self._updates = 0
        self._output_present = True

    # ---- public API
    def start(self, output_sink=None):
        """Build the HRIRs and the graph, start PipeWire's filter-chain and the control pipe.
        Raises RuntimeError (after cleaning up) when it cannot."""
        if self._proc:
            return
        self._stop.clear()
        self.error = None
        try:
            self._start(output_sink)
        except Exception as e:
            self.error = str(e)
            self._teardown(restore_default=True)
            raise RuntimeError(str(e)) from e

    def set_pose(self, yaw, pitch):
        """Latest head pose (degrees; yaw + = turned right, pitch + = looking up). Cheap and
        thread-safe; the worker smooths it and sends gains at most 100 times a second. Never
        raises."""
        try:
            y, p = float(yaw), float(pitch)
            if not (math.isfinite(y) and math.isfinite(p)):
                return
        except (TypeError, ValueError):
            return
        with self._lock:
            self._pose = (y, p)
        self._wake.set()

    def set_output(self, output_sink=None):
        """Render onto another sink (e.g. the AirPods just connected): restarts the filter, keeps
        the default-sink bookkeeping so stop() still restores the original default."""
        prev = (self._prev_default, self._changed_default)
        pose = self._pose
        self._teardown(restore_default=False)
        self._prev_default, self._changed_default = prev
        self._pose = pose
        self.start(output_sink)

    def stop(self):
        """Stop everything and give the previous default sink back. Safe to call twice."""
        self._teardown(restore_default=True)

    def status(self):
        alive = bool(self._proc and self._proc.poll() is None)
        return {"active": alive and self.node_id is not None, "output": self.output,
                "error": self.error, "directions": N_DIRS, "node_id": self.node_id,
                "output_present": self._output_present, "updates": self._updates,
                "hrtf": os.path.basename(self.hrtf_path) if self.hrtf_path else None}

    # ---- start / stop internals
    def _start(self, output_sink):
        if not shutil.which("pipewire") or not shutil.which("pw-cli"):
            raise RuntimeError("PipeWire (pipewire, pw-cli) not found")
        if not self.hrtf_path or not os.path.exists(self.hrtf_path):
            raise RuntimeError("no HRTF (.sofa) file found")
        os.makedirs(self.dir, exist_ok=True)
        self._kill_stale()
        meta = build_hrir_cache(self.hrtf_path, os.path.join(self.dir, "hrir"))
        dump = pw_dump()
        if find_node(dump, OWN_SINK) is not None:
            raise RuntimeError("another MousePods Spatial sink is already running")
        out = output_sink or self.cfg.get("output") or find_output_sink(True, dump)
        if not out or out in (OWN_SINK, OWN_OUTPUT):
            raise RuntimeError("no output sink to render onto")
        self.output = out
        gain = 10 ** (float(self.cfg.get("gain_db") or 0.0) / 20.0) / meta["factor"]
        with self._lock:
            yaw, pitch = self._pose
        initial = gain_matrix(yaw, pitch, self.width)
        initial = np.where(initial < THRESHOLD, 0.0, initial)
        conf = build_conf(meta["files"], gain, out, initial, int(self.cfg.get("blocksize") or 256))
        xdg = os.path.join(self.dir, "xdg")
        confd = os.path.join(xdg, "pipewire", "filter-chain.conf.d")
        os.makedirs(confd, exist_ok=True)
        for old in glob.glob(os.path.join(confd, "*.conf")):
            os.remove(old)
        with open(os.path.join(confd, "mousepods-spatial.conf"), "w") as f:
            f.write(conf)
        env = dict(os.environ, XDG_CONFIG_HOME=xdg)
        logf = open(os.path.join(self.dir, "pipewire.log"), "wb")
        try:
            self._proc = subprocess.Popen(["pipewire", "-c", "filter-chain.conf"], env=env,
                                          stdin=subprocess.DEVNULL, stdout=logf, stderr=logf,
                                          start_new_session=True)
        finally:
            logf.close()
        with open(os.path.join(self.dir, "pipewire.pid"), "w") as f:
            f.write(str(self._proc.pid))
        # wait for the sink to appear
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("PipeWire filter-chain exited: " + self._log_tail())
            self.node_id = find_node(pw_dump(), OWN_SINK)
            if self.node_id is not None:
                break
            time.sleep(0.15)
        if self.node_id is None:
            raise RuntimeError("the MousePods Spatial sink did not appear: " + self._log_tail())
        self._sent = initial.copy()
        self._spawn_cli()
        if self.cfg.get("make_default", True):
            self._make_default(dump)
        self._output_present = True
        self._thread = threading.Thread(target=self._worker, name="spatial-audio", daemon=True)
        self._thread.start()
        _live.add(self)
        log.info("spatial audio: node %s, output %s, IR %d samples", self.node_id, out, meta["length"])

    def _spawn_cli(self):
        self._cli = subprocess.Popen(["pw-cli"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)

    def _teardown(self, restore_default):
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
        cli, self._cli = self._cli, None
        if cli:
            try:
                cli.stdin.close()
            except OSError:
                pass
            _terminate(cli, group=True)
        proc, self._proc = self._proc, None
        if proc:
            _terminate(proc, group=True)
            try:
                os.remove(os.path.join(self.dir, "pipewire.pid"))
            except OSError:
                pass
        self.node_id = None
        if restore_default and self._changed_default:
            self._restore_default()
        _live.discard(self)

    def _kill_stale(self):
        """A previous MousePods that crashed may have left its filter process: stop it, but only
        if it really is our pipewire (same command line and private XDG_CONFIG_HOME)."""
        pid_file = os.path.join(self.dir, "pipewire.pid")
        try:
            pid = int(open(pid_file).read().strip())
            cmd = open(f"/proc/{pid}/cmdline", "rb").read()
            env = open(f"/proc/{pid}/environ", "rb").read()
        except (OSError, ValueError):
            return
        xdg = os.fsencode("XDG_CONFIG_HOME=" + os.path.join(self.dir, "xdg"))
        if b"pipewire\0" in cmd and b"filter-chain.conf" in cmd and xdg in env.split(b"\0"):
            log.info("spatial audio: stopping a stale filter process (%d)", pid)
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
            for _ in range(20):
                if not os.path.exists(f"/proc/{pid}"):
                    break
                time.sleep(0.1)
        try:
            os.remove(pid_file)
        except OSError:
            pass

    def _log_tail(self):
        try:
            with open(os.path.join(self.dir, "pipewire.log"), "rb") as f:
                lines = f.read().decode(errors="replace").strip().splitlines()
            return " | ".join(lines[-3:]) or "no message"
        except OSError:
            return "no log"

    # ---- default sink
    def _make_default(self, dump):
        if not self._changed_default:
            # remember the *configured* default (what the user chose), to put it back exactly
            self._prev_default = default_metadata(dump, "default.configured.audio.sink")
        ok = False
        if shutil.which("pactl"):
            ok = _run_ok(["pactl", "set-default-sink", OWN_SINK])
        if not ok and shutil.which("wpctl"):
            ok = _run_ok(["wpctl", "set-default", str(self.node_id)])
        if ok:
            self._changed_default = True
        else:
            log.warning("spatial audio: could not make it the default output")

    def _restore_default(self):
        self._changed_default = False
        now = default_metadata(pw_dump(), "default.configured.audio.sink")
        if now not in (None, OWN_SINK):
            return                      # the user picked another output meanwhile: leave it
        prev = self._prev_default
        if shutil.which("pw-metadata"):
            if prev:
                ok = _run_ok(["pw-metadata", "0", "default.configured.audio.sink",
                              json.dumps({"name": prev}), "Spa:String:JSON"])
            else:
                ok = _run_ok(["pw-metadata", "-d", "0", "default.configured.audio.sink"])
            if ok:
                return
        if prev and shutil.which("pactl"):
            _run_ok(["pactl", "set-default-sink", prev])

    # ---- worker: pose -> gains -> set-param
    def _worker(self):
        period = 1.0 / UPDATE_HZ
        next_health = time.monotonic() + HEALTH_S
        smooth = None
        busy = False                    # still gliding towards the target: do not sleep
        while not self._stop.is_set():
            if not busy:
                self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self._stop.is_set():
                break
            t0 = time.monotonic()
            try:
                with self._lock:
                    target = self._pose
                smooth = smooth_pose(smooth, target, period)
                want = gain_matrix(smooth[0], smooth[1], self.width)
                busy = self._send(want) or smooth != tuple(target)
                if t0 >= next_health:
                    next_health = t0 + HEALTH_S
                    self._health()
            except Exception as e:  # noqa: BLE001  (never let the worker die silently)
                self.error = f"spatial worker: {e}"
                log.warning("spatial audio: %s", e)
                busy = False
            dt = time.monotonic() - t0
            if dt < period:
                self._stop.wait(period - dt)

    def _send(self, want):
        """Move PipeWire's gains towards `want` (at most MAX_STEP each). Returns True while
        they have not reached it yet."""
        step = limit_step(self._sent, want)
        ups = diff_updates(self._sent, step)
        if ups and self.node_id is not None:
            cli = self._cli
            if cli is None or cli.poll() is not None:
                self._spawn_cli()           # pw-cli died: one new try per update
                cli = self._cli
            try:
                for line in set_param_lines(self.node_id, ups):
                    cli.stdin.write((line + "\n").encode())
                    cli.stdin.flush()
            except (OSError, ValueError) as e:
                self.error = f"pw-cli: {e}"
                return False
            for d, k, v in ups:
                self._sent[k, d] = v
            self._updates += 1
        goal = np.where(np.asarray(want) < THRESHOLD, 0.0, want)
        return bool(np.any(np.abs(goal - self._sent) > THRESHOLD))

    def _health(self):
        if self._proc is None or self._proc.poll() is not None:
            self.error = "PipeWire filter-chain stopped: " + self._log_tail()
            return
        dump = pw_dump()
        if not dump:
            return
        if find_node(dump, OWN_SINK) is None:
            self.error = "the MousePods Spatial sink disappeared"
        present = find_node(dump, self.output) is not None
        if present != self._output_present:
            self._output_present = present
            self.error = None if present else f"output {self.output} is gone"


def _run_ok(args, timeout=3.0):
    try:
        return subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=timeout).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _terminate(proc, group=False):
    """SIGTERM (to the process group), then SIGKILL after 2 s."""
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 1.0)):
        try:
            if group:
                os.killpg(proc.pid, sig)
            else:
                proc.send_signal(sig)
        except OSError:
            pass
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


# instances still running at interpreter exit are stopped (default sink given back)
_live = weakref.WeakSet()


@atexit.register
def _stop_all():
    for sp in list(_live):
        try:
            sp.stop()
        except Exception:  # noqa: BLE001
            pass
