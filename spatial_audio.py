# SPDX-License-Identifier: Apache-2.0
"""Head-tracked spatial audio on Linux with PipeWire, like Apple's spatial audio.

A virtual 7.1 output, "MousePods Spatial", is added to PipeWire. Audio sent to it is rendered
binaurally with an HRTF onto the AirPods (or any stereo sink), and the virtual speakers stay fixed
in the room while the head turns.

How it works (three helper processes, all PipeWire command-line tools, plus one render thread):
  * the sink: a long-lived `pw-cli` creates a null audio sink (8 channels) owned by its client
    (object.linger = false), so the sink disappears with that pw-cli, even if MousePods crashes;
  * capture: `pw-record` reads the sink's monitor (stream.capture.sink) as f32 frames on its
    stdout (PipeWire 1.2.4 writes them raw; a WAV or .au header is parsed if a version adds one);
  * render: spatial_render.Renderer, block by block (512 frames): HRIR of the exact direction of
    every virtual speaker relative to the head (libmysofa interpolates), old/new HRIR crossfade
    inside each block (no click when the head moves), diffuse-field EQ, dry/binaural mix;
  * playback: `pw-play` reads the binaural stereo from its stdin (raw f32) and plays it on
    the chosen output (AirPods) with node.dont-reconnect, so it never falls back onto our own sink.
The capture side runs on the null sink's clock, the playback side on the output's clock (both
follow the system clock). The pipes are kept short (~2 blocks queued) and trimmed by whole
blocks if they grow. Do NOT put the streams in one node.group: pw-play's blocking read would then
stall every graph cycle (measured: half the cycles skipped, garbled audio).

Conventions:
  * azimuth in degrees, positive = to the LEFT (counter-clockwise seen from above), 0 = front;
    elevation positive = up. SOFA coordinates: x front, y left, z up (verified with MIT KEMAR:
    azimuth +90 is louder in the left ear).
  * head pose: yaw positive = head turned RIGHT, pitch positive = looking UP (same as
    opentrack_out.py). Roll is ignored for now: tilting the head sideways does not move the
    speakers (a small error, much less audible than yaw).

Only the stdlib and numpy, libmysofa through ctypes, and pw-cli / pw-record / pw-play / pw-dump.
"""
from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import fcntl
import glob
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import struct
import subprocess
import termios
import threading
import time
import weakref

import numpy as np

import spatial_render as sr
from spatial_render import relative_direction, smooth_pose, speakers  # noqa: F401  (public helpers)

log = logging.getLogger(__name__)

RATE = sr.RATE
BLOCK = sr.BLOCK
CHANNELS = sr.CHANNELS
OWN_SINK = "mousepods_spatial"                  # node.name of our virtual sink
OWN_OUTPUT = OWN_SINK + ".output"               # node.name of our playback stream
DESCRIPTION = "MousePods Spatial"
SOFA_RADIUS = 1.2                               # m, distance asked to libmysofa
HEALTH_S = 2.0                                  # how often the health thread looks at the graph
RESPAWN_S = 1.0                                 # at most one restart of a stream per second
MAX_PIPE_BLOCKS = 4                             # trim a pipe that holds more blocks than this
PREFILL_BLOCKS = 2                              # silence written ahead to pw-play at start
CACHE_VERSION = 3                               # bump when the EQ computation changes
F_SETPIPE_SZ = 1031                             # fcntl (linux/fcntl.h)

DEFAULTS = {
    "width": 30.0,          # azimuth of the front left/right speakers (degrees)
    "intensity": 0.75,      # 1 = fully binaural, 0 = plain stereo downmix
    "make_default": True,   # make "MousePods Spatial" the default output while it runs
    "gain_db": 0.0,         # overall level (a limiter keeps it under full scale)
    "output": None,         # node.name of the sink to render onto; None = AirPods, else default
    "hrtf": None,           # path of a .sofa file; None = the first one found (see find_hrtf)
    "sink_name": OWN_SINK,  # node.name of the virtual sink (only tests change it)
}


# ------------------------------------------------------------------ directions for the EQ
def directions():
    """Grid used to measure the diffuse field: rings at -30° (12), 0° (36) and +30° (12)."""
    out = []
    for el, n in ((-30.0, 12), (0.0, 36), (30.0, 12)):
        for i in range(n):
            az = i * 360.0 / n
            out.append((az - 360.0 if az > 180.0 else az, el))
    return out


DIRECTIONS = directions()
N_DIRS = len(DIRECTIONS)


# ------------------------------------------------------------------ HRIRs (libmysofa)
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




class SofaHrtf:
    """HRIR source for the renderer: fetch(az, el) -> (2, n) with the ear delays prepended.
    Only used from one thread at a time (libmysofa's lookup is not thread-safe)."""

    def __init__(self, path, rate=RATE, mysofa=None):
        self.mysofa = mysofa or Mysofa()
        self.h, self.n = self.mysofa.open(path, rate)
        self.rate = rate

    def __call__(self, az, el):
        a, e = math.radians(az), math.radians(el)
        left, right, dl, dr = self.mysofa.getfilter(
            self.h, self.n, SOFA_RADIUS * math.cos(e) * math.cos(a),
            SOFA_RADIUS * math.cos(e) * math.sin(a), SOFA_RADIUS * math.sin(e))
        pl = max(0, int(round(float(dl) * self.rate)))
        pr = max(0, int(round(float(dr) * self.rate)))
        n = max(pl + len(left), pr + len(right))
        out = np.zeros((2, n))
        out[0, pl:pl + len(left)] = left
        out[1, pr:pr + len(right)] = right
        return out

    def close(self):
        if self.h:
            self.mysofa.close(self.h)
            self.h = None


# ------------------------------------------------------------------ diffuse-field EQ
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




def cache_key(sofa_path, rate=RATE):
    """Key of the EQ cache: SOFA file identity (path, size, mtime) + rate + version."""
    st = os.stat(sofa_path)
    blob = json.dumps({"v": CACHE_VERSION, "sofa": os.path.abspath(sofa_path), "size": st.st_size,
                       "mtime": st.st_mtime_ns, "rate": int(rate), "dirs": N_DIRS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def eq_for(sofa_path, cache_dir, rate=RATE, mysofa=None):
    """Diffuse-field EQ FIR for a SOFA file, computed once and kept in cache_dir (.npy)."""
    path = os.path.join(cache_dir, f"eq-{cache_key(sofa_path, rate)}.npy")
    try:
        h = np.load(path)
        if h.ndim == 1 and h.size:
            return h
    except (OSError, ValueError):
        pass
    irs = extract_hrirs(sofa_path, DIRECTIONS, rate, mysofa)
    h = diffuse_field_filter(irs, rate)
    os.makedirs(cache_dir, exist_ok=True)
    for old in glob.glob(os.path.join(cache_dir, "eq-*.npy")):
        try:
            os.remove(old)
        except OSError:
            pass
    tmp = path + ".tmp.npy"
    np.save(tmp, h)
    os.replace(tmp, path)
    return h


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
    def own(n):                         # our sink, its streams, test instances
        return bool(n) and n.startswith(OWN_SINK)
    names = [n for _, n, _ in sinks(dump) if n and not own(n)]
    default = default_metadata(dump, "default.audio.sink")
    if not default and shutil.which("pactl"):
        default = _run(["pactl", "get-default-sink"]).strip() or None
    if prefer_bluez:
        bt = [n for n in names if n.startswith("bluez_output.")]
        if bt:
            return default if default in bt else bt[0]     # several headsets: the default one
    if default and not own(default) and (not names or default in names):
        return default
    return names[0] if names else None


def _null_sink_plugin():
    for d in ("/usr/lib/x86_64-linux-gnu/spa-0.2", "/usr/lib64/spa-0.2", "/usr/lib/spa-0.2",
              "/usr/lib/aarch64-linux-gnu/spa-0.2"):
        p = os.path.join(d, "support", "libspa-support.so")
        if os.path.exists(p):
            return p
    found = glob.glob("/usr/lib*/**/spa-0.2/support/libspa-support.so", recursive=True)
    return found[0] if found else None


def available(data_dir=None, hrtf_path=None):
    """(True, "") when everything spatial audio needs is there, else (False, why)."""
    for tool in ("pw-cli", "pw-record", "pw-play", "pw-dump"):
        if not shutil.which(tool):
            return False, f"{tool} not found (PipeWire is needed)"
    if not _null_sink_plugin():
        return False, "PipeWire's null audio sink (spa support plugin) not found"
    try:
        Mysofa()
    except OSError:
        return False, "libmysofa not found (install libmysofa1)"
    path = hrtf_path or find_hrtf(data_dir)
    if not path or not os.path.exists(path):
        return False, "no HRTF (.sofa) file in ~/.local/share/mousepods/hrtf/"
    return True, ""


# ------------------------------------------------------------------ command lines
def _q(s):
    return json.dumps(str(s))


def sink_command(name=OWN_SINK, description=DESCRIPTION):
    """pw-cli line that creates our 8-channel null sink (gone when that pw-cli exits)."""
    pos = " ".join(CHANNELS)
    return (f"create-node adapter {{ factory.name = support.null-audio-sink node.name = {_q(name)} "
            f"node.description = {_q(description)} media.class = Audio/Sink audio.channels = 8 "
            f"audio.rate = {RATE} audio.position = [ {pos} ] object.linger = false "
            f"priority.driver = 1 monitor.channel-volumes = true }}")


def capture_command(sink=OWN_SINK):
    """pw-record of the sink's monitor: f32, 8 channels, WAV container on stdout."""
    props = (f"{{ stream.capture.sink = true node.name = {_q(sink + '.capture')} "
             f"node.description = {_q(DESCRIPTION + ' (capture)')} node.dont-reconnect = true "
             f"application.name = {_q(DESCRIPTION)} }}")
    return ["pw-record", "--target", sink, "-P", props, "--rate", str(RATE), "--channels", "8",
            "--channel-map", ",".join(CHANNELS), "--format", "f32", "--latency", str(BLOCK), "-"]


def playback_command(output, sink=OWN_SINK):
    """pw-play of the binaural stereo (raw f32 on stdin) onto `output`, never onto our own sink."""
    if not output or output == sink:
        raise ValueError("the spatial output needs a real sink, never itself")
    props = (f"{{ node.name = {_q(sink + '.output')} node.description = {_q(DESCRIPTION + ' (output)')} "
             f"node.dont-reconnect = true "
             f"application.name = {_q(DESCRIPTION)} media.role = Music }}")
    return ["pw-play", "--target", output, "-P", props, "--rate", str(RATE), "--channels", "2",
            "--format", "f32", "--latency", str(BLOCK), "-"]


def pipe_pending(fd):
    """Bytes waiting in a pipe (works on both ends on Linux), 0 if unknown."""
    try:
        return struct.unpack("i", fcntl.ioctl(fd, termios.FIONREAD, b"\0\0\0\0"))[0]
    except OSError:
        return 0


def _set_pipe_size(fd, size):
    try:
        fcntl.fcntl(fd, F_SETPIPE_SZ, size)
    except OSError:
        pass


def links_into(dump, src_name, dst_name):
    """True when node `src_name` has a link into node `dst_name`."""
    src, dst = find_node(dump, src_name), find_node(dump, dst_name)
    if src is None or dst is None:
        return False
    for o in dump:
        if str(o.get("type", "")).endswith(":Link"):
            info = o.get("info") or {}
            if info.get("output-node-id") == src and info.get("input-node-id") == dst:
                return True
    return False


# ------------------------------------------------------------------ the running service
class SpatialAudio:
    """Owns the sink (pw-cli), the capture (pw-record), the playback (pw-play) and the render
    thread. cfg keys: see DEFAULTS. Typical use:
        sp = SpatialAudio(cfg, data_dir); sp.start(); sp.set_pose(yaw, pitch) ...; sp.stop()
    """

    def __init__(self, cfg: dict, data_dir: str, hrtf_path: str | None = None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update({k: v for k, v in (cfg or {}).items() if v is not None})
        self.width = float(self.cfg.get("width") or 30.0)
        self.sink = str(self.cfg.get("sink_name") or OWN_SINK)
        self.data_dir = data_dir
        self.dir = os.path.join(data_dir, "spatial")
        self.hrtf_path = hrtf_path or self.cfg.get("hrtf") or find_hrtf(data_dir)
        self.output = None
        self.error = None
        self.node_id = None
        self.renderer = None
        self._hrtf = None
        self._sink_cli = None       # pw-cli holding the null sink
        self._rec = None            # pw-record
        self._play = None           # pw-play
        self._play_lock = threading.Lock()
        self._threads = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pose = (0.0, 0.0)
        self._prev_default = None   # configured default sink before we changed it
        self._changed_default = False
        self._output_present = True
        self._sink_present = False
        self._last_rec_spawn = 0.0
        self._last_play_spawn = 0.0
        self._blocks = 0
        self._dropped = 0
        self._cpu_s = 0.0

    # ---- public API
    def start(self, output_sink=None):
        """Create the sink, start capture, renderer and playback. Raises RuntimeError (after
        cleaning up) when it cannot."""
        if self._sink_cli:
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
        thread-safe; the renderer smooths it and uses it at the next block. Never raises."""
        try:
            y, p = float(yaw), float(pitch)
            if not (math.isfinite(y) and math.isfinite(p)):
                return
        except (TypeError, ValueError):
            return
        with self._lock:
            self._pose = (y, p)

    def set_output(self, output_sink=None):
        """Render onto another sink (e.g. the AirPods just connected). Only the playback
        stream is restarted; None = pick again (AirPods, else the default sink)."""
        out = output_sink or find_output_sink(True)
        if not out or out == self.sink:
            self.error = "no output sink to render onto"
            return
        self.output = out
        self._output_present = True
        self.error = None
        self._kill_play()               # the render thread starts a new one at the next block

    def stop(self):
        """Stop everything and give the previous default sink back. Safe to call twice."""
        self._teardown(restore_default=True)

    def status(self):
        alive = bool(self._sink_cli and self._sink_cli.poll() is None
                     and any(t.is_alive() for t in self._threads))
        return {"active": alive and self.node_id is not None, "output": self.output,
                "error": self.error, "directions": "any",
                "node_id": self.node_id, "output_present": self._output_present,
                "blocks": self._blocks, "dropped": self._dropped,
                "render_cpu_s": round(self._cpu_s, 3),
                "hrtf": os.path.basename(self.hrtf_path) if self.hrtf_path else None,
                "intensity": self.renderer.intensity if self.renderer else None}

    # ---- start / stop internals
    def _start(self, output_sink):
        for tool in ("pw-cli", "pw-record", "pw-play", "pw-dump"):
            if not shutil.which(tool):
                raise RuntimeError(f"{tool} not found (PipeWire is needed)")
        if not self.hrtf_path or not os.path.exists(self.hrtf_path):
            raise RuntimeError("no HRTF (.sofa) file found")
        os.makedirs(self.dir, exist_ok=True)
        mysofa = Mysofa()
        eq = eq_for(self.hrtf_path, self.dir, RATE, mysofa)
        self._hrtf = SofaHrtf(self.hrtf_path, RATE, mysofa)
        self.renderer = sr.Renderer(self._hrtf, eq, width=self.width,
                                    intensity=float(self.cfg.get("intensity", 0.75)),
                                    gain_db=float(self.cfg.get("gain_db") or 0.0))
        dump = pw_dump()
        if find_node(dump, self.sink) is not None:
            raise RuntimeError(f"a sink named {self.sink} already exists")
        out = output_sink or self.cfg.get("output") or find_output_sink(True, dump)
        if not out or out == self.sink:
            raise RuntimeError("no output sink to render onto")
        self.output = out
        # the sink: lives as long as this pw-cli
        self._sink_cli = subprocess.Popen(["pw-cli"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, start_new_session=True)
        self._sink_cli.stdin.write((sink_command(self.sink) + "\n").encode())
        self._sink_cli.stdin.flush()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._sink_cli.poll() is not None:
                raise RuntimeError("pw-cli exited while creating the sink")
            self.node_id = find_node(pw_dump(), self.sink)
            if self.node_id is not None:
                break
            time.sleep(0.1)
        if self.node_id is None:
            raise RuntimeError("the MousePods Spatial sink did not appear")
        self._sink_present = True
        if self.cfg.get("make_default", True):
            self._make_default(dump)
        self._output_present = True
        self._threads = [threading.Thread(target=self._render_loop, name="spatial-render", daemon=True),
                         threading.Thread(target=self._health_loop, name="spatial-health", daemon=True)]
        for t in self._threads:
            t.start()
        _live.add(self)
        log.info("spatial audio: sink %s (node %s), output %s", self.sink, self.node_id, out)

    def _teardown(self, restore_default):
        self._stop.set()
        for p in (self._rec, self._play):         # unblocks the render thread's read / write
            if p:
                _terminate(p, group=True)
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=3.0)
        self._threads = []
        self._kill_play()
        rec, self._rec = self._rec, None
        if rec:
            _terminate(rec, group=True)
        cli, self._sink_cli = self._sink_cli, None
        if cli:
            try:                                # pw-cli quits, the sink goes with it
                cli.stdin.write(b"quit\n")
                cli.stdin.close()
            except OSError:
                pass
            try:
                cli.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _terminate(cli, group=True)
        if self._hrtf:
            self._hrtf.close()
            self._hrtf = None
        self.node_id = None
        self._sink_present = False
        if restore_default and self._changed_default:
            self._restore_default()
        _live.discard(self)

    # ---- streams
    def _spawn_rec(self):
        self._last_rec_spawn = time.monotonic()
        p = subprocess.Popen(capture_command(self.sink), stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             start_new_session=True, bufsize=0)
        _set_pipe_size(p.stdout.fileno(), 65536)
        info, left = sr.read_stream_header(p.stdout.read)
        if info["channels"] not in (None, 8):
            _terminate(p, group=True)
            raise RuntimeError(f"capture has {info['channels']} channels")
        self._rec, self._rec_info, self._rec_left = p, info, left

    def _spawn_play(self):
        self._last_play_spawn = time.monotonic()
        p = subprocess.Popen(playback_command(self.output, self.sink), stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, bufsize=0)
        _set_pipe_size(p.stdin.fileno(), 16384)
        p.stdin.write(sr.encode_f32(np.zeros((BLOCK * PREFILL_BLOCKS, 2))))
        with self._play_lock:
            self._play = p

    def _kill_play(self):
        with self._play_lock:
            p, self._play = self._play, None
        if p:
            try:
                p.stdin.close()
            except OSError:
                pass
            _terminate(p, group=True)

    def _read_block(self):
        """One block of 8-channel frames from pw-record, or None when the stream ended."""
        info = self._rec_info
        size = BLOCK * 8 * np.dtype(info["dtype"]).itemsize
        fd = self._rec.stdout
        # fell behind (CPU hiccup, clock drift): throw whole blocks away instead of lagging
        while pipe_pending(fd.fileno()) > MAX_PIPE_BLOCKS * size:
            if len(fd.read(size)) < size:
                return None
            self._dropped += 1
        buf = self._rec_left
        while len(buf) < size:
            chunk = fd.read(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        self._rec_left = buf[size:]
        return sr.decode_frames(buf[:size], info["dtype"], 8)

    def _write_block(self, out):
        """Send one rendered block to pw-play; (re)start it when needed. Never raises."""
        with self._play_lock:
            p = self._play
        if p is None or p.poll() is not None:
            if p is not None:
                self._kill_play()
            if (self._stop.is_set() or not self._output_present
                    or time.monotonic() - self._last_play_spawn < RESPAWN_S):
                return
            try:
                self._spawn_play()
            except (OSError, ValueError) as e:
                self.error = f"pw-play: {e}"
                return
            with self._play_lock:
                p = self._play
        data = sr.encode_f32(out)
        try:
            if pipe_pending(p.stdin.fileno()) > MAX_PIPE_BLOCKS * len(data):
                self._dropped += 1            # the output is slower than us: shorten the queue
                return
            p.stdin.write(data)
        except (OSError, ValueError):
            self._kill_play()

    def _render_loop(self):
        while not self._stop.is_set():
            try:
                if self._rec is None or self._rec.poll() is not None:
                    if self._rec is not None:
                        _terminate(self._rec, group=True)
                        self._rec = None
                    wait = RESPAWN_S - (time.monotonic() - self._last_rec_spawn)
                    if self._stop.wait(max(0.0, wait)):
                        break
                    self._spawn_rec()
                x = self._read_block()
                if x is None:
                    if self._rec is not None:
                        _terminate(self._rec, group=True)
                    self._rec = None
                    continue
                with self._lock:
                    yaw, pitch = self._pose
                t0 = time.thread_time()
                out = self.renderer.process(x, yaw, pitch)
                self._cpu_s += time.thread_time() - t0
                self._blocks += 1
                self._write_block(out)
            except EOFError:
                self._rec = None
            except Exception as e:  # noqa: BLE001  (never let the render thread die)
                if not self._stop.is_set():
                    self.error = f"spatial render: {e}"
                    log.warning("spatial audio: %s", e)
                    self._stop.wait(0.5)

    def _health_loop(self):
        while not self._stop.wait(HEALTH_S):
            try:
                self._health()
            except Exception as e:  # noqa: BLE001
                self.error = f"spatial health: {e}"

    def _health(self):
        if self._sink_cli is None or self._sink_cli.poll() is not None:
            self.error = "the MousePods Spatial sink is gone (pw-cli exited)"
            self._sink_present = False
            return
        dump = pw_dump()
        if not dump:
            return
        self._sink_present = find_node(dump, self.sink) is not None
        if not self._sink_present:
            self.error = "the MousePods Spatial sink disappeared"
            return
        # never let our output feed our own input (it would howl)
        if links_into(dump, self.sink + ".output", self.sink):
            log.warning("spatial audio: output was linked into its own sink, stopping it")
            self._kill_play()
        present = find_node(dump, self.output) is not None
        if present != self._output_present:
            self._output_present = present
            self.error = None if present else f"output {self.output} is gone"
            if not present:
                self._kill_play()

    # ---- default sink
    def _make_default(self, dump):
        if not self._changed_default:
            # remember the *configured* default (what the user chose), to put it back exactly
            self._prev_default = default_metadata(dump, "default.configured.audio.sink")
        ok = False
        if shutil.which("pactl"):
            ok = _run_ok(["pactl", "set-default-sink", self.sink])
        if not ok and shutil.which("wpctl"):
            ok = _run_ok(["wpctl", "set-default", str(self.node_id)])
        if ok:
            self._changed_default = True
        else:
            log.warning("spatial audio: could not make it the default output")

    def _restore_default(self):
        self._changed_default = False
        now = default_metadata(pw_dump(), "default.configured.audio.sink")
        if now not in (None, self.sink):
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
