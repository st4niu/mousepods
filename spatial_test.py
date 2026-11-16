# SPDX-License-Identifier: Apache-2.0
"""Test sound for spatial audio: a chime that walks around the head, one virtual speaker after the
other (front, front right, right, back right, back left, left, front left, front), played into the
"MousePods Spatial" output. With head tracking on, the chime stays where it is in the room while
the head turns.

The file is generated once (numpy, no recording shipped) and played with pw-play.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import wave

import numpy as np

RATE = 48000
CHANNELS = ["FL", "FR", "FC", "LFE", "RL", "RR", "SL", "SR"]       # the spatial sink's channel order
WALK = ["FC", "FR", "SR", "RR", "RL", "SL", "FL", "FC"]             # clockwise seen from above
NOTE_S, GAP_S = 0.55, 0.12
SINK = "mousepods_spatial"
_player = None


def chime(freq: float, seconds: float = NOTE_S, rate: int = RATE) -> np.ndarray:
    """A short bell-like note with a broadband attack: the noise burst helps the ears place it."""
    t = np.arange(int(seconds * rate)) / rate
    tone = (np.sin(2 * np.pi * freq * t) + 0.45 * np.sin(2 * np.pi * 2.01 * freq * t)
            + 0.2 * np.sin(2 * np.pi * 3.02 * freq * t)) * np.exp(-t * 6.0)
    rng = np.random.default_rng(int(freq))
    burst = rng.standard_normal(t.size) * np.exp(-t * 90.0) * 0.35
    out = tone + burst
    fade = min(int(0.01 * rate), t.size)                              # no click at either end
    out[:fade] *= np.linspace(0.0, 1.0, fade)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    return out


def build(path: str) -> str:
    """Write the 8-channel walk as 16-bit WAV (stdlib wave)."""
    notes = [220.0 * 2 ** (k / 12) for k in (0, 2, 4, 5, 7, 9, 11, 12)]
    step = int((NOTE_S + GAP_S) * RATE)
    data = np.zeros((step * len(WALK) + RATE // 2, len(CHANNELS)))
    for i, (ch, f) in enumerate(zip(WALK, notes)):
        n = chime(f)
        data[i * step:i * step + n.size, CHANNELS.index(ch)] += n
    data *= 0.5 / max(1e-9, float(np.abs(data).max()))               # about -6 dBFS
    pcm = (data * 32767).astype("<i2")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(len(CHANNELS))
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)
    return path


def play(data_dir: str) -> str | None:
    """Play the walk into the spatial output; returns an error text, or None when it started."""
    global _player
    pw_play = shutil.which("pw-play")
    if not pw_play:
        return "pw-play not found"
    path = os.path.join(data_dir, "spatial", "test-walk-8ch.wav")
    if not os.path.exists(path):
        build(path)
    if _player and _player.poll() is None:                           # one at a time
        _player.terminate()
    _player = subprocess.Popen([pw_play, "--target", SINK, "--channels", str(len(CHANNELS)),
                                "--channel-map", ",".join(CHANNELS), path],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return None
