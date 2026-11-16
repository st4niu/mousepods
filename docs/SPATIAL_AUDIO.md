# Spatial audio

Head-tracked spatial audio on Linux, like Apple's spatial audio on an iPhone: MousePods adds an
output called **MousePods Spatial**. Anything you play through it (Spotify, YouTube, games, films)
is rendered in 3D in your AirPods, and the sound stays in place in the room while you turn your
head: stereo sounds like two speakers on your desk, 5.1 / 7.1 comes from all around.

## Use it

1. Overview page: switch **Spatial audio** on. The first time, MousePods downloads the HRTF
   (1.1 MB, see below). "MousePods Spatial" becomes the default output; your previous output comes
   back when you switch it off.
2. Click **Play a test sound**: a chime walks around your head (front, right, back, left). Turn
   your head while it plays: it stays where it is in the room.
3. Play anything. If something was already playing, pause and play again so it moves to the new
   output.
4. Look straight at your screen and press **Recenter** (or Ctrl+Alt+C).

Settings, Spatial audio:

| Setting | What it does |
|---|---|
| Intensity | 0 = plain stereo, 1 = full 3D. Around 0.5 to 0.75 keeps music clear and still spatial. Changes live |
| Head tracking | Off: the sound turns with your head, like normal headphones |
| Width | Angle of the front speakers |
| Use it as the default output | Switch the default output automatically |

## Good things to try

- **Browser games**: [Krunker.io](https://krunker.io) or [Shell Shockers](https://shellshock.io):
  steps and shots stay anchored to the screen when you turn your head.
- **Music and videos** (Spotify, YouTube): two virtual speakers in front of you.
- **Real surround**: a Steam game through Proton (it outputs 7.1 when the output is 7.1, which
  MousePods Spatial is) or a 5.1 film in VLC or mpv: sounds really come from behind and the sides.
- **Binaural recordings and "8D audio"** are already 3D: set the intensity to 0 for those.

Most things in a browser are stereo, so you get the two front speakers there; surround needs a
5.1 / 7.1 source.

## How it works

- A virtual 7.1 sink (PipeWire null sink) receives the sound; `pw-record` reads it block by block
  (512 frames at 48 kHz) and `pw-play` sends the result to the AirPods.
- Every block, each speaker's direction relative to the head is computed from the head angles,
  and the HRIR for that exact direction comes from libmysofa (interpolated, not a fixed grid).
- The block is convolved with the new HRIR and with the previous one, and the two are crossfaded:
  turning the head never clicks.
- Diffuse-field equalisation keeps the timbre close to the original; a limiter prevents clipping.
- Extra latency: about 30 to 50 ms. CPU: a few percent of one core for stereo.

Code: `spatial_audio.py` (sink, streams, control) and `spatial_render.py` (the DSP).

## HRTF

MousePods uses the **MIT KEMAR** dummy-head measurements by Bill Gardner and Keith Martin (MIT
Media Lab), in the SOFA format from the [SOFA conventions database](https://sofacoustics.org/data/database/mit/).
The file is not shipped with MousePods: it is downloaded on first use from sofacoustics.org and
checked against a fixed SHA-256 (`hrtf_download.py`). Any other `.sofa` file placed in the
`hrtf` folder of MousePods' data folder can be used instead.

## Requirements

PipeWire (with `pw-cli`, `pw-record`, `pw-play`, `pw-dump`), libmysofa and numpy. The Flatpak has
them all; it needs access to PipeWire and to the sound server for this
(`--filesystem=xdg-run/pipewire-0`, `--socket=pulseaudio`).
