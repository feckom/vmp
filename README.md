# VMP — Cyberpunk Circular Music Player (Python)

A fast, single-file music player with a neon/cyberpunk circular visualizer, precise FFmpeg-based seeking, smooth cosine crossfades, and a lightweight web terminal for remote key controls.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Performance Notes](#performance-notes)
- [Troubleshooting](#troubleshooting)


---

## Features

- **Optimized FFT & visualization**
  - Reused Hann windows and reusable FFT buffers
  - Log-spaced **64** bands (40 Hz … 16 kHz), band-weighting
  - Bass/voice masks with smoothed roll-offs
  - “Vocal pulse” center disc + dotted ring + energy-colored progress arc

- **Responsive audio engine**
  - **Precise seek** by decoding a **30 s** segment via **FFmpeg** to `Channel(0)`
  - **Smooth cosine crossfade** using overlap on `Channel(1)`
  - **ThreadPoolExecutor** for async decoding & prefetch
  - LRU cache for analyzed tracks (normalized cache keys)

- **Scheduling & UI**
  - Robust overlap scheduling near track end
  - Fake fullscreen (borderless) with correct rescaling after toggle
  - View presets (HUD/FPS/TIME/TITLE), background scaling cache
  - Lower CPU usage when paused; low-CPU viz mode available

- **Web terminal mode**
  - Optional `--webterm` WebSocket server on `ws://localhost:3030`
  - Streams logs and accepts basic remote commands

---

## Requirements

- **Python** 3.11+
- **FFmpeg** available in your `PATH`
- Python packages:
  - `pygame`, `numpy`, `mutagen`, `pydub`, `websockets`
- Windows (recommended/primary target):
  - Environment hints set automatically:
    - `SDL_RENDER_DRIVER=direct3d`
    - `SDL_HINT_RENDER_SCALE_QUALITY=1`
- Optional: `cyberpunk.ttf` in the same folder as `vmp.py` (falls back to system fonts)

> The player supports audio files: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`.

---

## Installation

### Windows (PowerShell or CMD)

```bat
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install pygame numpy mutagen pydub websockets
mkdir music
mkdir backgrounds
```
---

## Quick-start

python vmp.py --debug --music-dir "music" --backgrounds "backgrounds"

## Usage

--music-dir PATH        Path to your music library (recursively scanned).
--backgrounds PATH      Optional backgrounds folder (.jpg/.jpeg/.png).
--debug                 Verbose logs + rotating log file at logs/player.log.
--shuffle               Shuffle after the first track.
--repeat-all            Repeat entire list.
--no-fft                Disable FFT (for testing).
--viz-lowcpu            Lightweight visual placebo pattern.
--ext EXT,EXT,...       Override extensions (default: mp3,wav,flac,ogg,m4a).
--ignore DIR,DIR,...    Ignore directory names during scan (case-insensitive).
--no-tags               Ignore ID3 tags; guess title/artist from filenames.
--webterm               Start WebSocket terminal at ws://localhost:3030.

Controls

View & window

H help overlay

1..5 preset views

V next preset, F2 cycle all

F fake fullscreen (borderless); scales correctly after toggle

T always-on-top

B toggle backgrounds, [ / ] previous/next background

O opacity −5% (Shift+O +5%)

Mouse: wheel = volume, drag with LMB to move window (when no BG)

Playback

Space pause/resume

N / P next/prev track

S shuffle toggle

R repeat-all toggle

← / → seek −5s / +5s (precise seek via FFmpeg segment)

Esc or Q quit

## How It Works

Audio + decode

Streaming via pygame.mixer.music in “music” mode.

For precise seeking, decodes a 30 s FFmpeg segment to PCM and plays it on Channel(0) (“channel” mode), while tracking position with a monotonic clock.

Crossfade & overlap

When close to the end, the next track’s head is decoded asynchronously and played on Channel(1).

A cosine curve fades the current track out and the next track in; once the overlap finishes, playback switches back to music mode with an offset equal to the overlap length.

FFT & visualization

A dedicated FFT thread responds to requests (event + timeout) and never performs decode.

Uses cached mono analysis per track (LRU).

BandMapper groups FFT bins into 64 log-spaced bands with per-band weighting; bass and voice masks compute envelopes.

UI draws a dotted circular ring, glow halos, outward bars, a progress arc colored by energy, and a central “vocal pulse” that scales/alpha-blends with voice envelope.

Caching & threading

AnalysisCache stores (samples, sample rate, duration) for quick FFT access.

Priority loader threads (NOW, PREFETCH, BULK) warm the cache for the current and upcoming tracks.

Text, background, and geometry caches minimize allocations.

## Performance Notes

Paused state intentionally skips heavy work (FFT etc.).

--viz-lowcpu reduces visual load on low-end setups.

Pygame’s vsync support depends on your GPU/driver.

If you experience stutter, try smaller --backgrounds images or skip backgrounds entirely.

## Troubleshooting

FFmpeg not found

Ensure ffmpeg is installed and on your PATH. On Windows, check by running ffmpeg -version in the same terminal.

Black/empty window

Some GPU/driver combos don’t like the selected SDL renderer. On non-Windows platforms, remove/adjust the SDL env vars at the top of vmp.py.

No sound / seek issues

Some formats don’t honor pygame.mixer.music.play(start=...). That’s why precise seeking uses FFmpeg segments. If you still have issues, convert your files to .wav or .flac to test.

High CPU

Disable backgrounds or use --viz-lowcpu. Close other GPU-intensive apps.


