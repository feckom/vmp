VMP — Visualized Music Player

Fast single-file music player with a neon/cyberpunk circular visualizer, precise FFmpeg seeking, and smooth cosine crossfades. Runs on Python + Pygame.

Features

64 log-spaced FFT bands (40 Hz–16 kHz), bass/voice envelopes

Amorphous “flubber” center shape driven by audio

Cosine crossfade to next track, precise seek via FFmpeg segment

View presets (HUD/FPS/TIME/TITLE), optional backgrounds, fake fullscreen

Lightweight async analysis + LRU cache for low latency

Requirements

Python 3.11+

FFmpeg in PATH

Python packages: pygame, numpy, mutagen, pydub, websockets (optional)

Install (Windows)
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install pygame numpy mutagen pydub websockets
mkdir music backgrounds

Run
python vmp.py --music-dir "music" --backgrounds "backgrounds" --debug


Icon (optional): place vmp.png (32×32 or 64×64 PNG) next to vmp.py.

Controls

Space pause/resume · N/P next/prev · ←/→ seek −/+5 s

S shuffle · R repeat-all

V / F2 next preset / cycle all · F fake fullscreen · T always-on-top

B toggle BG · [ / ] prev/next BG · H help

Wheel/↑/↓ volume · M mute · Esc/Q quit

CLI Options
--music-dir PATH        Library root (recursive scan)
--backgrounds PATH      Optional images (.jpg/.jpeg/.png)
--debug                 Verbose logs + logs/player.log
--shuffle               Shuffle after the first track
--repeat-all            Repeat the whole list
--no-fft                Disable FFT (testing)
--viz-lowcpu            Lightweight placebo viz
--ext EXT,EXT,...       Override extensions (default: mp3,wav,flac,ogg,m4a)
--ignore DIR,DIR,...    Skip directories by name
--no-tags               Ignore ID3; guess from filenames

Notes

Environment hints (Windows) are set in code:
SDL_RENDER_DRIVER=direct3d, SDL_HINT_RENDER_SCALE_QUALITY=1.

If seeking via pygame.mixer.music isn’t supported by a codec, FFmpeg segment playback is used automatically.
