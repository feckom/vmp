VMP: Visualized Music Player
A fast, single-file music player built with Python and Pygame, featuring a stunning neon/cyberpunk-themed circular audio visualizer. It includes precise seeking via FFmpeg and smooth cosine crossfades for seamless track transitions.
Features
Dynamic Visualization: An amorphous "flubber" shape in the center is driven by the audio, complemented by 64 log-spaced FFT bands (40 Hz–16 kHz), along with bass and voice envelopes.
Smooth Playback: Enjoy cosine crossfades between tracks and precise seeking thanks to FFmpeg segment playback.
Customization & Control: Choose from various view presets (HUD/FPS/TIME/TITLE), toggle optional background images, and use fake fullscreen mode.
Optimized Performance: Features lightweight asynchronous audio analysis and an LRU cache for low-latency visualization.
Effortless Management: Create a dedicated music folder for your audio files and an optional backgrounds folder for images.
Requirements
Python 3.11+
FFmpeg installed and accessible in your system's PATH.
The following Python packages: pygame, numpy, mutagen, pydub, and websockets (optional).
Installation (Windows)
Open a terminal and create a virtual environment:
Bash
python -m venv .venv
Activate the virtual environment:
Bash
call .venv\Scripts\activate.bat
Upgrade pip and install the required packages:
Bash
python -m pip install --upgrade pip
pip install pygame numpy mutagen pydub websockets
Create the necessary directories:
Bash
mkdir music backgrounds
Running the Player
To start the player, specify the directories for your music and backgrounds:
Bash
python vmp.py --music-dir "music" --backgrounds "backgrounds"
To run it in debug mode, which provides verbose logs and a logs/player.log file:
Bash
python vmp.py --music-dir "music" --backgrounds "backgrounds" --debug
Optional: Add a Custom Icon
Place a vmp.png file (32×32 or 64×64) in the same directory as vmp.py.
Controls
Key/ActionDescription
SpacePause / resume playback
N / PNext / previous track
← / →Seek backward / forward by 5 seconds
SToggle shuffle mode
RToggle repeat-all mode
V / F2Cycle to the next view preset
FToggle fake fullscreen
TToggle always-on-top mode
BToggle background images
[ / ]Go to the previous / next background image
HShow a help overlay
Wheel / ↑ / ↓Adjust volume
MMute / unmute
Esc / QQuit the application
Exportovať do Tabuliek
Command-Line Options
OptionDescription
--music-dir PATH(Required) Specifies the root directory to recursively scan for music files.
--backgrounds PATH(Optional) Specifies the directory for background images (.jpg, .jpeg, .png).
--debugEnables verbose logging to the console and a logs/player.log file.
--shuffleStarts playback in shuffle mode after the first track.
--repeat-allStarts playback in repeat-all mode.
--no-fftDisables the FFT visualizer (useful for testing).
--viz-lowcpuUses a lightweight, low-impact placebo visualizer.
--ext EXT,EXT,...Overrides the default file extensions to scan for (e.g., --ext mp3,wav). Default: mp3,wav,flac,ogg,m4a.
--ignore DIR,DIR,...Skips specified directories by name during the scan.
--no-tagsIgnores ID3 metadata and attempts to guess track titles from filenames.
Exportovať do Tabuliek
Notes
Windows-specific optimizations: The code includes environment hints to use Direct3D rendering (SDL_RENDER_DRIVER=direct3d) and high-quality scaling (SDL_HINT_RENDER_SCALE_QUALITY=1) for a smoother experience.
Automatic FFmpeg Fallback: If a music file's codec isn't supported by pygame.mixer.music for seeking, the player automatically switches to FFmpeg segment playback to ensure precise seeking functionality.
