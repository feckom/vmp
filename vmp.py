from __future__ import annotations
import os, sys, math, random, re, time, argparse, logging, threading, subprocess, ctypes, platform, shutil, queue, collections, itertools
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np
import pygame
import pygame.gfxdraw as gfxdraw
import pygame.sndarray
from mutagen import File as MutaFile
from mutagen.easyid3 import EasyID3
from pydub import AudioSegment
from collections import deque
# ========== CONSOLIDATED CONFIGURATION CONSTANTS (AMBER/CHARCOAL + CHILL FLUBBER) ==========
# Audio Configuration (unchanged)
audio_target_sample_rate = 44100
audio_bass_low_hz = 30
audio_bass_high_hz = 150
audio_analysis_lag_seconds = 0.030
audio_bass_attack_speed = 0.45
audio_bass_release_fast = 0.12
audio_bass_release_slow = 0.04
audio_beat_minimum_gap_seconds = 0.11
audio_voice_low_hz = 200
audio_voice_high_hz = 3800
audio_voice_attack_speed = 0.60
audio_voice_release_fast = 0.10
audio_voice_release_slow = 0.05
audio_voice_peak_decay = 0.995
# Visualization Configuration
visualization_target_fps = 60
visualization_fft_size = 2048
visualization_bass_fft_size = 1024
visualization_number_of_bands = 64
visualization_fft_every_n_frames = 1
# Subtler, cleaner geometry
visualization_progress_arc_width = 12
visualization_rays_thickness = 6
visualization_rays_max_length_fraction = 0.22
visualization_flubber_radius_fraction = 0.18
# --- Colorway: Neo-Amber on Charcoal ---
# base bg: very dark charcoal, UI text: warm white, accents: amber/sand/brass
visualization_rays_color = (212, 114, 22)            # amber rays (#D47216)
visualization_text_bright_color = (238, 238, 235)     # warm white (#EEE EEB)
visualization_alert_color = (255, 64, 48)             # alert red (#FF4030)
visualization_lyrics_color = (245, 208, 66)           # soft gold (#F5D042)
visualization_text_dim_color = (168, 170, 176)        # cool gray (#A8AAB0)
visualization_background_color = (11, 15, 20)         # deep charcoal (#0B0F14)
visualization_voice_circle_base_color = (168, 170, 176)  # dark brass/brown (#40301C)
visualization_title_font_size = 20
visualization_voice_alpha_min = 36
visualization_voice_alpha_max = 210
# Enhanced Flubber organism (centered, bass-driven size)
visualization_flubber_points = 128
visualization_flubber_base_radius_fraction = 0.42
# --- CHILL/LAID-BACK TUNING ---
# lower intensity + speed, more fluid smoothing, softer angular noise
visualization_flubber_amorphous_intensity = 0.06
visualization_flubber_voice_amorphous_gain = 0.035
visualization_flubber_amorphous_speed = 0.18
visualization_flubber_angular_noise_scale = 1.80
# Dynamic scaling based on bass (wider low end, but not jumpy)
visualization_flubber_min_size_multiplier = 0.78
visualization_flubber_max_size_multiplier = 1.85
# Words through flubber (scheduler) — slower cadence to match chill vibe
visualization_word_simple_min = 1
visualization_word_simple_max = 3
visualization_word_morph_min = 2
visualization_word_morph_max = 5
visualization_word_time_window_low = 0.25
visualization_word_time_window_high = 0.85
visualization_word_minimum_gap_seconds = 5.0
visualization_word_smoke_test = False
# Organic motion parameters — slower pulse, gentler amplitude, more fluidity
visualization_organic_pulse_speed = 0.25
visualization_organic_pulse_intensity = 0.10
visualization_flubber_fluidity_factor = 0.93   # higher = more smoothing
visualization_morph_duration_min_seconds = 1.8
visualization_morph_duration_max_seconds = 4.2
visualization_dwell_duration_min_seconds = 1.4
visualization_dwell_duration_max_seconds = 2.8
# Centered track info (unchanged)
visualization_show_centered_info = True
visualization_centered_info_font_size = 24
visualization_centered_info_time_font_size = 18
visualization_centered_info_offset_y = 120
# UI Configuration (unchanged)
ui_next_track_cooldown_seconds = 0.35
ui_volume_popup_duration_seconds = 0.9
ui_toast_duration_seconds = 1.4
ui_seek_segment_seconds = 30.0
# ========== NEW FLUBBER 3D EFFECT CONSTANTS ==========
# 3D lighting and depth parameters for Flubber
visualization_flubber_3d_intensity = 0.45
visualization_flubber_light_angle = 0.785  # 45 degrees in radians
visualization_flubber_specular_power = 8.0
visualization_flubber_depth_multiplier = 0.3
# ========== END NEW FLUBBER 3D EFFECT CONSTANTS ==========
# ========== END CONFIGURATION CONSTANTS ==========
# ========== CONFIGURATION CLASSES (DO NOT MODIFY) ==========
@dataclass
class AudioCfg:
    target_sr: int = audio_target_sample_rate
    bass_low_hz: int = audio_bass_low_hz
    bass_high_hz: int = audio_bass_high_hz
    analysis_lag_sec: float = audio_analysis_lag_seconds
    attack: float = audio_bass_attack_speed
    rel_fast: float = audio_bass_release_fast
    rel_slow: float = audio_bass_release_slow
    beat_min_gap: float = audio_beat_minimum_gap_seconds
    voice_low_hz: int = audio_voice_low_hz
    voice_high_hz: int = audio_voice_high_hz
    voice_attack: float = audio_voice_attack_speed
    voice_rel_fast: float = audio_voice_release_fast
    voice_rel_slow: float = audio_voice_release_slow
    voice_peak_decay: float = audio_voice_peak_decay
@dataclass
class VisualCfg:
    fps_target: int = visualization_target_fps
    fft_size: int = visualization_fft_size
    bass_fft: int = visualization_bass_fft_size
    n_bands: int = visualization_number_of_bands
    fft_every_n_frames: int = visualization_fft_every_n_frames
    progress_width: int = visualization_progress_arc_width
    bar_thickness: int = visualization_rays_thickness
    bar_max_len_frac: float = visualization_rays_max_length_fraction
    ring_radius_frac: float = visualization_flubber_radius_fraction
    bar_color: Tuple[int,int,int] = visualization_rays_color  # HLAVNÁ FARBA LÚČOV: SVETLOŠEDÁ
    white: Tuple[int,int,int] = visualization_text_bright_color      # Takmer biela, pre text
    red: Tuple[int,int,int] = visualization_alert_color          # DRUHÁ FARBA: KRVAVO ČERVENÁ (na dôležité prvky)
    yellow: Tuple[int,int,int] = visualization_lyrics_color       # ŽLTÁ: IBA NA STOPY A VAROVANIA (používa sa minimálne)
    text_dim: Tuple[int,int,int] = visualization_text_dim_color   # Tmavšia šedá pre menej dôležitý text
    bg_color: Tuple[int,int,int] = visualization_background_color      # Hlboká tmavosivá / takmer čierna
    voice_base_color: Tuple[int,int,int] = visualization_voice_circle_base_color  # Hlasový kruh: KOVOVÁ ŠEDÁ
    title_font_size: int = visualization_title_font_size   
    voice_alpha_min: int = visualization_voice_alpha_min
    voice_alpha_max: int = visualization_voice_alpha_max
    # Enhanced Flubber organism (centered, bass-driven size)
    flub_points: int = visualization_flubber_points
    flub_base_frac: float = visualization_flubber_base_radius_fraction
    flub_amorphous_intensity: float = visualization_flubber_amorphous_intensity
    flub_voice_amorphous_gain: float = visualization_flubber_voice_amorphous_gain
    flub_amorphous_speed: float = visualization_flubber_amorphous_speed
    flub_angular_noise_scale: float = visualization_flubber_angular_noise_scale
    # Dynamic scaling based on bass
    flub_min_size_multiplier: float = visualization_flubber_min_size_multiplier
    flub_max_size_multiplier: float = visualization_flubber_max_size_multiplier
    # Words through flubber (scheduler)
    word_simple_min: int = visualization_word_simple_min
    word_simple_max: int = visualization_word_simple_max
    word_morph_min: int = visualization_word_morph_min
    word_morph_max: int = visualization_word_morph_max
    word_window_lo: float = visualization_word_time_window_low
    word_window_hi: float = visualization_word_time_window_high
    word_min_gap: float = visualization_word_minimum_gap_seconds
    word_smoke_test: bool = visualization_word_smoke_test
    # New: Organic motion parameters
    organic_pulse_speed: float = visualization_organic_pulse_speed
    organic_pulse_intensity: float = visualization_organic_pulse_intensity
    fluidity_factor: float = visualization_flubber_fluidity_factor
    morph_duration_min: float = visualization_morph_duration_min_seconds
    morph_duration_max: float = visualization_morph_duration_max_seconds
    dwell_duration_min: float = visualization_dwell_duration_min_seconds
    dwell_duration_max: float = visualization_dwell_duration_max_seconds
    # New: Centered track info
    show_centered_info: bool = visualization_show_centered_info
    centered_info_font_size: int = visualization_centered_info_font_size
    centered_info_time_font_size: int = visualization_centered_info_time_font_size
    centered_info_offset_y: int = visualization_centered_info_offset_y
    # ========== NEW FLUBBER 3D EFFECT PARAMETERS ==========
    flub_3d_intensity: float = visualization_flubber_3d_intensity
    flub_light_angle: float = visualization_flubber_light_angle
    flub_specular_power: float = visualization_flubber_specular_power
    flub_depth_multiplier: float = visualization_flubber_depth_multiplier
    # ========== END NEW FLUBBER 3D EFFECT PARAMETERS ==========
@dataclass
class UiCfg:
    next_cooldown_sec: float = ui_next_track_cooldown_seconds
    volume_popup_sec: float = ui_volume_popup_duration_seconds
    toast_sec: float = ui_toast_duration_seconds
    seek_segment_sec: float = ui_seek_segment_seconds
AUDIO = AudioCfg()
VIS = VisualCfg()
UI = UiCfg()
# ========== END CONFIGURATION CLASSES ==========
MUSIC_DIR = "music"
BG_DIR = "backgrounds"
SCAN_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
FONT_PATH = Path(__file__).with_name("vmp.ttf")
log = logging.getLogger("vmp")
# ========== OPTIMIZED CACHE SYSTEMS ==========
_hann_cache: Dict[int, np.ndarray] = {}
_fft_window_cache: Dict[int, np.ndarray] = {}
_cached_glow: Dict[Tuple[int,int,Tuple[int,int,int],int], pygame.Surface] = {}
_glyph_cache: Dict[Tuple[str,int,int,int,int,bool,bool,int], np.ndarray] = {}
_glyph_cache_max_size = 512
_glyph_cache_access = collections.OrderedDict()
# ========== PRECOMPUTED MASKS ==========
_bass_mask_f: Optional[np.ndarray] = None
_voice_mask_f: Optional[np.ndarray] = None
# ========== LYRICS CACHE ========== (NOVÝ KÓD)
_lyrics_cache: Dict[str, List[Tuple[float, str]]] = {}
def load_lrc_file(lrc_path: Path) -> List[Tuple[float, str]]:
    """Načíta LRC súbor a vráti zoznam (časová značka, text)."""
    if not lrc_path.exists():
        return []
    try:
        with open(lrc_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lyrics = []
        for line in lines:
            line = line.strip()
            if line.startswith('[') and ']' in line:
                try:
                    time_str = line[1:line.index(']')]
                    minutes, seconds = time_str.split(':')
                    total_seconds = int(minutes) * 60 + float(seconds)
                    text = line[line.index(']')+1:].strip()
                    if text:
                        lyrics.append((total_seconds, text))
                except Exception:
                    continue
        # Zoradiť podľa času
        lyrics.sort(key=lambda x: x[0])
        return lyrics
    except Exception as e:
        log.debug("Failed to load LRC file %s: %s", lrc_path.name, e)
        return []
def get_current_lyric_line(lyrics: List[Tuple[float, str]], current_time: float) -> str:
    """Nájde aktuálny riadok textu na základe času."""
    if not lyrics:
        return ""
    current_line = ""
    for i, (timestamp, text) in enumerate(lyrics):
        if timestamp <= current_time:
            current_line = text
        else:
            break
    return current_line
# ========== END LYRICS ========== (NOVÝ KÓD)
def setup_logging(debug=False):
    import logging.handlers
    lg = logging.getLogger("vmp")
    for h in list(lg.handlers): lg.removeHandler(h)
    lg.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(logging.DEBUG if debug else logging.INFO)
    lg.addHandler(ch)
    try:
        log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(log_dir/"player.log", maxBytes=4_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); lg.addHandler(fh)
    except Exception as e:
        print("Log file init failed:", e, file=sys.stderr)
    logging.getLogger().setLevel(logging.WARNING)
    lg.debug("Logging initialized. Debug=%s", debug)
    return lg
def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        log.critical("FFmpeg not found in PATH.")
        return False
    try:
        subprocess.run(["ffmpeg","-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        log.debug("FFmpeg available.")
        return True
    except Exception as e:
        log.critical("FFmpeg probe failed: %s", e)
        return False
def format_time(sec: float) -> str:
    sec = max(0, int(sec))
    if sec >= 3600:
        h, m = divmod(sec, 3600)
        m, s = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    else:
        m, s = divmod(sec, 60)
        return f"{m:02d}:{s:02d}"
_CLEAN_PAT = re.compile(r"(?i)\b(official\s*video|lyrics?|audio|hd|hq|remaster(?:ed)?|live|clip|mv|visualizer|topic|karaoke|instrumental|mono|stereo|remix|mix|edit|radio\s*edit|feat\.?.+|ft\.?.+)\b")
_BRACKETS = re.compile(r"[\(\[\{].*?[\)\]\}]")
_MULTI_SPACE = re.compile(r"\s{2,}")
def _smart_titlecase(s: str) -> str:
    s = s.strip().title()
    for w in ["And","Or","The","Of","Ft.","Feat.","Vs.","A","An","To","In","On","For","With","From","By","At","As","But"]:
        s = re.sub(rf"\b{w}\b", w.lower(), s)
    return s
def guess_from_filename(path: Path) -> Tuple[Optional[str], Optional[str]]:
    name = path.stem.replace("_"," ").replace("."," ")
    name = re.sub(r"^\s*\d{1,3}\s*[-. ]\s*","",name)
    name = _BRACKETS.sub(" ", name); name = _CLEAN_PAT.sub(" ", name)
    name = _MULTI_SPACE.sub(" ", name).strip(" -_")
    artist, title = None, None
    for sep in (" - ", " – "):
        if sep in name:
            p = [x.strip() for x in name.split(sep,1)]
            if len(p)==2 and all(p):
                artist = _smart_titlecase(p[0]); title = _smart_titlecase(p[1]); break
    if not title and name: title = _smart_titlecase(name)
    if not artist:
        parent = _MULTI_SPACE.sub(" ", _BRACKETS.sub(" ", path.parent.name.replace("_"," "))).strip()
        if parent and parent.lower() not in ("music","mp3","tracks","songs"): artist = _smart_titlecase(parent)
    return (title or None, artist or None)
def _norm_key(path: Path | str) -> str:
    try:
        return str(Path(path).resolve()).lower()
    except Exception:
        return str(path).replace("\\", "/").lower()
class Track:
    def __init__(self, path: Path, no_tags: bool=False):
        self.path = path
        self.title, self.artist = (guess_from_filename(path) if no_tags else self._read_or_guess_tags())
        self.duration_sec = self._fast_length()
        self.samples: Optional[np.ndarray] = None
        self.sample_rate = AUDIO.target_sr
        self.analysis_ready = False
        log.debug("Track created: %s | title=%s artist=%s dur=%.2fs",
                  path.name, self.title, self.artist, self.duration_sec)
    def _read_or_guess_tags(self) -> Tuple[Optional[str], Optional[str]]:
        title=None; artist=None
        try:
            tags=EasyID3(str(self.path))
            title = tags.get("title",[None])[0]
            artist= tags.get("artist",[None])[0]
        except Exception as e:
            log.debug("ID3 read failed for %s: %s", self.path.name, e)
        if not title or not str(title).strip() or not artist or not str(artist).strip():
            g_title, g_artist = guess_from_filename(self.path)
            title  = title  if title  and str(title).strip()  else g_title
            artist = artist if artist and str(artist).strip() else g_artist
        title  = str(title).strip()  if title  and str(title).strip()  else None
        artist = str(artist).strip() if artist and str(artist).strip() else None
        return (title, artist)
    def _fast_length(self) -> float:
        try:
            mf = MutaFile(str(self.path))
            if mf and mf.info and getattr(mf.info,"length",None): return float(mf.info.length)
        except Exception: pass
        return 0.0
class AnalysisCache:
    def __init__(self, capacity:int=64):
        self.cap = capacity
        self.od: "collections.OrderedDict[str,Tuple[np.ndarray,int,float]]" = collections.OrderedDict()
        self.lock = threading.Lock()
    def get(self, key:str) -> Optional[Tuple[np.ndarray,int,float]]:
        with self.lock:
            v = self.od.get(key)
            if v is not None:
                self.od.move_to_end(key)
            return v
    def put(self, key:str, samples:np.ndarray, sr:int, dur:float):
        with self.lock:
            if key in self.od:
                self.od.move_to_end(key)
            self.od[key] = (samples, sr, dur)
            while len(self.od) > self.cap:
                k,_ = self.od.popitem(last=False)
                log.debug("AnalysisCache: evict %s", Path(k).name)
AN_CACHE: Optional[AnalysisCache] = None
class Library:
    def __init__(self, exts: Tuple[str,...], ignore_dirs: List[str], no_tags: bool):
        self.tracks: List[Track] = []
        self.exts = tuple(x.lower().strip() for x in exts)
        self.ignore_dirs = set(d.lower() for d in ignore_dirs)
        self.no_tags = no_tags
    def scan(self, base_dir: Path):
        log.info("Scanning library: %s", base_dir)
        add = self.tracks.append
        for p in base_dir.rglob("*"):
            try:
                if any(part.lower() in self.ignore_dirs for part in p.parts): continue
                if p.is_file() and p.suffix.lower() in self.exts:
                    try:
                        add(Track(p, no_tags=self.no_tags))
                    except Exception as e_track:
                        log.warning("Track init failed for %s: %s", p, e_track)
            except Exception as e:
                log.warning("Scan skip (path issue) %s: %s", p, e)
        self.tracks.sort(key=lambda t: str(t.path).lower())
        log.info("Scan complete: %d tracks", len(self.tracks))
def decode_pcm_segment(path: Path, start_sec: float, dur_sec: float, sr=AUDIO.target_sr) -> np.ndarray:
    ss = max(0.0, start_sec)
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-ss", f"{ss:.3f}","-t", f"{float(dur_sec):.3f}",
           "-i", str(path), "-f","s16le","-acodec","pcm_s16le","-ac","2","-ar",str(sr), "-"]
    try:
        log.debug("FFmpeg segment: %s [%.3f..+%.3f]", path.name, ss, dur_sec)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            log.debug("ffmpeg decode non-zero exit (%s): %s", p.returncode, err.decode(errors="ignore")[:500])
            return np.zeros((int(dur_sec*sr),2), np.float32)
    except FileNotFoundError:
        log.critical("ffmpeg not available.")
        return np.zeros((int(dur_sec*sr),2), np.float32)
    except Exception as e:
        log.debug("ffmpeg segment decode failed for %s at %.2fs: %s", path.name, start_sec, e)
        return np.zeros((int(dur_sec*sr),2), np.float32)
    data = np.frombuffer(out, dtype=np.int16)
    if data.size == 0: return np.zeros((int(dur_sec*sr),2), np.float32)
    # return (data.reshape(-1,2).ast(np.float32))/32768.0
    # ensure stereo float32 in [-1.0, 1.0]
    if data is None or data.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # trim to even number of samples (interleaved stereo)
    data = data[: (data.size // 2) * 2]
    # reshape -> stereo, convert -> float32, scale
    seg = data.reshape(-1, 2).astype(np.float32, copy=False)
    seg *= (1.0 / 32768.0)
    return seg
def decode_pcm_segment_robust(path: Path, start_sec: float, dur_sec: float, sr=AUDIO.target_sr, max_retries=2) -> np.ndarray:
    for attempt in range(max_retries):
        try:
            return decode_pcm_segment(path, start_sec, dur_sec, sr)
        except Exception as e:
            if attempt == max_retries - 1:
                log.warning(f"FFmpeg failed after {max_retries} attempts: {e}")
                return np.zeros((int(dur_sec*sr),2), np.float32)
            time.sleep(0.1 * attempt)
    return np.zeros((int(dur_sec*sr),2), np.float32)
def numpy_to_sound(arr_float_stereo: np.ndarray) -> pygame.mixer.Sound:
    arr = np.clip(arr_float_stereo, -1.0, 1.0)
    arr16 = (arr * 32767.0).astype(np.int16, copy=False)
    arr16 = np.ascontiguousarray(arr16)
    if arr16.ndim == 1:
        arr16 = arr16.reshape(-1, 2)
    return pygame.sndarray.make_sound(arr16)
def _hann(size: int) -> np.ndarray:
    h = _hann_cache.get(size)
    if h is None:
        log.debug("Create Hann window: %d", size)
        h = np.hanning(size).astype(np.float32)
        _hann_cache[size] = h
    return h
def make_fft_window(samples, sr, pos_sec, size=VIS.fft_size):
    key = size
    if key not in _fft_window_cache:
        _fft_window_cache[key] = np.zeros(size, dtype=np.float32)
    window = _fft_window_cache[key]
    if samples is None or len(samples) == 0:
        window.fill(0.0)
        return window
    idx = int(pos_sec * sr)
    start = max(0, idx - size)
    end = min(len(samples), start + size)
    sl = end - start
    if sl >= size:
        window[:] = samples[end - size:end]
    elif sl > 0:
        window[:sl] = samples[start:end]
        if sl < size: window[sl:] = 0.0
    else:
        window.fill(0.0)
    return window
class BandMapper:
    def __init__(self, n_time: int, sr: int, n_bands: int):
        self.sr = sr
        self.n_bands = n_bands
        fmin, fmax = 40.0, 16000.0
        self.edges = np.geomspace(fmin, fmax, n_bands+1)
        freqs = np.fft.rfftfreq(n_time, 1.0/self.sr)
        idxs = []
        for i in range(n_bands):
            mask = np.where((freqs>=self.edges[i]) & (freqs<self.edges[i+1]))[0]
            if mask.size == 0:
                closest = np.argmin(np.abs(freqs - (self.edges[i]+self.edges[i+1])/2))
                mask = np.array([closest], dtype=int)
            idxs.append(mask)
        self.idxs = idxs
        self._cat_idx = np.concatenate(self.idxs) if len(self.idxs) else np.array([], dtype=int)
        self._split_points = np.cumsum([len(ix) for ix in self.idxs])[:-1] if len(self.idxs) else np.array([], dtype=int)
        self.band_weights = np.linspace(1.15, 0.85, n_bands).astype(np.float32)
    def map(self, spectrum_abs: np.ndarray) -> np.ndarray:
        if self._cat_idx.size == 0:
            return np.zeros(self.n_bands, np.float32)
        all_values = spectrum_abs[self._cat_idx]
        chunks = np.split(all_values, self._split_points) if self._split_points.size else [all_values]
        band_rms = np.array([
            np.sqrt(np.mean(chunk * chunk)) if chunk.size > 0 else 0.0
            for chunk in chunks
        ], dtype=np.float32)
        vmax = band_rms.max()
        if vmax > 0:
            band_rms /= vmax
        band_rms *= self.band_weights
        return np.clip(band_rms, 0.0, 1.0)
def build_glow_circle_surface(radius:int, glow:int, color_rgb:Tuple[int,int,int], thickness:int=2) -> pygame.Surface:
    key=(radius,glow,color_rgb,thickness)
    if key in _cached_glow: return _cached_glow[key]
    size = (radius+glow+thickness+2)*2
    surf = pygame.Surface((size,size), pygame.SRCALPHA)
    center=(size//2,size//2)
    r,g,b=color_rgb
    for i in range(glow,0,-1):
        alpha=int(12*i)
        col=(r,g,b, min(255,alpha))
        gfxdraw.filled_circle(surf, center[0], center[1], radius+i, col)
    _cached_glow[key]=surf
    return surf
def blit_center(dst: pygame.Surface, src: pygame.Surface, pos: Tuple[int,int]):
    dst.blit(src, src.get_rect(center=pos))
def draw_progress_arc_aa(surf, rect, start_angle, fraction, color_rgb, width):
    fraction = max(0.0, min(1.0, fraction))
    color=(*color_rgb,255)
    end_angle=start_angle+2*math.pi*fraction
    pygame.draw.arc(surf,color,rect,start_angle,end_angle,width)
    for d in (-width//2, width//2):
        pygame.draw.arc(surf, color, rect.inflate(d,d), start_angle, end_angle, 1)
def render_ui_and_text(
    text_surf, w, h, cx, cy,
    pos_now, dur_now, volume,
    last_volume_popup_t, toast_msg, toast_until,
    show_title, show_time, show_hud, show_fps,
    v_mode, shuffle, repeat_all, repeat_one,  # <-- Pridané repeat_one
    current_track, title_font, font_small_sys, vol_font_sys
):
    """
    Bottom-center UI stack with guaranteed fit of up to 4 rows:
      R1: TITLE
      R2: TIME
      R3: HUD
      R4: FPS + BPM + Flubber + Next Word
    - TITLE & TIME: yellow fill with WHITE outline.
    - HUD/FPS remain in VIS.bar_color.
    - Rows uniformly downscale if needed to fit vertical space.
    Returns: (vol_shown: bool, y_next: int)
    """
    import time as _time
    import pygame
    # ---- per-function cache ----
    if not hasattr(render_ui_and_text, "_cache"):
        render_ui_and_text._cache = {
            "title_key": None, "title_surf": None,
            "time_key":  None, "time_surf":  None,
            "hud_key":   None, "hud_surf":   None,
            "fps_key":   None, "fps_surf":   None,
            "vol_txt":   "",   "vol_surf":   None,
        }
    C = render_ui_and_text._cache
    # Clear target surface (transparent)
    text_surf.fill((0, 0, 0, 0))
    # ---------- helpers ----------
    def _ellipsize(font, txt, max_w):
        if not txt:
            return ""
        if font.size(txt)[0] <= max_w:
            return txt
        ell = "…"
        lo, hi = 0, len(txt)
        while lo < hi:
            mid = (lo + hi) // 2
            if font.size(txt[:mid] + ell)[0] <= max_w:
                lo = mid + 1
            else:
                hi = mid
        cut = max(0, lo - 1)
        return txt[:cut] + ell
    def _clamp255(x):
        return max(0, min(255, int(x)))
    def _lighten(rgb, amt=0.25):
        r, g, b = rgb
        return (
            _clamp255(r + (255 - r) * amt),
            _clamp255(g + (255 - g) * amt),
            _clamp255(b + (255 - b) * amt),
        )
    def _render_with_outline(font, txt, fill_rgb, outline_rgb, outline_px):
        """Render text with an outline by over-blitting the outline text around the fill."""
        if not txt:
            return None
        base = font.render(txt, True, fill_rgb)
        if outline_px <= 0:
            return base
        tw = base.get_width() + 2 * outline_px
        th = base.get_height() + 2 * outline_px
        surf = pygame.Surface((tw, th), pygame.SRCALPHA)
        out = font.render(txt, True, outline_rgb)
        offs = (-outline_px, 0, outline_px)
        for dx in offs:
            for dy in offs:
                if dx == 0 and dy == 0:
                    continue
                surf.blit(out, (dx + outline_px, dy + outline_px))
        surf.blit(base, (outline_px, outline_px))
        return surf
    # ---------- colors ----------
    YELLOW = (255, 220, 0)  # fill for Title/Time
    WHITE  = (0, 0, 0)  # outline for Title/Time
    beam_color = getattr(VIS, "bar_color", (220, 220, 220))  # HUD/FPS color
    title_fill    = YELLOW
    time_fill     = YELLOW
    title_outline = WHITE
    time_outline  = WHITE
    outline_px    = 2  # stroke thickness (Title), Time gets outline_px-1
    hud_color = (255, 220, 0)
    fps_color = (255, 220, 0)
    # compact color keys for cache invalidation
    def _ck(rgb): return f"{rgb[0]:03}{rgb[1]:03}{rgb[2]:03}"
    tkey_c = _ck(title_fill) + _ck(title_outline) + f"|o{outline_px}"
    ikey_c = _ck(time_fill)  + _ck(time_outline)  + f"|o{max(1, outline_px-1)}"
    hkey_c = _ck(hud_color)
    fkey_c = _ck(fps_color)
    # ---------- layout ----------
    x_center   = w // 2
    gap        = 8
    max_text_w = w - 40
    # ---------- build row surfaces ----------
    rows = []
    # TITLE (yellow fill + white outline)
    if show_title:
        title_txt = ""
        if current_track and (current_track.title or current_track.artist):
            if current_track.artist and current_track.title:
                title_txt = f"{current_track.artist} — {current_track.title}"
            elif current_track.title:
                title_txt = current_track.title
        if title_txt:
            t_ell = _ellipsize(title_font, title_txt, max_text_w)
            key = f"{t_ell}|w{w}|{tkey_c}"
            if C["title_key"] != key:
                C["title_key"] = key
                C["title_surf"] = _render_with_outline(
                    title_font, t_ell, title_fill, title_outline, outline_px
                )
            if C["title_surf"]:
                rows.append(C["title_surf"])
    # TIME (yellow fill + white outline)
    if show_time:
        time_txt = f"{format_time(pos_now)} / {format_time(dur_now)}"
        key = f"{time_txt}|{ikey_c}"
        if C["time_key"] != key:
            C["time_key"] = key
            C["time_surf"] = _render_with_outline(
                title_font, time_txt, time_fill, time_outline, max(1, outline_px - 1)
            )
        if C["time_surf"]:
            rows.append(C["time_surf"])
    # HUD
    # Normalize repeat_status first (ONE > ALL > OFF)
    if repeat_one:
        repeat_status = "ONE"
    elif repeat_all:
        repeat_status = "ALL"
    else:
        repeat_status = "OFF"
     # HUD
    # Normalize repeat_status first (ONE > ALL > OFF)
    if repeat_one:
        repeat_status = "ONE"
    elif repeat_all:
        repeat_status = "ALL"
    else:
        repeat_status = "OFF"
    if show_hud:
        hud_txt = (
            f"[S]huffle: {'ON' if shuffle else 'OFF'}  "
            f"[R]epeat: {repeat_status}  "
            f"[F] FakeFS  [V/F2] View: {v_mode+1}  [H] Help  [1..5] Presets"
        )
        base = hud_txt if font_small_sys.size(hud_txt)[0] <= max_text_w else _ellipsize(font_small_sys, hud_txt, max_text_w)
        key = f"{base}|{hkey_c}"
        if C["hud_key"] != key:
            C["hud_key"]  = key
            C["hud_surf"] = font_small_sys.render(base, True, hud_color)
        if C["hud_surf"]:
            rows.append(C["hud_surf"])
    # FPS + BPM + Flubber + Next Word
    if show_fps:
        fps_val = getattr(render_ui_and_text, "_last_fps", 60.0)
        bpm_txt = "--"
        flub_state = "reactive"
        next_word = "-"
        try:
            bpm_val = float(getattr(draw_visuals, "_bpm_ema", 0.0) or 0.0)
            bpm_txt = f"{bpm_val:5.1f}" if bpm_val > 0.0 else "--"
            t_now = pygame.time.get_ticks() / 1000.0
            shape_active = bool(getattr(draw_visuals, "_shape_active", False))
            if shape_active:
                shape_name = getattr(draw_visuals, "_shape_name", "") or "morph"
                t1 = float(getattr(draw_visuals, "_shape_t1", t_now))
                rem = max(0.0, t1 - t_now)
                flub_state = f"{shape_name} {rem:0.1f}s"
            else:
                m_times = list(getattr(draw_visuals, "_morph_times", []))
                m_used = set(getattr(draw_visuals, "_morph_used", set()))
                next_eta = None
                for i, mt in enumerate(m_times):
                    if i not in m_used and pos_now < float(mt):
                        next_eta = max(0.0, float(mt) - pos_now)
                        break
                flub_state = f"reactive +{next_eta:.0f}s" if next_eta is not None else "reactive"
            next_word = getattr(draw_visuals, "_current_word", "") or "-"
        except Exception:
            pass
        dbg = f"FPS {fps_val:4.1f} | BPM {bpm_txt} | Flub: {flub_state} | Next: {next_word}"
        base = dbg if font_small_sys.size(dbg)[0] <= max_text_w else _ellipsize(font_small_sys, dbg, max_text_w)
        key = f"{base}|{fkey_c}"
        if C["fps_key"] != key:
            C["fps_key"]  = key
            C["fps_surf"] = font_small_sys.render(base, True, fps_color)
        if C["fps_surf"]:
            rows.append(C["fps_surf"])
    # ---- ensure fit (uniform downscale) ----
    rows = rows[:4]
    n = len(rows)
    gap = 8
    y_stack_top = None
    y_stack_bottom = None
    if n:
        total_h = sum(s.get_height() for s in rows) + gap * (n - 1)
        top_pad = gap
        bot_pad = gap
        avail_h = max(1, h - top_pad - bot_pad)
        if total_h > avail_h:
            scale = (avail_h - gap * (n - 1)) / max(1, sum(s.get_height() for s in rows))
            scale = max(0.55, min(1.0, scale))
            if scale < 1.0:
                rows = [
                    pygame.transform.smoothscale(
                        s,
                        (max(1, int(s.get_width() * scale)),
                         max(1, int(s.get_height() * scale)))
                    )
                    for s in rows
                ]
                total_h = sum(s.get_height() for s in rows) + gap * (n - 1)
        # stack is bottom-anchored
        y_start = h - bot_pad - total_h
        y = y_start
        for s in rows:
            text_surf.blit(s, s.get_rect(centerx=x_center, y=y))
            y += s.get_height() + gap
        y_stack_top = y_start
        y_stack_bottom = y_start + total_h
    else:
        y_stack_top = h - gap
        y_stack_bottom = h - gap
    # ---------- VOL & TOAST (VOL styled like TITLE) ----------
    now = _time.time()
    vol_active = (now - last_volume_popup_t) < UI.volume_popup_sec
    if vol_active:
        vol_txt = f"VOL {int(volume * 100)}%"
        vkey = f"{vol_txt}|{_ck(title_fill)}{_ck(title_outline)}|o{outline_px}"
        if C.get("vol_key") != vkey:
            C["vol_key"]  = vkey
            C["vol_surf"] = _render_with_outline(
                vol_font_sys, vol_txt, title_fill, title_outline, outline_px
            )
        vs = C.get("vol_surf")
        if vs:
            # draw ABOVE the 4-line stack (use y_stack_top computed above)
            vy = max(gap, y_stack_top - vs.get_height() - gap)
            text_surf.blit(vs, vs.get_rect(centerx=x_center, y=vy))
            y_stack_top = vy  # so toast stacks above volume if present
    if now < toast_until and toast_msg:
        tmsg = _ellipsize(vol_font_sys, toast_msg, max_text_w)
        toast_surf = vol_font_sys.render(tmsg, True, hud_color)
        ty = max(gap, y_stack_top - toast_surf.get_height() - 4)
        text_surf.blit(toast_surf, toast_surf.get_rect(centerx=x_center, y=ty))
        y_stack_top = ty
    return vol_active, int(y_stack_bottom + gap)
def lerp(a,b,t): return a+(b-a)*max(0.0,min(1.0,t))
def lerp_color(c1,c2,t): return (int(lerp(c1[0],c2[0],t)),int(lerp(c1[1],c2[1],t)),int(lerp(c1[2],c2[2],t)))
def load_system(size:int, bold=True):
    name = pygame.font.match_font('segoe ui,segoeui,arial,tahoma,verdana,calibri,dejavusans', bold=bold)
    return pygame.font.Font(name, size) if name else pygame.font.SysFont(None, size, bold=bold)
def load_cyber(size:int):
    try: return pygame.font.Font(str(FONT_PATH), size)
    except Exception: return load_system(size, bold=True)
def parse_args():
    ap = argparse.ArgumentParser(description="Visual Music Player v0.1")
    ap.add_argument("--music-dir", default=MUSIC_DIR)
    ap.add_argument("--backgrounds", default=BG_DIR)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--shuffle", action="store_true", help="Enable randomized order after the first track")
    ap.add_argument("--repeat-all", action="store_true", help="Repeat the whole list")
    ap.add_argument("--no-fft", action="store_true")
    ap.add_argument("--viz-lowcpu", action="store_true")
    ap.add_argument("--ext", default=",".join(SCAN_EXTS))
    ap.add_argument("--ignore", default="")
    ap.add_argument("--no-tags", action="store_true")
    ap.add_argument("--cache-cap", type=int, default=16, help="Max analyzed tracks in memory")
    return ap.parse_args()
def win_toggle_topmost():
    try:
        if platform.system() != "Windows":
            log.debug("Topmost toggle ignored (not Windows).")
            return
        wm_info = pygame.display.get_wm_info() if pygame.display.get_init() else {}
        hwnd = wm_info.get("window")
        if not hwnd:
            log.debug("Topmost toggle failed: no window handle.")
            return
        u32 = ctypes.windll.user32
        GWL_EXSTYLE   = -20
        WS_EX_TOPMOST = 0x00000008
        SWP_NOMOVE    = 0x0002
        SWP_NOSIZE    = 0x0001
        SWP_SHOWWINDOW= 0x0040
        HWND_TOPMOST     = -1
        HWND_NOTOPMOST   = -2
        is_top = bool(u32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOPMOST)
        target_hwnd = HWND_NOTOPMOST if is_top else HWND_TOPMOST
        ok = u32.SetWindowPos(hwnd, target_hwnd, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
        if not ok:
            raise ctypes.WinError()
        log.debug("Topmost toggled -> %s", "OFF" if is_top else "ON")
    except Exception as e:
        log.debug("Topmost toggle failed: %s", e)
def win_drag_window():
    try:
        if platform.system() != "Windows":
            return
        wm_info = pygame.display.get_wm_info() if pygame.display.get_init() else {}
        hwnd = wm_info.get("window")
        if not hwnd:
            return
        u32 = ctypes.windll.user32
        WM_NCLBUTTONDOWN = 0x00A1
        HTCAPTION        = 0x0002
        u32.ReleaseCapture()
        u32.SendMessageW(hwnd, WM_NCLBUTTONDOWN, HTCAPTION, 0)
    except Exception as e:
        log.debug("Drag failed: %s", e)
def get_desktop_size() -> Tuple[int, int]:
    try:
        info = pygame.display.Info()
        return (max(640, int(info.current_w)), max(480, int(info.current_h)))
    except Exception:
        return (1920, 1080)
def _fbm1(x):
    """Cheap 1D fractal noise (no deps). Returns ~[-1, 1]."""
    return (
        np.sin(x) * 0.60 +
        np.sin(2.0 * x + 1.7) * 0.28 +
        np.sin(4.0 * x + 0.9) * 0.12
    )
def _fbm2(x, y):
    val = 0.0
    amp = 1.0
    freq = 1.0
    for i in range(3):
        val += amp * math.sin(freq * x + 0.3 * i) * math.sin(freq * y + 0.7 * i)
        amp *= 0.5
        freq *= 2.0
    return val * 0.7
class BPMState:
    def __init__(self, max_beats: int = 12):
        self.beat_times = deque(maxlen=max_beats)
        self.bpm_ema = 0.0
        self.last_beat_time = 0.0
        self.confidence = 0.0
def update_bpm_from_flash(
    t: float,
    cur_flash: float,
    prev_flash: float,
    bass_env: float,
    *,
    state: BPMState,
    min_bpm: float = 55.0,
    max_bpm: float = 220.0,
    min_abs_env: float = 0.12,
    beat_min_gap: Optional[float] = None,
    decay_after_sec: float = 2.5,
    decay_per_sec: float = 0.4,
    flash_threshold_low: float = 0.25,
    flash_threshold_high: float = 0.75,
) -> tuple[float, float]:
    if beat_min_gap is None:
        try:
            gap = float(getattr(globals().get('AUDIO', None), 'beat_min_gap', 0.12))
        except (AttributeError, TypeError, ValueError):
            gap = 0.12
        beat_min_gap = float(np.clip(gap, 0.08, 0.30))
    rising = (prev_flash < flash_threshold_low) and (cur_flash > flash_threshold_high)
    energetic_enough = (bass_env >= min_abs_env)
    strong_rising = (cur_flash - prev_flash) > 0.3
    if rising and energetic_enough and strong_rising:
        if (not state.beat_times) or ((t - state.beat_times[-1]) >= beat_min_gap):
            state.beat_times.append(t)
            state.last_beat_time = t
            if len(state.beat_times) >= 3:
                intervals = np.diff(np.asarray(state.beat_times, dtype=np.float32))
                min_interval = 60.0 / max_bpm
                max_interval = 60.0 / min_bpm
                valid = intervals[(intervals >= min_interval) & (intervals <= max_interval)]
                if len(valid) >= 2:
                    if len(valid) >= 3:
                        q75, q25 = np.percentile(valid, [75, 25])
                        iqr = q75 - q25
                        if iqr > 0:
                            mask = (valid >= (q25 - 1.5*iqr)) & (valid <= (q75 + 1.5*iqr))
                            if np.any(mask):
                                valid = valid[mask]
                    tail = valid[-4:]
                    weights = np.exp(np.linspace(0, 1, len(tail)))
                    period = float(np.average(tail, weights=weights))
                    bpm_inst = 60.0 / max(period, 1e-6)
                    tail3 = valid[-3:] if len(valid) >= 3 else valid
                    den = max(1e-6, float(np.mean(tail3)))
                    consistency = 1.0 - float(np.std(tail3)) / den
                    consistency = float(np.clip(consistency, 0.1, 1.0))
                    alpha = 0.3 + 0.3 * consistency
                    if state.bpm_ema == 0.0:
                        state.bpm_ema = bpm_inst
                    else:
                        state.bpm_ema = (1 - alpha) * state.bpm_ema + alpha * bpm_inst
                    state.confidence = float(min(1.0, consistency * len(state.beat_times) / 8.0))
    if state.last_beat_time > 0.0:
        time_since = t - state.last_beat_time
        if time_since > decay_after_sec:
            decay_factor = (1.0 - decay_per_sec * (1.0 - state.confidence * 0.5))
            state.bpm_ema *= decay_factor ** max(0.0, (time_since - decay_after_sec))
            state.confidence *= 0.95
            if state.bpm_ema < 1.0:
                state.bpm_ema = 0.0
                state.confidence = 0.0
    return float(np.clip(state.bpm_ema, 0.0, max_bpm)), float(state.confidence)
def calculate_beam_rotation_speed(
    bpm_est: float,
    confidence: float = 1.0,
    omega_min: float = 0.02,
    omega_max: float = 1.30,  # Zvýšená maximálna rýchlosť
    bpm_range: tuple[float, float] = (60.0, 180.0),
    confidence_influence: float = 0.1  # Znížený vplyv dôvery (viac responzívne)
) -> float:
    bpm_lo, bpm_hi = bpm_range
    if bpm_est <= 1.0:
        return 0.0
    x = (bpm_est - bpm_lo) / max(1e-6, (bpm_hi - bpm_lo))
    x = float(np.clip(x, 0.0, 1.0))
    x = x*x*(3 - 2*x)
    omega_base = omega_min + (omega_max - omega_min) * x
    confidence_factor = 1.0 - confidence_influence * (1.0 - float(np.clip(confidence, 0.0, 1.0)))
    return float(omega_base * confidence_factor)
from functools import lru_cache
def _npf32(x): 
    return np.asarray(x, dtype=np.float32)
@lru_cache(maxsize=32)
def _thetas_cached(N: int) -> np.ndarray:
    if N <= 0:
        return _npf32([])
    return _npf32(np.linspace(0.0, 2.0*np.pi, int(N), endpoint=False))
def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(1e-6, (edge1 - edge0)), 0.0, 1.0)
    return _npf32(t * t * (3.0 - 2.0 * t))
def _circ_smooth3(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    if arr.size == 0:
        return arr
    out = arr.astype(np.float32, copy=False)
    for _ in range(max(0, passes)):
        out = 0.5 * out + 0.25 * np.roll(out, 1) + 0.25 * np.roll(out, -1)
    return out
def _star_sharp(thetas: np.ndarray, k: int, p: float) -> np.ndarray:
    return _npf32(np.abs(np.cos(k * thetas)) ** p)
def _superellipse_r(theta: np.ndarray, n: float) -> np.ndarray:
    n = float(max(1e-3, n))
    c = np.abs(np.cos(theta)) ** n
    s = np.abs(np.sin(theta)) ** n
    denom = np.clip(c + s, 1e-6, None)
    return _npf32(denom ** (-1.0 / n))
def _regular_polygon_r(theta: np.ndarray, sides: int, eps: float = 1e-4) -> np.ndarray:
    n = max(3, int(sides))
    a = np.pi / n
    w = (theta + a) % (2.0 * a) - a
    denom = np.clip(np.cos(w), eps, None)
    return _npf32(np.cos(a) / denom)
SHAPE_MODELS = {
    # Základné kvety
    "flower": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * np.cos(6 * _thetas_cached(N)),
    # Prirodzené tvary
    "lotus": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * (
        0.5 * np.cos(5 * _thetas_cached(N)) + 0.3 * np.cos(10 * _thetas_cached(N) + 0.5)
    ),
    # Mechanické tvary - vylepšená verzia gear
    "gear": lambda N, intensity: (
        lambda thetas: (
            lambda teeth, angle_per_tooth: (
                lambda tooth_angles: 1.0 + (0.2 + intensity * 0.6) * 0.4 * (
                    _circ_smooth3(_npf32(np.where(
                        tooth_angles < angle_per_tooth * 0.4, 1.0,
                        np.where(tooth_angles < angle_per_tooth * 0.6, 0.6, 1.0)
                    )), passes=2) - 1.0
                )
            )(thetas % angle_per_tooth)
        )(12, 2.0 * np.pi / 12)
    )(_thetas_cached(N)),
    # Vlnové tvary
    "waves": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * (
        0.4 * np.sin(3 * _thetas_cached(N)) + 0.3 * np.sin(7 * _thetas_cached(N) + 1.2)
    ),
    # Geometrické tvary
    "diamond": lambda N, intensity: 0.9 + (0.2 + intensity * 0.6) * 0.35 * _superellipse_r(_thetas_cached(N), 1.0),
    "triangle": lambda N, intensity: 0.9 + (0.2 + intensity * 0.6) * 0.35 * _circ_smooth3(
        _regular_polygon_r(_thetas_cached(N), 3), passes=1
    ),
    "square": lambda N, intensity: 0.85 + (0.2 + intensity * 0.6) * 0.32 * _superellipse_r(_thetas_cached(N), 8.0),
    "hexagon": lambda N, intensity: 0.85 + (0.2 + intensity * 0.6) * 0.32 * _circ_smooth3(
        _regular_polygon_r(_thetas_cached(N), 6), passes=1
    ),
    "octagon": lambda N, intensity: 0.87 + (0.2 + intensity * 0.6) * 0.30 * _circ_smooth3(
        _regular_polygon_r(_thetas_cached(N), 8), passes=1
    ),
    # Kríž - opravená verzia
    "cross": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * 0.4 * np.maximum(
        smoothstep(np.abs(np.cos(2 * _thetas_cached(N))), 0.60, 0.85),
        smoothstep(np.abs(np.sin(2 * _thetas_cached(N))), 0.60, 0.85)
    ),
    # Organické tvary
    "blob": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * (
        0.4 * np.sin(2.3 * _thetas_cached(N) + 0.5) + 
        0.3 * np.sin(4.7 * _thetas_cached(N) + 1.2) + 
        0.2 * np.sin(7.1 * _thetas_cached(N) + 2.1)
    ),
    # Špirála - opravená verzia
    "spiral": lambda N, intensity: 1.0 + 0.15 * (_thetas_cached(N) / (2.0 * np.pi)) + (
        0.2 + intensity * 0.6
    ) * 0.1 * np.sin(12 * _thetas_cached(N)),
    # Prírodné motívy
    "petal": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * 0.6 * np.abs(np.sin(_thetas_cached(N))) * (
        0.8 + 0.2 * np.cos(2 * _thetas_cached(N))
    ),
    "burst": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * 0.6 * np.cos(16 * _thetas_cached(N)) * (
        0.5 + 0.5 * np.cos(_thetas_cached(N))
    ),
    # Komplexnejšie vlny
    "wave_complex": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * (
        0.3 * np.sin(2 * _thetas_cached(N)) + 
        0.2 * np.sin(5 * _thetas_cached(N) + 0.8) + 
        0.15 * np.sin(9 * _thetas_cached(N) + 1.5)
    ),
    # Špeciálne tvary
    "clover": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * 0.6 * np.abs(
        np.cos(2 * _thetas_cached(N)) * np.sin(2 * _thetas_cached(N))
    ),
    "donut": lambda N, intensity: 1.2 + (0.2 + intensity * 0.6) * 0.3 * (
        0.3 * np.sin(8 * _thetas_cached(N) + 0.5) + 
        0.2 * np.sin(12 * _thetas_cached(N) + 1.2)
    ),
    "sun": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * (
        (np.cos(24 * _thetas_cached(N)) ** 2) * 0.4 + 
        0.2 * np.sin(3 * _thetas_cached(N) + 0.3)
    ),
    # Organické tvary s evolúciou
    "leaf": lambda N, intensity: 0.8 + (0.2 + intensity * 0.6) * 0.5 * (
        np.abs(np.sin(_thetas_cached(N))) * (1.0 - 0.3 * _thetas_cached(N) / (2.0 * np.pi)) + 
        0.1 * np.sin(8 * _thetas_cached(N))
    ),
    "shell": lambda N, intensity: 0.9 + (0.2 + intensity * 0.6) * 0.4 * (
        np.exp(-0.3 * _thetas_cached(N) / (2.0 * np.pi)) + 
        0.3 * np.sin(10 * _thetas_cached(N))
    ),
    # Energetické tvary
    "lightning": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * 0.5 * np.sin(8 * _thetas_cached(N)) * np.abs(
        np.cos(_thetas_cached(N))
    ),
    "snowflake": lambda N, intensity: 0.9 + (0.2 + intensity * 0.6) * 0.4 * (
        (np.cos(6 * _thetas_cached(N)) ** 2) + 
        0.3 * np.cos(18 * _thetas_cached(N)) * np.abs(np.cos(6 * _thetas_cached(N)))
    ),
    # Predvolený tvar
    "default": lambda N, intensity: 1.0 + (0.2 + intensity * 0.6) * (
        0.25 * np.sin(3.1 * _thetas_cached(N) + 0.2) + 
        0.20 * np.sin(5.3 * _thetas_cached(N) + 1.3)
    )
}
def shape_profile(name: str, N: int, intensity: float = 0.6) -> np.ndarray:
    thetas = _thetas_cached(int(N))
    if thetas.size == 0:
        return _npf32([])
    # Unified amplitude and "crispness" from intensity
    t = float(np.clip(intensity, 0.0, 1.0))
    amp = 0.20 + t * 0.50       # 0.20 .. 0.70
    crisp = 1.3 + 2.0 * t       # used by star sharpness / polygon softness
    smooth_passes = 1 if t < 0.5 else 2
    # Base = circle
    prof = np.ones_like(thetas, dtype=np.float32)
    if name == "circle":
        prof = 1.0 + 0.0 * thetas
    elif name == "blob":
        # Soft organic wobble with controlled harmonics
        base = (0.45*np.sin(2.2*thetas + 0.3) +
                0.30*np.sin(3.7*thetas + 1.1) +
                0.20*np.sin(5.1*thetas + 2.0))
        prof = 1.0 + amp * _npf32(base)
    elif name == "flower6":
        # Classic rosette, smooth
        base = np.cos(6 * thetas)
        prof = 1.0 + amp * 0.85 * _npf32(base)
    elif name == "star":
        # Sharp but pleasant 5-point star (p grows with intensity)
        p = 1.6 + 1.6 * t
        base = 2.0 * _star_sharp(thetas, 5, p) - 1.0
        prof = 1.0 + amp * 0.95 * _npf32(base)
    elif name == "triangle":
        # Exact polygon radius, then slight circular blur
        r = _regular_polygon_r(thetas, 3)
        prof = 0.92 + amp * 0.35 * _circ_smooth3(r, passes=1)
    elif name == "square":
        # Superellipse as a rounded square (crisp controls squareness)
        r = _superellipse_r(thetas, n=6.0 + 4.0*t)  # 6..10
        prof = 0.90 + amp * 0.34 * _circ_smooth3(r, passes=1)
    elif name == "hexagon":
        r = _regular_polygon_r(thetas, 6)
        prof = 0.90 + amp * 0.32 * _circ_smooth3(r, passes=1)
    elif name == "heart":
        # Stable cardioid-style heart (always positive)
        r = 1.0 - 0.82*np.sin(thetas)
        r += 0.16*np.sin(thetas) * np.sqrt(np.abs(np.cos(thetas))) / 1.15
        prof = 0.82 + amp * 0.52 * _npf32(r)
    else:
        # Fallback: gentle blob
        base = 0.30*np.sin(3.1*thetas + 0.2) + 0.22*np.sin(5.3*thetas + 1.3)
        prof = 1.0 + amp * _npf32(base)
    # Global softening for niceness (keeps corners from aliasing)
    if smooth_passes > 0:
        prof = _circ_smooth3(_npf32(prof), passes=smooth_passes)
    # Robust normalization & guards
    prof = np.nan_to_num(prof, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32, copy=False)
    m = float(np.mean(prof))
    if m > 1e-6:
        prof = (prof / m).astype(np.float32)
    prof = np.maximum(prof, 0.10).astype(np.float32)  # strictly positive for renderer safety
    return prof
def glyph_profile(
    ch: str,
    N: int,
    font: Optional[pygame.font.Font] = None,
    size_px: int = 512,            # tvoje vyššie rozlíšenie
    pad_frac: float = 0.10,        # tvoj menší padding
    smooth_passes: int = 3,        # tvoj vyšší smoothing
    *,
    antialiasing: bool = True,
    edge_detection_threshold: int = 64,
    subpixel_precision: bool = True,
) -> np.ndarray:
    if not ch or len(ch) != 1:
        return np.ones(int(N), np.float32)
    font_id = 0 if font is None else id(font)
    key = (
        ord(ch.upper()), int(N), int(size_px), int(smooth_passes),
        int(edge_detection_threshold), bool(antialiasing), bool(subpixel_precision),
        font_id
    )
    if key in _glyph_cache:
        _glyph_cache_access.move_to_end(key)
        return _glyph_cache[key].copy()
    # Bezpečný výber fontu
    if font is None:
        try:
            font = pygame.font.Font(str(FONT_PATH), int(size_px * 0.85))
        except Exception:
            font = pygame.font.SysFont("arial", int(size_px * 0.85), bold=True)
    # --- Interný supersampling ×2 + gamma-korektný downscale (bez zmeny API) ---
    SS = 2
    big_px = size_px * SS
    box_big = int(big_px * (1.0 - 2 * pad_frac))
    gamma = 2.2
    surf_big = pygame.Surface((big_px, big_px), pygame.SRCALPHA)
    surf_big.fill((0, 0, 0, 0))
    ch_up = ch.upper()
    # Pokus o kvalitnejší render (pygame.freetype), fallback na pygame.font
    glyph_surf = None
    try:
        import pygame.freetype as ft
        # Skús odvodiť cestu k fontu; ak to nejde, free type použije default
        path = getattr(font, "name", None)
        ft_font = ft.Font(path or None, int(big_px * 0.70))
        ft_font.pad = True
        ft_font.kerning = True
        ft_font.antialiased = True
        ft_font.strong = True
        glyph_surf, _ = ft_font.render(ch_up, fgcolor=(255, 255, 255, 255), bgcolor=None)
    except Exception:
        pass
    if glyph_surf is None:
        # klasický pygame.font → najprv metriky, potom AA render a scale do boxu
        base_tmp = font.render(ch_up, True, (255, 255, 255))
        gw, gh = base_tmp.get_width(), base_tmp.get_height()
        if gw == 0 or gh == 0:
            return np.ones(int(N), np.float32)
        scale = min(box_big / max(1, gw), box_big / max(1, gh))
        new_w = max(1, int(gw * scale))
        new_h = max(1, int(gh * scale))
        base = font.render(ch_up, antialiasing, (255, 255, 255))
        try:
            glyph_surf = pygame.transform.smoothscale(base, (new_w, new_h))
        except Exception:
            glyph_surf = pygame.transform.scale(base, (new_w, new_h))
    # Presné centrovanie podľa reálneho bbox
    try:
        rect = glyph_surf.get_bounding_rect()
        glyph_crop = glyph_surf.subsurface(rect).copy()
    except Exception:
        glyph_crop = glyph_surf
    gw2, gh2 = glyph_crop.get_width(), glyph_crop.get_height()
    if gw2 == 0 or gh2 == 0:
        return np.ones(int(N), np.float32)
    # Ak presahuje box, prispôsob do boxu
    if gw2 > box_big or gh2 > box_big:
        scale = min(box_big / max(1, gw2), box_big / max(1, gh2))
        nw = max(1, int(gw2 * scale)); nh = max(1, int(gh2 * scale))
        try:
            glyph_crop = pygame.transform.smoothscale(glyph_crop, (nw, nh))
        except Exception:
            glyph_crop = pygame.transform.scale(glyph_crop, (nw, nh))
        gw2, gh2 = glyph_crop.get_width(), glyph_crop.get_height()
    rect2 = glyph_crop.get_rect(center=(big_px // 2, big_px // 2))
    surf_big.blit(glyph_crop, rect2)
    # Gamma→linear, downscale, linear→gamma (len pre alfa)
    a_big = pygame.surfarray.array_alpha(surf_big).astype(np.float32) / 255.0
    a_lin = np.where(a_big > 0.0, np.power(a_big, gamma), 0.0)
    tmp_big = pygame.Surface((big_px, big_px), pygame.SRCALPHA)
    pygame.surfarray.pixels_alpha(tmp_big)[:, :] = (a_lin * 255.0).clip(0, 255).astype(np.uint8)
    try:
        tmp_small = pygame.transform.smoothscale(tmp_big, (size_px, size_px))
    except Exception:
        tmp_small = pygame.transform.scale(tmp_big, (size_px, size_px))
    alpha = pygame.surfarray.array_alpha(tmp_small).astype(np.float32) / 255.0
    alpha = np.where(alpha > 0.0, np.power(alpha, 1.0 / gamma), 0.0)
    alpha = (alpha * 255.0).astype(np.float32)
    # Orientácia (W,H)
    if alpha.shape[0] != size_px and alpha.shape[1] == size_px:
        alpha = alpha.T
    W, H = alpha.shape
    # Jemný 1D blur (horizontálne aj vertikálne) pred hrano-detekciou
    kern = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    a = alpha
    a = np.pad(a, ((1, 1), (0, 0)), mode="edge")
    a = (kern[0] * a[:-2, :] + kern[1] * a[1:-1, :] + kern[2] * a[2:, :])
    a = np.pad(a, ((0, 0), (1, 1)), mode="edge")
    a = (kern[0] * a[:, :-2] + kern[1] * a[:, 1:-1] + kern[2] * a[:, 2:])
    # Radial profil
    cx = size_px // 2
    cy = size_px // 2
    rmax = int(min(cx, cy) - 2)
    thetas = _thetas_cached(int(N))
    prof = np.empty(int(N), np.float32)
    thr = float(edge_detection_threshold)
    for i, th in enumerate(thetas):
        ct = math.cos(th); st = math.sin(th)
        found = 0.0
        if subpixel_precision:
            v_prev = 0.0
            for r in range(rmax, 1, -1):
                xf = cx + r * ct; yf = cy + r * st
                x0 = int(xf); y0 = int(yf)
                x1 = x0 + 1;  y1 = y0 + 1
                if 0 <= x0 < W-1 and 0 <= y0 < H-1:
                    wx = xf - x0; wy = yf - y0
                    v = (
                        a[x0, y0] * (1-wx) * (1-wy) +
                        a[x1, y0] * wx     * (1-wy) +
                        a[x0, y1] * (1-wx) * wy     +
                        a[x1, y1] * wx     * wy
                    )
                    if v >= thr and v_prev < thr:
                        denom = max(1e-6, (v - v_prev))
                        t = (thr - v_prev) / denom  # 0..1 medzi (r-1, r)
                        found = r - (1.0 - float(np.clip(t, 0.0, 1.0)))
                        break
                    v_prev = v
        else:
            for r in range(rmax, 1, -1):
                x = int(cx + r * ct); y = int(cy + r * st)
                if 0 <= x < W and 0 <= y < H and a[x, y] >= thr:
                    found = float(r)
                    break
        prof[i] = (found / float(rmax)) if found > 0.0 else 0.20
    # Vyhladenie po kružnici
    prof = _circ_smooth3(prof, passes=max(0, smooth_passes))
    # Normalizácia + minimálne dno
    m = float(np.mean(prof)) if prof.size else 1.0
    s = float(np.std(prof))  if prof.size else 0.0
    if m > 1e-6:
        prof = (prof / m).astype(np.float32)
    min_thr = max(0.10, 0.50 - 0.30 * s)
    prof = np.maximum(prof.astype(np.float32, copy=False), np.float32(min_thr))
    _glyph_cache[key] = prof.copy()
    _glyph_cache_access[key] = True
    if len(_glyph_cache) > _glyph_cache_max_size:
        for k in list(_glyph_cache.keys())[:64]:
            _glyph_cache.pop(k, None)
            _glyph_cache_access.pop(k, None)
    return prof
def clear_glyph_cache():
    _glyph_cache.clear()
    _glyph_cache_access.clear()
def get_glyph_cache_stats() -> Dict[str, float]:
    return {
        "entries": float(len(_glyph_cache)),
        "memory_mb": float(sum(arr.nbytes for arr in _glyph_cache.values()) / (1024*1024))
    }
class CyberpunkLexicon:
    def __init__(self, tokens=None, uppercase=True, dedupe=True):
        self.uppercase = bool(uppercase)
        self.tokens = []
        if tokens:
            self.set_tokens(tokens, dedupe=dedupe)
    def set_tokens(self, tokens, dedupe=True):
        toks = [str(t).strip() for t in tokens if str(t).strip()]
        if self.uppercase:
            toks = [t.upper() for t in toks]
        if dedupe:
            seen, out = set(), []
            for t in toks:
                if t not in seen:
                    seen.add(t); out.append(t)
            toks = out
        self.tokens = toks
        return self
    def set_tokens_from_str(self, s, separators=None, dedupe=True):
        if separators is None:
            separators = [",", ";", "|", "", "\t"]
        parts = [s]
        for sep in separators:
            parts = sum((frag.split(sep) for frag in parts), [])
        return self.set_tokens(parts, dedupe=dedupe)
    def word(self, rng=None) -> str:
        import random as _r
        r = rng if rng is not None else _r
        if not self.tokens:
            return "NEON"
        return r.choice(self.tokens)
LEX = CyberpunkLexicon().set_tokens([
    # Artificial intelligence / machines
    "AI", "BOT", "DRONE", "ROBOT", "CORE", "NODE", "DATA", "CODE",
    # Hacking / networks
    "HACK", "CRACK", "NET", "GRID", "WORM", "PORT", "LINK", "BYTE", "CHIP",
    # Cyber aesthetics
    "NEON", "CYBER", "WIRE", "PULSE", "SIGNAL",
    # Augmentation / transhumanism
    "MOD", "GENE", "DNA", "CELL", "BONE",
    # Futuristic science
    "NANO", "ION",
    # Corporate dystopia
    "CORP", "ZONE", "SLUM",
    # Existential / mind
    "MIND", "SOUL", "GHOST"
])
def _precompute_masks():
    """Predpočíta masky pre FFT"""
    global _bass_mask_f, _voice_mask_f
    if _bass_mask_f is None:
        freqs_bass = np.fft.rfftfreq(VIS.bass_fft, 1/AUDIO.target_sr)
        _bass_mask_f = smooth_mask(freqs_bass, AUDIO.bass_low_hz, AUDIO.bass_high_hz, 0.25)
    if _voice_mask_f is None:
        freqs_voice = np.fft.rfftfreq(VIS.fft_size, 1/AUDIO.target_sr)
        _voice_mask_f = smooth_mask(freqs_voice, AUDIO.voice_low_hz, AUDIO.voice_high_hz, 0.20)
def smooth_mask(freqs, low, high, roll=0.15):
    m = np.zeros_like(freqs, dtype=np.float32)
    band = (freqs>=low) & (freqs<=high)
    m[band] = 1.0
    bw = high-low
    r = max(1.0, roll*bw)
    left = (freqs>=low-r) & (freqs<low)
    right= (freqs>high) & (freqs<=high+r)
    if left.any():
        x = (freqs[left]- (low-r))/r
        m[left] = 0.5*(1-np.cos(np.pi*x))
    if right.any():
        x = 1 - (freqs[right]-high)/r
        m[right] = 0.5*(1-np.cos(np.pi*x))
    return m
def _draw_beams(vis_surf, x0, y0, cur_angles, bands_in, bar_len_mul, bar_thick_mul, n_bands):
    """Optimalizované vykresľovanie lúčov"""
    step = 1
    for i in range(0, n_bands, step):
        v = float(bands_in[i])
        vv = np.clip(v * (0.85 + 0.30) * (0.9 + 0.3), 0.0, 1.0)
        L_raw = (4 + vv * int(min(vis_surf.get_width(), vis_surf.get_height()) * VIS.bar_max_len_frac) * (1.10*bar_len_mul))
        W_raw = (1 + vv * (VIS.bar_thickness * (1.0*bar_thick_mul)))
        L = int(max(1, L_raw))
        thickness = int(np.clip(W_raw, 1, VIS.bar_thickness * 2.1))
        ca = math.cos(cur_angles[i]); sa = math.sin(cur_angles[i])
        x1 = int(x0[i] + L * ca)
        y1 = int(y0[i] + L * sa)
        pygame.draw.line(vis_surf, (*VIS.bar_color, 255), (x0[i], y0[i]), (x1, y1), width=thickness)
def _draw_flubber_ring(vis_surf, outer_pts, inner_pts, n_points, fill_col, edge_col, fps):
    """Optimalizované vykresľovanie flubber kruhu"""
    for i in range(n_points):
        j = (i + 1) % n_points
        quad = [
            (int(outer_pts[i,0]), int(outer_pts[i,1])),
            (int(outer_pts[j,0]), int(outer_pts[j,1])),
            (int(inner_pts[j,0]), int(inner_pts[j,1])),
            (int(inner_pts[i,0]), int(inner_pts[i,1])),
        ]
        gfxdraw.filled_polygon(vis_surf, quad, (*fill_col, 255))
    outer_loop = [(int(outer_pts[i,0]), int(outer_pts[i,1])) for i in range(n_points)]
    inner_loop = [(int(inner_pts[i,0]), int(inner_pts[i,1])) for i in range(n_points)]
    if len(outer_loop) > 2 and (fps >= 50):
        pygame.draw.aalines(vis_surf, edge_col, True, outer_loop)
    if len(inner_loop) > 2 and (fps >= 50):
        pygame.draw.aalines(vis_surf, edge_col, True, inner_loop)
def _update_bpm_detection(t, cur_flash, prev_flash, bass_env, state):
    """Oddelená BPM detekcia"""
    return update_bpm_from_flash(
        t=t,
        cur_flash=cur_flash,
        prev_flash=prev_flash,
        bass_env=bass_env,
        state=state,
        beat_min_gap=AUDIO.beat_min_gap,
        min_abs_env=0.12,
    )
def draw_centered_track_info(screen, current_track, pos_now, dur_now, energy_color, show_title, show_time):
    """Vykreslí názov skladby a čas pod hlavným kruhom"""
    if not show_title or not show_time:
        return
    w, h = screen.get_size()
    cx, cy = w // 2, h // 2
    info_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    title_font = load_cyber(VIS.centered_info_font_size)
    time_font = load_system(VIS.centered_info_time_font_size, bold=True)
    title_text = ""
    if current_track and (current_track.title or current_track.artist):
        if current_track.artist and current_track.title:
            title_text = f"{current_track.artist} — {current_track.title}"
        elif current_track.title:
            title_text = current_track.title
    if title_text:
        title_surf = title_font.render(title_text, True, energy_color)
        title_rect = title_surf.get_rect(center=(cx, cy + VIS.centered_info_offset_y))
        info_surf.blit(title_surf, title_rect)
    time_text = f"{format_time(pos_now)} / {format_time(dur_now)}"
    time_surf = time_font.render(time_text, True, energy_color)
    time_rect = time_surf.get_rect(center=(cx, cy + VIS.centered_info_offset_y + 30))
    info_surf.blit(time_surf, time_rect)
    screen.blit(info_surf, (0, 0))
def draw_visuals(screen, vis_surf, state):
    # Predpočítať masky ak ešte neexistujú
    _precompute_masks()
    w, h = screen.get_size()
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * VIS.ring_radius_frac)
    max_bar_out = int(min(w, h) * VIS.bar_max_len_frac)
    rim_pad = 6
    t = pygame.time.get_ticks() / 1000.0
    fps = float(state.get("fps", VIS.fps_target))
    pos_s = float(state.get("pos", 0.0))
    dur_s = float(state.get("dur", 0.0))
    n_bands = int(getattr(VIS, "n_bands", 64))
    bands_in = state.get("bands", np.zeros(n_bands, np.float32))
    bands_in = np.clip(np.nan_to_num(np.asarray(bands_in, np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if bands_in.shape[0] != n_bands:
        bands_in = np.interp(np.linspace(0, 1, n_bands), np.linspace(0, 1, len(bands_in)), bands_in).astype(np.float32)
    bass_env = float(state.get("bass_env", 0.0))
    voice_env = float(state.get("voice_env", 0.0))
    cur_flash = float(state.get("flash", 0.0))
    frame_idx = int(state.get("frame_idx", 0))
    # Inicializácia stavu
    if not hasattr(draw_visuals, "_init"):
        draw_visuals._pos = np.array([0.0, 0.0], dtype=np.float32)
        draw_visuals._vel = np.array([0.0, 0.0], dtype=np.float32)
        draw_visuals._prev_flash = 0.0
        draw_visuals._rng = np.random.default_rng(1337)
        draw_visuals._ema = np.zeros(n_bands, dtype=np.float32)
        draw_visuals._bar_len_ema = np.zeros(n_bands, dtype=np.float32)
        draw_visuals._bar_w_ema = np.zeros(n_bands, dtype=np.float32)
        n_points = int(VIS.flub_points)
        draw_visuals._angles = _thetas_cached(n_points)
        draw_visuals._outer_pts = np.zeros((n_points, 2), dtype=np.float32)
        draw_visuals._inner_pts = np.zeros((n_points, 2), dtype=np.float32)
        draw_visuals._prev_outer = np.zeros((n_points, 2), dtype=np.float32)
        draw_visuals._prev_inner = np.zeros((n_points, 2), dtype=np.float32)
        draw_visuals._have_prev = False
        draw_visuals._ripples = []
        draw_visuals._satellites = []
        draw_visuals._body_ripple_active = False
        draw_visuals._body_ripple_phase = 0.0
        draw_visuals._current_mood = "neutral"
        # Organic texture
        ns = 128
        noise = pygame.Surface((ns, ns), pygame.SRCALPHA)
        noise.fill((255, 255, 255, 255))
        rng_local = np.random.default_rng(2024)
        for _ in range(800):
            x = int(rng_local.integers(0, ns)); y = int(rng_local.integers(0, ns))
            r = int(rng_local.integers(1, 3)); col = int(rng_local.integers(200, 245))
            gfxdraw.filled_circle(noise, x, y, r, (col, col, col, 255))
        draw_visuals._noise_small = noise
        draw_visuals._pulse_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        # BPM state
        draw_visuals._bpm_state = BPMState(max_beats=12)
        draw_visuals._bpm_ema = 0.0
        draw_visuals._bpm_conf = 0.0
        # Modes and shapes
        draw_visuals._modes = [
            {"k":8.0,"c":4.8,"j":0.0015,"w":0.8,"bl":0.9,"bt":1.1,"sx":1.02,"sy":1.05},
            {"k":8.0,"c":3.8,"j":0.0025,"w":1.00,"bl":1.05,"bt":1.25,"sx":1.03,"sy":1.08},
            {"k":7.5,"c":3.0,"j":0.0035,"w":1.25,"bl":1.20,"bt":1.45,"sx":1.05,"sy":1.12},
            {"k":7.0,"c":2.4,"j":0.0050,"w":1.55,"bl":1.35,"bt":1.70,"sx":1.06,"sy":1.18},
            {"k":6.8,"c":1.8,"j":0.0075,"w":1.85,"bl":1.55,"bt":1.95,"sx":1.08,"sy":1.25},
        ]
        draw_visuals._bpm_ranges = [(0,85),(80,115),(110,140),(135,170),(165,1e9)]
        draw_visuals._mode_idx = 0
        draw_visuals._mode_target = 0
        draw_visuals._mode_blend = 1.0
        draw_visuals._mode_tstart = 0.0
        draw_visuals._mode_tdur = 1.2
        draw_visuals._mode_from = draw_visuals._modes[0].copy()
        draw_visuals._mode_to = draw_visuals._modes[0].copy()
        draw_visuals._base_angles = np.linspace(0, 2*np.pi, n_bands, endpoint=False).astype(np.float32)
        draw_visuals._rot_phase = 0.0
        # Shape/morph state
        draw_visuals._shapes_points = int(getattr(VIS, "flub_points", 128))
        draw_visuals._shape_active = False
        # --- Shape morphing: curated, nicer, and less frequent -------------------
        draw_visuals._shape_profile = None
        draw_visuals._shape_initial = "circle"
        draw_visuals._shape_name = draw_visuals._shape_initial
        draw_visuals._shape_t0 = 0.0
        draw_visuals._shape_t1 = 0.0
        draw_visuals._shape_count = 0
        # Hold times to avoid too-frequent morphs (seconds)
        draw_visuals._shape_hold_min_s = 6.0
        draw_visuals._shape_hold_max_s = 14.0
        draw_visuals._shape_no_immediate_backtrack = True
        draw_visuals._shape_recent = []  # keep last few names; trim to 3 entries when updating
        # Minimal, clean set of recognizable profiles (matches shape_profile)
        draw_visuals._shape_all = [
            "circle", "blob", "flower6", "star",
            "triangle", "square", "hexagon", "heart",
        ]
        # Similarity map (to avoid switching to a near-identical look right away)
        draw_visuals._shape_similar = {
            "circle":   {"blob", "square", "hexagon"},
            "blob":     {"circle", "flower", "heart"},
            "flower":  {"heart", "blob", "star"},
            "star":     {"flower6", "triangle", "heart"},
            "triangle": {"star", "square"},
            "square":   {"hexagon", "triangle", "circle"},
            "hexagon":  {"square", "circle"},
            "heart":    {"flower6", "blob", "star"},
        }
        draw_visuals._shape_pool = []
        draw_visuals._shape_last = None
        draw_visuals._morph_times = []
        draw_visuals._morph_used = set()
        draw_visuals._morph_fade = 0.35
        draw_visuals._armed_threshold = 8.0 + np.random.uniform(-2.0, +2.0)
        draw_visuals._long_done = False
        draw_visuals._shorts_before_long = int(draw_visuals._rng.integers(1, 3))
        draw_visuals._freeze = {"base_r": None, "thickness": None, "sx": 1.0, "sy": 1.0, "profile": None}
        draw_visuals._last_pos = -1.0
        draw_visuals._noise_cache = {}
        # Letter mode
        draw_visuals._letter_enabled = True
        draw_visuals._letter_rng = np.random.default_rng(424242)
        draw_visuals._letter_font = load_cyber(260)
        draw_visuals._letter_timeline = []
        draw_visuals._letter_seg_idx = -1
        draw_visuals._letter_center = (0, 0)
        draw_visuals._letter_freeze = {"base_r": None, "thickness": None, "sx": 1.0, "sy": 1.0}
        draw_visuals._organic_phase = 0.0
        draw_visuals._pulse_intensity = 0.0
        draw_visuals._morph_progress = 0.0
        draw_visuals._current_word = ""
        if not hasattr(draw_visuals, "_initialized"):draw_visuals._init = True
    # Reset on seek back/new track
    if pos_s < draw_visuals._last_pos - 1.0:
        draw_visuals._shape_active = False
        draw_visuals._shape_profile = None
        draw_visuals._shape_name = ""
        draw_visuals._shape_count = 0
        draw_visuals._morph_times = []
        draw_visuals._morph_used = set()
        draw_visuals._freeze = {"base_r": None, "thickness": None, "sx": 1.0, "sy": 1.0, "profile": None}
        draw_visuals._armed_threshold = 12.0 + np.random.uniform(-3.0, +3.0)
        draw_visuals._long_done = False
        draw_visuals._shorts_before_long = int(draw_visuals._rng.integers(2, 5))
        draw_visuals._shape_pool = []
        draw_visuals._shape_last = None
        draw_visuals._letter_timeline = []
        draw_visuals._letter_seg_idx = -1
        draw_visuals._letter_center = (0, 0)
        draw_visuals._letter_freeze = {"base_r": None, "thickness": None, "sx": 1.0, "sy": 1.0}
        draw_visuals._organic_phase = 0.0
        draw_visuals._pulse_intensity = 0.0
        draw_visuals._morph_progress = 0.0
        draw_visuals._current_word = ""
        draw_visuals._satellites = []
        draw_visuals._body_ripple_active = False
        draw_visuals._body_ripple_phase = 0.0
    draw_visuals._last_pos = pos_s
    # BPM detection
    if not hasattr(draw_visuals, "_bpm_state"):
        draw_visuals._bpm_state = BPMState(max_beats=12)
    prev_flash = float(getattr(draw_visuals, "_prev_flash", 0.0))
    prev_last_beat = draw_visuals._bpm_state.last_beat_time
    bpm_est, bpm_conf = _update_bpm_detection(t, cur_flash, prev_flash, bass_env, draw_visuals._bpm_state)
    # Satelity pri bite
    if draw_visuals._bpm_state.last_beat_time != prev_last_beat:
        base_r_tmp = radius * 0.9
        draw_visuals._ripples.append({"r": base_r_tmp, "a": 200, "w": 2, "dr": 14.0, "da": 8})
        draw_visuals._ripples.append({"r": base_r_tmp*0.9, "a": 150, "w": 1, "dr": 18.0, "da": 10})
        num_satellites = draw_visuals._rng.integers(1, 4)
        for _ in range(num_satellites):
            satellite = {
                'angle': draw_visuals._rng.uniform(0.0, 2*np.pi),
                'distance': radius * draw_visuals._rng.uniform(1.2, 1.8),
                'size': draw_visuals._rng.integers(3, 8),
                'color': (218, 180, 80), 
                'orbit_speed': draw_visuals._rng.uniform(1.0, 3.0) * (bpm_est / 120.0)
            }
            draw_visuals._satellites.append(satellite)
    draw_visuals._prev_flash = cur_flash
    draw_visuals._bpm_ema = float(bpm_est)
    draw_visuals._bpm_conf = float(bpm_conf)
    # Mode blend
    target_idx = draw_visuals._mode_target
    for i, (lo, hi) in enumerate(draw_visuals._bpm_ranges):
        if lo <= bpm_est < hi:
            target_idx = i; break
    if target_idx != draw_visuals._mode_target:
        draw_visuals._mode_from = draw_visuals._modes[draw_visuals._mode_idx].copy()
        draw_visuals._mode_to = draw_visuals._modes[target_idx].copy()
        draw_visuals._mode_tstart = t
        draw_visuals._mode_blend = 0.0
        draw_visuals._mode_target = target_idx
    if draw_visuals._mode_blend < 1.0:
        x = (t - draw_visuals._mode_tstart) / max(0.001, draw_visuals._mode_tdur)
        b = min(1.0, x*x*(3-2*x))
        draw_visuals._mode_blend = b
        if b >= 0.999:
            draw_visuals._mode_blend = 1.0
            draw_visuals._mode_idx = draw_visuals._mode_target
            draw_visuals._mode_from = draw_visuals._modes[draw_visuals._mode_idx].copy()
    b = draw_visuals._mode_blend
    F, Tm = draw_visuals._mode_from, draw_visuals._mode_to
    k = (1-b)*F["k"] + b*Tm["k"]
    c = (1-b)*F["c"] + b*Tm["c"]
    jitter = (1-b)*F["j"] + b*Tm["j"]
    wobble_mul = (1-b)*F["w"] + b*Tm["w"]
    bar_len_mul = (1-b)*F["bl"] + b*Tm["bl"]
    bar_thick_mul = (1-b)*F["bt"] + b*Tm["bt"]
    squash_h = (1-b)*F["sx"] + b*Tm["sx"]
    squash_v = (1-b)*F["sy"] + b*Tm["sy"]
    # Physics of center
    # Physics of center (with NaN/inf guards)
        # Physics of center (ULTRA-ROBUST version)
    if (prev_flash < 0.3) and (cur_flash > 0.7):
        ang = draw_visuals._rng.uniform(0.0, 2*np.pi)
        dirv = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
        px = 0.07 * min(w, h) * (0.5 + 0.5 * bass_env)
        impulse = dirv * px
        # Guard against NaN/inf in impulse
        if not np.all(np.isfinite(impulse)):
            impulse = np.zeros_like(impulse)
        draw_visuals._vel += impulse
    dt = 1.0 / max(30.0, fps)
    # Calculate acceleration with clamping
    acc = -k * draw_visuals._pos - c * draw_visuals._vel
    # Clamp acceleration to prevent runaway values
    acc = np.clip(acc, -1000.0, 1000.0)
    if not np.all(np.isfinite(acc)):
        acc = np.zeros_like(acc)
    draw_visuals._vel += acc * dt
    if jitter > 0.0:
        noise = draw_visuals._rng.normal(0.0, jitter * (0.4 + 0.6 * bass_env), size=2)
        # Clamp and guard noise
        noise = np.clip(noise, -5.0, 5.0)
        if not np.all(np.isfinite(noise)):
            noise = np.zeros_like(noise)
        draw_visuals._vel += noise
    # Clamp velocity to prevent extreme speeds
    draw_visuals._vel = np.clip(draw_visuals._vel, -200.0, 200.0)
    if not np.all(np.isfinite(draw_visuals._vel)):
        draw_visuals._vel = np.zeros_like(draw_visuals._vel)
    draw_visuals._pos += draw_visuals._vel * dt
    # Hard clamp position to prevent overflow
    lim = 0.10 * min(w, h)
    draw_visuals._pos = np.clip(draw_visuals._pos, -lim, lim)
    if not np.all(np.isfinite(draw_visuals._pos)):
        draw_visuals._pos = np.zeros_like(draw_visuals._pos)
    # Final safe conversion to screen coordinates
    cx_off = cx + int(np.clip(draw_visuals._pos[0], -lim, lim))
    cy_off = cy + int(np.clip(draw_visuals._pos[1], -lim, lim)) 
    letter_prof = None
    letter_static = False
    # Morph planning
    if (dur_s > 0) and (not draw_visuals._morph_times):
        rng = draw_visuals._rng
        min_gap = 15
        margin = 10.0
        t_first = min(8.0, 0.18 * dur_s) + rng.uniform(-2.0, +2.0)
        n_target = max(3, min(8, int(dur_s // 18)))
        anchors = np.linspace(0.30*dur_s, 0.88*dur_s, max(0, n_target-1))
        jitter = rng.uniform(-5, +5, size=anchors.shape)
        candidates = [t_first] + list(anchors + jitter)
        cleaned = []
        for x in sorted(float(np.clip(c, margin, max(margin, dur_s - margin))) for c in candidates):
            if not cleaned or (x - cleaned[-1]) >= min_gap:
                cleaned.append(x)
        draw_visuals._morph_times = cleaned[:8]
    # Letter mode timeline
    if draw_visuals._letter_enabled and (dur_s > 0) and (not draw_visuals._letter_timeline):
        rng = draw_visuals._letter_rng
        n_words = int(rng.integers(1, 4))
        t_lo = 0.22 * dur_s
        t_hi = 0.90 * dur_s
        min_gap = 6.0
        def _pick_word():
            for _ in range(30):
                w = LEX.word(rng).upper()
                w = re.sub(r"[^A-Z]", "", w)
                if 3 <= len(w) <= 8:
                    return w
            return "NEON"
        starts = []
        tries = 0
        while len(starts) < n_words and tries < 200:
            tries += 1
            x = float(rng.uniform(t_lo, t_hi))
            if all(abs(x - y) >= min_gap for y in starts):
                starts.append(x)
        starts.sort()
        dwell_min, dwell_max = VIS.dwell_duration_min, VIS.dwell_duration_max
        morph_min, morph_max = VIS.morph_duration_min, VIS.morph_duration_max
        timeline = []
        the_N = int(getattr(VIS, "flub_points", 128))
        neutral = np.ones(the_N, np.float32)
        for t0 in starts:
            word = _pick_word()
            prev_prof = neutral
            t_cursor = float(t0)
            for j, ch in enumerate(word):
                profB = glyph_profile(ch, the_N, draw_visuals._letter_font)
                morph_dur = float(rng.uniform(morph_min, morph_max))
                timeline.append({
                    "t0": t_cursor, "t1": t_cursor + morph_dur,
                    "type": "morph",
                    "profA": prev_prof, "profB": profB,
                    "is_static": False,
                    "word": word, "char": ch, "j": j, "n": len(word),
                })
                t_cursor += morph_dur
                dwell_dur = float(rng.uniform(dwell_min, dwell_max))
                timeline.append({
                    "t0": t_cursor, "t1": t_cursor + dwell_dur,
                    "type": "dwell",
                    "profA": profB, "profB": profB,
                    "is_static": True,
                    "word": word, "char": ch, "j": j, "n": len(word),
                })
                t_cursor += dwell_dur
                prev_prof = profB
            t_cursor += float(rng.uniform(0.7, 1.4))
        draw_visuals._letter_timeline = timeline
        draw_visuals._letter_seg_idx = -1
        draw_visuals._current_word = ""
    # Letter mode active segment
    seg_idx = -1
    if draw_visuals._letter_enabled and draw_visuals._letter_timeline:
        tl = draw_visuals._letter_timeline
        for i, seg in enumerate(tl):
            if seg["t0"] <= pos_s < seg["t1"]:
                seg_idx = i
                if seg["type"] == "morph":
                    u = (pos_s - seg["t0"]) / max(1e-6, (seg["t1"] - seg["t0"]))
                    u = u*u*(3-2*u)
                    letter_prof = ((1.0 - u) * seg["profA"] + u * seg["profB"]).astype(np.float32)
                    letter_static = False
                else:
                    letter_prof = seg["profB"]
                    letter_static = True
                if seg["word"] != draw_visuals._current_word:
                    draw_visuals._current_word = seg["word"]
                break
        if seg_idx != draw_visuals._letter_seg_idx:
            draw_visuals._letter_seg_idx = seg_idx
            if seg_idx >= 0:
                seg = draw_visuals._letter_timeline[seg_idx]
                if seg["is_static"]:
                    draw_visuals._letter_center = (cx_off, cy_off)
                    bpm_scale = 0.5 + 0.5*min(1.0, bpm_est/140.0)
                    base_r_now = radius * VIS.flub_base_frac * 1.18 * bpm_scale
                    thickness_now = radius * 0.085
                    orient = 0.5 + 0.5 * math.sin(t * 0.5 * wobble_mul)
                    sx_now = squash_h * (1.0 + 0.02 * bass_env * (1.0 - orient))
                    sy_now = squash_v * (1.0 + 0.04 * bass_env * orient)
                    draw_visuals._letter_freeze.update({
                        "base_r": float(base_r_now),
                        "thickness": float(thickness_now),
                        "sx": float(sx_now),
                        "sy": float(sy_now),
                    })
    def _refill_shape_pool():
        pool = list(draw_visuals._shape_all)
        idx = draw_visuals._rng.permutation(len(pool))
        pool = [pool[i] for i in idx]
        if draw_visuals._shape_last and pool and pool[0] == draw_visuals._shape_last:
            pool.append(pool.pop(0))
        draw_visuals._shape_pool = pool
    def _pick_unique_shape() -> str:
        if not draw_visuals._shape_pool:
            _refill_shape_pool()
        name = draw_visuals._shape_pool.pop(0)
        sim = draw_visuals._shape_similar.get(draw_visuals._shape_last, set())
        if draw_visuals._shape_last and name in sim and draw_visuals._shape_pool:
            swap_idx = None
            for i, cand in enumerate(draw_visuals._shape_pool):
                if cand not in sim:
                    swap_idx = i; break
            if swap_idx is not None:
                alt = draw_visuals._shape_pool.pop(swap_idx)
                draw_visuals._shape_pool.insert(0, name)
                name = alt
        draw_visuals._shape_last = name
        return name
    def _start_morph():
        name = _pick_unique_shape()
        inten = float(draw_visuals._rng.uniform(0.4, 0.8))
        prof = shape_profile(name, int(draw_visuals._shapes_points), inten)
        remaining = max(0, len(draw_visuals._morph_times) - len(draw_visuals._morph_used))
        need_long = (not draw_visuals._long_done) and (
            draw_visuals._shape_count >= draw_visuals._shorts_before_long or
            remaining <= 1
        )
        if need_long:
            dur = float(draw_visuals._rng.uniform(7.0, 7.4))
            draw_visuals._long_done = True
        else:
            dur = float(draw_visuals._rng.uniform(2.0, 5.0))
        draw_visuals._shape_profile = prof
        draw_visuals._shape_name = name
        draw_visuals._shape_t0 = t
        draw_visuals._shape_t1 = t + dur
        draw_visuals._shape_active = True
        draw_visuals._shape_count += 1
        bpm_scale = 0.5 + 0.5*min(1.0, bpm_est/140.0)
        base_r_now = radius * VIS.flub_base_frac * 1.18 * bpm_scale
        thickness_now = radius * 0.08 * (0.85 + 0.60 * voice_env)
        orient = 0.5 + 0.5 * math.sin(t * 0.5 * wobble_mul)
        sx_now = squash_h * (1.0 + 0.04 * bass_env * (1.0 - orient))
        sy_now = squash_v * (1.0 + 0.10 * bass_env * orient)
        draw_visuals._freeze.update({
            "profile": prof,
            "base_r": float(base_r_now),
            "thickness": float(thickness_now),
            "sx": float(sx_now),
            "sy": float(sy_now),
        })
    if (draw_visuals._shape_count < 8) and (not draw_visuals._shape_active) and draw_visuals._morph_times:
        for idx, tt in enumerate(draw_visuals._morph_times):
            if idx in draw_visuals._morph_used:
                continue
            if pos_s >= float(tt):
                draw_visuals._morph_used.add(idx)
                _start_morph()
                break
    shape_w = 0.0
    if draw_visuals._shape_active and draw_visuals._shape_profile is not None:
        if t >= draw_visuals._shape_t1:
            draw_visuals._shape_active = False
            draw_visuals._shape_profile = None
            draw_visuals._freeze = {"base_r": None, "thickness": None, "sx": 1.0, "sy": 1.0, "profile": None}
        else:
            fin = min(1.0, (t - draw_visuals._shape_t0) / max(1e-3, draw_visuals._morph_fade))
            fout = min(1.0, (draw_visuals._shape_t1 - t) / max(1e-3, draw_visuals._morph_fade))
            shape_w = fin*fin*(3-2*fin) * fout*fout*(3-2*fout)
    draw_visuals._organic_phase = (draw_visuals._organic_phase + dt * VIS.organic_pulse_speed) % (2*np.pi)
    draw_visuals._pulse_intensity = 0.5 + 0.5 * math.sin(draw_visuals._organic_phase)
    if draw_visuals._body_ripple_active:
        draw_visuals._body_ripple_phase += 5.0 * dt
        if draw_visuals._body_ripple_phase > 2 * math.pi:
            draw_visuals._body_ripple_active = False
    omega = calculate_beam_rotation_speed(
        bpm_est=float(bpm_est),
        confidence=float(getattr(draw_visuals, "_bpm_conf", 1.0)),
        omega_min=0.02,
        omega_max=1.30,           # Musí zodpovedať omega_max v definícii funkcie
        bpm_range=(60.0, 180.0),
        confidence_influence=0.1  # Musí zodpovedať confidence_influence v definícii funkcie
    )
    draw_visuals._rot_phase = (draw_visuals._rot_phase + omega*dt) % (2*np.pi)
    cur_angles = (draw_visuals._base_angles + draw_visuals._rot_phase).astype(np.float32)
    x0 = (cx_off + (radius + rim_pad) * np.cos(cur_angles)).astype(np.int32)
    y0 = (cy_off + (radius + rim_pad) * np.sin(cur_angles)).astype(np.int32)
    # Draw beams
    # Draw beams WITH YELLOW PEAKS
    if not hasattr(draw_visuals, "_beam_peaks"):
        draw_visuals._beam_peaks = np.zeros(n_bands, dtype=np.float32)
        draw_visuals._peak_decay = 0.98  # Pomalé klesanie špičiek (nastavte 0.95-0.995)
    draw_visuals._ema[:] = (1.0 - 0.25) * draw_visuals._ema + 0.25 * bands_in
    bar_len_ema = draw_visuals._bar_len_ema
    bar_w_ema = draw_visuals._bar_w_ema
    alpha_len, alpha_w = 0.35, 0.45
    bpm_pulse = 0.5 + 0.5 * math.sin(t * (0.7 + 0.012*max(0.0, bpm_est)))
    step = 1 if fps >= 30 else 2
    for i in range(0, n_bands, step):
        v = float(draw_visuals._ema[i])
        vv = np.clip(v * (0.85 + 0.30*bpm_pulse) * (0.9 + 0.3*bass_env), 0.0, 1.0)
        L_raw = (4 + vv * max_bar_out * (1.10*bar_len_mul))
        W_raw = (1 + vv * (VIS.bar_thickness * (1.0*bar_thick_mul)))
        bar_len_ema[i] = (1-alpha_len)*bar_len_ema[i] + alpha_len*L_raw
        bar_w_ema[i] = (1-alpha_w)*bar_w_ema[i] + alpha_w*W_raw
        L = int(max(1, bar_len_ema[i]))
        thickness = int(np.clip(bar_w_ema[i], 1, VIS.bar_thickness * 2.1))
        ca = math.cos(cur_angles[i]); sa = math.sin(cur_angles[i])
        x1 = int(x0[i] + L * ca)
        y1 = int(y0[i] + L * sa)
        # Draw main beam (orange/red)
        pygame.draw.line(vis_surf, (*VIS.bar_color, 255), (x0[i], y0[i]), (x1, y1), width=thickness)
        # Update and draw YELLOW PEAK
        # Ak je aktuálna hodnota vyššia, resetuj peak
        if vv > draw_visuals._beam_peaks[i]:
            draw_visuals._beam_peaks[i] = vv
        else:
            # Pomaly klesaj
            draw_visuals._beam_peaks[i] *= draw_visuals._peak_decay
        # Vypočítaj dĺžku pre peak (trochu dlhšia ako hlavný lúč pre efekt)
        peak_L_raw = (4 + draw_visuals._beam_peaks[i] * max_bar_out * (1.15*bar_len_mul))  # 1.15 namiesto 1.10
        peak_L = int(max(1, peak_L_raw * 1.05))  # o 5% dlhší ako hlavný lúč
        px1 = int(x0[i] + peak_L * ca)
        py1 = int(y0[i] + peak_L * sa)
        # Nakresli žltý peak (tenší a kratší)
        if peak_L > L + 2:  # Iba ak je peak aspoň o 2 pixely dlhší ako hlavný lúč
             pygame.draw.line(vis_surf, (*VIS.red, 255), (x1, y1), (px1, py1), width=max(1, thickness//2))
    # Flubber ring
    n_points = int(VIS.flub_points)
    if (not hasattr(draw_visuals, "_angles")) or len(draw_visuals._angles) != n_points:
        draw_visuals._angles = _thetas_cached(n_points).astype(np.float32)
    angles = draw_visuals._angles
    if (not hasattr(draw_visuals, "_cos_angles")) or len(draw_visuals._cos_angles) != n_points:
        draw_visuals._cos_angles = np.cos(angles).astype(np.float32)
        draw_visuals._sin_angles = np.sin(angles).astype(np.float32)
    cos_a = draw_visuals._cos_angles
    sin_a = draw_visuals._sin_angles
    if (not hasattr(draw_visuals, "_band_idx")) or len(draw_visuals._band_idx) != n_points or getattr(draw_visuals, "_band_idx_nbands", -1) != n_bands:
        draw_visuals._band_idx = (np.arange(n_points, dtype=np.int32) % n_bands).astype(np.int32)
        draw_visuals._band_idx_nbands = n_bands
    bi = draw_visuals._band_idx
    if (not hasattr(draw_visuals, "_phases")) or len(draw_visuals._phases) != n_points:
        rng2 = np.random.default_rng(12345)
        draw_visuals._phases = rng2.uniform(0.0, 1000.0, size=n_points).astype(np.float32)
        draw_visuals._phase_offsets = rng2.uniform(-0.5, 0.5, size=n_points).astype(np.float32)     
        draw_visuals._drift_seeds = rng2.uniform(0.0, 100.0, size=n_points).astype(np.float32)
        draw_visuals._stretch_seeds = rng2.uniform(0.0, 50.0, size=n_points).astype(np.float32)
        draw_visuals._blob_seeds = rng2.uniform(0.0, 200.0, size=n_points).astype(np.float32)
        draw_visuals._pseudopod_phases = rng2.uniform(0.0, 10.0, size=n_points).astype(np.float32)
        draw_visuals._organic_phases = rng2.uniform(0.0, 2*np.pi, size=n_points).astype(np.float32)
    phases = draw_visuals._phases
    phase_offsets = draw_visuals._phase_offsets
    drift_seeds = draw_visuals._drift_seeds
    stretch_seeds = draw_visuals._stretch_seeds
    blob_seeds = draw_visuals._blob_seeds
    pseudopod_phases = draw_visuals._pseudopod_phases
    organic_phases = draw_visuals._organic_phases
    bpm_scale = 0.5 + 0.5 * min(1.0, bpm_est/140.0)
    base_r_normal = radius * VIS.flub_base_frac * 1.18 * bpm_scale
    size_multiplier = np.interp(bass_env**3, [0.0, 1.0],
                                [VIS.flub_min_size_multiplier, VIS.flub_max_size_multiplier])
    orient = 0.5 + 0.5 * math.sin(t * 0.5 * wobble_mul)
    sx_normal = squash_h * (1.0 + 0.04 * bass_env * (1.0 - orient))
    sy_normal = squash_v * (1.0 + 0.10 * bass_env * orient)
    primary_breath = 0.5 + 0.5 * math.sin(t * (0.8*wobble_mul + 0.01*max(0.0, bpm_est)) + 6.5)
    secondary_breath = 0.5 + 0.5 * math.sin(t * (0.3*wobble_mul) + 2.1)
    tertiary_breath = 0.5 + 0.5 * math.sin(t * (1.2*wobble_mul) + 4.7)
    breath = 0.6 * primary_breath + 0.25 * secondary_breath + 0.15 * tertiary_breath
    breath = max(0.88, breath)
    global_breathe = size_multiplier * breath
    thickness_base_normal = radius * 0.08 * (0.85 + 0.60 * voice_env)
    global_drift_x = 0.003 * radius * math.sin(t * 0.2 * wobble_mul + 1.3)
    global_drift_y = 0.003 * radius * math.cos(t * 0.15 * wobble_mul + 2.7)
    global_stretch_x = 1.0 + 0.08 * math.sin(t * 0.25 * wobble_mul + 3.2) * (0.3 + 0.7 * (bass_env**2))
    global_stretch_y = 1.0 + 0.06 * math.cos(t * 0.35 * wobble_mul + 1.8) * (0.4 + 0.6 * voice_env)
    asymmetry_wave1 = 0.03 * math.sin(t * 0.4 * wobble_mul + 2.5)
    asymmetry_wave2 = 0.02 * math.cos(t * 0.6 * wobble_mul + 4.1)
    outer_pts = draw_visuals._outer_pts
    inner_pts = draw_visuals._inner_pts
    prof_shape = draw_visuals._shape_profile if (draw_visuals._shape_active and draw_visuals._shape_profile is not None) else None
    prof = None
    mode = "reactive"
    if letter_prof is not None:
        prof = letter_prof
        mode = "letter_static" if letter_static else "letter_morph"
    elif prof_shape is not None:
        prof = prof_shape
        mode = "stable_shape"
    else:
        mode = "reactive"
    # Vektorizované výpočty bodov
    thetas = angles
    ca_all = cos_a
    sa_all = sin_a
    # Lokálne organické variácie
    local_time_scales = 1.0 + 0.1 * phase_offsets
    local_drifts = 0.002 * radius * np.sin(t * 0.25 * local_time_scales + drift_seeds)
    # Intenzita pseudopód
    stretch_intensity = np.where(mode == "reactive", 0.15,
                               np.where(mode == "letter_morph", 0.06, 0.03))
    stretch_intensity = np.where(mode == "letter_static", 0.015, stretch_intensity)
    pseudopod_waves = np.sin(t * 0.3 * wobble_mul + pseudopod_phases)
    pseudopod_strengths = np.maximum(0.0, pseudopod_waves * pseudopod_waves - 0.3)
    local_pseudopods = 1.0 + stretch_intensity * pseudopod_strengths
    # Bubliny
    blob_waves1 = np.sin(t * 0.45 * wobble_mul + blob_seeds * 0.1)
    blob_waves2 = np.cos(t * 0.28 * wobble_mul + stretch_seeds * 0.15)
    local_blobs = 1.0 + stretch_intensity * 0.6 * blob_waves1 * blob_waves2
    # Organické pulzovanie
    organic_pulses = 1.0 + VIS.organic_pulse_intensity * (
        0.3 * np.sin(t * 1.0 + organic_phases) * bass_env +
        0.2 * np.sin(t * 1.5 + organic_phases * 0.7) * voice_env
    )
    # Asymetrické napätie
    angle_factors = np.sin(thetas * 2.0 + t * 0.1)
    local_stretch_x = global_stretch_x * (1.0 + 0.03 * angle_factors + asymmetry_wave1 * np.cos(thetas))
    local_stretch_y = global_stretch_y * (1.0 + 0.02 * angle_factors + asymmetry_wave2 * np.sin(thetas))
    combined_growth = local_pseudopods * local_blobs * organic_pulses
    # Ripple efekt
    ripple_effects = np.zeros_like(thetas)
    if draw_visuals._body_ripple_active:
        ripple_waves = np.sin(draw_visuals._body_ripple_phase + thetas * 4.0)
        ripple_effects = 0.05 * np.maximum(0.0, ripple_waves)
    # Výpočet polomerov podľa módu
    if mode == "letter_static":
        base_r0 = np.full_like(thetas, float(draw_visuals._letter_freeze["base_r"]))
        thick0 = np.full_like(thetas, float(draw_visuals._letter_freeze["thickness"]))
        sx0 = np.full_like(thetas, float(draw_visuals._letter_freeze["sx"]))
        sy0 = np.full_like(thetas, float(draw_visuals._letter_freeze["sy"]))
        subtle_pulses = 1.0 + 0.015 * np.sin(t * 0.6 * wobble_mul + phases * 0.1)
        micro_variations = 1.0 + 0.008 * np.sin(t * 1.1 * wobble_mul + drift_seeds * 0.1)
        r_core = base_r0 * subtle_pulses * micro_variations * combined_growth * (1.0 + (prof - 1.0)) * (1.0 + ripple_effects)
        thick = np.maximum(1.0, 0.5 * thick0 * combined_growth)
        r_out = r_core + thick
        r_in = np.maximum(2.0, r_core - thick)
        cx_fix, cy_fix = draw_visuals._letter_center
        ox = cx_fix + r_out * ca_all * sx0 * local_stretch_x + local_drifts * 0.3
        oy = cy_fix + r_out * sa_all * sy0 * local_stretch_y + local_drifts * 0.3
        ix = cx_fix + r_in * ca_all * sx0 * local_stretch_x + local_drifts * 0.3
        iy = cy_fix + r_in * sa_all * sy0 * local_stretch_y + local_drifts * 0.3
    elif mode == "letter_morph":
        base_r0 = np.full_like(thetas, radius * VIS.flub_base_frac * 1.18 * bpm_scale)
        thick0 = np.full_like(thetas, radius * 0.085 * (0.95 + 0.10 * voice_env))
        breathing1 = 0.96 + 0.06 * (0.5 + 0.5*np.sin(t*0.9*wobble_mul + phases * 0.05))
        breathing2 = 0.98 + 0.03 * (0.5 + 0.5*np.sin(t*0.4*wobble_mul + drift_seeds * 0.1))
        breathing = breathing1 * breathing2
        local_variance = 1.0 + 0.02 * np.sin(t * 1.3 * wobble_mul + phases * 0.2)
        r_core = base_r0 * breathing * local_variance * combined_growth * (1.0 + (prof - 1.0)) * (1.0 + ripple_effects)
        thick = np.maximum(1.0, 0.5 * thick0 * combined_growth)
        r_out = r_core + thick
        r_in = np.maximum(2.0, r_core - thick)
        ox = cx_off + r_out * ca_all * squash_h * local_stretch_x + global_drift_x + local_drifts
        oy = cy_off + r_out * sa_all * squash_v * local_stretch_y + global_drift_y + local_drifts
        ix = cx_off + r_in * ca_all * squash_h * local_stretch_x + global_drift_x + local_drifts
        iy = cy_off + r_in * sa_all * squash_v * local_stretch_y + global_drift_y + local_drifts
    elif mode == "stable_shape":
        base_r0 = np.full_like(thetas, float(draw_visuals._freeze["base_r"]))
        thick0 = np.full_like(thetas, float(draw_visuals._freeze["thickness"]))
        sx0 = np.full_like(thetas, float(draw_visuals._freeze["sx"]))
        sy0 = np.full_like(thetas, float(draw_visuals._freeze["sy"]))
        organic_variation = 1.0 + 0.01 * np.sin(t * 0.7 * wobble_mul + phases * 0.15)
        micro_shimmer = 1.0 + 0.005 * np.sin(t * 2.1 * wobble_mul + drift_seeds * 0.3)
        r_core = base_r0 * organic_variation * micro_shimmer * combined_growth * (1.0 + shape_w * (prof - 1.0)) * (1.0 + ripple_effects)
        thick = np.maximum(1.0, 0.5 * thick0 * combined_growth)
        r_out = r_core + thick
        r_in = np.maximum(2.0, r_core - thick)
        ox = cx_off + r_out * ca_all * sx0 * local_stretch_x + global_drift_x * 0.5 + local_drifts * 0.5
        oy = cy_off + r_out * sa_all * sy0 * local_stretch_y + global_drift_y * 0.5 + local_drifts * 0.5
        ix = cx_off + r_in * ca_all * sx0 * local_stretch_x + global_drift_x * 0.5 + local_drifts * 0.5
        iy = cy_off + r_in * sa_all * sy0 * local_stretch_y + global_drift_y * 0.5 + local_drifts * 0.5
    else:  # reactive mode
        n_local1 = _fbm1(t * VIS.flub_amorphous_speed*wobble_mul + phases*0.17 + VIS.flub_angular_noise_scale*thetas)
        n_local2 = _fbm1(t * VIS.flub_amorphous_speed*wobble_mul*0.60 + drift_seeds*0.10 + thetas*0.50) * 0.4
        n_local3 = _fbm1(t * VIS.flub_amorphous_speed*wobble_mul*1.80 + phases*0.05) * 0.2
        n_combined = n_local1 + n_local2 + n_local3
        infl = 0.5 + 0.5 * n_combined
        local_breath_phase = phases * 0.10 + drift_seeds * 0.05
        local_breathing = 0.95 + 0.10 * np.sin(t * 0.9 * wobble_mul + local_breath_phase)
        feeding_wave = np.sin(t * 0.15 * wobble_mul + blob_seeds * 0.05)
        feeding_burst = 1.0 + 0.12 * np.maximum(0.0, feeding_wave**3) * bass_env
        r_core = base_r_normal * global_breathe * local_breathing * combined_growth * feeding_burst * (1.0 + (VIS.flub_amorphous_intensity + VIS.flub_voice_amorphous_gain*voice_env) * infl) * (1.0 + ripple_effects)
        r_core += 0.35 * base_r_normal * bands_in[bi] * (0.4 + 0.6 * bass_env)
        r_core = np.maximum(base_r_normal * 0.70, r_core)
        thick_variation = 1.0 + 0.10 * np.sin(t * 1.5 * wobble_mul + phases * 0.2)
        organic_thick = combined_growth * (0.8 + 0.4 * pseudopod_strengths)
        thick = thickness_base_normal * (0.85 + 0.30 * infl) * thick_variation * organic_thick
        thick = np.maximum(1.0, thick)
        r_out = r_core + 0.5 * thick
        r_in = np.maximum(2.0, r_core - 0.5 * thick)
        sx_local = sx_normal * local_stretch_x * (1.0 + 0.02 * np.sin(t * 0.8 * wobble_mul + drift_seeds))
        sy_local = sy_normal * local_stretch_y * (1.0 + 0.02 * np.cos(t * 0.6 * wobble_mul + phases * 0.1))
        ox = cx_off + r_out * ca_all * sx_local + global_drift_x + local_drifts
        oy = cy_off + r_out * sa_all * sy_local + global_drift_y + local_drifts
        ix = cx_off + r_in * ca_all * sx_local + global_drift_x + local_drifts
        iy = cy_off + r_in * sa_all * sy_local + global_drift_y + local_drifts
    # Uloženie bodov
    outer_pts[:, 0] = ox
    outer_pts[:, 1] = oy
    inner_pts[:, 0] = ix
    inner_pts[:, 1] = iy
    # Mikro-inércia
    if draw_visuals._have_prev and not draw_visuals._shape_active and mode != "letter_static":
        if mode == "letter_morph":
            inertia_factor, responsiveness = 0.75, 0.25
        elif mode == "reactive":
            inertia_factor, responsiveness = 0.82, 0.18
        else:
            inertia_factor, responsiveness = 0.85, 0.15
        outer_pts *= responsiveness
        outer_pts += inertia_factor * draw_visuals._prev_outer
        inner_pts *= responsiveness
        inner_pts += inertia_factor * draw_visuals._prev_inner
    draw_visuals._prev_outer[:] = outer_pts
    draw_visuals._prev_inner[:] = inner_pts
    draw_visuals._have_prev = True
    # Farby podľa energie
    # >>>> FARBY FLUBBERA PRENESIETE DO KONŠTÁNT <<<<
    # fill_col = VIS.yellow  # Výplň
    # edge_col = lerp_color(VIS.yellow, (255, 255, 255), 0.3)  # Okraj je o niečo svetlejší
    # Farby flubbera - oranžová ako rays
    # fill_col = (212, 114, 22)   # oranžová výplň (rovnaká ako rays_color)
    # edge_col = (255, 200, 80)   # svetlejší oranžovo-žltý okraj (ako peaks highlight)
    # fill_col = VIS.rays_color   # oranžová výplň (rovnaká ako rays_color)
    fill_col = VIS.bar_color   # oranžová výplň (rovnaká ako rays_color)
    # edge_col = lerp_color(VIS.rays_color, (255, 255, 255), 0.3)   # svetlejší okraj
    edge_col = lerp_color(VIS.bar_color, (255, 255, 255), 0.3)   # svetlejší okraj
    # ========== RANDOM 3D EFFECTS FOR FLUBBER ==========
    # Apply 3D lighting and depth effect
    if not hasattr(draw_visuals, "_3d_phases"):
        draw_visuals._3d_phases = np.random.uniform(0, 2*np.pi, n_points).astype(np.float32)
        draw_visuals._3d_depths = np.random.uniform(-1, 1, n_points).astype(np.float32)
    
    # Calculate 3D lighting based on position and light source
    light_dir = np.array([math.cos(VIS.flub_light_angle), math.sin(VIS.flub_light_angle)], dtype=np.float32)
    normals = np.zeros((n_points, 2), dtype=np.float32)
    
    # Calculate normals for each point
    for i in range(n_points):
        prev_i = (i - 1) % n_points
        next_i = (i + 1) % n_points
        # Calculate tangent
        tangent = np.array([
            outer_pts[next_i, 0] - outer_pts[prev_i, 0],
            outer_pts[next_i, 1] - outer_pts[prev_i, 1]
        ], dtype=np.float32)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-6:
            tangent /= tangent_norm
            # Calculate normal (perpendicular to tangent)
            normals[i] = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        else:
            normals[i] = np.array([0, 1], dtype=np.float32)
    
    # Normalize normals
    normal_magnitudes = np.linalg.norm(normals, axis=1)
    normal_magnitudes[normal_magnitudes < 1e-6] = 1.0
    normals /= normal_magnitudes[:, np.newaxis]
    
    # Calculate lighting intensity
    light_intensity = np.dot(normals, light_dir)
    light_intensity = np.clip(light_intensity, 0.0, 1.0)
    
    # Add specular highlights
    view_dir = np.array([0, 0], dtype=np.float32)  # Simplified view direction
    half_vector = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir + 1e-6)
    specular = np.power(np.clip(np.sum(normals * half_vector, axis=1), 0.0, 1.0), VIS.flub_specular_power)
    
    # Add random 3D depth effect
    depth_factor = 1.0 + VIS.flub_depth_multiplier * draw_visuals._3d_depths * bass_env
    
    # Animate 3D phases for dynamic effect
    draw_visuals._3d_phases += dt * 0.5 * (1.0 + bass_env)
    depth_animation = 0.1 * np.sin(draw_visuals._3d_phases + t * 2.0) * bass_env
    depth_factor += depth_animation
    
    # Apply 3D effects to points
    center = np.array([cx_off, cy_off], dtype=np.float32)
    for i in range(n_points):
        # Calculate vector from center to point
        to_point = np.array([outer_pts[i, 0] - cx_off, outer_pts[i, 1] - cy_off], dtype=np.float32)
        distance = np.linalg.norm(to_point)
        if distance > 1e-6:
            direction = to_point / distance
            # Apply depth factor (points move in/out based on depth)
            new_distance = distance * depth_factor[i]
            outer_pts[i, 0] = cx_off + direction[0] * new_distance
            outer_pts[i, 1] = cy_off + direction[1] * new_distance
            
            # Apply same effect to inner points
            to_inner = np.array([inner_pts[i, 0] - cx_off, inner_pts[i, 1] - cy_off], dtype=np.float32)
            inner_distance = np.linalg.norm(to_inner)
            if inner_distance > 1e-6:
                inner_direction = to_inner / inner_distance
                new_inner_distance = inner_distance * depth_factor[i]
                inner_pts[i, 0] = cx_off + inner_direction[0] * new_inner_distance
                inner_pts[i, 1] = cy_off + inner_direction[1] * new_inner_distance
    
    # Create 3D lighting effect on fill color
    r, g, b = fill_col
    # Base lighting - make fill color brighter on lit areas
    lit_fill_col = (
        min(255, int(r * (1.0 + VIS.flub_3d_intensity * light_intensity.mean()))),
        min(255, int(g * (1.0 + VIS.flub_3d_intensity * light_intensity.mean()))),
        min(255, int(b * (1.0 + VIS.flub_3d_intensity * light_intensity.mean())))
    )
    
    # Add specular highlights to edge color
    spec_intensity = specular.mean()
    edge_r, edge_g, edge_b = edge_col
    highlighted_edge_col = (
        min(255, int(edge_r + 100 * spec_intensity)),
        min(255, int(edge_g + 100 * spec_intensity)),
        min(255, int(edge_b + 100 * spec_intensity))
    )
    # ========== END RANDOM 3D EFFECTS FOR FLUBBER ==========
    # Vykreslenie flubber kruhu
    _draw_flubber_ring(vis_surf, outer_pts, inner_pts, n_points, lit_fill_col, highlighted_edge_col, fps)
    # Vykreslenie satelitov
    for satellite in list(draw_visuals._satellites):
        satellite['angle'] += satellite['orbit_speed'] * dt
        sx = cx_off + satellite['distance'] * math.cos(satellite['angle'])
        sy = cy_off + satellite['distance'] * math.sin(satellite['angle'])
        pygame.draw.circle(vis_surf, satellite['color'], (int(sx), int(sy)), satellite['size'])
        if satellite['distance'] < radius * 0.5 or np.random.random() < 0.005:
            draw_visuals._satellites.remove(satellite)
    # Hlasový kruh
    if voice_env > 0.01:
        voice_radius = int(radius * 0.70 * (1.0 + 0.5 * voice_env))
        voice_alpha = int(np.interp(voice_env, [0,1], [VIS.voice_alpha_min, VIS.voice_alpha_max]))
        voice_color = (*VIS.voice_base_color, voice_alpha)
        pygame.draw.circle(vis_surf, voice_color, (cx_off, cy_off), voice_radius, width=3)
    # Glow effect
    def _q8(x, steps=16):
        return int(round(x * (steps-1)) * (255 // (steps-1)))
    glow_color = (_q8(fill_col[0]/255.0), _q8(fill_col[1]/255.0), _q8(fill_col[2]/255.0))
    glow_near = build_glow_circle_surface(radius, glow=10, color_rgb=glow_color, thickness=2)
    glow_far = build_glow_circle_surface(int(radius*1.1), glow=18, color_rgb=lerp_color(glow_color, (255,255,255), 0.25), thickness=1)
    blit_center(vis_surf, glow_far, (cx_off, cy_off))
    blit_center(vis_surf, glow_near, (cx_off, cy_off))
    # Texture overlay
    key = (w, h)
    noise_big = draw_visuals._noise_cache.get(key)
    if noise_big is None:
        noise_big = pygame.transform.smoothscale(draw_visuals._noise_small, (w, h))
        draw_visuals._noise_cache[key] = noise_big
    vis_surf.blit(noise_big, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    # Progress arc
    pad = VIS.progress_width // 2 + 8
    rect = pygame.Rect(cx_off - radius - pad, cy_off - radius - pad, 2 * (radius + pad), 2 * (radius + pad))
    frac = 0.0 if dur_s <= 0 else min(1.0, pos_s / dur_s)
    # Bezpečný výpočet energy_color — ak chcete fixnú krvavo červenú, jednoducho toto zakomentujte a odkomentujte riadok nižšie
    try:
        hi_energy = float(np.mean(bands_in[int(n_bands*0.6):])) if n_bands > 0 else 0.0
        energy_t = np.clip(0.5*voice_env + 0.5*hi_energy, 0.0, 1.0)
        # energy_color = lerp_color(lerp_color(VIS.red, VIS.yellow, min(1.0, bass_env ** 0.5)), VIS.white, 0.35*energy_t)
        energy_color = VIS.red
    except Exception:
        energy_color = (139, 0, 0)  # Fallback: krvavo červená
    # Ak chcete VŽDY krvavo červenú, odkomentujte nasledujúci riadok a zakomentujte celý try-except blok vyššie:
    # energy_color = (139, 0, 0)
    draw_progress_arc_aa(vis_surf, rect, -math.pi / 2, frac, energy_color, VIS.progress_width)
    # Beat ripples
    for rr in list(draw_visuals._ripples):
        rr["r"] += rr["dr"]; rr["a"] = max(0, rr["a"] - rr["da"])
        if rr["a"] <= 0 or rr["r"] > max(w, h):
            draw_visuals._ripples.remove(rr); continue
        pygame.draw.circle(vis_surf, (*energy_color, rr["a"]), (cx_off, cy_off), int(rr["r"]), width=rr["w"])
    # Spustenie ripple shocku
    if draw_visuals._bpm_state.last_beat_time != prev_last_beat:
        draw_visuals._body_ripple_active = True
        draw_visuals._body_ripple_phase = 0.0
def apply_sepia(surface: pygame.Surface, strength: float = 0.8) -> None:
    w, h = surface.get_size()
    s = max(0.0, min(1.0, float(strength)))
    mul = pygame.Surface((w, h), pygame.SRCALPHA)
    mul.fill((210, 180, 140, int(255 * s)))
    surface.blit(mul, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    add = pygame.Surface((w, h), pygame.SRCALPHA)
    add.fill((40, 20, 0, int(60 * s)))
    surface.blit(add, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
def main():
    args = parse_args()
    setup_logging(args.debug)
    ffok = ensure_ffmpeg()
    pygame.init()
    pygame.display.set_caption("VMP — Visualized Music Player")
    try:
        icon = pygame.image.load("vmp.png")
        pygame.display.set_icon(icon)
    except Exception:
        pass
    flags_windowed = pygame.RESIZABLE | pygame.DOUBLEBUF
    def try_mode(size, flags, vsync):
        try: return pygame.display.set_mode(size, flags, vsync=vsync)
        except Exception: return None
    screen = try_mode((1280,720), flags_windowed, 1) or try_mode((1280,720), flags_windowed, 0) or pygame.display.set_mode((1280,720))
    pygame.mixer.init(frequency=AUDIO.target_sr, channels=2, size=-16, buffer=1024)
    log.debug("Pygame mixer initialized @ %d Hz", AUDIO.target_sr)
    chan_main = pygame.mixer.Channel(0)
    # Async segment prefetch
    seg_prefetch_q: "queue.Queue[Tuple[Path, float, float, np.ndarray]]" = queue.Queue(maxsize=2)
    seg_req_q: "queue.Queue[Tuple[Path, float, float]]" = queue.Queue(maxsize=3)
    seg_worker_should_run = True
    def _clear_seg_prefetch():
        try:
            while True:
                _ = seg_prefetch_q.get_nowait()
                seg_prefetch_q.task_done()
        except queue.Empty:
            pass
        try:
            while True:
                _ = seg_req_q.get_nowait()
                seg_req_q.task_done()
        except queue.Empty:
            pass
    def segment_prefetch_worker():
        log.debug("Segment prefetch worker started")
        while seg_worker_should_run:
            try:
                path, start_at, length = seg_req_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not seg_worker_should_run:
                break
            try:
                pcm = decode_pcm_segment_robust(path, start_at, length, sr=AUDIO.target_sr)
                if pcm.shape[0] > 0:
                    try:
                        seg_prefetch_q.put_nowait((path, start_at, length, pcm))
                    except queue.Full:
                        pass
            except Exception as e:
                log.debug("segment_prefetch error: %s", e)
            finally:
                seg_req_q.task_done()
        log.debug("Segment prefetch worker exit")
    seg_worker = threading.Thread(target=segment_prefetch_worker, daemon=True)
    seg_worker.start()
    playback_mode = "music"
    current_main_sound: Optional[pygame.mixer.Sound] = None
    fake_fullscreen = False
    prev_window_pos = (100, 100)
    prev_window_size = (1280, 720)
    def remember_window_rect():
        nonlocal prev_window_pos, prev_window_size
        try: prev_window_pos = pygame.display.get_window_position()
        except Exception: pass
        prev_window_size = screen.get_size()
        log.debug("Remember window rect pos=%s size=%s", prev_window_pos, prev_window_size)
    def get_display_size_for_current():
        try:
            idx = pygame.display.get_window_display_index()
            sizes = pygame.display.get_desktop_sizes()
            if 0 <= idx < len(sizes): return sizes[idx]
        except Exception: pass
        return get_desktop_size()
    vis_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    text_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    def after_mode_change_rescale():
        nonlocal vis_surf, text_surf
        w,h = screen.get_size()
        vis_surf = pygame.Surface((w,h), pygame.SRCALPHA)
        text_surf = pygame.Surface((w,h), pygame.SRCALPHA)
        if original_bg_surface is not None:
            rescale_background_for_size()
        else:
            choose_background(True)
        log.debug("Rescaled after mode change -> %dx%d", w, h)
    def exit_fake_fullscreen():
        nonlocal screen, fake_fullscreen
        # Obnovíme predchádzajúcu veľkosť a pozíciu
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{prev_window_pos[0]},{prev_window_pos[1]}"
        flags = flags_windowed  # pygame.RESIZABLE | pygame.DOUBLEBUF
        screen = try_mode(prev_window_size, flags, 1) or try_mode(prev_window_size, flags, 0) or pygame.display.set_mode(prev_window_size, flags)
        fake_fullscreen = False
        if 'SDL_VIDEO_WINDOW_POS' in os.environ:
            del os.environ['SDL_VIDEO_WINDOW_POS']
        after_mode_change_rescale()
        log.debug("Exited borderless fullscreen -> windowed: %dx%d at %s", prev_window_size[0], prev_window_size[1], prev_window_pos)
    def enter_fake_fullscreen():
        nonlocal screen, fake_fullscreen
        try:
            x, y = pygame.display.get_window_position()
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
        except Exception: pass
        dw, dh = get_display_size_for_current()
        flags = pygame.NOFRAME | pygame.DOUBLEBUF
        screen = try_mode((dw, dh), flags, 1) or try_mode((dw, dh), flags, 0) or pygame.display.set_mode((dw, dh), flags)
        fake_fullscreen = True
        if 'SDL_VIDEO_WINDOW_POS' in os.environ: del os.environ['SDL_VIDEO_WINDOW_POS']
        after_mode_change_rescale()
    def exit_fake_fullscreen():
        nonlocal screen, fake_fullscreen
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{prev_window_pos[0]},{prev_window_pos[1]}"
        flags = flags_windowed
        screen = try_mode(prev_window_size, flags, 1) or try_mode(prev_window_size, flags, 0) or pygame.display.set_mode(prev_window_size, flags)
        fake_fullscreen = False
        if 'SDL_VIDEO_WINDOW_POS' in os.environ: del os.environ['SDL_VIDEO_WINDOW_POS']
        after_mode_change_rescale()
    def toggle_fake_fullscreen():
        if fake_fullscreen: exit_fake_fullscreen()
        else: remember_window_rect(); enter_fake_fullscreen()
    font_small_sys = load_system(16, bold=True)
    font_mid_sys = load_system(50, bold=True)
    title_font = load_cyber(VIS.title_font_size)
    vol_font_sys = load_system(18, bold=True)
    base = Path(args.music_dir)
    if not base.exists():
        print(f"MUSIC_DIR not found: {base}")
        sys.exit(1)
    ext_tuple = tuple(e if e.startswith(".") else f".{e}" for e in args.ext.split(",") if e.strip())
    ignore_dirs = [d.strip() for d in args.ignore.split(",") if d.strip()]
    lib = Library(ext_tuple, ignore_dirs, args.no_tags); lib.scan(base)
    if not lib.tracks:
        print("No supported audio files found."); sys.exit(1)
    bg_paths: List[Path] = []
    if args.backgrounds:
        bgdir = Path(args.backgrounds).expanduser()
        if bgdir.exists():
            for p in sorted(bgdir.rglob("*")):
                if p.suffix.lower() in (".jpg",".jpeg",".png"): bg_paths.append(p)
    bg_enabled = bool(bg_paths)
    bg_index = 0
    current_bg: Optional[pygame.Surface] = None
    bg_dark: Optional[pygame.Surface] = None
    original_bg_surface: Optional[pygame.Surface] = None
    bg_size_cache: Dict[Tuple[int,int], pygame.Surface] = {}
    def choose_background(force=False):
        nonlocal current_bg, bg_dark, bg_index, original_bg_surface, bg_size_cache
        if not bg_enabled or not bg_paths:
            current_bg=None; bg_dark=None; original_bg_surface=None; bg_size_cache.clear(); return
        if force:
            bg_index = max(0, min(bg_index, len(bg_paths)-1))
            p = bg_paths[bg_index]
        else:
            p = random.choice(bg_paths); bg_index = bg_paths.index(p)
        try:
            original_bg_surface = pygame.image.load(str(p)).convert_alpha() if p.suffix.lower() == '.png' else pygame.image.load(str(p)).convert()
            bg_size_cache.clear()
            w,h=screen.get_size()
            iw,ih=original_bg_surface.get_width(),original_bg_surface.get_height()
            scale=max(w/iw,h/ih)
            img=pygame.transform.smoothscale(original_bg_surface,(int(iw*scale),int(ih*scale)))
            x=(img.get_width()-w)//2; y=(img.get_height()-h)//2
            current_bg=img.subsurface(pygame.Rect(x,y,w,h)).copy()
            bg_dark=pygame.Surface((w,h), pygame.SRCALPHA); bg_dark.fill((0,0,0,110))
            log.debug("Background loaded: %s", p.name)
        except Exception as e:
            log.debug("BG load fail: %s", e); current_bg=None; bg_dark=None; original_bg_surface=None; bg_size_cache.clear()
    def rescale_background_for_size():
        nonlocal current_bg, bg_dark
        if original_bg_surface is None: return
        w,h=screen.get_size()
        key=(w,h)
        if key in bg_size_cache:
            current_bg = bg_size_cache[key]
        else:
            iw,ih=original_bg_surface.get_width(),original_bg_surface.get_height()
            scale=max(w/iw,h/ih)
            img=pygame.transform.smoothscale(original_bg_surface,(int(iw*scale),int(ih*scale)))
            x=(img.get_width()-w)//2; y=(img.get_height()-h)//2
            current_bg=img.subsurface(pygame.Rect(x,y,w,h)).copy()
            bg_size_cache[key]=current_bg
        bg_dark=pygame.Surface((w,h), pygame.SRCALPHA); bg_dark.fill((0,0,0,110))
        log.debug("Background rescaled -> %s", key)
    v_states = []
    for show_hud in (False, True):
        for show_fps in (False, True):
            for show_time in (False, True):
                for show_title in (False, True):
                    v_states.append({"hud":show_hud,"fps":show_fps,"time":show_time,"title":show_title})
    def find_state_idx(hud,fps,timev,title):
        for i,s in enumerate(v_states):
            if (s["hud"],s["fps"],s["time"],s["title"])==(hud,fps,timev,title): return i
        return 0
    preset_indices = [
        find_state_idx(True, False, False, False),
        find_state_idx(True, True,  False, False),
        find_state_idx(False,False, True,  False),
        find_state_idx(False,False, True,  True),
        find_state_idx(False,False, False, False),
    ]
    v_mode = preset_indices[3]
    def v_mode_desc(m):
        st=v_states[m%len(v_states)]
        return f"HUD {'ON' if st['hud'] else 'OFF'} / FPS {'ON' if st['fps'] else 'OFF'} / TIME {'ON' if st['time'] else 'OFF'} / TITLE {'ON' if st['title'] else 'OFF'}"
    help_visible = False
    sepia_enabled = False
    help_cache_surf: Optional[pygame.Surface] = None
    def build_help_surface(w: int, h: int) -> pygame.Surface:
        pad = 14
        lines = [
            "Visualized Music Player",
            "-----------------------",
            "H – Help (toggle)",
            "V – Next preset",
            "F2 – Cycle all view combinations",
            "F – Toggle Fake Fullscreen",
            "T – Toggle Always on Top",
            "B – Toggle Backgrounds",
            "[ – Previous Background",
            "] – Next Background",
            "Space – Pause/Resume",
            "N – Next Track",
            "P – Previous Track",
            "S – Toggle Shuffle (after first)",
            "R – Toggle Repeat All",
            "← – Seek -5 seconds",
            "→ – Seek +5 seconds",
            "↑ – Volume Up",
            "↓ – Volume Down",
            "M – Mute/Unmute",
            "Mouse Wheel – Volume",
            "LMB drag – Move window (when no background)",
            "C – Toggle Sepia",
            "L – Toggle Lyrics (LRC)",
            "Esc / Q – Quit"
        ]
        col_bg = (0, 0, 0, 180)
        col_frame = (255, 255, 255, 60)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        texts = [font_small_sys.render(s, True, VIS.white) for s in lines]
        tw = max(t.get_width() for t in texts)
        th = sum(t.get_height()+4 for t in texts)
        box_w = tw + pad*2
        box_h = th + pad*2 + 6
        x = 20; y = 20
        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box.fill(col_bg)
        pygame.draw.rect(box, col_frame, box.get_rect(), width=2, border_radius=8)
        cy2 = pad + 3
        for t in texts:
            box.blit(t, (pad, cy2)); cy2 += t.get_height() + 4
        surf.blit(box, (x, y))
        return surf
    n = len(lib.tracks)
    shuffle = bool(args.shuffle)
    repeat_all = bool(args.repeat_all)
    repeat_one = False  # <-- NOVÁ PREMENNÁ: REPEAT ONE
    if shuffle and n > 1:
        tail = list(range(1, n)); random.shuffle(tail); order = [0] + tail
    else:
        order = list(range(n))
    index_in_order = 0
    def display_title(t: Track) -> str:
        title = t.title; artist= t.artist
        if not title:
            g_title, _ = guess_from_filename(t.path); title = g_title
        if not title: return ""
        return f"{artist} — {title}" if artist else title
    current_track: Track = lib.tracks[order[index_in_order]]
    log.debug("Initial track: %s", current_track.path.name)
    def stop_all_playback():
        nonlocal current_main_sound
        try: pygame.mixer.music.stop()
        except Exception: pass
        try: chan_main.stop()
        except Exception: pass
        current_main_sound = None
    def set_play_start(start_sec: float):
        nonlocal play_start_monotonic, paused_accum, pause_started
        play_start_monotonic = time.monotonic() - start_sec
        paused_accum = 0.0
        pause_started = None
        log.debug("Set play start: %.3f", start_sec)
    def get_play_pos() -> float:
        if paused:
            if pause_started is not None:
                return max(0.0, pause_started - play_start_monotonic - paused_accum)
            return max(0.0, time.monotonic() - play_start_monotonic - paused_accum)
        if playback_mode == "music":
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms is not None and pos_ms >= 0:
                return pos_ms / 1000.0
        return max(0.0, time.monotonic() - play_start_monotonic - paused_accum)
    def seek_relative(delta_sec: float):
        cur = get_play_pos()
        dur = current_track.duration_sec or max(0.0, cur + 1.0)
        new_pos = max(0.0, min(max(0.0, dur - 0.05), cur + delta_sec))
        set_play_start(new_pos)
        seek_and_play_with_ffmpeg(current_track, new_pos, volume)
        log.debug("Seek relative %.2f -> new_pos=%.2f (mode=%s)", delta_sec, new_pos, playback_mode)
        return new_pos
    def play_track_music(track: Track, start_sec: float = 0.0, volume: float = 0.85):
        nonlocal playback_mode
        stop_all_playback()
        _clear_seg_prefetch()
        pygame.mixer.music.load(str(track.path))
        if start_sec > 0.0:
            seek_and_play_with_ffmpeg(track, start_sec, volume)
            return
        pygame.mixer.music.play()
        set_play_start(0.0)
        playback_mode = "music"
        pygame.mixer.music.set_volume(volume)
    def seek_and_play_with_ffmpeg(track: Track, start_sec: float, volume: float = 0):
        nonlocal playback_mode, current_main_sound, seg_cur_start, seg_cur_len
        stop_all_playback()
        # clear async prefetch state (to avoid old chunks)
        _clear_seg_prefetch()
        if not ffok:
            try:
                pygame.mixer.music.load(str(track.path))
                pygame.mixer.music.play()
                # try to honor the seek if backend supports it
                try:
                    pygame.mixer.music.set_pos(max(0.0, float(start_sec)))
                except Exception:
                    pass
                pygame.mixer.music.set_volume(volume)
                playback_mode = "music"
                # keep monotonic clock aligned even in fallback
                set_play_start(max(0.0, float(start_sec)))
            except Exception:
                playback_mode = "music"
            return
        dur_left = min(UI.seek_segment_sec, max(0.1, (track.duration_sec or 0.0) - start_sec))
        seg = decode_pcm_segment(track.path, start_sec, dur_left, sr=AUDIO.target_sr)
        if seg.shape[0] == 0:
            play_track_music(track, 0.0, volume)
            return
        current_main_sound = numpy_to_sound(seg)
        chan_main.play(current_main_sound)
        chan_main.set_volume(volume, volume)
        playback_mode = "channel"
        # keep monotonic clock aligned on successful ffmpeg path as well
        set_play_start(max(0.0, float(start_sec)))
        seg_cur_start = start_sec
        seg_cur_len = dur_left
    def maybe_queue_next_segment(pos_now: float):
        nonlocal seg_cur_start, seg_cur_len, playback_mode
        if playback_mode != "channel":
            _clear_seg_prefetch()
            return
        seg_end = seg_cur_start + seg_cur_len
        # 1) Try to receive ready prefetched chunk
        try:
            pth, start_at, length, pcm = seg_prefetch_q.get_nowait()
            try:
                if (pth == current_track.path) and abs(start_at - seg_end) < 1e-3 and length > 0.05 and pcm.shape[0]:
                    snd = numpy_to_sound(pcm)
                    if not chan_main.get_busy():
                        # channel is stopped -> play immediately and reset segment window
                        chan_main.play(snd)
                        seg_cur_start = start_at
                        seg_cur_len = length
                    else:
                        chan_main.queue(snd)
                        seg_cur_len += length
                # otherwise result is old/doesn't match → discard
            finally:
                # keep internal counters in order
                seg_prefetch_q.task_done()
        except queue.Empty:
            pass
        # 2) Decide if need to request next chunk
        #    - dynamic threshold based on length of already "covered" segment
        seg_window = max(0.1, seg_cur_len)
        THRESH = max(3.0, min(10.0, 0.6 * seg_window))
        MAX_CHUNK = 12.0
        # if duration is unknown, pretend it's "infinite" → stream while ffmpeg provides data
        dur_known = current_track.duration_sec if (current_track.duration_sec and current_track.duration_sec > 0.1) else float("inf")
        remain_total = dur_known - seg_end
        # if nothing is in queues and approaching end of covered segment, request next chunk
        no_backlog = seg_req_q.empty() and seg_prefetch_q.empty()
        # when channel is idle, want next chunk sooner and shorter (faster "takeover" of playback)
        idle = not chan_main.get_busy()
        need_more = (no_backlog and remain_total > 0.05 and ((seg_end - pos_now) < THRESH or idle))
        if need_more:
            next_len = min(MAX_CHUNK if not idle else 4.0, remain_total if math.isfinite(remain_total) else MAX_CHUNK)
            if next_len > 0.12:
                try:
                    seg_req_q.put_nowait((current_track.path, seg_end, float(next_len)))
                except queue.Full:
                    pass
    volume = 0.85
    last_unmuted_volume = 0.85
    muted = False
    play_track_music(current_track, 0.0, volume)
    play_start_monotonic = time.monotonic()
    paused = False
    paused_accum = 0.0
    pause_started = None
    band_mapper_full = BandMapper(VIS.fft_size, AUDIO.target_sr, VIS.n_bands)
    hann_full = _hann(VIS.fft_size)
    hann_bass = _hann(VIS.bass_fft)
    def smooth_mask(freqs, low, high, roll=0.15):
        m = np.zeros_like(freqs, dtype=np.float32)
        band = (freqs>=low) & (freqs<=high)
        m[band] = 1.0
        bw = high-low
        r = max(1.0, roll*bw)
        left = (freqs>=low-r) & (freqs<low)
        right= (freqs>high) & (freqs<=high+r)
        if left.any():
            x = (freqs[left]- (low-r))/r
            m[left] = 0.5*(1-np.cos(np.pi*x))
        if right.any():
            x = 1 - (freqs[right]-high)/r
            m[right] = 0.5*(1-np.cos(np.pi*x))
        return m
    freqs_bass = np.fft.rfftfreq(VIS.bass_fft, 1/AUDIO.target_sr)
    bass_mask_f = smooth_mask(freqs_bass, AUDIO.bass_low_hz, AUDIO.bass_high_hz, roll=0.25)
    freqs_voice = np.fft.rfftfreq(VIS.fft_size, 1/AUDIO.target_sr)
    voice_mask_f = smooth_mask(freqs_voice, AUDIO.voice_low_hz, AUDIO.voice_high_hz, roll=0.20)
    fft_lock = threading.Lock()
    fft_req_pos = 0.0
    fft_should_run = True
    fft_event = threading.Event()
    fft_last_bands = np.zeros(VIS.n_bands, np.float32)
    fft_last_bass_energy = 0.0
    fft_last_voice_energy = 0.0
    def load_track_samples_quick(path: Path) -> Tuple[np.ndarray,int,float]:
        try:
            seg = AudioSegment.from_file(path).set_frame_rate(AUDIO.target_sr).set_channels(2).set_sample_width(2)
            arr = np.array(seg.get_array_of_samples()).reshape(-1,2).astype(np.float32)
            mono = arr.mean(axis=1)/32768.0
            # Convert to float16 to save memory
            mono = mono.astype(np.float16)
            dur = len(mono)/float(AUDIO.target_sr)
            return mono, AUDIO.target_sr, dur
        except Exception as e:
            log.debug("load_track_samples_quick failed for %s: %s", path.name, e)
            return np.zeros(0, np.float16), AUDIO.target_sr, 0.0
    PRIO_NOW = 0
    def fft_worker():
        nonlocal fft_last_bands, fft_last_bass_energy, fft_last_voice_energy
        log.debug("FFT thread started")
        while fft_should_run:
            _ = fft_event.wait(timeout=0.12)
            fft_event.clear()
            if not fft_should_run: break
            if paused:
                continue
            with fft_lock:
                key = _norm_key(current_track.path)
                pos = fft_req_pos
            try:
                if args.no_fft:
                    with fft_lock:
                        fft_last_bands.fill(0.0)
                        fft_last_bass_energy = 0.0
                        fft_last_voice_energy = 0.0
                    continue
                if args.viz_lowcpu and random.random() < 0.6:
                    t = time.time()
                    base = 0.15 + 0.08*math.sin(t*1.4)
                    with fft_lock:
                        fft_last_bands.fill(base)
                        fft_last_bass_energy = base
                        fft_last_voice_energy = base
                    continue
                cached = AN_CACHE.get(key)
                if cached is None:
                    _put_load(PRIO_NOW, "analyze", current_track.path)
                    t = time.time()
                    base = 0.12 + 0.06*math.sin(t*2.0)
                    with fft_lock:
                        fft_last_bands.fill(base)
                        fft_last_bass_energy = base
                        fft_last_voice_energy = base
                    continue
                samples, sr, _dur = cached
                # >>> NEW: write detected length if was unknown / significantly different
                if (_dur and _dur > 0.1) and (current_track.duration_sec <= 0.1 or abs(current_track.duration_sec - _dur) > 0.5):
                    current_track.duration_sec = float(_dur)
                if samples is None or samples.size == 0:
                    t = time.time()
                    base = 0.12 + 0.06*math.sin(t*2.0)
                    with fft_lock:
                        fft_last_bands.fill(base)
                        fft_last_bass_energy = base
                        fft_last_voice_energy = base
                    continue
                # Convert from float16 to float32 for FFT
                samples = samples.astype(np.float32)
                pos_an = max(0.0, pos - AUDIO.analysis_lag_sec)
                win_full = make_fft_window(samples, sr, pos_an, VIS.fft_size)
                sp_full = np.fft.rfft(win_full*hann_full)
                mag_full = np.abs(sp_full)
                bands = band_mapper_full.map(mag_full)
                win_bass = make_fft_window(samples, sr, pos_an, VIS.bass_fft)
                sp_b = np.fft.rfft(win_bass*hann_bass)
                bb = np.abs(sp_b) * bass_mask_f
                if bb.size > 0:
                    thresh = np.median(bb) * 0.3
                    bb_clean = bb[bb > thresh]
                    if bb_clean.size > 0:
                        bass_energy = float(np.sqrt(np.mean(bb_clean*bb_clean)))
                    else:
                        bass_energy = 0.0
                else:
                    bass_energy = 0.0
                vb = mag_full * voice_mask_f
                voice_energy = float(np.sqrt(np.mean(vb*vb))) if vb.size else 0.0
                with fft_lock:
                    fft_last_bands[:] = bands
                    fft_last_bass_energy = bass_energy * 1.5
                    fft_last_voice_energy = voice_energy
            except Exception as e:
                t = time.time()
                base = 0.12 + 0.06*math.sin(t*2.0)
                with fft_lock:
                    fft_last_bands.fill(base)
                    fft_last_bass_energy = base
                    fft_last_voice_energy = base
                log.debug("fft_worker error: %s", e)
        log.debug("FFT thread exit")
    from queue import PriorityQueue
    load_q: "PriorityQueue[tuple[int,int,str,Path]]" = PriorityQueue()
    load_should_run = True
    _load_seq = itertools.count()
    PRIO_NOW = 0
    PRIO_PREFETCH = 5
    PRIO_BULK = 9
    def _put_load(prio: int, action: str, path: Path):
        load_q.put((prio, next(_load_seq), action, path))
    def loader_worker_fn():
        log.debug("Loader worker started")
        while load_should_run:
            try:
                prio, _, action, path = load_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not load_should_run:
                break
            try:
                key = _norm_key(path)
                if action in ("analyze","prefetch"):
                    if AN_CACHE.get(key) is not None:
                        pass
                    else:
                        samples, sr, dur = load_track_samples_quick(path)
                        AN_CACHE.put(key, samples, sr, dur)
                        # >>> NEW: if it's current track and has no length, write it
                        if path == current_track.path and (current_track.duration_sec <= 0.1) and (dur > 0.1):
                            current_track.duration_sec = float(dur)
                        log.debug("Loader: %s %s (%.1fs)", action, path.name, dur)
            except Exception as e:
                log.debug("Loader error: %s", e)
            finally:
                load_q.task_done()
        log.debug("Loader worker exit")
    # Initialize cache with configured capacity (moved BEFORE starting threads)
    global AN_CACHE
    AN_CACHE = AnalysisCache(capacity=max(1, args.cache_cap))
    _put_load(PRIO_NOW, "analyze", current_track.path)
    fft_thread = threading.Thread(target=fft_worker, daemon=True); fft_thread.start()
    loader_threads: List[threading.Thread] = []
    num_workers = max(2, min(4, (os.cpu_count() or 4)//2))
    for _ in range(num_workers):
        t = threading.Thread(target=loader_worker_fn, daemon=True)
        t.start()
        loader_threads.append(t)
    time_last_sec = -1
    time_cache_surf: Optional[pygame.Surface] = None
    title_cache_key = None
    title_cache_surf: Optional[pygame.Surface] = None
    hud_cache_key = None
    hud_cache_surf: Optional[pygame.Surface] = None
    last_volume_popup_t = 0.0
    toast_msg = ""
    toast_until = 0.0
    def show_toast(msg: str):
        nonlocal toast_msg, toast_until
        toast_msg = msg
        toast_until = time.time() + UI.toast_sec
        log.debug("Toast: %s", msg)
    def build_hud():
        ui=f"[S]huffle: {'ON' if shuffle else 'OFF'}  [R]epeat: {'ALL' if repeat_all else 'OFF'}  [F] Fake Fullscreen  [V/F2] View: {v_mode_desc(v_mode)}  [H] Help  [1..5] Presets"
        return font_small_sys.render(ui, True, VIS.text_dim)
    choose_background(False)
    flash=0.0
    bass_env = 0.0
    bass_peak = 1e-6
    hist_len = int(0.7*VIS.fps_target)
    bass_hist = np.zeros(hist_len, dtype=np.float32)
    hist_idx = 0
    hist_filled = 0
    bass_last_beat = 0.0
    ema = 0.0
    ema_decay = 0.92
    mad_ema = 0.0
    mad_decay = 0.90
    voice_env = 0.0
    voice_peak = 1e-6
    last_nav_time = 0.0
    clock=pygame.time.Clock()
    frame_idx = 0
    running=True
    bg_switch_pending = False
    resize_pending = False
    last_resize_time = 0.0
    end_of_queue_reached = False
    seg_cur_start = 0.0
    seg_cur_len = 0.0
    # ========== LYRICS STATE ========== (NOVÝ KÓD)
    lyrics_enabled = False
    current_lyrics = []
    lyrics_font = load_cyber(24)  # Font pre text piesne
    # ========== END LYRICS STATE ========== (NOVÝ KÓD)
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running=False; break
            if ev.type == pygame.VIDEORESIZE:
                resize_pending = True; last_resize_time = time.time()
                vis_surf = pygame.Surface(ev.size, pygame.SRCALPHA)
                text_surf = pygame.Surface(ev.size, pygame.SRCALPHA)
                log.debug("VIDEORESIZE -> %s", ev.size)
            if ev.type == pygame.KEYDOWN:
                log.debug("KeyDown: %s", pygame.key.name(ev.key))
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running=False; break
                if pygame.K_1 <= ev.key <= pygame.K_5:
                    v_mode = preset_indices[ev.key - pygame.K_1]; hud_cache_surf=None; hud_cache_key=None
                elif ev.key == pygame.K_f:
                    toggle_fake_fullscreen()
                elif ev.key == pygame.K_v:
                    i = preset_indices.index(v_mode) if v_mode in preset_indices else 0
                    v_mode = preset_indices[(i+1)%len(preset_indices)]; hud_cache_surf=None; hud_cache_key=None
                elif ev.key == pygame.K_F2:
                    v_mode = (v_mode + 1) % len(v_states); hud_cache_surf=None; hud_cache_key=None
                elif ev.key == pygame.K_h:
                    help_visible = not help_visible; help_cache_surf = None
                # ========== LYRICS TOGGLE ========== (NOVÝ KÓD)
                elif ev.key == pygame.K_l:
                    lyrics_enabled = not lyrics_enabled
                    if lyrics_enabled:
                        # Načítať LRC súbor pre aktuálnu skladbu
                        lrc_path = current_track.path.with_suffix('.lrc')
                        cache_key = str(lrc_path)
                        if cache_key in _lyrics_cache:
                            current_lyrics = _lyrics_cache[cache_key]
                        else:
                            current_lyrics = load_lrc_file(lrc_path)
                            _lyrics_cache[cache_key] = current_lyrics
                        show_toast("Lyrics ON")
                    else:
                        current_lyrics = []
                        show_toast("Lyrics OFF")
                # ========== END LYRICS TOGGLE ========== (NOVÝ KÓD)
                elif ev.key == pygame.K_t:
                    win_toggle_topmost()
                elif ev.key == pygame.K_b:
                    bg_enabled = not bg_enabled
                    if bg_enabled:
                        choose_background(True)
                    else:
                        # turn off background completely, don't draw and enable LMB-drag
                        current_bg = None
                        bg_dark = None
                        original_bg_surface = None
                        bg_size_cache.clear()
                elif ev.key == pygame.K_LEFTBRACKET:
                    if bg_paths:
                        bg_index = (bg_index - 1) % len(bg_paths); choose_background(True)
                elif ev.key == pygame.K_RIGHTBRACKET:
                    if bg_paths:
                        bg_index = (bg_index + 1) % len(bg_paths); choose_background(True)
                elif ev.key == pygame.K_SPACE:
                    if paused:
                        if playback_mode == "music":
                            pygame.mixer.music.unpause()
                        else:
                            pygame.mixer.unpause()
                        paused=False
                        if pause_started is not None:
                            paused_accum += time.monotonic() - pause_started; pause_started=None
                        if playback_mode == "music":
                            pos_ms = pygame.mixer.music.get_pos()
                            if pos_ms is not None and pos_ms >= 0:
                                set_play_start(pos_ms / 1000.0)
                        with fft_lock:
                            bass_peak_val = max(bass_peak*0.995, fft_last_bass_energy)
                            bass_norm_val = fft_last_bass_energy / (bass_peak_val + 1e-9)
                            bass_env = bass_norm_val
                            voice_peak_val = max(voice_peak * AUDIO.voice_peak_decay, fft_last_voice_energy)
                            voice_norm_val = fft_last_voice_energy / (voice_peak_val + 1e-9)
                            voice_env = voice_norm_val
                        log.debug("Playback resumed")
                    else:
                        if playback_mode == "music":
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.pause()
                        paused=True
                        if pause_started is None: pause_started = time.monotonic()
                        bass_env = 0.0; bass_peak = 1e-6; hist_idx = 0; hist_filled = 0; bass_hist.fill(0)
                        ema = 0.0; mad_ema = 0.0
                        log.debug("Playback paused")
                elif ev.key in (pygame.K_n, pygame.K_p):
                    if end_of_queue_reached and ev.key == pygame.K_p:
                        target_idx = len(order) - 1
                        end_of_queue_reached = False
                    elif end_of_queue_reached and ev.key == pygame.K_n:
                        target_idx = 0
                        end_of_queue_reached = False
                    else:
                        nowk=time.time()
                        if nowk - last_nav_time < UI.next_cooldown_sec: continue
                        last_nav_time = nowk
                        if ev.key==pygame.K_n:
                            if (index_in_order+1<len(order)) or repeat_all:
                                target_idx = (index_in_order+1) % len(order)
                            else:
                                show_toast("End of queue")
                                end_of_queue_reached = True
                                stop_all_playback()
                                continue
                        else:
                            target_idx = (index_in_order - 1) % len(order)
                    target_track = lib.tracks[order[target_idx]]
                    current_track = target_track
                    index_in_order = target_idx
                    set_play_start(0.0)
                    play_track_music(current_track, 0.0, volume)
                    bg_switch_pending = True
                    _clear_seg_prefetch()
                    _put_load(PRIO_NOW, "analyze", current_track.path)
                    if (index_in_order+1<len(order)) or repeat_all:
                        nxt = lib.tracks[order[(index_in_order+1)%len(order)]]
                        _put_load(PRIO_PREFETCH, "prefetch", nxt.path)
                    log.debug("Switched track -> %s", current_track.path.name)
                    # ========== RESET LYRICS ON TRACK CHANGE ========== (NOVÝ KÓD)
                    # ========== RESET LYRICS ON TRACK CHANGE ========== (OPRAVENÝ KÓD)
                    current_lyrics = []
                    if lyrics_enabled:
                        # Načítať LRC súbor pre novú skladbu
                        lrc_path = current_track.path.with_suffix('.lrc')
                        cache_key = str(lrc_path)
                        if cache_key in _lyrics_cache:
                            current_lyrics = _lyrics_cache[cache_key]
                        else:
                            current_lyrics = load_lrc_file(lrc_path)
                            _lyrics_cache[cache_key] = current_lyrics
                    # ========== END RESET LYRICS ========== (OPRAVENÝ KÓD)
                    # ========== END RESET LYRICS ========== (NOVÝ KÓD)
                elif ev.key == pygame.K_s:
                    shuffle = not shuffle
                    head = [order[index_in_order]]
                    remaining = [i for i in range(n) if i != order[index_in_order]]
                    if shuffle:
                        random.shuffle(remaining)
                    order = head + remaining
                    index_in_order = 0
                    msg = f"Shuffle {'ON' if shuffle else 'OFF'}"
                    show_toast(msg)
                    log.debug("Shuffle -> %s", "ON" if shuffle else "OFF")
                elif ev.key == pygame.K_r:
                    # Cyklus: OFF -> ALL -> ONE -> OFF
                    if not repeat_all and not repeat_one:
                        repeat_all = True
                        repeat_one = False
                        msg = "Repeat ALL"
                    elif repeat_all and not repeat_one:
                        repeat_all = False
                        repeat_one = True
                        msg = "Repeat ONE"
                    else:  # repeat_one is True
                        repeat_all = False
                        repeat_one = False
                        msg = "Repeat OFF"
                    show_toast(msg)
                    log.debug("Repeat mode -> %s", msg)
                elif ev.key == pygame.K_RIGHT:
                    seek_relative(+5.0)
                elif ev.key == pygame.K_LEFT:
                    seek_relative(-5.0)
                elif ev.key == pygame.K_m:
                    if muted:
                        volume = last_unmuted_volume
                        muted = False
                        if playback_mode == "music":
                            pygame.mixer.music.set_volume(volume)
                        else:
                            chan_main.set_volume(volume, volume)
                        log.debug("Unmuted. Volume restored to %.2f", volume)
                    else:
                        last_unmuted_volume = volume
                        volume = 0.0
                        muted = True
                        if playback_mode == "music":
                            pygame.mixer.music.set_volume(volume)
                        else:
                            chan_main.set_volume(volume, volume)
                        log.debug("Muted. Volume set to 0.0")
                    last_volume_popup_t = time.time()
                elif ev.key == pygame.K_UP:
                    volume = min(1.0, volume + 0.05)
                    if playback_mode == "music":
                        pygame.mixer.music.set_volume(volume)
                    else:
                        chan_main.set_volume(volume, volume)
                    muted = False
                    last_volume_popup_t = time.time()
                    log.debug("Volume Up -> %.2f", volume)
                elif ev.key == pygame.K_DOWN:
                    volume = max(0.0, volume - 0.05)
                    if playback_mode == "music":
                        pygame.mixer.music.set_volume(volume)
                    else:
                        chan_main.set_volume(volume, volume)
                    muted = (volume <= 0.0)
                    last_volume_popup_t = time.time()
                    log.debug("Volume Down -> %.2f", volume)
                elif ev.key == pygame.K_c:
                    sepia_enabled = not sepia_enabled
                    show_toast(f"Sepia {'ON' if sepia_enabled else 'OFF'}")
            elif ev.type == pygame.MOUSEWHEEL:
                volume = float(np.clip(volume + ev.y * 0.03, 0.0, 1.0))
                if playback_mode == "music": pygame.mixer.music.set_volume(volume)
                else: chan_main.set_volume(volume, volume)
                last_volume_popup_t = time.time()
                log.debug("Volume (wheel) -> %.0f%%", volume*100)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 4:
                    volume = min(1.0, volume + 0.03)
                    if playback_mode == "music": pygame.mixer.music.set_volume(volume)
                    else: chan_main.set_volume(volume, volume)
                    muted = False
                    last_volume_popup_t = time.time(); log.debug("Volume + -> %.0f%%", volume*100)
                elif ev.button == 5:
                    volume = max(0.0, volume - 0.03)
                    if playback_mode == "music": pygame.mixer.music.set_volume(volume)
                    else: chan_main.set_volume(volume, volume)
                    if volume <= 0.0:
                        muted = True
                    last_volume_popup_t = time.time(); log.debug("Volume - -> %.0f%%", volume*100)
                elif ev.button == 1 and not current_bg:
                    win_drag_window()
        if resize_pending and (time.time() - last_resize_time) > 0.15:
            rescale_background_for_size() if original_bg_surface is not None else choose_background(True)
            resize_pending = False
        pos_now = get_play_pos()
        dur_now = current_track.duration_sec or 0.0
        time_left = (dur_now - pos_now) if dur_now>0 else 999.0
        if (not paused) and (dur_now > 0):
            if time_left < 30.0 and ((index_in_order+1<len(order)) or repeat_all):
                nxt = lib.tracks[order[(index_in_order+1)%len(order)]]
                _put_load(PRIO_PREFETCH, "prefetch", nxt.path)
        if bg_switch_pending and ((playback_mode == "music" and pygame.mixer.music.get_busy()) or (playback_mode == "channel" and chan_main.get_busy())):
            choose_background(False); bg_switch_pending = False
        w,h = screen.get_size()
        if current_bg and (current_bg.get_width()!=w or current_bg.get_height()!=h):
            rescale_background_for_size()
        if bg_enabled and current_bg:
            screen.blit(current_bg,(0,0))
            if bg_dark: screen.blit(bg_dark,(0,0))
        else:
            screen.fill(VIS.bg_color)
        vis_surf.fill((0,0,0,0))
        # text_surf: will be cleared in render_ui_and_text()
        if (not paused) and ((frame_idx % max(1, int(VIS.fft_every_n_frames))) == 0):
            with fft_lock: fft_req_pos = pos_now
            fft_event.set()
        with fft_lock:
            bands = fft_last_bands.copy()
            bass_energy = float(fft_last_bass_energy)
            voice_energy = float(fft_last_voice_energy)
        if not paused:
            bass_peak = max(bass_peak*0.995, bass_energy)
            bass_norm = bass_energy / (bass_peak + 1e-9)
            if bass_norm > bass_env: bass_env += AUDIO.attack*(bass_norm - bass_env)
            else: bass_env += (AUDIO.rel_slow if bass_norm<0.1 else AUDIO.rel_fast)*(bass_norm - bass_env)
            bass_env = float(np.clip(bass_env, 0.0, 1.0))
            bass_hist[hist_idx] = bass_env
            hist_idx = (hist_idx + 1) % hist_len
            hist_filled = min(hist_filled+1, hist_len)
            hist_view = bass_hist[:hist_filled]
            if hist_filled > 4:
                ema = ema_decay*ema + (1-ema_decay)*bass_env
                mad = np.median(np.abs(hist_view - np.median(hist_view)))
                mad_ema = mad_decay*mad_ema + (1-mad_decay)*mad
                thresh = ema + (1.5 if hist_filled<30 else 1.2) * (mad_ema*1.4826 + 1e-6)
            else:
                thresh = 0.8
            nowt = time.time()
            if (bass_env > thresh) and ((nowt - bass_last_beat) > AUDIO.beat_min_gap):
                flash = 1.0; bass_last_beat = nowt
            flash = max(0.0, flash*0.85)
            voice_peak = max(voice_peak * AUDIO.voice_peak_decay, voice_energy)
            voice_norm = voice_energy / (voice_peak + 1e-9)
            if voice_norm > voice_env:
                voice_env += AUDIO.voice_attack * (voice_norm - voice_env)
            else:
                voice_env += (AUDIO.voice_rel_slow if voice_norm < 0.1 else AUDIO.voice_rel_fast) * (voice_norm - voice_env)
            voice_env = max(0.0, min(1.0, voice_env))
        state = {
            "pos": pos_now,
            "dur": dur_now,
            "bass_env": bass_env,
            "flash": flash,
            "bands": bands,
            "fps": clock.get_fps(),
            "voice_env": voice_env,
            "frame_idx": frame_idx,
            "lyrics_next_word": getattr(draw_visuals, "_current_word", "")
        }
        draw_visuals(screen, vis_surf, state)
        screen.blit(vis_surf, (0, 0))
        w, h = screen.get_size()
        cx, cy = w // 2, h // 2
        st_view = v_states[v_mode]
        show_time = st_view["time"]
        show_title = st_view["title"]
        show_hud = st_view["hud"]
        show_fps = st_view["fps"]
        any_text_allowed = show_time or show_title or show_hud or show_fps
        # === TEXT LAYER (top-left stack: VOL, TOAST, HUD, TITLE, TIME; FPS we'll add below) ===
        vol_show, y_cursor = render_ui_and_text(
            text_surf, w, h, cx, cy,
            pos_now, dur_now, volume,
            last_volume_popup_t, toast_msg, toast_until,
            show_title, show_time, show_hud, show_fps,
            v_mode, shuffle, repeat_all, repeat_one,  # <-- Pridané repeat_one
            current_track, title_font, font_small_sys, vol_font_sys
        )
        # Final (single) blit of text layer
        if any_text_allowed or (time.time() < toast_until) or vol_show:
            screen.blit(text_surf, (0, 0))
        # HELP overlay (separate layer)
        if help_visible:
            if (help_cache_surf is None) or (help_cache_surf.get_width()!=w or help_cache_surf.get_height()!=h):
                help_cache_surf = build_help_surface(w,h)
            screen.blit(help_cache_surf, (0,0))
        # ========== RENDER LYRICS ========== (NOVÝ KÓD)
        if lyrics_enabled and current_lyrics:
            current_line = get_current_lyric_line(current_lyrics, pos_now)
            if current_line:
                lyrics_surf = lyrics_font.render(current_line, True, VIS.yellow)  # Žltá farba
                # Umiestniť text na vrch obrazovky
                screen.blit(lyrics_surf, (w//2 - lyrics_surf.get_width()//2, 10))
        # ========== END RENDER LYRICS ========== (NOVÝ KÓD)
        # Check if we need to queue next segment
        maybe_queue_next_segment(pos_now)
        # Apply sepia filter if enabled
        # Apply sepia filter if enabled
        if sepia_enabled:
            apply_sepia(screen, 0.8)
        # >>>> AUTO-NEXT LOGIC - PRESUNUTÉ SEM <<<<
        # Auto-next
        if dur_now > 0:
            track_ended = False
        # 1) sme úplne na konci skladby (malý buffer 50 ms)
        if pos_now >= (dur_now - 0.05):
            if playback_mode == "music":
                track_ended = not pygame.mixer.music.get_busy()
            else:
                track_ended = not chan_main.get_busy()
        if track_ended:
            # 2) vyrátaj next_idx jednoznačne (repeat_one má prednosť)
            if repeat_one:
                next_idx = index_in_order
            else:
                has_next = (index_in_order + 1) < len(order)
                if has_next or repeat_all:
                    next_idx = (index_in_order + 1) % len(order)
                else:
                    next_idx = index_in_order  # žiadna ďalšia skladba a repeat_all je OFF
            # 3) rozhodni: preskoč na ďalšiu alebo ukonči queue
            if next_idx != index_in_order:
                next_track = lib.tracks[order[next_idx]]
                current_track = next_track
                index_in_order = next_idx
                set_play_start(0.0)
                play_track_music(current_track, 0.0, volume)
                bg_switch_pending = True
                _clear_seg_prefetch()
                _put_load(PRIO_NOW, "analyze", current_track.path)
                if (index_in_order + 1) < len(order) or repeat_all:
                    nxt = lib.tracks[order[(index_in_order + 1) % len(order)]]
                    _put_load(PRIO_PREFETCH, "prefetch", nxt.path)
                log.debug("Auto next -> %s", current_track.path.name)
                # ========== RESET LYRICS ON AUTO-NEXT ========== (NOVÝ KÓD)
                # ========== RESET LYRICS ON AUTO-NEXT ========== (OPRAVENÝ KÓD)
                current_lyrics = []
                if lyrics_enabled:
                    # Načítať LRC súbor pre novú skladbu
                    lrc_path = current_track.path.with_suffix('.lrc')
                    cache_key = str(lrc_path)
                    if cache_key in _lyrics_cache:
                        current_lyrics = _lyrics_cache[cache_key]
                    else:
                        current_lyrics = load_lrc_file(lrc_path)
                        _lyrics_cache[cache_key] = current_lyrics
                # ========== END RESET LYRICS ========== (OPRAVENÝ KÓD)
                # ========== END RESET LYRICS ========== (NOVÝ KÓD)
            else:
                show_toast("End of queue")
                end_of_queue_reached = True
                stop_all_playback()
        # >>>> KONIEC AUTO-NEXT BLOKU <<<<
        pygame.display.flip()
        frame_idx += 1
        # Use regular tick instead of busy_loop for lower CPU usage
        clock.tick(VIS.fps_target)
    # >>>> TU UŽ JE KONIEC HLAVNEJ WHILE SLUČKY <<<<
    # Všetko nižšie sa vykoná až po stlačení ESC/Q
    try:
        log.debug("Shutting down…")
        fft_should_run = False
        fft_event.set()
        load_should_run = False
        seg_worker_should_run = False
        # wake worker if sleeping
        try: seg_req_q.put_nowait((current_track.path, 0.0, 0.0))
        except Exception: pass
        try:
            for _ in loader_threads:
                _put_load(PRIO_BULK, "prefetch", current_track.path)
        except Exception:
            pass
        try:
            chan_main.stop()
        except Exception:
            pass
        time.sleep(0)
    finally:
        try:
            if fft_thread.is_alive(): fft_thread.join(timeout=0.5)
        except Exception: pass
        for t in loader_threads:
            try: t.join(timeout=0.5)
            except Exception: pass
        if seg_worker.is_alive():
            seg_worker.join(timeout=0.5)
        pygame.quit()
        log.debug("Exited cleanly")
if __name__=="__main__":
    main()
