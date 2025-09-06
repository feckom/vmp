from __future__ import annotations
import os, sys, math, random, re, time, argparse, logging, threading, subprocess, ctypes, platform, shutil, queue, collections, itertools
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import io
import socket
import json
import base64
import asyncio
import websockets
os.environ.setdefault("SDL_RENDER_DRIVER", "direct3d")
os.environ.setdefault("SDL_HINT_RENDER_SCALE_QUALITY", "1")
import numpy as np
import pygame
from pygame import gfxdraw
import pygame.sndarray
from mutagen import File as MutaFile
from mutagen.easyid3 import EasyID3
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
# ---------------- Config ----------------
@dataclass
class AudioCfg:
    target_sr: int = 44100
    crossfade_sec: float = 12.0
    bass_low_hz: int = 30
    bass_high_hz: int = 150
    analysis_lag_sec: float = 0.030
    attack: float = 0.45
    rel_fast: float = 0.12
    rel_slow: float = 0.04
    beat_min_gap: float = 0.11
    eq_default: dict = None
    softclip: bool = True
    voice_low_hz: int = 200
    voice_high_hz: int = 3800
    voice_attack: float = 0.60
    voice_rel_fast: float = 0.10
    voice_rel_slow: float = 0.05
    voice_peak_decay: float = 0.995
@dataclass
class VisualCfg:
    fps_target: int = 60
    fft_size: int = 2048
    bass_fft: int = 1024
    n_bands: int = 64
    fft_every_n_frames: int = 2
    ui_alpha: int = 153  # ~60% transparency
    dotted_count: int = 96
    progress_width: int = 14
    bar_thickness: int = 5
    bar_max_len_frac: float = 0.26
    ring_radius_frac: float = 0.20
    glow_color: Tuple[int,int,int]=(255,40,0)
    bar_color: Tuple[int,int,int]=(255,64,0)
    white: Tuple[int,int,int]=(255,255,255)
    red: Tuple[int,int,int]=(255,32,32)
    yellow: Tuple[int,int,int]=(255,220,0)
    text_dim: Tuple[int,int,int]=(230,230,230)
    bg_color: Tuple[int,int,int]=(8,8,8)
    title_font_size: int = 18
    voice_base_color: Tuple[int,int,int]=(255,80,40)
    voice_peak_color: Tuple[int,int,int]=(255,60,60)
    voice_base_radius_scale: float = 0.62
    voice_max_pulse_px: int = 18
    voice_alpha_min: int = 40
    voice_alpha_max: int = 200
@dataclass
class UiCfg:
    next_cooldown_sec: float = 0.35
    volume_popup_sec: float = 0.9
    toast_sec: float = 1.4
    seek_segment_sec: float = 30.0
AUDIO = AudioCfg(eq_default={"low":+1.5,"mid":0.0,"high":+0.5}, softclip=True)
VIS = VisualCfg()
UI = UiCfg()
MUSIC_DIR = r"C:\Music"
BG_DIR    = ""
SCAN_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
FONT_PATH  = Path(__file__).with_name("cyberpunk.ttf")
log = logging.getLogger("vmp")
# ---------------- Logging ----------------
class WebTermHandler(logging.Handler):
    """Custom log handler that sends messages to the web terminal."""
    def __init__(self):
        super().__init__()
        self.clients = set()
        self.lock = threading.Lock()
    def emit(self, record):
        try:
            msg = self.format(record)
            with self.lock:
                for client in list(self.clients):
                    try:
                        asyncio.run_coroutine_threadsafe(client.send(msg), asyncio.get_event_loop())
                    except Exception:
                        pass
        except Exception:
            pass
    def add_client(self, client):
        with self.lock:
            self.clients.add(client)
    def remove_client(self, client):
        with self.lock:
            self.clients.discard(client)
def setup_logging(debug=False, webterm_handler=None):
    import logging.handlers
    lg = logging.getLogger("vmp")
    for h in list(lg.handlers): lg.removeHandler(h)
    lg.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(logging.DEBUG if debug else logging.INFO)
    lg.addHandler(ch)
    if webterm_handler:
        webterm_handler.setFormatter(fmt)
        webterm_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        lg.addHandler(webterm_handler)
    try:
        log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(log_dir/"player.log", maxBytes=4_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); lg.addHandler(fh)
    except Exception as e:
        print("Log file init failed:", e, file=sys.stderr)
    logging.getLogger().setLevel(logging.WARNING)
    lg.debug("Logging initialized. Debug=%s", debug)
    return lg
# ---------------- FFmpeg ----------------
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
# ---------------- Utils ----------------
def format_time(sec: float) -> str:
    sec = max(0, int(sec)); m, s = divmod(sec, 60); return f"{m:02d}:{s:02d}"
_CLEAN_PAT   = re.compile(r"(?i)\b(official\s*video|lyrics?|audio|hd|hq|remaster(?:ed)?|live|clip|mv|visualizer|topic|karaoke|instrumental|mono|stereo|remix|mix|edit|radio\s*edit|feat\.?.+|ft\.?.+)\b")
_BRACKETS    = re.compile(r"[\(\[\{].*?[\)\]\}]")
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
# ---------------- Cache Key Normalize ----------------
def _norm_key(path: Path | str) -> str:
    try:
        return str(Path(path).resolve()).lower()
    except Exception:
        return str(path).replace("\\", "/").lower()
# ---------------- Track / Library ----------------
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
# LRU cache for analyzed mono samples
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
AN_CACHE = AnalysisCache(capacity=64)
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
# ---------------- DSP (EQ/Limiter) ----------------
def biquad_low_shelf(f0, sr, gain_db, S=1.0):
    A = 10**(gain_db/40); w0=2*math.pi*f0/sr; alpha=math.sin(w0)/2*math.sqrt((A+1/A)*(1/S-1)+2)
    cosw=math.cos(w0)
    b0 =    A*((A+1)-(A-1)*cosw + 2*np.sqrt(A)*alpha)
    b1 =  2*A*((A-1)-(A+1)*cosw)
    b2 =    A*((A+1)-(A-1)*cosw - 2*np.sqrt(A)*alpha)
    a0 =        (A+1+(A-1)*cosw + 2*np.sqrt(A)*alpha)
    a1 =   -2*((A-1)+(A+1)*cosw)
    a2 =        (A+1+(A-1)*cosw - 2*np.sqrt(A)*alpha)
    return np.array([b0/a0,b1/a0,b2/a0,a1/a0,a2/a0], dtype=np.float64)
def biquad_high_shelf(f0, sr, gain_db, S=1.0):
    A = 10**(gain_db/40); w0=2*math.pi*f0/sr; alpha=math.sin(w0)/2*math.sqrt((A+1/A)*(1/S-1)+2)
    cosw=math.cos(w0)
    b0 =    A*( (A+1)+(A-1)*cosw + 2*np.sqrt(A)*alpha)
    b1 = -2*A*( (A-1)+(A+1)*cosw )
    b2 =    A*( (A+1)+(A-1)*cosw - 2*np.sqrt(A)*alpha)
    a0 =        (A+1-(A-1)*cosw + 2*np.sqrt(A)*alpha)
    a1 =    2*( (A-1)-(A+1)*cosw )
    a2 =        (A+1-(A-1)*cosw - 2*np.sqrt(A)*alpha)
    return np.array([b0/a0,b1/a0,b2/a0,a1/a0,a2/a0], dtype=np.float64)
def biquad_peaking(f0, sr, gain_db, Q=1.0):
    A = 10**(gain_db/40); w0=2*math.pi*f0/sr; alpha=math.sin(w0)/(2*Q); cosw=math.cos(w0)
    b0 = 1 + alpha*A
    b1 = -2*cosw
    b2 = 1 - alpha*A
    a0 = 1 + alpha/A
    a1 = -2*cosw
    a2 = 1 - alpha/A
    return np.array([b0/a0,b1/a0,b2/a0,a1/a0,a2/a0], dtype=np.float64)
def biquad_filter(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    b0,b1,b2,a1,a2 = coeffs
    y = np.zeros_like(x, dtype=np.float32)
    x1=x2=0.0; y1=y2=0.0
    for i in range(len(x)):
        xn = x[i]
        yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2
        y[i]=yn; x2=x1; x1=xn; y2=y1; y1=yn
    return y
def apply_eq_limiter_stereo(buf: np.ndarray, sr: int, eq_db: dict, softclip=True) -> np.ndarray:
    if eq_db:
        ls = biquad_low_shelf(120, sr, eq_db.get("low",0.0))
        pk = biquad_peaking(1000, sr, eq_db.get("mid",0.0), Q=0.8)
        hs = biquad_high_shelf(8000, sr, eq_db.get("high",0.0))
        for ch in (0,1):
            c = buf[:,ch]; c = biquad_filter(c, ls); c = biquad_filter(c, pk); c = biquad_filter(c, hs); buf[:,ch] = c
    if softclip:
        buf = np.tanh(2.2*buf) / np.tanh(2.2)
    return np.clip(buf, -0.999, 0.999)
# ---------------- Decode helpers ----------------
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
    return (data.reshape(-1,2).astype(np.float32))/32768.0
def numpy_to_sound(arr_float_stereo: np.ndarray) -> pygame.mixer.Sound:
    arr = np.clip(arr_float_stereo, -1.0, 1.0)
    arr16 = (arr * 32767.0).astype(np.int16)
    return pygame.sndarray.make_sound(arr16)
# ---------------- FFT Window & Mapping ----------------
_hann_cache: Dict[int, np.ndarray] = {}
def _hann(size: int) -> np.ndarray:
    h = _hann_cache.get(size)
    if h is None:
        log.debug("Create Hann window: %d", size)
        h = np.hanning(size).astype(np.float32)
        _hann_cache[size] = h
    return h
# Reusable window buffers to avoid allocations
_fft_window_cache: Dict[int, np.ndarray] = {}
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
    """Log-spaced bands with precomputed index slices and vectorized RMS."""
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
        self.band_weights = np.linspace(1.15, 0.85, n_bands).astype(np.float32)
    def map(self, spectrum_abs: np.ndarray) -> np.ndarray:
        # Pre-allocate result array
        all_indices = np.concatenate(self.idxs)
        all_values = spectrum_abs[all_indices]
        # Calculate start and end indices for each band
        lens = np.array([len(idx) for idx in self.idxs])
        split_points = np.cumsum(lens[:-1])  # Points to split the concatenated array
        # Split and calculate RMS for each band
        band_rms = np.array([
            np.sqrt(np.mean(chunk * chunk)) if len(chunk) > 0 else 0.0
            for chunk in np.split(all_values, split_points)
        ], dtype=np.float32)
        vmax = band_rms.max()
        if vmax > 0:
            band_rms /= vmax
        band_rms *= self.band_weights
        return np.clip(band_rms, 0.0, 1.0)
# ---------------- Surfaces Cache ----------------
_cached_dotted: Dict[int, pygame.Surface] = {}
_cached_glow: Dict[Tuple[int,int,Tuple[int,int,int],int], pygame.Surface] = {}
def build_dotted_ring_surface(radius: int, count: int, color_rgb: Tuple[int,int,int]) -> pygame.Surface:
    if radius in _cached_dotted: return _cached_dotted[radius]
    size = radius*2+6
    surf = pygame.Surface((size,size), pygame.SRCALPHA)
    cx=cy=size//2
    col = (*color_rgb,255)
    for i in range(count):
        a=2*math.pi*i/count-math.pi/2
        x=int(cx+radius*math.cos(a)); y=int(cy+radius*math.sin(a))
        gfxdraw.filled_circle(surf,x,y,2,col)
    _cached_dotted[radius]=surf
    log.debug("Build dotted ring: r=%d", radius)
    return surf
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
        pygame.draw.circle(surf, col, center, radius+i, width=thickness)
    _cached_glow[key]=surf
    # intentionally no debug log here (to reduce spam)
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
def lerp(a,b,t): return a+(b-a)*max(0.0,min(1.0,t))
def lerp_color(c1,c2,t): return (int(lerp(c1[0],c2[0],t)),int(lerp(c1[1],c2[1],t)),int(lerp(c1[2],c2[2],t)))
def load_system(size:int, bold=True):
    name = pygame.font.match_font('segoe ui,segoeui,arial,tahoma,verdana,calibri,dejavusans', bold=bold)
    return pygame.font.Font(name, size) if name else pygame.font.SysFont(None, size, bold=bold)
def load_cyber(size:int):
    try: return pygame.font.Font(str(FONT_PATH), size)
    except Exception: return load_system(size, bold=True)
# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Cyberpunk circular music player (optimized)")
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
    ap.add_argument("--webterm", action="store_true", help="Start web terminal on port 3030")
    return ap.parse_args()
# ---------------- Win helpers ----------------
def win_toggle_topmost():
    if platform.system() != "Windows": return
    try:
        hwnd = pygame.display.get_wm_info().get("window")
        u32 = ctypes.windll.user32
        GWL_EXSTYLE = -20; WS_EX_TOPMOST = 0x00000008
        SWP_NOMOVE=0x0002; SWP_NOSIZE=0x0001; SWP_SHOWWINDOW=0x0040
        is_top = bool(u32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOPMOST)
        u32.SetWindowPos(hwnd, -1 if not is_top else -2, 0,0,0,0, SWP_NOMOVE|SWP_NOSIZE|SWP_SHOWWINDOW)
        log.debug("Topmost toggled -> %s", "ON" if not is_top else "OFF")
    except Exception as e:
        log.debug("Topmost toggle failed: %s", e)
def win_drag_window():
    if platform.system() != "Windows": return
    try:
        hwnd = pygame.display.get_wm_info().get("window")
        u32 = ctypes.windll.user32
        u32.ReleaseCapture()
        u32.SendMessageW(hwnd, 0x00A1, 2, 0)
    except Exception as e:
        log.debug("Drag failed: %s", e)
def get_desktop_size() -> Tuple[int,int]:
    try:
        info = pygame.display.Info()
        return (max(640, int(info.current_w)), max(480, int(info.current_h)))
    except Exception:
        return (1920,1080)
# ---------------- Precomputes ----------------
COS_ARR = np.cos(2*np.pi*np.arange(VIS.n_bands)/VIS.n_bands - np.pi/2.0).astype(np.float32)
SIN_ARR = np.sin(2*np.pi*np.arange(VIS.n_bands)/VIS.n_bands - np.pi/2.0).astype(np.float32)
# ---------------- Visualization ----------------
def draw_visuals(screen, vis_surf, state):
    w,h = screen.get_size()
    cx,cy = w//2, h//2
    radius = int(min(w,h)*VIS.ring_radius_frac)
    max_bar = int(min(w,h)*VIS.bar_max_len_frac)
    if state["bars"]["radius"] != radius:
        x0 = (cx + (radius+4)*COS_ARR).astype(np.int32)
        y0 = (cy + (radius+4)*SIN_ARR).astype(np.int32)
        state["bars"].update({"radius":radius,"x0":x0,"y0":y0})
        log.debug("Bars layout updated (radius=%d)", radius)
    dotted = build_dotted_ring_surface(radius, VIS.dotted_count, VIS.bar_color)
    blit_center(vis_surf, dotted, (cx,cy))
    glow1 = build_glow_circle_surface(radius, glow=8, color_rgb=VIS.glow_color, thickness=2)
    blit_center(vis_surf, glow1, (cx,cy))
    bass_pulse = min(1.0, state["bass_env"] * 4.0)
    pulse_color = lerp_color((30,30,30), VIS.bar_color, bass_pulse)
    glow2 = build_glow_circle_surface(int(radius*0.7), glow=6, color_rgb=pulse_color, thickness=0)
    glow2.set_alpha(int(40 + 180*bass_pulse))
    blit_center(vis_surf, glow2, (cx,cy))
    ve = max(0.0, min(1.0, state.get("voice_env", 0.0)))
    base_r = int(radius * VIS.voice_base_radius_scale)
    dyn_r  = base_r + int(ve * VIS.voice_max_pulse_px)
    alpha  = int(lerp(VIS.voice_alpha_min, VIS.voice_alpha_max, ve))
    col    = lerp_color(VIS.voice_base_color, VIS.voice_peak_color, ve)
    gfxdraw.filled_circle(vis_surf, cx, cy, dyn_r, (*col, alpha))
    gfxdraw.aacircle(vis_surf, cx, cy, dyn_r, (*col, min(255, alpha+30)))
    pad=VIS.progress_width//2+8
    rect=pygame.Rect(cx-radius-pad, cy-radius-pad, 2*(radius+pad), 2*(radius+pad))
    frac = 0.0 if state["dur"]<=0 else min(1.0, state["pos"]/state["dur"])
    energy_t = min(1.0, state["bass_env"]**0.5)
    energy_color = lerp_color(lerp_color(VIS.red, VIS.yellow, energy_t), VIS.white, energy_t*0.5)
    draw_progress_arc_aa(vis_surf, rect, -math.pi/2, frac, energy_color, VIS.progress_width)
    if state["flash"] > 0:
        halo = build_glow_circle_surface(radius+8, glow=36, color_rgb=energy_color, thickness=3)
        halo.set_alpha(int(255*state["flash"]))
        blit_center(vis_surf, halo, (cx,cy))
    x0_arr, y0_arr = state["bars"]["x0"], state["bars"]["y0"]
    step = 1 if state["fps"] >= 30 else 2
    bands = state["bands"]
    for i in range(0, VIS.n_bands, step):
        v = bands[i]
        L=int(6 + v*max_bar)
        x1=int(x0_arr[i] + L*COS_ARR[i])
        y1=int(y0_arr[i] + L*SIN_ARR[i])
        pygame.draw.line(vis_surf, (*VIS.bar_color,255), (x0_arr[i],y0_arr[i]), (x1,y1), width=VIS.bar_thickness)
# ---------------- Main ----------------
def main():
    args = parse_args()
    # Initialize web terminal handler if requested
    webterm_handler = WebTermHandler() if args.webterm else None
    setup_logging(args.debug, webterm_handler)
    ffok = ensure_ffmpeg()
    pygame.init()
    flags_windowed = pygame.RESIZABLE | pygame.DOUBLEBUF
    def try_mode(size, flags, vsync):
        try: return pygame.display.set_mode(size, flags, vsync=vsync)
        except Exception: return None
    screen = try_mode((1280,720), flags_windowed, 1) or try_mode((1280,720), flags_windowed, 0) or pygame.display.set_mode((1280,720))
    pygame.display.set_caption("VMP – Cyberpunk Ring Player (Optimized)")
    # playback: we will use pygame.mixer.music for normal playback,
    # Channel(0) (chan_main) for precise seek segments, Channel(1) (chan_overlap) for overlap next
    pygame.mixer.init(frequency=AUDIO.target_sr, channels=2, size=-16, buffer=1024)
    log.debug("Pygame mixer initialized @ %d Hz", AUDIO.target_sr)
    chan_main = pygame.mixer.Channel(0)
    chan_overlap = pygame.mixer.Channel(1)
    playback_mode = "music"  # "music" or "channel"
    current_main_sound: Optional[pygame.mixer.Sound] = None  # when mode == "channel"
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
    # surfaces that depend on size
    vis_surf  = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    text_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    def after_mode_change_rescale():
        """Force-rescale visuals & caches after fake-fullscreen toggle or resize."""
        nonlocal vis_surf, text_surf
        w,h = screen.get_size()
        vis_surf  = pygame.Surface((w,h), pygame.SRCALPHA)
        text_surf = pygame.Surface((w,h), pygame.SRCALPHA)
        rescale_background_for_size() if original_bg_surface is not None else choose_background(True)
        bars_state["radius"] = None  # force recompute ring geometry
        reset_text_caches()
        log.debug("Rescaled after mode change -> %dx%d", w, h)
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
    font_mid_sys   = load_system(50, bold=True)
    title_font     = load_cyber(VIS.title_font_size)
    vol_font_sys   = load_system(18, bold=True)
    base = Path(args.music_dir)
    if not base.exists():
        print(f"MUSIC_DIR not found: {base}")
        sys.exit(1)
    ext_tuple = tuple(e if e.startswith(".") else f".{e}" for e in args.ext.split(",") if e.strip())
    ignore_dirs = [d.strip() for d in args.ignore.split(",") if d.strip()]
    lib = Library(ext_tuple, ignore_dirs, args.no_tags); lib.scan(base)
    if not lib.tracks:
        print("No supported audio files found."); sys.exit(1)
    # Backgrounds
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
            original_bg_surface = pygame.image.load(str(p)).convert()
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
    # View presets
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
    v_mode = preset_indices[0]
    def v_mode_desc(m):
        st=v_states[m%len(v_states)]
        return f"HUD {'ON' if st['hud'] else 'OFF'} / FPS {'ON' if st['fps'] else 'OFF'} / TIME {'ON' if st['time'] else 'OFF'} / TITLE {'ON' if st['title'] else 'OFF'}"
    help_visible = False
    help_cache_surf: Optional[pygame.Surface] = None
    def build_help_surface(w: int, h: int) -> pygame.Surface:
        pad = 14
        lines = [
            "H – Help (toggle)",
            "1..5 – View presets   |   V – Next preset   |   F2 – All combinations",
            "F – Fake Fullscreen     T – Always on Top",
            "B – Toggle Backgrounds     [ / ] – Prev/Next BG",
            "O – Opacity -5%     Shift+O – Opacity +5%",
            "Space – Pause/Resume     N / P – Next / Prev Track",
            "S – Shuffle (after first)     R – Repeat All",
            "← / → – Seek -5s / +5s",
            "Mouse Wheel – Volume     LMB drag – Move window (no BG)",
            "Esc / Q – Quit"
        ]
        if args.webterm:
            lines.append("Web Terminal: http://localhost:3030")
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
        cy = pad + 3
        for t in texts:
            box.blit(t, (pad, cy)); cy += t.get_height() + 4
        surf.blit(box, (x, y))
        return surf
    # Order
    n = len(lib.tracks)
    shuffle = bool(args.shuffle)
    repeat_all = bool(args.repeat_all)
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
    # --- Playback helpers (music vs channel) ---
    def stop_all_playback():
        nonlocal current_main_sound
        try: pygame.mixer.music.stop()
        except Exception: pass
        try: chan_main.stop()
        except Exception: pass
        current_main_sound = None
    def play_track_music(track: Track, start_sec: float = 0.0, volume: float = 0.85):
        """Default streaming playback using mixer.music (works for whole track)."""
        nonlocal playback_mode
        stop_all_playback()
        pygame.mixer.music.load(str(track.path))
        if start_sec > 0.0:
            # set_pos is unreliable for mp3, so use 0 and fall back to channel below if needed.
            try:
                pygame.mixer.music.play(start=start_sec)
                playback_mode = "music"
            except TypeError:
                # fall back to channel seek
                seek_and_play_with_ffmpeg(track, start_sec, volume)
                return
        else:
            pygame.mixer.music.play()
            playback_mode = "music"
        pygame.mixer.music.set_volume(volume)
    def seek_and_play_with_ffmpeg(track: Track, start_sec: float, volume: float = 0.85):
        """Precise seek: decode remainder with ffmpeg and play on Channel(0)."""
        nonlocal playback_mode, current_main_sound
        stop_all_playback()
        if not ffok:
            # Without ffmpeg, try best-effort
            try:
                pygame.mixer.music.load(str(track.path))
                pygame.mixer.music.play()
                pygame.mixer.music.set_volume(volume)
                playback_mode = "music"
            except Exception:
                playback_mode = "music"
            return
        # FIXED: Clamp duration to track's actual remaining length
        dur_left = min(UI.seek_segment_sec, max(0.1, (track.duration_sec or 0.0) - start_sec))
        seg = decode_pcm_segment(track.path, start_sec, dur_left, sr=AUDIO.target_sr)
        if seg.shape[0] == 0:
            # fallback to normal play
            play_track_music(track, 0.0, volume)
            return
        current_main_sound = numpy_to_sound(seg)
        chan_main.play(current_main_sound)
        chan_main.set_volume(volume, volume)
        playback_mode = "channel"
    # Initial playback
    volume = 0.85
    play_track_music(current_track, 0.0, volume)
    # Playback time accounting (works for both modes)
    play_start_monotonic = time.monotonic()
    paused = False
    paused_accum = 0.0
    pause_started = None
    def set_play_start(start_sec: float):
        nonlocal play_start_monotonic, paused_accum, pause_started
        play_start_monotonic = time.monotonic() - start_sec
        paused_accum = 0.0
        pause_started = None
        log.debug("Set play start: %.3f", start_sec)
    def get_play_pos() -> float:
        # Prefer mixer.music.get_pos when in 'music' mode & valid
        if playback_mode == "music":
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms is not None and pos_ms >= 0:
                return pos_ms/1000.0
        # Generic monotonic fallback (also for 'channel' mode)
        now = time.monotonic()
        extra = (now - pause_started) if (paused and pause_started is not None) else 0.0
        return max(0.0, now - play_start_monotonic - (paused_accum + extra))
    def seek_relative(delta_sec: float):
        """Use precise seek via ffmpeg segment on Channel(0)."""
        cur = get_play_pos()
        dur = current_track.duration_sec or max(0.0, cur+1.0)
        new_pos = max(0.0, min(max(0.0, dur-0.05), cur + delta_sec))
        set_play_start(new_pos)
        seek_and_play_with_ffmpeg(current_track, new_pos, volume)
        log.debug("Seek relative %.2f -> new_pos=%.2f (mode=%s)", delta_sec, new_pos, playback_mode)
        return new_pos
    # Overlap for next track (same as pred, Channel(1))
    overlap_active = False
    overlap_end_time = 0.0
    overlap_seg_len = 0.0
    next_track_pending: Optional[Track] = None
    # Global decoder executor
    decoder_executor = ThreadPoolExecutor(max_workers=2)
    def start_overlap_next(next_track: Track) -> bool:
        nonlocal overlap_active, overlap_end_time, overlap_seg_len, next_track_pending
        seg_len = min(AUDIO.crossfade_sec, max(0.5, (next_track.duration_sec or AUDIO.crossfade_sec)))
        try:
            # Submit decode job asynchronously
            future = decoder_executor.submit(decode_pcm_segment, next_track.path, 0.0, seg_len, sr=AUDIO.target_sr)
            # Store future for later handling
            overlap_future = future
            overlap_active   = True
            overlap_end_time = time.monotonic() + seg_len
            overlap_seg_len  = seg_len
            next_track_pending = next_track
            log.debug("Overlap start -> %s (len=%.2fs)", next_track.path.name, seg_len)
            return True
        except Exception as e:
            log.debug("Overlap start failed: %s", e)
            return False
    def finish_overlap_switch():
        """After overlap completes: switch to next track using mixer.music at offset."""
        nonlocal overlap_active, next_track_pending, current_track, overlap_seg_len, bg_switch_pending, playback_mode
        if not next_track_pending: return
        # stop current (both music and channel)
        stop_all_playback()
        pygame.mixer.music.load(str(next_track_pending.path))
        start_off = max(0.0, float(overlap_seg_len))
        try:
            pygame.mixer.music.play(start=start_off)
            playback_mode = "music"
        except TypeError:
            pygame.mixer.music.play()
            playback_mode = "music"
            start_off = 0.0
        pygame.mixer.music.set_volume(volume)
        set_play_start(start_off)
        current_track = next_track_pending
        next_track_pending = None
        overlap_active = False
        overlap_seg_len = 0.0
        reset_text_caches()
        bg_switch_pending = True
        log.debug("Overlap finished. Current -> %s (mode=%s)", current_track.path.name, playback_mode)
    # --------- FFT setup & improved masks ----------
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
    # --------- FFT thread (non-blocking, no decode) ----------
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
            dur = len(mono)/float(AUDIO.target_sr)
            return mono, AUDIO.target_sr, dur
        except Exception as e:
            log.debug("load_track_samples_quick failed for %s: %s", path.name, e)
            return np.zeros(0, np.float32), AUDIO.target_sr, 0.0
    def fft_worker():
        nonlocal fft_last_bands, fft_last_bass_energy, fft_last_voice_energy
        log.debug("FFT thread started")
        while fft_should_run:
            _ = fft_event.wait(timeout=0.12)
            fft_event.clear()
            if not fft_should_run: break
            # If paused: keep last values, skip heavy work
            if paused:
                continue
            with fft_lock:
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
                key = _norm_key(current_track.path)
                cached = AN_CACHE.get(key)
                if cached is None:
                    # Ask loader with highest priority, do not block
                    _put_load(PRIO_NOW, "analyze", current_track.path)
                    t = time.time()
                    base = 0.12 + 0.06*math.sin(t*2.0)
                    with fft_lock:
                        fft_last_bands.fill(base)
                        fft_last_bass_energy = base
                        fft_last_voice_energy = base
                    continue
                samples, sr, _dur = cached
                if samples is None or samples.size == 0:
                    t = time.time()
                    base = 0.12 + 0.06*math.sin(t*2.0)
                    with fft_lock:
                        fft_last_bands.fill(base)
                        fft_last_bass_energy = base
                        fft_last_voice_energy = base
                    continue
                pos_an = max(0.0, pos - AUDIO.analysis_lag_sec)
                win_full = make_fft_window(samples, sr, pos_an, VIS.fft_size)
                sp_full = np.fft.rfft(win_full*hann_full)
                mag_full = np.abs(sp_full)
                bands = band_mapper_full.map(mag_full)
                win_bass = make_fft_window(samples, sr, pos_an, VIS.bass_fft)
                sp_b = np.fft.rfft(win_bass*hann_bass)
                bb = np.abs(sp_b) * bass_mask_f
                bass_energy = float(np.sqrt(np.mean(bb*bb))) if bb.size else 0.0
                vb = mag_full * voice_mask_f
                voice_energy = float(np.sqrt(np.mean(vb*vb))) if vb.size else 0.0
                with fft_lock:
                    fft_last_bands[:] = bands  # Copy into existing array
                    fft_last_bass_energy = bass_energy
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
    # --------- Loader threads (priority queue) ----------
    from queue import PriorityQueue
    load_q: "PriorityQueue[tuple[int,int,str,Path]]" = PriorityQueue()
    load_should_run = True
    _load_seq = itertools.count()  # stable ordering
    # lower prio value = higher priority
    PRIO_NOW      = 0   # current track
    PRIO_PREFETCH = 5   # next track
    PRIO_BULK     = 9   # warm-up rest
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
                        log.debug("Loader: %s %s (%.1fs)", action, path.name, dur)
            except Exception as e:
                log.debug("Loader error: %s", e)
            finally:
                load_q.task_done()
        log.debug("Loader worker exit")
    fft_thread = threading.Thread(target=fft_worker, daemon=True); fft_thread.start()
    loader_threads: List[threading.Thread] = []
    num_workers = max(2, min(4, (os.cpu_count() or 4)//2))
    for _ in range(num_workers):
        t = threading.Thread(target=loader_worker_fn, daemon=True)
        t.start()
        loader_threads.append(t)
    # Immediately queue analysis for current + prefetch next
    _put_load(PRIO_NOW, "analyze", current_track.path)
    # UI state caches
    bars_state = {"radius": None, "x0": None, "y0": None}
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
        ui=f"[S]huffle: {'ON' if shuffle else 'OFF'}  [R]epeat: {'ALL' if repeat_all else 'OFF'}  [F] Fake Fullscreen  [V/F2] View: {v_mode_desc(v_mode)}  [H] Help"
        return font_small_sys.render(ui, True, VIS.text_dim)
    def reset_text_caches():
        nonlocal time_last_sec, time_cache_surf, title_cache_key, title_cache_surf, hud_cache_surf, hud_cache_key, help_cache_surf
        time_last_sec = -1; time_cache_surf = None
        title_cache_key = None; title_cache_surf = None
        hud_cache_surf = None; hud_cache_key = None
        help_cache_surf = None
    choose_background(False)
    # Beat & history (optimized ring buffers)
    flash=0.0
    bass_env = 0.0
    bass_peak = 1e-6
    hist_len = int(0.7*VIS.fps_target)
    bass_hist = np.zeros(hist_len, dtype=np.float32)
    hist_idx = 0
    hist_filled = 0
    bass_last_beat = 0.0
    # Robust adaptive threshold using EMA + MAD
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
    help_visible = False
    # >>> NEW: End of Queue State <<<
    end_of_queue_reached = False
    # Start web terminal server if requested
    if args.webterm:
        start_web_terminal_server(webterm_handler)
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running=False; break
            if ev.type == pygame.VIDEORESIZE:
                resize_pending = True; last_resize_time = time.time()
                vis_surf  = pygame.Surface(ev.size, pygame.SRCALPHA)
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
                elif ev.key == pygame.K_t:
                    win_toggle_topmost()
                elif ev.key == pygame.K_b:
                    bg_enabled = not bg_enabled
                    if bg_enabled: choose_background(True)
                elif ev.key == pygame.K_LEFTBRACKET:
                    if bg_paths:
                        bg_index = (bg_index - 1) % len(bg_paths); choose_background(True)
                elif ev.key == pygame.K_RIGHTBRACKET:
                    if bg_paths:
                        bg_index = (bg_index + 1) % len(bg_paths); choose_background(True)
                elif ev.key == pygame.K_o:
                    try:
                        cur = pygame.display.get_window_alpha() if hasattr(pygame.display, "get_window_alpha") else 1.0
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT: window_opacity = min(1.0, (cur or 1.0) + 0.05)
                        else: window_opacity = max(0.2, (cur or 1.0) - 0.05)
                        pygame.display.set_window_alpha(window_opacity)
                        log.debug("Opacity -> %.2f", window_opacity)
                    except Exception: pass
                elif ev.key == pygame.K_SPACE:
                    if paused:
                        pygame.mixer.unpause()
                        paused=False
                        if pause_started is not None:
                            paused_accum += time.monotonic() - pause_started; pause_started=None
                        # >>> SET ENVELOPES TO CURRENT FFT VALUES FOR SMOOTHER RESUME <<<
                        with fft_lock:
                            bass_peak_val = max(bass_peak*0.995, fft_last_bass_energy)
                            bass_norm_val = fft_last_bass_energy / (bass_peak_val + 1e-9)
                            bass_env = bass_norm_val
                            voice_peak_val = max(voice_peak * AUDIO.voice_peak_decay, fft_last_voice_energy)
                            voice_norm_val = fft_last_voice_energy / (voice_peak_val + 1e-9)
                            voice_env = voice_norm_val
                        reset_text_caches()
                        log.debug("Playback resumed")
                    else:
                        pygame.mixer.pause()  # pauses both music and channels
                        paused=True
                        if pause_started is None: pause_started = time.monotonic()
                        bass_env = 0.0; bass_peak = 1e-6; hist_idx = 0; hist_filled = 0; bass_hist.fill(0)
                        ema = 0.0; mad_ema = 0.0
                        log.debug("Playback paused")
                elif ev.key in (pygame.K_n, pygame.K_p):
                    # >>> FIXED: Allow navigation after "End of Queue" <<<
                    if end_of_queue_reached and ev.key == pygame.K_p:
                        # Go to last track
                        target_idx = len(order) - 1
                        end_of_queue_reached = False
                    elif end_of_queue_reached and ev.key == pygame.K_n:
                        # Go to first track
                        target_idx = 0
                        end_of_queue_reached = False
                    else:
                        nowk=time.time()
                        if nowk - last_nav_time < UI.next_cooldown_sec: continue
                        last_nav_time = nowk
                        if overlap_active:
                            try: chan_overlap.stop()
                            except Exception: pass
                            overlap_active = False; next_track_pending = None; overlap_seg_len = 0.0
                        if ev.key==pygame.K_n:
                            if (index_in_order+1<len(order)) or repeat_all:
                                target_idx = (index_in_order+1) % len(order)
                            else:
                                show_toast("End of queue")
                                end_of_queue_reached = True
                                # >>> Stop playback and timing <<<
                                stop_all_playback()
                                continue
                        else:
                            target_idx = (index_in_order - 1) % len(order)
                    target_track = lib.tracks[order[target_idx]]
                    current_track = target_track
                    index_in_order = target_idx
                    set_play_start(0.0)
                    play_track_music(current_track, 0.0, volume)  # start fresh in music mode
                    reset_text_caches()
                    bg_switch_pending = True
                    _put_load(PRIO_NOW, "analyze", current_track.path)
                    if (index_in_order+1<len(order)) or repeat_all:
                        nxt = lib.tracks[order[(index_in_order+1)%len(order)]]
                        _put_load(PRIO_PREFETCH, "prefetch", nxt.path)
                    log.debug("Switched track -> %s", current_track.path.name)
                elif ev.key == pygame.K_s:
                    shuffle = not shuffle
                    head = [order[index_in_order]]
                    remaining = [i for i in range(n) if i != order[index_in_order]]
                    if shuffle: random.shuffle(remaining)
                    order = head + remaining
                    index_in_order = 0
                    reset_text_caches()
                    log.debug("Shuffle -> %s", "ON" if shuffle else "OFF")
                elif ev.key == pygame.K_r:
                    repeat_all = not repeat_all
                    log.debug("Repeat all -> %s", "ON" if repeat_all else "OFF")
                elif ev.key == pygame.K_RIGHT:
                    seek_relative(+5.0); reset_text_caches()
                elif ev.key == pygame.K_LEFT:
                    seek_relative(-5.0); reset_text_caches()
            if ev.type == pygame.MOUSEWHEEL:
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
                    last_volume_popup_t = time.time(); log.debug("Volume + -> %.0f%%", volume*100)
                elif ev.button == 5:
                    volume = max(0.0, volume - 0.03)
                    if playback_mode == "music": pygame.mixer.music.set_volume(volume)
                    else: chan_main.set_volume(volume, volume)
                    last_volume_popup_t = time.time(); log.debug("Volume - -> %.0f%%", volume*100)
                elif ev.button == 1 and not current_bg:
                    win_drag_window()
        if resize_pending and (time.time() - last_resize_time) > 0.15:
            rescale_background_for_size() if original_bg_surface is not None else choose_background(True)
            resize_pending = False
            bars_state["radius"] = None
            reset_text_caches()
        pos_now = get_play_pos()
        dur_now = current_track.duration_sec or 0.0
        time_left = (dur_now - pos_now) if dur_now>0 else 999.0
        # Prefetch & overlap scheduling
        if (not paused) and (not overlap_active) and (dur_now > 0):
            if time_left < 30.0 and ((index_in_order+1<len(order)) or repeat_all):
                nxt = lib.tracks[order[(index_in_order+1)%len(order)]]
                _put_load(PRIO_PREFETCH, "prefetch", nxt.path)
            # Vylepšené spúšťanie crossfadeu
            crossfade_start_threshold = max(0.5, AUDIO.crossfade_sec)  # Minimalný čas pre spustenie je 0.5s
            if (time_left <= crossfade_start_threshold + 0.2) and (time_left > 0.1) and (not overlap_active):
                next_idx_auto = (index_in_order+1) % len(order) if (index_in_order+1<len(order) or repeat_all) else index_in_order
                if next_idx_auto != index_in_order:
                    next_track_auto = lib.tracks[order[next_idx_auto]]
                    if ffok and start_overlap_next(next_track_auto):
                        index_in_order = next_idx_auto
        if overlap_active:
            t_left = overlap_end_time - time.monotonic()
            seg_total = max(0.001, overlap_seg_len)
            t_elapsed = seg_total - max(0.0, t_left)
            frac = max(0.0, min(1.0, t_elapsed / seg_total))
            # >>> VYLEPŠENÝ SMOOTH COSINE CROSSFADE <<<
            out_curve = 0.5 * (1 + math.cos(math.pi * frac))  # Cosine fade out (začína pri 1, končí pri 0)
            in_curve = 0.5 * (1 - math.cos(math.pi * frac))   # Cosine fade in (začína pri 0, končí pri 1)
            # fade out current main (music or channel)
            if playback_mode == "music":
                pygame.mixer.music.set_volume(out_curve * volume)
            else:
                chan_main.set_volume(out_curve * volume, out_curve * volume)
            # fade in overlap
            chan_overlap.set_volume(in_curve * volume, in_curve * volume)
            if t_left <= 0.0:
                finish_overlap_switch()
        if bg_switch_pending and ((playback_mode == "music" and pygame.mixer.music.get_busy()) or (playback_mode == "channel" and chan_main.get_busy())):
            choose_background(False); bg_switch_pending = False
        w,h = screen.get_size()
        if current_bg and (current_bg.get_width()!=w or current_bg.get_height()!=h):
            rescale_background_for_size()
        if current_bg:
            screen.blit(current_bg,(0,0))
            if bg_dark: screen.blit(bg_dark,(0,0))
        else:
            screen.fill(VIS.bg_color)
        # clear reusable surfaces
        vis_surf.fill((0,0,0,0))
        text_surf.fill((0,0,0,0))
        # Request FFT periodically (skip if paused to save CPU)
        if (not paused) and ((frame_idx % max(1, int(VIS.fft_every_n_frames))) == 0):
            with fft_lock: fft_req_pos = pos_now
            fft_event.set()
        # Fetch FFT results
        with fft_lock:
            bands = fft_last_bands.copy()
            bass_energy = float(fft_last_bass_energy)
            voice_energy = float(fft_last_voice_energy)
        # Envelope followers (keep simple when paused)
        if not paused:
            bass_peak = max(bass_peak*0.995, bass_energy)
            bass_norm = bass_energy / (bass_peak + 1e-9)
            if bass_norm > bass_env: bass_env += AUDIO.attack*(bass_norm - bass_env)
            else: bass_env += (AUDIO.rel_slow if bass_norm<0.1 else AUDIO.rel_fast)*(bass_norm - bass_env)
            bass_env = float(np.clip(bass_env, 0.0, 1.0))
            # History ring buffer
            bass_hist[hist_idx] = bass_env
            hist_idx = (hist_idx + 1) % hist_len
            hist_filled = min(hist_filled+1, hist_len)
            hist_view = bass_hist[:hist_filled]
            # Adaptive threshold (EMA + MAD)
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
            # Voice env
            voice_peak = max(voice_peak * AUDIO.voice_peak_decay, voice_energy)
            voice_norm = voice_energy / (voice_peak + 1e-9)
            if voice_norm > voice_env:
                voice_env += AUDIO.voice_attack * (voice_norm - voice_env)
            else:
                voice_env += (AUDIO.voice_rel_slow if voice_norm < 0.1 else AUDIO.voice_rel_fast) * (voice_norm - voice_env)
            voice_env = max(0.0, min(1.0, voice_env))
        else:
            # keep visuals steady when paused
            flash = max(0.0, flash*0.9)
            bass_env *= 0.98
            voice_env *= 0.98
        state = {"pos":pos_now,"dur":dur_now,"bass_env":bass_env,"flash":flash,"bands":bands,"fps":clock.get_fps(),"bars":bars_state,"voice_env":voice_env}
        draw_visuals(screen, vis_surf, state)
        # -------- Text/UI --------
        st_view = v_states[v_mode]
        show_time = st_view["time"]
        show_title = st_view["title"]
        show_hud = st_view["hud"]
        show_fps = st_view["fps"]
        any_text_allowed = show_time or show_title or show_hud or show_fps
        cx,cy = w//2, h//2
        if show_time:
            if time_cache_surf is None or int(pos_now) != time_last_sec:
                time_last_sec = int(pos_now)
                time_cache_surf = font_mid_sys.render(f"{format_time(pos_now)} / {format_time(dur_now)}", True, VIS.white)
            text_surf.blit(time_cache_surf, time_cache_surf.get_rect(center=(cx, cy)))
        if show_title:
            meta_text = display_title(current_track)
            if meta_text:
                if title_cache_key != meta_text:
                    title_cache_key = meta_text
                    title_cache_surf = title_font.render(meta_text, True, VIS.white)
                text_surf.blit(title_cache_surf, title_cache_surf.get_rect(center=(cx, cy + int(min(w,h)*VIS.ring_radius_frac) + 110)))
        if show_hud:
            key_now = (v_mode, shuffle, repeat_all)
            if hud_cache_surf is None or hud_cache_key != key_now:
                hud_cache_key = key_now; hud_cache_surf = build_hud()
            text_surf.blit(hud_cache_surf, (14,10))
        if any_text_allowed and ((time.time() - last_volume_popup_t) < UI.volume_popup_sec):
            vol_txt = vol_font_sys.render(f"VOL {int(volume*100)}%", True, VIS.text_dim)
            text_surf.blit(vol_txt, (w - vol_txt.get_width()-10, 10))
        if time.time() < toast_until:
            toast = vol_font_sys.render(toast_msg, True, VIS.text_dim)
            text_surf.blit(toast, (w - toast.get_width()-10, 10 + 24))
        if show_fps:
            fps = clock.get_fps()
            dbg = f"FPS {fps:4.1f} | pos {pos_now:6.2f}/{dur_now:6.2f} | overlap {'Y' if overlap_active else 'N'} | idx {index_in_order+1}/{len(order)} | V {v_mode+1}/{len(v_states)} | mode {playback_mode}"
            ds = font_small_sys.render(dbg, True, (200,200,200))
            text_surf.blit(ds, (14, h-10-ds.get_height()))
        # composite with transparency
        vis_surf.set_alpha(VIS.ui_alpha)
        screen.blit(vis_surf,(0,0))
        if any_text_allowed or help_visible or (time.time()<toast_until):
            screen.blit(text_surf,(0,0))
        if help_visible:
            if (help_cache_surf is None) or (help_cache_surf.get_width()!=w or help_cache_surf.get_height()!=h):
                help_cache_surf = build_help_surface(w,h)
            screen.blit(help_cache_surf, (0,0))
        pygame.display.flip()
        frame_idx += 1
        clock.tick_busy_loop(VIS.fps_target)
        # End-of-track fallback (if no overlap)
        if (not overlap_active) and (dur_now > 0):
            # >>> FIXED: Robust end-of-track detection <<<
            track_ended = False
            if pos_now >= dur_now - 0.05:
                if playback_mode == "music":
                    if not pygame.mixer.music.get_busy():
                        track_ended = True
                else: # channel mode
                    if not chan_main.get_busy():
                        track_ended = True
            if track_ended:
                next_idx = (index_in_order+1) % len(order) if (index_in_order+1<len(order) or repeat_all) else index_in_order
                if next_idx != index_in_order:
                    next_track = lib.tracks[order[next_idx]]
                    current_track = next_track; index_in_order = next_idx
                    set_play_start(0.0)
                    play_track_music(current_track, 0.0, volume)  # start next in music mode
                    reset_text_caches()
                    bg_switch_pending = True
                    _put_load(PRIO_NOW, "analyze", current_track.path)
                    if (index_in_order+1<len(order)) or repeat_all:
                        nxt = lib.tracks[order[(index_in_order+1)%len(order)]]
                        _put_load(PRIO_PREFETCH, "prefetch", nxt.path)
                    log.debug("Auto next -> %s", current_track.path.name)
                else:
                    show_toast("End of queue")
                    end_of_queue_reached = True
                    stop_all_playback()
    # -------- Cleanup --------
    try:
        log.debug("Shutting down…")
        fft_should_run = False
        fft_event.set()
        load_should_run = False
        try:
            # wake workers
            for _ in loader_threads:
                _put_load(PRIO_BULK, "prefetch", current_track.path)
        except Exception:
            pass
        try:
            chan_overlap.stop()
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
        decoder_executor.shutdown(wait=False)
        pygame.quit()
        log.debug("Exited cleanly")
# ---------------- Web Terminal Server ----------------
async def webterm_handler(websocket, path, log_handler):
    """Handle a single WebSocket connection for the web terminal."""
    log_handler.add_client(websocket)
    try:
        # Send welcome message
        await websocket.send("Welcome to VMP Web Terminal!\n")
        await websocket.send("You can send commands like 'n', 'p', ' ', 'q' to control the player.\n")
        async for message in websocket:
            try:
                # Simulate key press based on message
                cmd = message.strip().lower()
                if cmd == 'n':
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_n}))
                elif cmd == 'p':
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_p}))
                elif cmd == ' ':
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE}))
                elif cmd == 'q':
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_q}))
                elif cmd == 'right':
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT}))
                elif cmd == 'left':
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_LEFT}))
                else:
                    await websocket.send(f"Unknown command: {cmd}\n")
            except Exception as e:
                await websocket.send(f"Error processing command: {e}\n")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        log_handler.remove_client(websocket)
def start_web_terminal_server(log_handler):
    """Start the web terminal server in a separate thread."""
    def run_server():
        async def server():
            async with websockets.serve(lambda ws, path: webterm_handler(ws, path, log_handler), "localhost", 3030):
                await asyncio.Future()  # run forever
        asyncio.run(server())
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    log.info("Web terminal server started on ws://localhost:3030")
if __name__=="__main__":
    main()
