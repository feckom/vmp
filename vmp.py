from __future__ import annotations
import os, sys, math, random, re, time, argparse, logging, threading, subprocess, ctypes, platform, shutil, queue, collections, itertools
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
if platform.system() == "Windows":
    os.environ.setdefault("SDL_RENDER_DRIVER", "direct3d")
os.environ.setdefault("SDL_HINT_RENDER_SCALE_QUALITY", "1")
import numpy as np
import pygame
from pygame import gfxdraw
import pygame.sndarray
from mutagen import File as MutaFile
from mutagen.easyid3 import EasyID3
from pydub import AudioSegment
@dataclass
class AudioCfg:
    target_sr: int = 44100
    bass_low_hz: int = 30
    bass_high_hz: int = 150
    analysis_lag_sec: float = 0.030
    attack: float = 0.45
    rel_fast: float = 0.12
    rel_slow: float = 0.04
    beat_min_gap: float = 0.11
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
    fft_every_n_frames: int = 1
    progress_width: int = 14
    bar_thickness: int = 8
    bar_max_len_frac: float = 0.26
    ring_radius_frac: float = 0.18
    bar_color: Tuple[int,int,int]=(255,80,0)
    white: Tuple[int,int,int]=(255,255,255)
    red: Tuple[int,int,int]=(255,32,32)
    yellow: Tuple[int,int,int]=(255,220,0)
    text_dim: Tuple[int,int,int]=(230,230,230)
    bg_color: Tuple[int,int,int]=(8,8,8)
    title_font_size: int = 20   
    voice_base_color: Tuple[int,int,int]=(255,80,40)
    voice_alpha_min: int = 40
    voice_alpha_max: int = 220
    # Flubber amorphous shape (centered, bass-driven size)
    flub_points: int = 96                  # polygon detail (64–128 OK)
    flub_base_frac: float = 0.42           # base_r = radius * flub_base_frac
    flub_amorphous_intensity: float = 0.22 # overall wobble amplitude
    flub_voice_amorphous_gain: float = 0.15# extra wobble from voice_env
    flub_amorphous_speed: float = 0.60     # time speed of wobble
    flub_angular_noise_scale: float = 2.00 # how fast shape changes around ring
    # Dynamické zväčšovanie podľa basov
    flub_min_size_multiplier: float = 0.7  # Minimálna veľkosť (keď bass_env = 0.0)
    flub_max_size_multiplier: float = 2.0  # Maximálna veľkosť (keď bass_env = 1.0)
@dataclass
class UiCfg:
    next_cooldown_sec: float = 0.35
    volume_popup_sec: float = 0.9
    toast_sec: float = 1.4
    seek_segment_sec: float = 30.0
AUDIO = AudioCfg()
VIS = VisualCfg()
UI = UiCfg()
MUSIC_DIR = "music"
BG_DIR    = "backgrounds"
SCAN_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
FONT_PATH  = Path(__file__).with_name("vmp.ttf")
log = logging.getLogger("vmp")
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
    return (data.reshape(-1,2).astype(np.float32))/32768.0
def numpy_to_sound(arr_float_stereo: np.ndarray) -> pygame.mixer.Sound:
    arr = np.clip(arr_float_stereo, -1.0, 1.0)
    arr16 = (arr * 32767.0).astype(np.int16)
    return pygame.sndarray.make_sound(arr16)
_hann_cache: Dict[int, np.ndarray] = {}
def _hann(size: int) -> np.ndarray:
    h = _hann_cache.get(size)
    if h is None:
        log.debug("Create Hann window: %d", size)
        h = np.hanning(size).astype(np.float32)
        _hann_cache[size] = h
    return h
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
        all_indices = np.concatenate(self.idxs)
        all_values = spectrum_abs[all_indices]
        lens = np.array([len(idx) for idx in self.idxs])
        split_points = np.cumsum(lens[:-1])
        band_rms = np.array([
            np.sqrt(np.mean(chunk * chunk)) if len(chunk) > 0 else 0.0
            for chunk in np.split(all_values, split_points)
        ], dtype=np.float32)
        vmax = band_rms.max()
        if vmax > 0:
            band_rms /= vmax
        band_rms *= self.band_weights
        return np.clip(band_rms, 0.0, 1.0)
_cached_glow: Dict[Tuple[int,int,Tuple[int,int,int],int], pygame.Surface] = {}
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
    v_mode, shuffle, repeat_all,
    current_track, title_font, font_small_sys, vol_font_sys
):
    """
    Draws all non-HELP text in the top-left "stack":
    R0: VOL (reserved slot)
    R1: TOAST (reserved slot)
    R2+: HUD / TITLE / TIME / (FPS is drawn by caller right after this function)
    Returns: (vol_shown: bool, y_cursor: int) where y_cursor is the next y to draw at.
    """
    import time as _time
    # ---- per-function cache ----
    if not hasattr(render_ui_and_text, "_cache"):
        render_ui_and_text._cache = {
            "title_key": None, "title_surf": None,
            "time_key":  None, "time_surf":  None,
            "hud_key":   None, "hud_surf":   None,
            "vol_txt":   "",   "vol_surf":   None,
        }
    C = render_ui_and_text._cache
    # ---- layout constants (top-left stack) ----
    x0 = 14
    y  = 12
    gap = 4
    # Clear target surface area (we clear whole layer anyway)
    text_surf.fill((0, 0, 0, 0))
    # Helper: ellipsize to fit width
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
    max_w = max(0, w - 2 * x0)
    vol_shown = False
    # ---------- R0: VOL (reserved slot) ----------
    now = _time.time()
    vol_active = (now - last_volume_popup_t) < UI.volume_popup_sec
    # always reserve line height for stability
    if vol_active:
        vol_txt = f"VOL {int(volume * 100)}%"
        if C["vol_txt"] != vol_txt:
            C["vol_txt"] = vol_txt
            C["vol_surf"] = vol_font_sys.render(vol_txt, True, VIS.text_dim)
        vs = C["vol_surf"]
        if vs:
            text_surf.blit(vs, (x0, y))
            vol_shown = True
    y += vol_font_sys.get_height() + gap
    # ---------- R1: TOAST (reserved slot) ----------
    if now < toast_until and toast_msg:
        tmsg = _ellipsize(vol_font_sys, toast_msg, max_w)
        toast_surf = vol_font_sys.render(tmsg, True, VIS.text_dim)
        text_surf.blit(toast_surf, (x0, y))
    y += vol_font_sys.get_height() + gap
    # ---------- HUD ----------
    if show_hud:
        hud_txt = f"[S]huffle: {'ON' if shuffle else 'OFF'}  [R]epeat: {'ALL' if repeat_all else 'OFF'}  [F] FakeFS  [V/F2] View: {v_mode+1}  [H] Help  [1..5] Presets"
        if C["hud_key"] != hud_txt:
            C["hud_key"]  = hud_txt
            C["hud_surf"] = font_small_sys.render(hud_txt, True, VIS.text_dim)
        hs = C["hud_surf"]
        if hs:
            # ellipsize HUD if too long
            hud_to_draw = hud_txt
            if hs.get_width() > max_w:
                hud_to_draw = _ellipsize(font_small_sys, hud_txt, max_w)
                hs = font_small_sys.render(hud_to_draw, True, VIS.text_dim)
            text_surf.blit(hs, (x0, y))
            y += hs.get_height() + gap
    # ---------- TITLE ----------
    title_txt = ""
    if show_title:
        if current_track and (current_track.title or current_track.artist):
            if current_track.artist and current_track.title:
                title_txt = f"{current_track.artist} — {current_track.title}"
            elif current_track.title:
                title_txt = current_track.title
    if show_title and title_txt:
        key = f"{title_txt}|w{max_w}"
        if C["title_key"] != key:
            C["title_key"]  = key
            t_ell = _ellipsize(title_font, title_txt, max_w)
            C["title_surf"] = title_font.render(t_ell, True, VIS.white)
        ts = C["title_surf"]
        if ts:
            text_surf.blit(ts, (x0, y))
            y += ts.get_height() + (gap + 2)  # slightly larger gap after title
    # ---------- TIME ----------
    if show_time:
        time_txt = f"{format_time(pos_now)} / {format_time(dur_now)}"
        if C["time_key"] != time_txt:
            C["time_key"]  = time_txt
            C["time_surf"] = font_small_sys.render(time_txt, True, VIS.white)
        ts = C["time_surf"]
        if ts:
            # time is short, no need to ellipsize normally
            text_surf.blit(ts, (x0, y))
            y += ts.get_height() + gap
    # Return next y for FPS (caller draws FPS right below)
    return vol_shown, y
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
    """Toggle Always-on-Top (Windows). Safe no-op on other OS."""
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
    """Begin window drag (Windows, borderless). Safe no-op elsewhere."""
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
    """Safe desktop size fetch with sane fallback."""
    try:
        info = pygame.display.Info()
        return (max(640, int(info.current_w)), max(480, int(info.current_h)))
    except Exception:
        return (1920, 1080)
# === Helper functions for amorphous flubber ===
def _fbm1(x):
    """Cheap 1D fractal noise (no deps). Returns ~[-1, 1]."""
    return (
        math.sin(x) * 0.60 +
        math.sin(2.0 * x + 1.7) * 0.28 +
        math.sin(4.0 * x + 0.9) * 0.12
    )
from functools import lru_cache
def _npf32(x): 
    return np.asarray(x, dtype=np.float32)
@lru_cache(maxsize=32)
def _thetas_cached(N: int) -> np.ndarray:
    if N <= 0:
        return _npf32([])
    return _npf32(np.linspace(0.0, 2.0*np.pi, int(N), endpoint=False))
def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    # Hermite smoothstep clamped 0..1
    t = np.clip((x - edge0) / max(1e-6, (edge1 - edge0)), 0.0, 1.0)
    return _npf32(t * t * (3.0 - 2.0 * t))
def _circ_smooth3(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    # tiny circular blur [0.25, 0.5, 0.25] to soften hard edges
    if arr.size == 0:
        return arr
    out = arr.astype(np.float32, copy=False)
    for _ in range(max(0, passes)):
        out = 0.5 * out + 0.25 * np.roll(out, 1) + 0.25 * np.roll(out, -1)
    return out
def _star_sharp(thetas: np.ndarray, k: int, p: float) -> np.ndarray:
    # |cos(kθ)|^p produces controllable sharp peaks (no sign flips)
    return _npf32(np.abs(np.cos(k * thetas)) ** p)
def _superellipse_r(theta: np.ndarray, n: float) -> np.ndarray:
    """
    Superellipse polar radius for |x|^n + |y|^n = 1 (a=b=1).
    r = (|cos θ|^n + |sin θ|^n)^(-1/n)
    n=1 → diamant, n≈2 → kruh, n≫2 → takmer štvorec.
    """
    n = float(max(1e-3, n))
    c = np.abs(np.cos(theta)) ** n
    s = np.abs(np.sin(theta)) ** n
    denom = np.clip(c + s, 1e-6, None)
    return _npf32(denom ** (-1.0 / n))
def _regular_polygon_r(theta: np.ndarray, sides: int, eps: float = 1e-4) -> np.ndarray:
    """
    Presné 'circumradius' vyjadrenie pravidelného n-uholníka:
    r(θ) = cos(π/n) / cos( (θ mod 2π/n) - π/n )
    + eps clamp proti singularitám pri rohoch.
    """
    n = max(3, int(sides))
    a = np.pi / n
    # map to wedge [-a, a]
    w = (theta + a) % (2.0 * a) - a
    denom = np.clip(np.cos(w), eps, None)
    return _npf32(np.cos(a) / denom)
def shape_profile(name: str, N: int, intensity: float = 0.6) -> np.ndarray:
    """
    Recognizable profiles for flubber morphs. Returns length-N float32,
    mean-normalized (~1.0) with positive values.
    """
    thetas = _thetas_cached(int(N))
    if thetas.size == 0:
        return _npf32([])
    # unified amplitude; keep your original range behavior
    amp = 0.2 + float(np.clip(intensity, 0.0, 1.0)) * 0.6
    prof = np.ones_like(thetas, dtype=np.float32)
    # === Classic / your originals (tuned for smoothness) ===
    if name == "flower6":
        prof = 1.0 + amp * np.cos(6 * thetas)
    elif name == "flower8":
        prof = 1.0 + amp * 0.8 * np.cos(8 * thetas)
    elif name == "star5":
        # p grows with intensity → sharper tips, but smooth
        p = 1.6 + 1.4 * np.clip(intensity, 0.0, 1.0)
        prof = 1.0 + amp * (2.0 * _star_sharp(thetas, 5, p) - 1.0)
    elif name == "star7":
        p = 1.5 + 1.3 * np.clip(intensity, 0.0, 1.0)
        prof = 1.0 + amp * 0.9 * (2.0 * _star_sharp(thetas, 7, p) - 1.0)
    elif name == "lotus":
        prof = 1.0 + amp * (0.5*np.cos(5*thetas) + 0.3*np.cos(10*thetas + 0.5))
    elif name == "gear":
        teeth = 12
        angle_per_tooth = 2.0*np.pi / teeth
        tooth_angles = thetas % angle_per_tooth
        # trapezoid-like (centered “flat” tooth) then softly blurred
        base = np.where(tooth_angles < angle_per_tooth * 0.4, 1.0,
                        np.where(tooth_angles < angle_per_tooth * 0.6, 0.6, 1.0))
        base = _circ_smooth3(_npf32(base), passes=2)
        prof = 1.0 + amp * 0.4 * (base - 1.0)
    elif name == "waves":
        prof = 1.0 + amp * (0.4*np.sin(3*thetas) + 0.3*np.sin(7*thetas + 1.2))
    elif name == "diamond":
        # superellipse n=1 is true diamond; raise a bit for crispness
        r = _superellipse_r(thetas, n=1.0)
        prof = 0.9 + amp * 0.35 * r
    elif name == "triangle":
        # exact regular polygon radius (n=3), mildly softened
        r = _regular_polygon_r(thetas, 3)
        prof = 0.9 + amp * 0.35 * _circ_smooth3(r, passes=1)
    elif name == "heart_real":
        # stable cardioid-style heart (smooth, positive)
        r = 1.0 - 0.85*np.sin(thetas)
        r += 0.15*np.sin(thetas) * np.sqrt(np.abs(np.cos(thetas))) / 1.2
        prof = 0.8 + amp * 0.5 * _npf32(r)
    elif name == "cross":
        # smooth “plus” via soft thresholds on sin/cos lobes
        h = _smoothstep(np.abs(np.cos(2*thetas)), 0.60, 0.85)
        v = _smoothstep(np.abs(np.sin(2*thetas)), 0.60, 0.85)
        cross_soft = np.maximum(h, v)
        prof = 1.0 + amp * 0.4 * cross_soft
    elif name == "hexagon":
        r = _regular_polygon_r(thetas, 6)
        prof = 0.85 + amp * 0.32 * _circ_smooth3(r, passes=1)
    elif name == "blob":
        prof = 1.0 + amp * (0.4*np.sin(2.3*thetas + 0.5) +
                            0.3*np.sin(4.7*thetas + 1.2) +
                            0.2*np.sin(7.1*thetas + 2.1))
    elif name == "spiral":
        spiral = 1.0 + 0.15 * (thetas / (2.0*np.pi))
        ripple = 0.1 * np.sin(12 * thetas)
        prof = spiral + amp * ripple
    elif name == "petal":
        prof = 1.0 + amp * 0.6 * np.abs(np.sin(thetas)) * (0.8 + 0.2*np.cos(2*thetas))
    elif name == "burst":
        prof = 1.0 + amp * 0.6 * np.cos(16*thetas) * (0.5 + 0.5*np.cos(thetas))
    elif name == "wave_complex":
        prof = 1.0 + amp * (0.3*np.sin(2*thetas) +
                            0.2*np.sin(5*thetas + 0.8) +
                            0.15*np.sin(9*thetas + 1.5))
    # === Newer shapes (polygons via superellipse/polygon formulas) ===
    elif name == "square":
        # superellipse with large n ≈ rounded square
        r = _superellipse_r(thetas, n=8.0)
        prof = 0.85 + amp * 0.32 * r
    elif name == "octagon":
        r = _regular_polygon_r(thetas, 8)
        prof = 0.87 + amp * 0.30 * _circ_smooth3(r, passes=1)
    elif name == "clover":
        prof = 1.0 + amp * 0.6 * np.abs(np.cos(2*thetas) * np.sin(2*thetas))
    elif name == "butterfly":
        wing1 = np.abs(np.sin(thetas)) * (1.0 + 0.3*np.cos(4*thetas))
        wing2 = 0.3 * np.abs(np.cos(thetas)) * (1.0 + 0.2*np.sin(6*thetas))
        prof = 0.7 + amp * 0.5 * (wing1 + wing2)
    elif name == "donut":
        ripple = 0.3 * np.sin(8*thetas + 0.5) + 0.2 * np.sin(12*thetas + 1.2)
        prof = 1.2 + amp * 0.3 * ripple
    elif name == "sun":
        rays = (np.cos(24*thetas) ** 2) * 0.4
        corona = 0.2 * np.sin(3*thetas + 0.3)
        prof = 1.0 + amp * (rays + corona)
    elif name == "leaf":
        leaf_shape = np.abs(np.sin(thetas)) * (1.0 - 0.3*thetas/(2.0*np.pi))
        veins = 0.1 * np.sin(8*thetas)
        prof = 0.8 + amp * 0.5 * (leaf_shape + veins)
    elif name == "shell":
        spiral_decay = np.exp(-0.3 * thetas / (2.0*np.pi))
        ridges = 0.3 * np.sin(10*thetas)
        prof = 0.9 + amp * 0.4 * (spiral_decay + ridges)
    elif name == "lightning":
        zigzag = np.sin(8*thetas) * np.abs(np.cos(thetas))
        prof = 1.0 + amp * 0.5 * zigzag
    elif name == "snowflake":
        main_arms = (np.cos(6*thetas) ** 2)
        branches = 0.3 * np.cos(18*thetas) * np.abs(np.cos(6*thetas))
        prof = 0.9 + amp * 0.4 * (main_arms + branches)
    else:
        # fallback: gentle blob
        prof = 1.0 + amp * (0.25*np.sin(3.1*thetas + 0.2) +
                            0.20*np.sin(5.3*thetas + 1.3))
    # --- Robust normalization & guards ---
    prof = prof.astype(np.float32, copy=False)
    # Replace NaN/Inf defensively
    prof = np.nan_to_num(prof, nan=1.0, posinf=1.0, neginf=1.0)
    m = float(np.mean(prof))
    if m > 1e-6:
        prof = (prof / m).astype(np.float32)
    # Ensure strictly positive (renderer safety)
    prof = np.maximum(prof, 0.1).astype(np.float32)
    return prof
def draw_visuals(screen, vis_surf, state):
    # ---------- helpers ----------
    def _to_lin(c):
        return (pow(c[0]/255.0, 2.2), pow(c[1]/255.0, 2.2), pow(c[2]/255.0, 2.2))
    def _to_srgb(c):
        return (int(np.clip(pow(c[0], 1/2.2), 0, 1)*255),
                int(np.clip(pow(c[1], 1/2.2), 0, 1)*255),
                int(np.clip(pow(c[2], 1/2.2), 0, 1)*255))
    def lerp_color_gamma(c1, c2, t):
        t = max(0.0, min(1.0, float(t)))
        l1, l2 = _to_lin(c1), _to_lin(c2)
        mix = ((1-t)*l1[0] + t*l2[0], (1-t)*l1[1] + t*l2[1], (1-t)*l1[2] + t*l2[2])
        return _to_srgb(mix)
    def lerp_color(c1, c2, t):
        t = max(0.0, min(1.0, float(t)))
        return (int((1-t)*c1[0] + t*c2[0]),
                int((1-t)*c1[1] + t*c2[1]),
                int((1-t)*c1[2] + t*c2[2]))
    def smoothstep(x):
        x = max(0.0, min(1.0, x));  return x*x*(3-2*x)
    # ---------- per-frame inputs ----------
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
    bass_env  = float(state.get("bass_env", 0.0))
    voice_env = float(state.get("voice_env", 0.0))
    cur_flash = float(state.get("flash", 0.0))
    frame_idx = int(state.get("frame_idx", 0))
    # ---------- persistent state ----------
    if not hasattr(draw_visuals, "_init"):
        draw_visuals._pos = np.array([0.0, 0.0], dtype=np.float32)
        draw_visuals._vel = np.array([0.0, 0.0], dtype=np.float32)
        draw_visuals._prev_flash = 0.0
        draw_visuals._rng = np.random.default_rng(1337)
        draw_visuals._ema = np.zeros(n_bands, dtype=np.float32)
        draw_visuals._bar_len_ema = np.zeros(n_bands, dtype=np.float32)
        draw_visuals._bar_w_ema   = np.zeros(n_bands, dtype=np.float32)
        n_points = int(VIS.flub_points)
        draw_visuals._angles = _thetas_cached(n_points)
        draw_visuals._outer_pts = np.zeros((n_points, 2), dtype=np.float32)
        draw_visuals._inner_pts = np.zeros((n_points, 2), dtype=np.float32)
        draw_visuals._prev_outer = np.zeros((n_points, 2), dtype=np.float32)
        draw_visuals._prev_inner = np.zeros((n_points, 2), dtype=np.float32)
        draw_visuals._have_prev  = False
        from collections import deque
        draw_visuals._beat_times = deque(maxlen=10)
        draw_visuals._bpm_ema  = 0.0
        draw_visuals._bpm_last_t = 0.0
        draw_visuals._ripples = []
        # jemná textúra
        ns = 96
        noise = pygame.Surface((ns, ns), pygame.SRCALPHA)
        noise.fill((255, 255, 255, 255))
        rng_local = np.random.default_rng(2024)
        for _ in range(500):
            x = int(rng_local.integers(0, ns)); y = int(rng_local.integers(0, ns))
            r = int(rng_local.integers(1, 3)); col = int(rng_local.integers(210, 245))
            pygame.gfxdraw.filled_circle(noise, x, y, r, (col, col, col, 255))
        draw_visuals._noise_small = noise
        draw_visuals._pulse_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        # BPM → dynamika
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
        draw_visuals._mode_to   = draw_visuals._modes[0].copy()
        draw_visuals._base_angles = np.linspace(0, 2*np.pi, n_bands, endpoint=False).astype(np.float32)
        draw_visuals._rot_phase = 0.0
        # SHAPE / MORPH state
        draw_visuals._shapes_points = int(getattr(VIS, "flub_points", 128))
        draw_visuals._shape_active = False
        draw_visuals._shape_profile = None
        draw_visuals._shape_name = ""
        draw_visuals._shape_t0 = 0.0
        draw_visuals._shape_t1 = 0.0
        draw_visuals._shape_count = 0  # <= 5 na track
        # Jedinečný výber tvarov: definícia univerza + „similarity“ mapa
        draw_visuals._shape_all = [
            "flower6","flower8","star5","star7","lotus","petal",
            "waves","wave_complex","blob","diamond","hexagon",
            "burst","spiral","square","octagon","clover","butterfly",
            "donut","sun","leaf","shell","lightning","snowflake","triangle","gear"
        ]
        draw_visuals._shape_similar = {
            "flower6": {"flower8","lotus","petal","clover"},
            "flower8": {"flower6","lotus","petal","clover"},
            "star5":   {"star7","burst","sun"},
            "star7":   {"star5","burst","sun"},
            "waves":   {"wave_complex","spiral","donut"},
            "wave_complex": {"waves","spiral","donut"},
            "diamond": {"square","triangle","hexagon","octagon"},
            "hexagon": {"octagon","square","diamond","triangle"},
            "triangle":{"diamond","square","hexagon","octagon"},
            "square":  {"diamond","triangle","hexagon","octagon"},
            "spiral":  {"shell","donut","waves","wave_complex"},
            "burst":   {"sun","star5","star7"},
            "sun":     {"burst","star5","star7"},
            "petal":   {"flower6","flower8","lotus","clover"},
            "lotus":   {"flower6","flower8","petal","clover"},
            "clover":  {"lotus","flower6","flower8","petal"},
            "leaf":    {"shell"},
            "shell":   {"spiral"},
            "donut":   {"spiral","waves","wave_complex"},
            # the rest are distinct enough
        }
        draw_visuals._shape_pool = []
        draw_visuals._shape_last = None
        # plán morphov
        draw_visuals._morph_times = []     # absolútne časy (s)
        draw_visuals._morph_used  = set()  # indexy už spustené
        draw_visuals._morph_fade  = 0.40
        draw_visuals._armed_threshold = 12.0 + np.random.uniform(-3.0, +3.0)
        draw_visuals._long_done = False
        draw_visuals._shorts_before_long = int(draw_visuals._rng.integers(2, 5))
        # freeze balík pre STABLE morph
        draw_visuals._freeze = {"base_r": None, "thickness": None, "sx": 1.0, "sy": 1.0, "profile": None}
        draw_visuals._last_pos = -1.0
        # Keš pre škálovanie textúry
        draw_visuals._noise_cache = {}
        draw_visuals._init = True
    # reset pri seeku späť/nový track
    if pos_s < draw_visuals._last_pos - 1.0:
        draw_visuals._shape_active = False
        draw_visuals._shape_profile = None
        draw_visuals._shape_name = ""
        draw_visuals._shape_count = 0
        draw_visuals._morph_times = []
        draw_visuals._morph_used  = set()
        draw_visuals._freeze = {"base_r": None, "thickness": None, "sx": 1.0, "sy": 1.0, "profile": None}
        draw_visuals._armed_threshold = 12.0 + np.random.uniform(-3.0, +3.0)
        draw_visuals._long_done = False
        draw_visuals._shorts_before_long = int(draw_visuals._rng.integers(2, 5))
        # reset jedinečného výberu
        draw_visuals._shape_pool = []
        draw_visuals._shape_last = None
    draw_visuals._last_pos = pos_s
    # ---------- BPM detekcia ----------
    prev_flash = float(draw_visuals._prev_flash)
    if (prev_flash < 0.3) and (cur_flash > 0.7):
        from math import cos, sin
        draw_visuals._beat_times.append(t)
        base_r_tmp = radius * 0.9
        draw_visuals._ripples.append({"r": base_r_tmp,    "a": 200, "w": 2, "dr": 14.0, "da": 8})
        draw_visuals._ripples.append({"r": base_r_tmp*0.9,"a": 150, "w": 1, "dr": 18.0, "da": 10})
        if len(draw_visuals._beat_times) >= 4:
            intervals = np.diff(np.array(draw_visuals._beat_times, dtype=np.float32))
            valid = intervals[(intervals >= 60/220.0) & (intervals <= 60/55.0)]
            if valid.size >= 3:
                bpm_inst = 60.0 / float(np.mean(valid[-3:]))
                alpha = 0.45
                if draw_visuals._bpm_last_t == 0.0:
                    draw_visuals._bpm_ema = bpm_inst
                else:
                    draw_visuals._bpm_ema = (1 - alpha)*draw_visuals._bpm_ema + alpha*bpm_inst
                draw_visuals._bpm_last_t = t
    bpm_est = float(draw_visuals._bpm_ema)
    # ---------- mode blend ----------
    target_idx = draw_visuals._mode_target
    for i, (lo, hi) in enumerate(draw_visuals._bpm_ranges):
        if lo <= bpm_est < hi:
            target_idx = i; break
    if target_idx != draw_visuals._mode_target:
        draw_visuals._mode_from = draw_visuals._modes[draw_visuals._mode_idx].copy()
        draw_visuals._mode_to   = draw_visuals._modes[target_idx].copy()
        draw_visuals._mode_tstart = t
        draw_visuals._mode_blend = 0.0
        draw_visuals._mode_target = target_idx
    if draw_visuals._mode_blend < 1.0:
        x = (t - draw_visuals._mode_tstart) / max(0.001, draw_visuals._mode_tdur)
        b = smoothstep(x)
        draw_visuals._mode_blend = b
        if b >= 0.999:
            draw_visuals._mode_blend = 1.0
            draw_visuals._mode_idx = draw_visuals._mode_target
            draw_visuals._mode_from = draw_visuals._modes[draw_visuals._mode_idx].copy()
    b = draw_visuals._mode_blend
    F, Tm = draw_visuals._mode_from, draw_visuals._mode_to
    k  = (1-b)*F["k"]  + b*Tm["k"]
    c  = (1-b)*F["c"]  + b*Tm["c"]
    jitter = (1-b)*F["j"] + b*Tm["j"]
    wobble_mul = (1-b)*F["w"] + b*Tm["w"]
    bar_len_mul = (1-b)*F["bl"] + b*Tm["bl"]
    bar_thick_mul = (1-b)*F["bt"] + b*Tm["bt"]
    squash_h = (1-b)*F["sx"] + b*Tm["sx"]
    squash_v = (1-b)*F["sy"] + b*Tm["sy"]
    # ---------- fyzika stredu ----------
    if (prev_flash < 0.3) and (cur_flash > 0.7):
        ang = draw_visuals._rng.uniform(0.0, 2*np.pi)
        dirv = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
        px = 0.07 * min(w, h) * (0.5 + 0.5 * bass_env)
        draw_visuals._vel += dirv * px
    dt = 1.0 / max(30.0, fps)
    acc = -k * draw_visuals._pos - c * draw_visuals._vel
    draw_visuals._vel += acc * dt
    if jitter > 0.0:
        draw_visuals._vel += draw_visuals._rng.normal(0.0, jitter * (0.4 + 0.6 * bass_env), size=2)
    draw_visuals._pos += draw_visuals._vel * dt
    lim = 0.10 * min(w, h)
    norm = float(np.linalg.norm(draw_visuals._pos))
    if norm > lim:
        draw_visuals._pos[:] = draw_visuals._pos * (lim / (norm + 1e-9))
    draw_visuals._prev_flash = cur_flash
    cx_off = cx + int(draw_visuals._pos[0])
    cy_off = cy + int(draw_visuals._pos[1])
    # ---------- plánovanie morphov ----------
    if (dur_s > 0) and (not draw_visuals._morph_times):
        MAX = 5; margin = 2.0
        rng = draw_visuals._rng
        t1_base = min(12.0, 0.28 * dur_s)
        t1 = t1_base + rng.uniform(-3.0, +3.0)
        candidates = [t1]
        if dur_s >= 45:  candidates.append(0.55*dur_s + rng.uniform(-4.0, +4.0))
        if dur_s >= 65:  candidates.append(0.72*dur_s + rng.uniform(-3.0, +3.0))
        if dur_s >= 85:  candidates.append(0.86*dur_s + rng.uniform(-2.5, +2.5))
        if dur_s >= 110: candidates.append(0.93*dur_s + rng.uniform(-2.0, +2.0))
        cleaned = sorted(max(1.0, min(dur_s - margin, float(x))) for x in candidates)
        times = []
        for x in cleaned:
            if not times or abs(x - times[-1]) >= 5.0:
                times.append(x)
            if len(times) >= MAX: break
        if not times:
            times = [max(1.0, min(dur_s - margin, 0.30*dur_s))]
        draw_visuals._morph_times = times
    if (dur_s <= 0) and (not draw_visuals._morph_times):
        draw_visuals._morph_times = [float(draw_visuals._armed_threshold)]
    # ---------- pomocné: jedinečný výber tvaru ----------
    def _refill_shape_pool():
        pool = list(draw_visuals._shape_all)
        idx = draw_visuals._rng.permutation(len(pool))
        pool = [pool[i] for i in idx]
        # vyhni sa okamžitému opakovaniu posledného
        if draw_visuals._shape_last and pool and pool[0] == draw_visuals._shape_last:
            pool.append(pool.pop(0))
        draw_visuals._shape_pool = pool
    def _pick_unique_shape() -> str:
        if not draw_visuals._shape_pool:
            _refill_shape_pool()
        name = draw_visuals._shape_pool.pop(0)
        # ak je príliš podobný poslednému, vezmi prvý „odlišný“, ak existuje
        sim = draw_visuals._shape_similar.get(draw_visuals._shape_last, set())
        if draw_visuals._shape_last and name in sim and draw_visuals._shape_pool:
            # nájdi prvý kandidát, ktorý nie je v „sim“
            swap_idx = None
            for i, cand in enumerate(draw_visuals._shape_pool):
                if cand not in sim:
                    swap_idx = i; break
            if swap_idx is not None:
                # prehoď a použi odlišný
                alt = draw_visuals._shape_pool.pop(swap_idx)
                draw_visuals._shape_pool.insert(0, name)  # vráť pôvodný späť na začiatok
                name = alt
        draw_visuals._shape_last = name
        return name
    # ---------- štart morphu (STABLE) ----------
    def _start_morph():
        # jedinečný výber tvaru
        name  = _pick_unique_shape()
        inten = float(draw_visuals._rng.uniform(0.4, 0.8))
        prof  = shape_profile(name, int(draw_visuals._shapes_points), inten)
        # --- trvanie: 2–5 s, garantuj jeden dlhý (~7 s) na skladbu ---
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
        # aktivácia morphu
        draw_visuals._shape_profile = prof
        draw_visuals._shape_name    = name
        draw_visuals._shape_t0      = t
        draw_visuals._shape_t1      = t + dur
        draw_visuals._shape_active  = True
        draw_visuals._shape_count  += 1
        # STABLE morph – zmraz geometriu
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
    if (draw_visuals._shape_count < 5) and (not draw_visuals._shape_active) and draw_visuals._morph_times:
        for idx, tt in enumerate(draw_visuals._morph_times):
            if idx in draw_visuals._morph_used:
                continue
            if pos_s >= float(tt):
                draw_visuals._morph_used.add(idx)
                _start_morph()
                break
    # ---------- fade weight (in/out) ----------
    shape_w = 0.0
    if draw_visuals._shape_active and draw_visuals._shape_profile is not None:
        if t >= draw_visuals._shape_t1:
            draw_visuals._shape_active = False
            draw_visuals._shape_profile = None
            draw_visuals._freeze = {"base_r": None, "thickness": None, "sx": 1.0, "sy": 1.0, "profile": None}
        else:
            fin  = min(1.0, (t - draw_visuals._shape_t0) / max(1e-3, draw_visuals._morph_fade))
            fout = min(1.0, (draw_visuals._shape_t1 - t) / max(1e-3, draw_visuals._morph_fade))
            shape_w = smoothstep(fin) * smoothstep(fout)
    # ---------- rotácia lúčov ----------
    omega = 0.15 + 0.95 * min(1.0, max(0.0, bpm_est/180.0))
    draw_visuals._rot_phase = (draw_visuals._rot_phase + omega*dt) % (2*np.pi)
    cur_angles = (draw_visuals._base_angles + draw_visuals._rot_phase).astype(np.float32)
    x0 = (cx_off + (radius + rim_pad) * np.cos(cur_angles)).astype(np.int32)
    y0 = (cy_off + (radius + rim_pad) * np.sin(cur_angles)).astype(np.int32)
    # ---------- lúče (von) ----------
    draw_visuals._ema[:] = (1.0 - 0.25) * draw_visuals._ema + 0.25 * bands_in
    bar_len_ema = draw_visuals._bar_len_ema
    bar_w_ema   = draw_visuals._bar_w_ema
    alpha_len, alpha_w = 0.35, 0.45
    bpm_pulse = 0.5 + 0.5 * math.sin(t * (0.7 + 0.012*max(0.0, bpm_est)))
    step = 1 if fps >= 30 else 2
    for i in range(0, n_bands, step):
        v = float(draw_visuals._ema[i])
        vv = np.clip(v * (0.85 + 0.30*bpm_pulse) * (0.9 + 0.3*bass_env), 0.0, 1.0)
        L_raw = (4 + vv * max_bar_out * (1.10*bar_len_mul))
        W_raw = (1 + vv * (VIS.bar_thickness * (1.0*bar_thick_mul)))
        bar_len_ema[i] = (1-alpha_len)*bar_len_ema[i] + alpha_len*L_raw
        bar_w_ema[i]   = (1-alpha_w)  *bar_w_ema[i]   + alpha_w  *W_raw
        L = int(max(1, bar_len_ema[i]))
        thickness = int(np.clip(bar_w_ema[i], 1, VIS.bar_thickness * 2.1))
        ca = math.cos(cur_angles[i]); sa = math.sin(cur_angles[i])
        x1 = int(x0[i] + L * ca)
        y1 = int(y0[i] + L * sa)
        pygame.draw.line(vis_surf, (*VIS.bar_color, 255), (x0[i], y0[i]), (x1, y1), width=thickness)
    # ---------- flubber ring (reactive vs STABLE morph) ----------
    n_points = int(VIS.flub_points)
    angles = draw_visuals._angles  # cached
    if not hasattr(draw_visuals, "_phases") or len(draw_visuals._phases) != n_points:
        rng2 = np.random.default_rng(12345)
        draw_visuals._phases = rng2.uniform(0.0, 1000.0, size=n_points).astype(np.float32)
    phases = draw_visuals._phases
    # pre-počty pre "reactive" vetvu
    bpm_scale = 0.5 + 0.5*min(1.0, bpm_est/140.0)
    base_r_normal = radius * VIS.flub_base_frac * 1.18 * bpm_scale
    size_multiplier = np.interp(bass_env**3, [0.0,1.0],
                                [VIS.flub_min_size_multiplier, VIS.flub_max_size_multiplier])
    orient = 0.5 + 0.5 * math.sin(t * 0.5 * wobble_mul)
    sx_normal = squash_h * (1.0 + 0.04 * bass_env * (1.0 - orient))
    sy_normal = squash_v * (1.0 + 0.10 * bass_env * orient)
    breath = 0.5 + 0.5 * math.sin(t * (0.8*wobble_mul + 0.01*max(0.0, bpm_est)) + 6.5)
    breath = max(0.90, breath)
    global_breathe = size_multiplier * breath
    thickness_base_normal = radius * 0.08 * (0.85 + 0.60 * voice_env)
    outer_pts = draw_visuals._outer_pts
    inner_pts = draw_visuals._inner_pts
    prof = draw_visuals._shape_profile if (draw_visuals._shape_active and draw_visuals._shape_profile is not None) else None
    for i, a in enumerate(angles):
        if draw_visuals._shape_active and prof is not None:
            # === STABLE MORPH PATH (žiadna hudobná reakcia) ===
            base_r0 = float(draw_visuals._freeze["base_r"])
            thick0  = float(draw_visuals._freeze["thickness"])
            sx0     = float(draw_visuals._freeze["sx"])
            sy0     = float(draw_visuals._freeze["sy"])
            r_core = base_r0 * (1.0 + shape_w * (prof[i] - 1.0))
            r_out = r_core + 0.5 * thick0
            r_in  = r_core - 0.5 * thick0
            ca, sa = math.cos(a), math.sin(a)
            ox = cx_off + r_out * ca * sx0
            oy = cy_off + r_out * sa * sy0
            ix = cx_off + r_in  * ca * sx0
            iy = cy_off + r_in  * sa * sy0
        else:
            # === REACTIVE PATH ===
            n_local = _fbm1(t * VIS.flub_amorphous_speed*wobble_mul + phases[i]*0.17 + VIS.flub_angular_noise_scale*a)
            infl = 0.5 + 0.5 * n_local
            r_core = base_r_normal * global_breathe * (1.0 + (VIS.flub_amorphous_intensity + VIS.flub_voice_amorphous_gain*voice_env) * infl)
            r_core += 0.35 * base_r_normal * float(bands_in[i % n_bands]) * (0.4 + 0.6 * bass_env)
            r_core = max(base_r_normal * 0.70, r_core)
            thick = thickness_base_normal * (0.85 + 0.30 * infl)
            r_out = r_core + 0.5 * thick
            r_in  = r_core - 0.5 * thick
            ca, sa = math.cos(a), math.sin(a)
            ox = cx_off + r_out * ca * sx_normal
            oy = cy_off + r_out * sa * sy_normal
            ix = cx_off + r_in  * ca * sx_normal
            iy = cy_off + r_in  * sa * sy_normal
        outer_pts[i, 0] = ox; outer_pts[i, 1] = oy
        inner_pts[i, 0] = ix; inner_pts[i, 1] = iy
    # mikro-inercia (iba mimo morphu)
    if draw_visuals._have_prev and not draw_visuals._shape_active:
        outer_pts = 0.85 * draw_visuals._prev_outer + 0.15 * outer_pts
        inner_pts = 0.85 * draw_visuals._prev_inner + 0.15 * inner_pts
    draw_visuals._prev_outer[:] = outer_pts
    draw_visuals._prev_inner[:] = inner_pts
    draw_visuals._have_prev = True
    # farby podľa energie
    hi_energy = float(np.mean(bands_in[int(n_bands*0.6):])) if n_bands > 0 else 0.0
    energy_t = np.clip(0.5*voice_env + 0.5*hi_energy, 0.0, 1.0)
    fill_col = lerp_color_gamma((255, 70, 0), (255, 230, 120), energy_t)
    edge_col = lerp_color_gamma((255, 110, 40), (255, 255, 180), energy_t*0.7)
    # výplň prstenca
    for i in range(n_points):
        j = (i + 1) % n_points
        quad = [
            (int(outer_pts[i,0]), int(outer_pts[i,1])),
            (int(outer_pts[j,0]), int(outer_pts[j,1])),
            (int(inner_pts[j,0]), int(inner_pts[j,1])),
            (int(inner_pts[i,0]), int(inner_pts[i,1])),
        ]
        gfxdraw.filled_polygon(vis_surf, quad, (*fill_col, 255))
    # AA kontúry
    outer_loop = [(int(outer_pts[i,0]), int(outer_pts[i,1])) for i in range(n_points)]
    inner_loop = [(int(inner_pts[i,0]), int(inner_pts[i,1])) for i in range(n_points)]
    if len(outer_loop) > 2 and (fps >= 50 or (frame_idx % 2 == 0)):
        pygame.draw.aalines(vis_surf, edge_col, True, outer_loop)
    if len(inner_loop) > 2 and (fps >= 50 or (frame_idx % 2 == 0)):
        pygame.draw.aalines(vis_surf, edge_col, True, inner_loop)
    # jemný voice ring
    if voice_env > 0.01:
        voice_radius = int(radius * 0.70 * (1.0 + 0.5 * voice_env))
        voice_alpha = int(np.interp(voice_env, [0,1], [VIS.voice_alpha_min, VIS.voice_alpha_max]))
        voice_color = (*VIS.voice_base_color, voice_alpha)
        pygame.draw.circle(vis_surf, voice_color, (cx_off, cy_off), voice_radius, width=3)
    # kvantizuj farbu (16 úrovní) aby cache nerástla donekonečna
    def _q8(x, steps=16):
        return int(round(x * (steps-1)) * (255 // (steps-1)))
    glow_color = (_q8(fill_col[0]/255.0), _q8(fill_col[1]/255.0), _q8(fill_col[2]/255.0))
    glow_near = build_glow_circle_surface(radius, glow=10, color_rgb=glow_color, thickness=2)
    glow_far  = build_glow_circle_surface(int(radius*1.1), glow=18, color_rgb=lerp_color_gamma(glow_color, (255,255,255), 0.25), thickness=1)
    blit_center(vis_surf, glow_far,  (cx_off, cy_off))
    blit_center(vis_surf, glow_near, (cx_off, cy_off))
    # textúra
    key = (w, h)
    noise_big = draw_visuals._noise_cache.get(key)
    if noise_big is None:
        noise_big = pygame.transform.smoothscale(draw_visuals._noise_small, (w, h))
        draw_visuals._noise_cache[key] = noise_big
    vis_surf.blit(noise_big, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    # progress arc
    pad = VIS.progress_width // 2 + 8
    rect = pygame.Rect(cx_off - radius - pad, cy_off - radius - pad, 2 * (radius + pad), 2 * (radius + pad))
    frac = 0.0 if dur_s <= 0 else min(1.0, pos_s / dur_s)
    energy_color = lerp_color_gamma(lerp_color(VIS.red, VIS.yellow, min(1.0, bass_env ** 0.5)), VIS.white, 0.35*energy_t)
    draw_progress_arc_aa(vis_surf, rect, -math.pi / 2, frac, energy_color, VIS.progress_width)
    # beat ripples
    for rr in list(draw_visuals._ripples):
        rr["r"] += rr["dr"]; rr["a"] = max(0, rr["a"] - rr["da"])
        if rr["a"] <= 0 or rr["r"] > max(w, h):
            draw_visuals._ripples.remove(rr); continue
        pygame.draw.circle(vis_surf, (*energy_color, rr["a"]), (cx_off, cy_off), int(rr["r"]), width=rr["w"])
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
    vis_surf  = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    text_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    def after_mode_change_rescale():
        nonlocal vis_surf, text_surf
        w,h = screen.get_size()
        vis_surf  = pygame.Surface((w,h), pygame.SRCALPHA)
        text_surf = pygame.Surface((w,h), pygame.SRCALPHA)
        rescale_background_for_size() if original_bg_surface is not None else choose_background(True)
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
    v_mode = preset_indices[0]
    def v_mode_desc(m):
        st=v_states[m%len(v_states)]
        return f"HUD {'ON' if st['hud'] else 'OFF'} / FPS {'ON' if st['fps'] else 'OFF'} / TIME {'ON' if st['time'] else 'OFF'} / TITLE {'ON' if st['title'] else 'OFF'}"
    help_visible = False
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
    def play_track_music(track: Track, start_sec: float = 0.0, volume: float = 0.85):
        nonlocal playback_mode
        stop_all_playback()
        pygame.mixer.music.load(str(track.path))
        if start_sec > 0.0:
            try:
                pygame.mixer.music.play(start=start_sec)
                playback_mode = "music"
            except TypeError:
                seek_and_play_with_ffmpeg(track, start_sec, volume)
                return
        else:
            pygame.mixer.music.play()
            playback_mode = "music"
        pygame.mixer.music.set_volume(volume)
    def seek_and_play_with_ffmpeg(track: Track, start_sec: float, volume: float = 0.85):
        nonlocal playback_mode, current_main_sound, seg_cur_start, seg_cur_len
        stop_all_playback()
        if not ffok:
            try:
                pygame.mixer.music.load(str(track.path))
                pygame.mixer.music.play()
                pygame.mixer.music.set_volume(volume)
                playback_mode = "music"
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
        seg_cur_start = start_sec
        seg_cur_len   = dur_left
    def play_segment_at(at: float, seg_len: float) -> bool:
        nonlocal playback_mode, current_main_sound, seg_cur_start, seg_cur_len
        seg_len = max(0.1, min(UI.seek_segment_sec, (current_track.duration_sec or 0.0) - at))
        if seg_len <= 0.05: 
            return False
        pcm = decode_pcm_segment(current_track.path, at, seg_len, sr=AUDIO.target_sr)
        if pcm.shape[0] == 0:
            return False
        current_main_sound = numpy_to_sound(pcm)
        chan_main.play(current_main_sound)  # pri prvom segmente; pre ďalšie použijeme .queue
        chan_main.set_volume(volume, volume)
        playback_mode = "channel"
        seg_cur_start = at
        seg_cur_len   = seg_len
        return True
    def maybe_queue_next_segment(pos_now: float):
        nonlocal seg_cur_len  # <- dôležité, meníme premennú z obalujúceho scope

        if playback_mode != "channel":
            return

        seg_end = seg_cur_start + seg_cur_len

        # ak nič nie je v queue a koniec segmentu je blízko, doqueue-uj ďalší
        if (chan_main.get_queue() is None
            and (seg_end - pos_now) < 4.0
            and pos_now < (current_track.duration_sec or 0.0) - 0.1):

            next_len = min(UI.seek_segment_sec, (current_track.duration_sec or 0.0) - seg_end)
            if next_len > 0.05:
                pcm = decode_pcm_segment(current_track.path, seg_end, next_len, sr=AUDIO.target_sr)
                if pcm.shape[0]:
                    chan_main.queue(numpy_to_sound(pcm))
                    seg_cur_len += next_len

    volume = 0.85
    last_unmuted_volume = 0.85
    muted = False
    play_track_music(current_track, 0.0, volume)
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
        dur = current_track.duration_sec or max(0.0, cur+1.0)
        new_pos = max(0.0, min(max(0.0, dur-0.05), cur + delta_sec))
        set_play_start(new_pos)
        seek_and_play_with_ffmpeg(current_track, new_pos, volume)
        log.debug("Seek relative %.2f -> new_pos=%.2f (mode=%s)", delta_sec, new_pos, playback_mode)
        return new_pos
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
    PRIO_NOW      = 0
    PRIO_PREFETCH = 5
    PRIO_BULK     = 9
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
    # Initialize cache with configured capacity (presunuté PRED štart threadov)
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
    seg_cur_len   = 0.0
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
                    if bg_enabled:
                        choose_background(True)
                    else:
                        # vypni pozadie úplne, nech sa nekreslí a ide LMB-drag
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
                    log.debug("Shuffle -> %s", "ON" if shuffle else "OFF")
                elif ev.key == pygame.K_r:
                    repeat_all = not repeat_all
                    log.debug("Repeat all -> %s", "ON" if repeat_all else "OFF")
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
                    if volume <= 0.0:
                        muted = True
                    last_volume_popup_t = time.time()
                    log.debug("Volume Down -> %.2f", volume)
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
        # text_surf: nechá sa vyčistiť v render_ui_and_text()
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
        state = {"pos":pos_now,"dur":dur_now,"bass_env":bass_env,"flash":flash,"bands":bands,"fps":clock.get_fps(),"voice_env":voice_env,"frame_idx":frame_idx}
        draw_visuals(screen, vis_surf, state)
        screen.blit(vis_surf, (0, 0))
        w, h = screen.get_size()
        cx, cy = w // 2, h // 2
        st_view = v_states[v_mode]
        show_time  = st_view["time"]
        show_title = st_view["title"]
        show_hud   = st_view["hud"]
        show_fps   = st_view["fps"]
        any_text_allowed = show_time or show_title or show_hud or show_fps
        # === TEXT VRSTVA (top-left stack: VOL, TOAST, HUD, TITLE, TIME; FPS dopíšeme pod to) ===
        vol_show, y_cursor = render_ui_and_text(
            text_surf, w, h, cx, cy,
            pos_now, dur_now, volume,
            last_volume_popup_t, toast_msg, toast_until,
            show_title, show_time, show_hud, show_fps,
            v_mode, shuffle, repeat_all,
            current_track, title_font, font_small_sys, vol_font_sys
        )
        # FPS → pod stack, ak je zapnutý presetom
        if show_fps:
            fps_val = clock.get_fps()
            dbg = f"FPS {fps_val:4.1f} | pos {pos_now:6.2f}/{dur_now:6.2f} | idx {index_in_order+1}/{len(order)} | V {v_mode+1}/{len(v_states)} | mode {playback_mode}"
            fps_surf = font_small_sys.render(dbg, True, VIS.text_dim)
            text_surf.blit(fps_surf, (14, y_cursor))
            y_cursor += fps_surf.get_height() + 4
        # Finálne (jediné) blitnutie textovej vrstvy
        if any_text_allowed or (time.time() < toast_until) or vol_show:
            screen.blit(text_surf, (0, 0))
        # HELP overlay (samostatná vrstva)
        if help_visible:
            if (help_cache_surf is None) or (help_cache_surf.get_width()!=w or help_cache_surf.get_height()!=h):
                help_cache_surf = build_help_surface(w,h)
            screen.blit(help_cache_surf, (0,0))
        # Check if we need to queue next segment
        maybe_queue_next_segment(pos_now)
        pygame.display.flip()
        frame_idx += 1
        # Use regular tick instead of busy_loop for lower CPU usage
        clock.tick(VIS.fps_target)
        # Auto-next
        if dur_now > 0:
            track_ended = False
            if pos_now >= dur_now - 0.05:
                if playback_mode == "music":
                    if not pygame.mixer.music.get_busy():
                        track_ended = True
                else:
                    if not chan_main.get_busy():
                        track_ended = True
            if track_ended:
                next_idx = (index_in_order+1) % len(order) if (index_in_order+1<len(order) or repeat_all) else index_in_order
                if next_idx != index_in_order:
                    next_track = lib.tracks[order[next_idx]]
                    current_track = next_track; index_in_order = next_idx
                    set_play_start(0.0)
                    play_track_music(current_track, 0.0, volume)
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
    try:
        log.debug("Shutting down…")
        fft_should_run = False
        fft_event.set()
        load_should_run = False
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
        pygame.quit()
        log.debug("Exited cleanly")
if __name__=="__main__":
    main()
