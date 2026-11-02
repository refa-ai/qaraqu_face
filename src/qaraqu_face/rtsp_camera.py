"""RTSP main-поток камеры: один поток на камеру, NVDEC-декод, слот последнего кадра.

Поток захвата = поток логики камеры. Открытие RTSP делается в этом же потоке,
отдельных потоков на открытие и на декодирование нет.

Интерфейс совместим с GpuRtspCapture: seq, read, read_full, _full_seq, motion,
motion_threshold, fps, watch, isOpened, release, _ts, _opened_at.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("qaraqu_face.rtsp_camera")

RTSP_OPTS = {
    "rtsp_transport": "tcp",
    "stimeout": "4000000",
    "rw_timeout": "4000000",
    "fflags": "nobuffer",
    "flags": "low_delay",
    "max_delay": "500000",
    "threads": "1",
}
OPEN_SEM = threading.Semaphore(4)
UNREACHABLE_BACKOFF_S = 60.0
RETRY_MAX_S = 8.0
# слот превью: кадр не шире этого (JPEG и рамки работают в этих координатах)
DISPLAY_MAX_SIDE = 960
# полный 1080p для детекции и 1:N — реже
FULL_EVERY_S = 0.80
FULL_WATCH_S = 0.35
# без просмотра превью — редкая выборка (CPU-конвертация на всех камерах)
IDLE_STORE_S = 0.12
WATCH_STORE_S = 0.04
THUMB = (96, 72)
MOTION_REF_S = 0.20
MOTION_K = 3.0
MOTION_FLOOR = 1.0
MOTION_CEIL = 6.0
MOTION_WIN = 120


def safe_err(exc: BaseException) -> str:
    return re.sub(r"(rtsp://[^:/?#]+):[^@/]+@", r"\1:***@", str(exc), flags=re.I)


def nv12_to_bgr(frame: Any) -> np.ndarray:
    """nv12 с GPU → BGR через OpenCV: libav-конвертер в bgr24 поднимает ~15 потоков на камеру."""
    return cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_YUV2BGR_NV12)


def fit_side(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return arr
    scale = max_side / float(side)
    return cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


class RtspCamera:
    """Один поток на камеру. Слоты: превью (≤960) и полный кадр (1080p для детекции)."""

    def __init__(self, url: str, *, name: str, device: int = 0, use_cuda: bool = True) -> None:
        self.url = url
        self.name = name
        self.device = int(device)
        self.use_cuda = bool(use_cuda)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._full: np.ndarray | None = None
        self._full_seq = 0
        self._seq = 0
        self._ts = 0.0
        self._opened_at = 0.0
        self._watch_until = 0.0
        self._motion = 0.0
        self._motion_hist: deque[float] = deque(maxlen=MOTION_WIN)
        self._motion_ref: np.ndarray | None = None
        self._motion_ref_ts = 0.0
        self.fps = 0.0
        self._rate_n = 0
        self._rate_t = time.time()
        self._opened = False
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{name}", daemon=True)
        self._thread.start()

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def motion(self) -> float:
        return self._motion

    @property
    def motion_blk(self) -> float:
        return getattr(self, "_motion_blk", 0.0)

    @property
    def motion_threshold(self) -> float:
        hist = self._motion_hist
        if len(hist) < 20:
            return MOTION_FLOOR * MOTION_K
        noise = float(np.median(np.fromiter(hist, dtype=np.float64)))
        return float(min(MOTION_CEIL, max(MOTION_FLOOR, MOTION_K * noise)))

    def isOpened(self) -> bool:
        return self._thread.is_alive() and not self._stop.is_set() and self.error is None

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def read_full(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._full is None:
                return False, None
            return True, self._full.copy()

    def grab(self) -> bool:
        return False

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        return self.read()

    def watch(self, seconds: float = 1.0) -> None:
        self._watch_until = time.time() + max(0.2, float(seconds))

    def release(self) -> None:
        self._stop.set()
        self._opened = False

    def _hw(self, av: Any) -> Any:
        if not self.use_cuda:
            return None
        try:
            return av.codec.hwaccel.HWAccel("cuda", device=self.device, allow_software_fallback=True)
        except Exception as e:  # noqa: BLE001
            log.warning("cam %s: CUDA hwaccel недоступен (%s)", self.name, safe_err(e))
            return None

    def _run(self) -> None:
        import av

        backoff = 1.0
        while not self._stop.is_set():
            if not OPEN_SEM.acquire(timeout=30.0):
                continue
            cont = None
            try:
                kw: dict[str, Any] = {"options": dict(RTSP_OPTS)}
                hw = self._hw(av)
                if hw is not None:
                    kw["hwaccel"] = hw
                cont = av.open(self.url, **kw)
            except Exception as e:  # noqa: BLE001
                self.error = safe_err(e)
                log.warning("cam %s: open: %s", self.name, self.error)
                if "Errno 111" in str(e) or "No route" in str(e):
                    backoff = UNREACHABLE_BACKOFF_S
            finally:
                OPEN_SEM.release()
            if cont is None:
                if self._stop.wait(backoff):
                    break
                backoff = min(max(backoff * 1.5, 1.0), max(RETRY_MAX_S, backoff))
                continue

            backoff = 1.0
            vs = cont.streams.video[0]
            # Один поток декодера: NVDEC иначе держит ~19 потоков на камеру.
            # thread_type=NONE обязателен — без него один thread_count даёт сегфолт.
            try:
                vs.codec_context.thread_count = 1
                vs.thread_type = "NONE"
            except Exception:  # noqa: BLE001
                pass
            self._opened = True
            self._opened_at = time.time()
            self.error = None
            log.info("cam %s: открыт", self.name)
            try:
                self._pump(cont, vs)
            except Exception as e:  # noqa: BLE001
                self.error = safe_err(e)
                log.warning("cam %s: поток: %s", self.name, self.error)
            finally:
                self._opened = False
                try:
                    cont.close()
                except Exception:  # noqa: BLE001
                    pass
            if self._stop.wait(1.0):
                break

    def _pump(self, cont: Any, vs: Any) -> None:
        next_full = time.time() + (hash(self.name) % 500) / 1000.0
        next_thumb = 0.0
        for packet in cont.demux(vs):
            if self._stop.is_set():
                return
            try:
                frames = packet.decode()
            except Exception:  # noqa: BLE001
                continue
            for frame in frames:
                now = time.time()
                watched = now < self._watch_until
                want_full = now >= next_full
                try:
                    if want_full:
                        next_full = now + (FULL_WATCH_S if watched else FULL_EVERY_S)
                        full = nv12_to_bgr(frame)
                        self._store_full(full)
                        self._store(fit_side(full, DISPLAY_MAX_SIDE), now)
                    elif watched:
                        self._store(fit_side(nv12_to_bgr(frame), DISPLAY_MAX_SIDE), now)
                        next_thumb = now + WATCH_STORE_S
                    elif now >= next_thumb:
                        self._store(fit_side(nv12_to_bgr(frame), DISPLAY_MAX_SIDE), now)
                        next_thumb = now + IDLE_STORE_S
                except Exception:  # noqa: BLE001
                    continue

    def _store_full(self, arr: np.ndarray) -> None:
        with self._lock:
            if self._full is None or self._full.shape != arr.shape:
                self._full = np.empty_like(arr)
            np.copyto(self._full, arr)
            self._full_seq += 1

    def _store(self, arr: np.ndarray, now: float) -> None:
        small = cv2.resize(arr, THUMB, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        ref = self._motion_ref
        if ref is not None and ref.shape == gray.shape:
            diff = np.abs(gray.astype(np.int16) - ref.astype(np.int16))
            self._motion = float(np.mean(diff))
            # максимум по сетке 4×4: небольшой человек даёт заметную разницу в своём блоке
            h, w = diff.shape
            blocks = diff[: h - h % 4, : w - w % 4].reshape(4, h // 4, 4, w // 4).mean(axis=(1, 3))
            self._motion_blk = float(blocks.max())
        else:
            self._motion = 0.0
            self._motion_blk = 0.0
        self._motion_hist.append(self._motion)
        if ref is None or now - self._motion_ref_ts >= MOTION_REF_S:
            self._motion_ref = gray
            self._motion_ref_ts = now
        with self._lock:
            if self._frame is None or self._frame.shape != arr.shape:
                self._frame = np.empty_like(arr)
            np.copyto(self._frame, arr)
            self._ts = now
            self._seq += 1
            self._rate_n += 1
            dt = now - self._rate_t
            if dt >= 1.0:
                self.fps = self._rate_n / dt
                self._rate_n = 0
                self._rate_t = now
