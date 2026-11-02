"""Превью стены: sub-поток камеры отдельным процессом ffmpeg.

Sub не участвует в детекции и распознавании. Процесс запускается только когда камеру
смотрят (стена/превью) и останавливается, когда никто не смотрит.
Кадры — JPEG, читаются из stdout по маркерам SOI/EOI, без потоков Python на чтение.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Callable

log = logging.getLogger("qaraqu_face.sub_preview")

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"
_IDLE_STOP_S = 6.0
_MAX_BUF = 4 * 1024 * 1024


class SubPreview:
    """Один ffmpeg на камеру. Вызывать watch() при каждом запросе превью."""

    def __init__(self, url: str, on_jpeg: Callable[[bytes], None], fps: int = 12, width: int = 640) -> None:
        self.url = url
        self._on_jpeg = on_jpeg
        self._fps = max(1, int(fps))
        self._width = max(160, int(width))
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._want_until = 0.0
        self._stop = False

    def watch(self) -> None:
        self._want_until = time.time() + _IDLE_STOP_S
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start_locked()

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start_locked(self) -> None:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay",
            "-i", self.url,
            "-vf", f"fps={self._fps},scale={self._width}:-2",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "6", "-",
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        except OSError as e:
            log.warning("sub ffmpeg не запущен: %s", e)
            self._proc = None
            return
        self._reader = threading.Thread(target=self._read, args=(self._proc,), name="sub-ffmpeg", daemon=True)
        self._reader.start()

    def _read(self, proc: subprocess.Popen) -> None:
        buf = bytearray()
        out = proc.stdout
        try:
            while True:
                if time.time() > self._want_until:
                    break
                chunk = out.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    s = buf.find(_SOI)
                    if s < 0:
                        buf.clear()
                        break
                    e = buf.find(_EOI, s + 2)
                    if e < 0:
                        if s > 0:
                            del buf[:s]
                        break
                    frame = bytes(buf[s : e + 2])
                    del buf[: e + 2]
                    self._on_jpeg(frame)
                if len(buf) > _MAX_BUF:
                    buf.clear()
        finally:
            self._kill(proc)

    def _kill(self, proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            if self._proc is proc:
                self._proc = None

    def stop(self) -> None:
        self._want_until = 0.0
        with self._lock:
            proc = self._proc
        if proc is not None:
            self._kill(proc)
