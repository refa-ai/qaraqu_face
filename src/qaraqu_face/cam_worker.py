"""Один процесс = одна камера. Декод NVDEC в главном потоке процесса, кадры в shared memory.

Аналитика читает слот из shared memory и не создаёт потоков на декодирование.
Слоты двойные: писатель пишет в слот, которого нет в заголовке, читатель копирует
последний завершённый кадр и проверяет, что заголовок не сменился.
"""

from __future__ import annotations

import os
import struct
import sys
import time
from collections import deque
from multiprocessing import shared_memory

import av
import cv2
import numpy as np

HDR = struct.Struct("<qdiiid")
HDR_BYTES = 64
MAX_W, MAX_H = 1920, 1080
SLOT_BYTES = MAX_W * MAX_H * 3
SHM_BYTES = HDR_BYTES + 2 * SLOT_BYTES
RTSP_OPTS = {
    "rtsp_transport": "tcp",
    "stimeout": "4000000",
    "rw_timeout": "4000000",
    "fflags": "nobuffer",
    "flags": "low_delay",
    "max_delay": "500000",
}
MOTION_REF_S = 0.20
THUMB = (96, 72)


def _log(msg: str) -> None:
    sys.stderr.write(f"{time.strftime('%H:%M:%S')} cam_worker {msg}\n")
    sys.stderr.flush()


def _fit(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    side = max(h, w)
    if side <= MAX_W and h * w * 3 <= SLOT_BYTES:
        return bgr
    scale = min(MAX_W / float(side), (SLOT_BYTES / float(h * w * 3)) ** 0.5)
    return cv2.resize(bgr, (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2), interpolation=cv2.INTER_AREA)


def _open(url: str, device: int) -> av.container.InputContainer:
    kw: dict = {"options": dict(RTSP_OPTS)}
    try:
        kw["hwaccel"] = av.codec.hwaccel.HWAccel("cuda", device=device, allow_software_fallback=True)
    except Exception as e:  # noqa: BLE001
        _log(f"cuda недоступна: {e}")
    return av.open(url, **kw)


def run(url: str, shm_name: str, min_dt: float, device: int = 0) -> None:
    parent = os.getppid()
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = shm.buf
    seq = 0
    motion_ref: np.ndarray | None = None
    motion_ref_ts = 0.0
    backoff = 1.0
    while True:
        if os.getppid() != parent:
            break
        try:
            cont = _open(url, device)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            _log(f"open: {msg}")
            time.sleep(60.0 if ("Errno 111" in msg or "No route" in msg) else backoff)
            backoff = min(backoff * 1.5, 8.0)
            continue
        backoff = 1.0
        vs = cont.streams.video[0]
        for setter in (
            lambda: setattr(vs, "thread_count", 1),
            lambda: setattr(vs.codec_context, "thread_count", 1),
        ):
            try:
                setter()
            except Exception:  # noqa: BLE001
                pass
        last = 0.0
        try:
            for packet in cont.demux(vs):
                if os.getppid() != parent:
                    return
                try:
                    frames = packet.decode()
                except Exception:  # noqa: BLE001
                    continue
                for frame in frames:
                    now = time.time()
                    if now - last < min_dt:
                        continue
                    last = now
                    bgr = _fit(cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_YUV2BGR_NV12))
                    h, w = bgr.shape[:2]
                    small = cv2.cvtColor(cv2.resize(bgr, THUMB, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
                    motion = 0.0
                    if motion_ref is not None and motion_ref.shape == small.shape:
                        motion = float(np.mean(np.abs(small.astype(np.int16) - motion_ref.astype(np.int16))))
                    if motion_ref is None or now - motion_ref_ts >= MOTION_REF_S:
                        motion_ref = small
                        motion_ref_ts = now
                    seq += 1
                    slot = seq % 2
                    off = HDR_BYTES + slot * SLOT_BYTES
                    view = np.ndarray((h, w, 3), dtype=np.uint8, buffer=buf, offset=off)
                    np.copyto(view, bgr)
                    HDR.pack_into(buf, 0, seq, now, h, w, slot, motion)
        except Exception as e:  # noqa: BLE001
            _log(f"поток: {e}")
        finally:
            try:
                cont.close()
            except Exception:  # noqa: BLE001
                pass
        time.sleep(1.0)


if __name__ == "__main__":
    url, shm_name, min_dt = sys.argv[1], sys.argv[2], float(sys.argv[3])
    dev = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    run(url, shm_name, min_dt, dev)
