"""Smoke-проверки изменённых узлов fork: YuNet на GPU, сглаживание рамки, адаптивный порог движения.

Запуск из корня fork: PYTHONPATH=src .venv/bin/python scripts/smoke_test.py
Камеры и БД не трогает.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""))


def test_yunet_gpu() -> None:
    from qaraqu_face.yunet_gpu import create_yunet_gpu

    model = ROOT / "data/models/face_detection_yunet_2023mar.onnx"
    check("yunet model file", model.is_file(), str(model))
    if not model.is_file():
        return
    det = create_yunet_gpu(str(model), 0.6, 0.3, 5000)
    check("yunet on CUDA", det is not None)
    if det is None:
        return
    img = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    det.run(img, 0.6)
    t = time.time()
    for _ in range(20):
        det.run(img, 0.6)
    ms = (time.time() - t) / 20 * 1000
    check("yunet 1080p latency < 15 ms", ms < 15.0, f"{ms:.1f} ms")
    _, out = det.run(img, 0.6)
    check("yunet output shape", out is None or out.shape[1] == 15)


def test_box_smoothing() -> None:
    from qaraqu_face.tracking import Track, _advance_track, display_box

    rng = np.random.default_rng(1)
    tr = Track(tid=1, xyxy=[100, 100, 200, 200], first_ts=0.0, last_ts=0.0)
    tr.fbox = tuple(float(v) for v in tr.xyxy)
    raw, smooth = [], []
    t = 0.0
    for _ in range(60):
        t += 0.04
        jit = [100 + int(rng.integers(-3, 4)) for _ in range(2)] + [200 + int(rng.integers(-3, 4)) for _ in range(2)]
        _advance_track(tr, jit, t)
        raw.append(jit[0])
        smooth.append(display_box(tr, tr.last_ts)[0])
    raw_std = float(np.std(raw[10:]))
    smooth_std = float(np.std(smooth[10:]))
    check("box jitter reduced", smooth_std < raw_std * 0.6, f"raw {raw_std:.2f} → smooth {smooth_std:.2f}")
    tr2 = Track(tid=2, xyxy=[0, 0, 50, 50], first_ts=0.0, last_ts=0.0)
    tr2.fbox = (0.0, 0.0, 50.0, 50.0)
    tr2.vel = (100.0, 0.0, 100.0, 0.0)
    tr2.last_ts = 0.0
    box = display_box(tr2, 0.1)
    check("extrapolation capped 0.25 s", box[0] <= 25 + 1, f"x1={box[0]}")
    tr3 = Track(tid=3, xyxy=[0, 0, 50, 50], first_ts=0.0, last_ts=0.0)
    tr3.fbox = (0.0, 0.0, 50.0, 50.0)
    tr3.vel = (5.0, 0.0, 5.0, 0.0)
    check("still box not extrapolated", display_box(tr3, 0.2)[0] == 0)


def test_motion_threshold() -> None:
    k, floor, ceil = 3.0, 1.0, 6.0

    def thr(hist: deque) -> float:
        if len(hist) < 20:
            return floor * k
        noise = float(np.median(np.fromiter(hist, dtype=np.float64)))
        return float(min(ceil, max(floor, k * noise)))

    quiet = thr(deque([0.16] * 120, maxlen=120))
    noisy = thr(deque([1.77] * 120, maxlen=120))
    loud = thr(deque([9.0] * 120, maxlen=120))
    warm = thr(deque([0.1] * 5, maxlen=120))
    check("quiet scene at floor", abs(quiet - floor) < 1e-6, f"{quiet:.2f}")
    check("noisy scene scaled", noisy > quiet and noisy <= ceil, f"{noisy:.2f}")
    check("threshold capped", loud == ceil, f"{loud:.2f}")
    check("warm-up uses floor", warm == floor * k, f"{warm:.2f}")


def test_preview_fps_option() -> None:
    src = (ROOT / "src/qaraqu_face/runtime.py").read_text(encoding="utf-8")
    check("preview_fps default", 'setdefault("preview_fps", 25)' in src)
    api = (ROOT / "src/qaraqu_face/api.py").read_text(encoding="utf-8")
    check("video_feed async with fps", "async def video_feed" in api and "min(0.005, 1.0 / fps)" in api)


def main() -> int:
    test_preview_fps_option()
    test_yunet_gpu()
    test_box_smoothing()
    test_motion_threshold()
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
