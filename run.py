#!/usr/bin/env python3
"""Запуск qaraqu_face API (:5020 по умолчанию)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ЦП: без ограничения OpenMP/MKL/OpenCV каждый пул берёт все ядра и плодит потоки
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from qaraqu_face.config import load_config  # noqa: E402


def main() -> None:
    import faulthandler
    faulthandler.enable(file=open(ROOT / "data" / "crash_trace.log", "a", buffering=1), all_threads=True)
    cfg = load_config()
    api = cfg.get("api", {})
    host = os.environ.get("QARAQU_FACE_HOST", api.get("host", "0.0.0.0"))
    port = int(os.environ.get("QARAQU_FACE_PORT", api.get("port", 5020)))
    import uvicorn

    uvicorn.run("qaraqu_face.api:app", host=host, port=port, reload=False, app_dir=str(ROOT / "src"))


if __name__ == "__main__":
    main()
