"""Эмбеддеры лиц на PyTorch: AdaFace IR-101 (по умолчанию) и ArcFace w600k_r50.

Оба принимают выровненное лицо 112×112 (norm_crop), отдают 512-мерный L2-вектор.
Отличие только в порядке каналов: ArcFace обучен на RGB, AdaFace — на BGR.
Граф тот же, что в ONNX: векторы картотеки не пересчитываются.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qaraqu_face.config import load_config, resolve

# Инференс из коротких потоков копит память; гоняем его в одном постоянном потоке
_embed_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arcface")


def l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


class ArcFaceEmbedder:
    name = "arcface"
    dim = 512
    rgb = True  # порядок каналов на входе сети

    def __init__(self, engine: Path):
        from qaraqu_face.torch_runtime import TorchRunner

        self.model_path = Path(engine)
        self.runner = TorchRunner(self.model_path)
        self.input_name = self.runner.input_names[0]
        self.output_name = self.runner.output_names[0]
        self._providers = ["PyTorch"]

    @property
    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "model": str(self.model_path),
            "providers": list(self._providers),
        }

    def embed_bgr(
        self,
        bgr: np.ndarray,
        *,
        landmarks=None,
        xyxy=None,
    ) -> np.ndarray:
        """Сначала norm_crop (InsightFace), потом PyTorch; (x − 127.5)/127.5."""
        from qaraqu_face.face_align import align_face

        img = align_face(bgr, xyxy=xyxy, landmarks=landmarks, image_size=112)
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(img, (2, 0, 1))[None, ...]
        out = _embed_pool.submit(self.runner.run, {self.input_name: blob}).result()[self.output_name]
        return l2(out)

    def embed_batch(self, items: list[tuple[np.ndarray, Any, Any]]) -> list[np.ndarray]:
        """Все лица кадра одним прогоном модели: (bgr, landmarks, xyxy) на каждое лицо."""
        from qaraqu_face.face_align import align_face

        if not items:
            return []
        blobs = []
        for bgr, landmarks, xyxy in items:
            img = align_face(bgr, xyxy=xyxy, landmarks=landmarks, image_size=112)
            if self.rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) - 127.5) / 127.5
            blobs.append(np.transpose(img, (2, 0, 1)))
        batch = np.stack(blobs).astype(np.float32)
        out = _embed_pool.submit(self.runner.run, {self.input_name: batch}).result()[self.output_name]
        return [l2(v) for v in out]


class AdaFaceEmbedder(ArcFaceEmbedder):
    """AdaFace IR-101 WebFace12M: обучен под низкое качество (IJB-S, TinyFace); вход BGR."""

    name = "adaface"
    rgb = False


_BACKENDS = {"arcface": ArcFaceEmbedder, "adaface": AdaFaceEmbedder}
_DEFAULT_ENGINES = {
    "arcface": "data/models/w600k_r50.onnx",
    "adaface": "data/models/adaface_ir101_webface12m.onnx",
}


def embedder_engine(cfg: dict) -> tuple[str, Path]:
    """(backend, путь к движку) из конфига recognize.backend / recognize.<backend>.engine."""
    rec = cfg.get("recognize", {}) if isinstance(cfg.get("recognize"), dict) else {}
    backend = str(rec.get("backend") or "adaface").lower()
    if backend not in _BACKENDS:
        raise ValueError(f"recognize.backend={backend!r}; допустимо {sorted(_BACKENDS)}")
    sub = rec.get(backend, {}) if isinstance(rec.get(backend), dict) else {}
    return backend, resolve(cfg, sub.get("weights") or sub.get("onnx") or sub.get("engine") or _DEFAULT_ENGINES[backend])


_embedders: dict[str, ArcFaceEmbedder] = {}
_embed_lock = __import__("threading").Lock()


def get_embedder(cfg: dict | None = None, backend: str | None = None) -> ArcFaceEmbedder:
    """По одной PyTorch-модели на бэкенд. Оба живут в процессе: сначала ArcFace, затем AdaFace.

    Без lock десятки камер грузят веса параллельно и забивают память.
    """
    cfg = cfg or load_config()
    if not backend:
        backend, _engine = embedder_engine(cfg)
    backend = str(backend).lower()
    if backend not in _BACKENDS:
        raise ValueError(f"recognize backend={backend!r}; допустимо {sorted(_BACKENDS)}")
    hit = _embedders.get(backend)
    if hit is not None:
        return hit
    with _embed_lock:
        hit = _embedders.get(backend)
        if hit is not None:
            return hit
        rec = cfg.get("recognize", {}) if isinstance(cfg.get("recognize"), dict) else {}
        sub = rec.get(backend, {}) if isinstance(rec.get(backend), dict) else {}
        engine = resolve(cfg, sub.get("weights") or sub.get("onnx") or sub.get("engine") or _DEFAULT_ENGINES[backend])
        _embedders[backend] = _BACKENDS[backend](engine)
        return _embedders[backend]
