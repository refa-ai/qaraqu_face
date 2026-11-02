"""Аугментации фото лица для картотеки (с пересчётом landmarks)."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def _brightness(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _contrast(image: np.ndarray, factor: float) -> np.ndarray:
    mean = float(np.mean(image))
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def _rotate_with_landmarks(image: np.ndarray, lm: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pts = cv2.transform(lm.reshape(1, -1, 2).astype(np.float32), m).reshape(-1, 2)
    return rot, pts


def _low_res(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else image


def augment_with_landmarks(image: np.ndarray, landmarks, n: int = 8) -> List[tuple[np.ndarray, np.ndarray]]:
    """Варианты фото с пересчитанными landmarks: поворот, свет, контраст, блюр и «камерное» качество.

    Landmarks сохраняют выравнивание ArcFace таким же, как в живом потоке; низкое разрешение и JPEG
    имитируют субпоток CCTV (704×576), на котором лицо ~40 px.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return []
    lm = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    side = max(8.0, 2.2 * float(np.linalg.norm(lm[0] - lm[1])))
    s_far = float(np.clip(48.0 / side, 0.05, 1.0))
    s_mid = float(np.clip(80.0 / side, 0.05, 1.0))
    ops = [
        lambda: _rotate_with_landmarks(image, lm, -8),
        lambda: _rotate_with_landmarks(image, lm, 8),
        lambda: (_brightness(image, 0.78), lm),
        lambda: (_brightness(image, 1.22), lm),
        lambda: (_contrast(image, 1.18), lm),
        lambda: (cv2.GaussianBlur(image, (3, 3), 0), lm),
        lambda: (_low_res(image, s_far), lm),
        lambda: (_jpeg(_low_res(image, s_mid), 35), lm),
    ]
    out: List[tuple[np.ndarray, np.ndarray]] = []
    for fn in ops:
        if len(out) >= n:
            break
        try:
            out.append(fn())
        except Exception:
            continue
    return out
