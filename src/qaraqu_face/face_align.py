"""ArcFace norm_crop (как InsightFace face_align) — без skimage."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

# эталон InsightFace / buffalo_l
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def landmarks_from_bbox(xyxy: Sequence[float]) -> np.ndarray:
    """Грубые 5 точек по bbox, если нет YuNet/SCRFD landmarks."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return np.array(
        [
            [x1 + 0.30 * w, y1 + 0.38 * h],
            [x1 + 0.70 * w, y1 + 0.38 * h],
            [x1 + 0.50 * w, y1 + 0.55 * h],
            [x1 + 0.35 * w, y1 + 0.78 * h],
            [x1 + 0.65 * w, y1 + 0.78 * h],
        ],
        dtype=np.float32,
    )


def estimate_norm(lmk: np.ndarray, image_size: int = 112) -> np.ndarray:
    lmk = np.asarray(lmk, dtype=np.float32).reshape(5, 2)
    ratio = float(image_size) / 112.0
    dst = ARCFACE_DST * ratio
    M, _ = cv2.estimateAffinePartial2D(lmk, dst, method=cv2.LMEDS)
    if M is None:
        # fallback similarity через lstsq (как FINDFACE face_attrs)
        src = lmk
        Mdst = dst.shape[0]
        x = dst[:, 0:1]
        y = dst[:, 1:2]
        X = np.vstack(
            (
                np.hstack((x, y, np.ones((Mdst, 1)), np.zeros((Mdst, 1)))),
                np.hstack((y, -x, np.zeros((Mdst, 1)), np.ones((Mdst, 1)))),
            )
        )
        U = np.vstack((src[:, 0:1], src[:, 1:2]))
        r, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        r = np.squeeze(r)
        sc, ss, tx, ty = float(r[0]), float(r[1]), float(r[2]), float(r[3])
        Tinv = np.array([[sc, -ss, 0.0], [ss, sc, 0.0], [tx, ty, 1.0]], dtype=np.float64)
        T = np.linalg.inv(Tinv)
        M = T[0:2, :].astype(np.float32)
    return M.astype(np.float32)


def norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Similarity warp лица в 112×112 под ArcFace."""
    M = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)


def align_face(
    bgr: np.ndarray,
    *,
    xyxy: Sequence[float] | None = None,
    landmarks: np.ndarray | Sequence[float] | None = None,
    image_size: int = 112,
) -> np.ndarray:
    """Выровнять лицо: landmarks предпочтительны, иначе из bbox / всего кадра."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    h, w = bgr.shape[:2]
    if landmarks is not None:
        lmk = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        if lmk.shape[0] == 5 and float(np.abs(lmk).sum()) > 1.0:
            return norm_crop(bgr, lmk, image_size)
    box = xyxy
    if box is None:
        box = (0, 0, w, h)
    return norm_crop(bgr, landmarks_from_bbox(box), image_size)


def landmarks_ok(landmarks, xyxy, min_eye: float = 0.18) -> bool:
    """Настоящие 5 точек лица, а не пустой бокс. Без них ArcFace путает подушку и затылок с карточкой."""
    if landmarks is None or xyxy is None:
        return False
    lm = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if lm.shape[0] != 5 or not np.isfinite(lm).all() or float(np.abs(lm).sum()) <= 1.0:
        return False
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = 0.4 * max(w, h)
    if np.any(lm[:, 0] < x1 - pad) or np.any(lm[:, 0] > x2 + pad):
        return False
    if np.any(lm[:, 1] < y1 - pad) or np.any(lm[:, 1] > y2 + pad):
        return False
    le, re, nose, ml, mr = lm
    if re[0] - le[0] < min_eye * w:
        return False
    if abs(float(le[1] - re[1])) > 0.28 * h:
        return False
    eye_y = float(le[1] + re[1]) / 2.0
    mouth_y = float(ml[1] + mr[1]) / 2.0
    if not (eye_y + 0.04 * h < float(nose[1]) < mouth_y):
        return False
    if mouth_y - eye_y < 0.12 * h:
        return False
    return True
