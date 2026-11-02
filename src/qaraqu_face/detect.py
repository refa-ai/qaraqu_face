"""Детекция лиц: YOLO PyTorch (.pt) + YuNet ONNX (landmarks)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qaraqu_face.config import load_config, resolve

_model = None
_model_path: Path | None = None
_yunet = None
_yunet_path: Path | None = None
_yunet_lock = __import__("threading").Lock()
# одна модель YOLO на все камеры: параллельный predict из разных потоков даёт сбои и пустые кадры.
# Инференс идёт в одном постоянном потоке: predict из новых коротких потоков течёт (~1 МБ на вызов).
_yolo_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
# YuNet на затылках и мешках даёт 0.55–0.6 с правдоподобными точками; у настоящих лиц на CCTV ≥0.8.
# Без поддержки YOLO верим только уверенному YuNet.
YUNET_ONLY_SCORE = 0.7
_YUNET_AGREE_SCORE = 0.5
YOLO_SUPPORT_SCORE = 0.4
_YUNET_REFINE_SCORE = 0.5
_YUNET_REFINE_IOU = 0.3
_YUNET_REFINE_TARGET_PX = 160
_YUNET_FULL_MAX_SIDE = 960
# YOLO ≥ этого + боковое лицо (глаза ближе 0.18 ширины бокса) — ещё пригодно для 1:N / Unknown
SIDE_FACE_YOLO_SCORE = 0.5
SIDE_EYE_RATIO = 0.14


_model_lock = __import__("threading").Lock()


def get_detector(cfg: dict | None = None, weights: str | Path | None = None):
    global _model, _model_path
    cfg = cfg or load_config()
    det = cfg.get("detect", {})
    path = Path(weights) if weights else resolve(cfg, det.get("weights", "data/models/yolov11n-face.pt"))
    if not path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {path}")
    with _model_lock:
        if _model is None or _model_path != path:
            from ultralytics import YOLO

            _model = YOLO(str(path))
            _model_path = path
        return _model, path


def _get_yunet(cfg: dict | None = None):
    """YuNet ONNX через OpenCV: 5 landmarks и доп. детект мелких лиц."""
    global _yunet, _yunet_path
    cfg = cfg or load_config()
    det = cfg.get("detect", {}) if isinstance(cfg.get("detect"), dict) else {}
    path = resolve(cfg, det.get("yunet", "data/models/face_detection_yunet_2023mar.onnx"))
    if not path.is_file():
        return None
    with _yunet_lock:
        if _yunet is None or _yunet_path != path:
            from qaraqu_face.yunet_gpu import create_yunet_gpu

            _yunet = create_yunet_gpu(str(path), 0.6, 0.3, 5000)
            if _yunet is None:
                _yunet = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.6, 0.3, 5000)
            _yunet_path = path
        return _yunet


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_yunet_bgr(
    bgr: np.ndarray,
    *,
    cfg: dict | None = None,
    conf: float = 0.45,
) -> list[dict[str, Any]]:
    """Полный кадр YuNet — ловит дальние/мелкие лица, которые YOLO пропускает."""
    det = _get_yunet(cfg)
    if det is None or bgr is None or getattr(bgr, "size", 0) == 0:
        return []
    h, w = bgr.shape[:2]
    out: list[dict[str, Any]] = []
    # YuNet на CPU под общим локом: 1080p занимает ~140 мс и душит фоновые камеры, на 960 px — ~35 мс
    s = min(1.0, _YUNET_FULL_MAX_SIDE / float(max(h, w)))
    src = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else bgr
    try:
        if hasattr(det, "run"):
            _, faces = det.run(src, float(conf))
        else:
            with _yunet_lock:
                det.setInputSize((src.shape[1], src.shape[0]))
                det.setScoreThreshold(float(conf))
                _, faces = det.detect(src)
    except Exception:
        return []
    if faces is None:
        return []
    for f in faces:
        f = np.asarray(f, dtype=np.float32).copy()
        if s < 1.0:
            f[:14] /= s
        x, y, bw, bh = (float(v) for v in f[:4])
        score = float(f[-1])
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
        if x2 <= x1 or y2 <= y1:
            continue
        landmarks = None
        try:
            lm = np.array(f[4:14], dtype=np.float32).reshape(5, 2)
            landmarks = lm.tolist()
        except Exception:
            pass
        out.append(
            {
                "xyxy": (x1, y1, x2, y2),
                "score": score,
                "kind": "face",
                "landmarks": landmarks,
                "backend": "yunet",
            }
        )
    return out


def refine_landmarks(
    bgr: np.ndarray,
    xyxy: tuple[int, int, int, int] | list[int],
    cfg: dict | None = None,
    min_score: float = _YUNET_REFINE_SCORE,
    allow_soft: bool = True,
    score_sink: list | None = None,
) -> list[list[float]] | None:
    """5 точек YuNet для YOLO-бокса; координаты в системе полного кадра.

    Берётся только лицо YuNet, которое действительно совпадает с боксом (IoU) —
    «ближайшее по центру» цеплялось за соседние предметы и чужие лица.
    """
    det = _get_yunet(cfg)
    if det is None or bgr is None or getattr(bgr, "size", 0) == 0:
        return None
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad = int(0.35 * max(bw, bh))
    xa, ya = max(0, x1 - pad), max(0, y1 - pad)
    xb, yb = min(w, x2 + pad), min(h, y2 + pad)
    crop = bgr[ya:yb, xa:xb]
    if crop.size == 0:
        return None
    def once(score: float, target: float, iou_min: float) -> tuple[list[list[float]], float] | None:
        scale = max(1.0, target / float(max(bw, bh)))
        view = crop
        if scale > 1.0:
            view = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ch, cw = view.shape[:2]
        try:
            if hasattr(det, "run"):
                _, found = det.run(view, float(score))
            else:
                with _yunet_lock:
                    det.setInputSize((cw, ch))
                    det.setScoreThreshold(float(score))
                    _, found = det.detect(view)
        except Exception:
            return None
        if found is None or len(found) == 0:
            return None
        box_in_crop = ((x1 - xa) * scale, (y1 - ya) * scale, (x2 - xa) * scale, (y2 - ya) * scale)
        best, best_iou = None, iou_min
        for f in found:
            fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            v = _iou_xyxy(box_in_crop, (fx, fy, fx + fw, fy + fh))
            if v >= best_iou:
                best_iou, best = v, f
        if best is None:
            return None
        try:
            lm = np.array(best[4:14], dtype=np.float32).reshape(5, 2) / scale
            lm[:, 0] += xa
            lm[:, 1] += ya
            conf = float(best[14]) if len(best) > 14 else float(score)
            return lm.tolist(), conf
        except Exception:
            return None

    # 30–50 px на полном кадре YuNet не видит. Кроп увеличиваем; если движение смазало лицо — второй, более крупный проход.
    # Мягкий проход 0.3 оставляем для рамки. Для номера «Неизвестный» его не вызывают:
    # на телефоне и бухте кабеля YuNet дорисовывает глаза.
    got = once(float(min_score), _YUNET_REFINE_TARGET_PX, _YUNET_REFINE_IOU)
    if got is None and allow_soft and float(min_score) <= 0.5:
        got = once(0.3, 256.0, 0.18)
    if got is None:
        return None
    if score_sink is not None:
        score_sink.append(got[1])
    return got[0]


def detect_bgr(
    bgr: np.ndarray,
    *,
    cfg: dict | None = None,
    conf: float | None = None,
    imgsz: int | None = None,
    weights: str | Path | None = None,
    with_landmarks: bool = True,
    merge_yunet: bool = False,
) -> list[dict[str, Any]]:
    cfg = cfg or load_config()
    det = cfg.get("detect", {})
    model, _path = get_detector(cfg, weights)
    if conf is None:
        from qaraqu_face.db import get_options
        conf = get_options().get("det_conf", 0.25)
    conf_v = float(conf)
    imgsz_v = int(imgsz if imgsz is not None else det.get("imgsz", 1280))
    device = det.get("device", "0")
    max_det = int(det.get("max_det", 300))

    def _yolo_one() -> Any:
        from qaraqu_face.torch_runtime import gpu_lock

        with gpu_lock():
            return model.predict(
                bgr,
                conf=conf_v,
                imgsz=imgsz_v,
                device=device,
                max_det=max_det,
                verbose=False,
            )[0]

    r = _yolo_pool.submit(_yolo_one).result()
    out: list[dict[str, Any]] = []
    if r.boxes is not None:
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            score = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            item: dict[str, Any] = {
                "xyxy": (x1, y1, x2, y2),
                "score": score,
                "kind": "face",
                "landmarks": None,
                "backend": "yolo",
            }
            if with_landmarks:
                item["landmarks"] = refine_landmarks(bgr, item["xyxy"], cfg=cfg)
            out.append(item)

    if merge_yunet:
        for yf in detect_yunet_bgr(bgr, cfg=cfg, conf=_YUNET_AGREE_SCORE):
            agreed = [o for o in out if _iou_xyxy(yf["xyxy"], o["xyxy"]) >= 0.35]
            if agreed:
                for o in agreed:
                    if not o.get("landmarks") and o["score"] >= YOLO_SUPPORT_SCORE:
                        o["landmarks"] = yf.get("landmarks")
                        break
                continue
            if yf["score"] >= YUNET_ONLY_SCORE:
                out.append(yf)

    out.sort(key=lambda f: f["score"], reverse=True)
    return out


def _nms_merge(faces: list[dict[str, Any]], iou_thr: float = 0.45) -> list[dict[str, Any]]:
    """Один бокс на лицо: тайлы перекрываются, одно лицо приходит дважды."""
    faces = sorted(faces, key=lambda f: float(f.get("score") or 0), reverse=True)
    kept: list[dict[str, Any]] = []
    for f in faces:
        dup = False
        for k in kept:
            if _iou_xyxy(f["xyxy"], k["xyxy"]) >= iou_thr:
                dup = True
                if not k.get("landmarks") and f.get("landmarks"):
                    k["landmarks"] = f["landmarks"]
                break
        if not dup:
            kept.append(f)
    return kept


def detect_tiled(
    bgr: np.ndarray,
    *,
    cfg: dict | None = None,
    conf: float,
    imgsz: int = 1280,
    grid: tuple[int, int] = (2, 2),
    overlap: float = 0.12,
    merge_yunet: bool = False,
    tile_max_side: int = 72,
    tile_conf: float | None = None,
) -> list[dict[str, Any]]:
    """Полный кадр плюс тайлы. На 1080p лицо 15 px в тайле становится ~20 px, и YOLO его видит.

    Тайлы добавляют только мелкие лица (до tile_max_side px) и только с подтверждением
    YuNet: увеличенный кусок захламлённой комнаты даёт YOLO 0.4 на спине и коробках.
    """
    from qaraqu_face.face_align import landmarks_ok

    h, w = bgr.shape[:2]
    faces = detect_bgr(bgr, cfg=cfg, conf=conf, imgsz=imgsz, with_landmarks=False, merge_yunet=merge_yunet)
    rows, cols = grid
    if rows * cols <= 1 or max(h, w) <= imgsz:
        return faces
    tw, th = w / cols, h / rows
    ox, oy = int(tw * overlap), int(th * overlap)
    # тайлы ищут только новые лица, а новый трек всё равно нужен не ниже ползунка
    t_conf = float(tile_conf if tile_conf is not None else conf)
    extra: list[dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tw) - ox)
            y1 = max(0, int(r * th) - oy)
            x2 = min(w, int((c + 1) * tw) + ox)
            y2 = min(h, int((r + 1) * th) + oy)
            tile = bgr[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            for f in detect_bgr(tile, cfg=cfg, conf=t_conf, imgsz=imgsz, with_landmarks=False, merge_yunet=False):
                fx1, fy1, fx2, fy2 = f["xyxy"]
                # лицо, срезанное краем тайла, даст кривой бокс; его найдёт соседний тайл или полный кадр
                tile_w, tile_h = x2 - x1, y2 - y1
                edge = 2
                if fx1 <= edge or fy1 <= edge or fx2 >= tile_w - edge or fy2 >= tile_h - edge:
                    continue
                if max(fx2 - fx1, fy2 - fy1) > tile_max_side:
                    continue
                item = dict(f)
                item["xyxy"] = (fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1)
                item["backend"] = "yolo"
                extra.append(item)
    if not extra:
        return faces
    extra = _nms_merge(extra)
    confirmed: list[dict[str, Any]] = []
    for item in extra:
        twin = next((o for o in faces if _iou_xyxy(item["xyxy"], o["xyxy"]) >= 0.45), None)
        if twin is not None and float(twin.get("score") or 0) >= t_conf:
            continue  # полный кадр уже уверен в этом лице
        # затылок и коробки в увеличенном тайле дают YOLO 0.4–0.5; без глаз и носа это не лицо
        lm = refine_landmarks(bgr, item["xyxy"], cfg=cfg, min_score=0.35)
        if lm is None or not landmarks_ok(lm, item["xyxy"], min_eye=0.11):
            continue
        item["landmarks"] = lm
        confirmed.append(item)
    if not confirmed:
        return faces
    merged = _nms_merge(confirmed + faces)
    merged.sort(key=lambda f: f["score"], reverse=True)
    return merged


def crop_face(bgr: np.ndarray, xyxy: tuple[int, int, int, int], margin: float = 0.18) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    xa = max(0, x1 - mx)
    ya = max(0, y1 - my)
    xb = min(w, x2 + mx)
    yb = min(h, y2 + my)
    return bgr[ya:yb, xa:xb].copy()


def face_crop_for_embed(bgr: np.ndarray, faces: list[dict[str, Any]] | None = None) -> np.ndarray:
    """Кроп для эмбеддинга. Если фото уже tight face-crop — не режем повторно (ломает 1:N)."""
    h, w = bgr.shape[:2]
    img_area = float(max(1, h * w))
    if faces is None:
        faces = detect_bgr(bgr)
    if not faces:
        return bgr
    faces = sorted(
        faces,
        key=lambda f: (f["xyxy"][2] - f["xyxy"][0]) * (f["xyxy"][3] - f["xyxy"][1]),
        reverse=True,
    )
    faces = [
        f
        for f in faces
        if (f["xyxy"][2] - f["xyxy"][0]) * (f["xyxy"][3] - f["xyxy"][1]) >= 0.02 * img_area
    ]
    if not faces:
        return bgr
    best = faces[0]
    fa = (best["xyxy"][2] - best["xyxy"][0]) * (best["xyxy"][3] - best["xyxy"][1])
    if fa / img_area >= 0.30 or max(h, w) <= 360:
        return bgr
    return crop_face(bgr, best["xyxy"])


def filter_faces_for_recog(
    faces: list[dict[str, Any]],
    frame_shape: tuple[int, ...],
    *,
    min_side: int = 18,
    min_area_frac: float = 0.0002,
) -> list[dict[str, Any]]:
    """Убрать крошечные FP для 1:N. На 1080p порог ~20 px: видимое лицо вдалеке ещё сравнивается с картотекой."""
    h, w = frame_shape[:2]
    img_area = float(max(1, h * w))
    out: list[dict[str, Any]] = []
    for f in faces:
        x1, y1, x2, y2 = f["xyxy"]
        bw, bh = max(0, x2 - x1), max(0, y2 - y1)
        if min(bw, bh) < min_side:
            continue
        if bw * bh < min_area_frac * img_area:
            continue
        out.append(f)
    return out


def best_valid_face(bgr: np.ndarray, cfg: dict | None = None) -> dict[str, Any] | None:
    """Самое крупное лицо с настоящими landmarks (глаза/нос/рот) — для картотеки.

    Тугие кропы без полей YOLO часто не видит, поэтому второй проход — с серой рамкой вокруг фото.
    Координаты в ответе всегда в системе исходного изображения.
    """
    from qaraqu_face.face_align import landmarks_ok

    if bgr is None or getattr(bgr, "size", 0) == 0:
        return None
    for border in (0.0, 0.5):
        img, off = bgr, 0
        if border:
            h, w = bgr.shape[:2]
            off = int(border * max(h, w))
            img = cv2.copyMakeBorder(bgr, off, off, off, off, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        faces = detect_bgr(img, cfg=cfg, conf=0.25, imgsz=1280, with_landmarks=True, merge_yunet=True)
        good = [f for f in faces if landmarks_ok(f.get("landmarks"), f["xyxy"])]
        if not good:
            continue
        best = max(good, key=lambda f: (f["xyxy"][2] - f["xyxy"][0]) * (f["xyxy"][3] - f["xyxy"][1]))
        if off:
            x1, y1, x2, y2 = best["xyxy"]
            best["xyxy"] = (x1 - off, y1 - off, x2 - off, y2 - off)
            best["landmarks"] = (np.asarray(best["landmarks"], dtype=np.float32) - off).tolist()
        return best
    return None
