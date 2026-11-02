"""Live runtime: RTSP → YOLO → трекинг → ArcFace 1:N → события и маршруты (PRINCIPLE.md)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

cv2.setNumThreads(0)

from qaraqu_face import cameras_store
from qaraqu_face import db as database
from qaraqu_face.config import load_config, resolve
from qaraqu_face.detect import (
    SIDE_EYE_RATIO,
    SIDE_FACE_YOLO_SCORE,
    YUNET_ONLY_SCORE,
    detect_bgr,
    detect_tiled,
    filter_faces_for_recog,
    get_detector,
    refine_landmarks,
)
from qaraqu_face.face_align import landmarks_ok
from qaraqu_face.gallery import add_observed_vec, list_people, load_gallery, match
from qaraqu_face.sub_preview import SubPreview
from qaraqu_face.rtsp_camera import RtspCamera
from qaraqu_face.recognize import get_embedder, l2
from qaraqu_face.tracking import (
    CameraTracker,
    Track,
    UnknownRegistry,
    _STILL_FACE_SCORE,
    display_box,
    latin,
    observe,
    observe_invalid,
)

log = logging.getLogger("qaraqu_face.runtime")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

_JPEG_QUALITY = 70
_DISPLAY_MAX_SIDE = 640
# 640 сжимал дальние лица → YOLO их не видел; 960 лучше ловит всех в кадре
_LIVE_IMGSZ = 1280
_LIVE_DET_CONF = 0.18  # рамки: ниже чем match; не путать с порогом 1:N
_COLOR_PENDING = (60, 160, 255)  # оранжевый BGR — лицо есть, картотека ещё не подтвердила
# рамку рисуем с порога детекции: отдельный отсев 0.35 прятал лица камер (0.28–0.40)
_COLOR_UNKNOWN = (60, 160, 255)  # оранжевый BGR — лицо есть, в базе нет
_COLOR_KNOWN = (40, 180, 80)  # зелёный — совпадение с картотекой
_SIGHTING_EVERY_S = 5.0
_INFER_SEM = threading.Semaphore(1)
# замер для диагностики: сколько ждём GPU и сколько он занят (мс, счётчики за окно)
PERF = {"sem_wait_ms": 0.0, "sem_wait_n": 0, "sem_miss": 0, "det_ms": 0.0, "det_n": 0, "rec_ms": 0.0, "rec_n": 0}
_PERF_LOCK = threading.Lock()


def _perf_add(key: str, ms: float) -> None:
    with _PERF_LOCK:
        PERF[key + "_ms"] += ms
        PERF[key + "_n"] += 1
# фоновые детекции — общий пул; GPU всё равно по одной за раз (_INFER_SEM)
_DETECT_POOL = ThreadPoolExecutor(max_workers=36, thread_name_prefix="detect")
# фильтр качества перед 1:N: меньше этого лица и профиль только в трек, не в сравнение.
# Замер на 100 камерных кропах (AdaFace, картотека по фото): <40 px — genuine 0.06, 40–60 — 0.14,
# 60–90 — 0.29, ≥90 — 0.52 при чужих ~0.14. Рабочий кадр — main 1080p; ниже 64 px лицо только
# трекается (оранжевая рамка), «Неизвестный #N» из него не заводится. Если кадр — sub,
# лица меньше _MAIN_SMALL_SIDE доуточняются по main независимо от score.
_MATCH_MIN_SIDE = 64
_MATCH_MIN_EYE = 0.18
_TRACK_VECS = 5
_MAIN_SMALL_SIDE = 90
# детекция на sub: кадр ≤ этого — считаем «малым» (вход YOLO 640, без тайлов);
# лицо на sub от _SUB_MIN_SIDE px (≈64 px на main 1080p) уходит в 1:N только через кроп main
_SMALL_FRAME_SIDE = 800
_SUB_IMGSZ = 640
_SUB_MIN_SIDE = 24
# самообучение карточки: камерный вектор дописывается в наблюдённые только при уверенном
# совпадении (чужие на 106 кропах не поднимались выше 0.30) и явном отрыве от второго
_LEARN_SCORE = 0.55
_LEARN_GAP = 0.20
# телефон на столе YuNet принимает за лицо с уверенностью 0.85, но кроп в разы резче
# любого лица с этих камер (лица до ~2000, экран и клавиатура от ~3100)
_OBJECT_SHARPNESS = 3000.0
# доля «телесных» пикселей в боксе. Лица на карточках 0.4–0.9, блик монитора/лампы ~0.06–0.22
_SKIN_MIN = 0.24
# известный трек без свежего YuNet и почти неподвижный — блик, не человек
_STALE_STILL_HITS = 6
# столько секунд рамка считается «сидящей»: блик/лампа висят минутами, человек так не стоит
_STATIC_HIDE_S = 25.0
# движение по серой миниатюре слота. Выше порога — человек вошёл, детекция и main сразу.
# Пустая неподвижная сцена почти не занимает видеокарту: редкий контрольный кадр, не поток.
_MOTION_ON = 2.0
# порог по самому активному блоку сетки 4×4: ловит небольшого человека, который среднее не поднимает
_BLK_ON = 4.0
_DETECT_STILL_S = 1.2
_DETECT_TRACK_S = 0.28
# Опорный множитель порога движения в rtsp_camera.MOTION_K — «середина» ползунка
# «Чувствительность к движению», чтобы 3.0 означало «как раньше».
MOTION_K_REF = 3.0
# эмбеддинг не на каждом боксе, а на лучшем кропе интервала (как realtime_post_interval).
# более крупное лицо внутри интервала всё равно считается сразу.
_EMBED_EVERY_S = 0.40
_EMBED_SHARPER = 1.15


def face_geometry_ok(f: dict[str, Any], lm) -> bool:
    """Точки лица годны для ArcFace. Бокс YOLO уже отсёк мусор, поэтому глаза могут быть чуть ближе.

    YuNet без YOLO (затылки, мешки) по-прежнему требует широкий фронт.
    """
    if landmarks_ok(lm, f["xyxy"]):
        return True
    if f.get("backend") == "yolo" and float(f.get("score") or 0) >= 0.28:
        return landmarks_ok(lm, f["xyxy"], min_eye=0.11)
    if f.get("backend") == "yolo" and float(f.get("score") or 0) >= SIDE_FACE_YOLO_SCORE:
        return landmarks_ok(lm, f["xyxy"], min_eye=SIDE_EYE_RATIO)
    return False


def unknown_display(no: int) -> str:
    return f"Неизвестный #{no}"


def _box_sharpness(frame: np.ndarray, xyxy) -> float:
    """Резкость бокса. Экран и клавиатура дают тысячи, лицо на этих камерах — до ~2000."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    side = max(gray.shape)
    if side > 96:
        scale = 96.0 / side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _box_skin_frac(frame: np.ndarray, xyxy) -> float:
    """Доля пикселей оттенка кожи. Блик экрана и лампа почти без saturation в телесном диапазоне."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    a = cv2.inRange(hsv, (0, 40, 40), (30, 200, 255))
    b = cv2.inRange(hsv, (150, 40, 40), (179, 200, 255))
    return float(np.mean((a | b) > 0))


def _is_glare_box(frame: np.ndarray, xyxy, score: float = 1.0) -> bool:
    """Монитор/лампа без кожи. Уверенный YOLO (≥0.40) не режем — на ИК кожа слабая."""
    if float(score) >= 0.40:
        return False
    return _box_skin_frac(frame, xyxy) < _SKIN_MIN


def _zones_for(options: dict, index: int) -> list:
    """Зоны-исключения камеры (нормализованные прямоугольники) из настроек."""
    data = options.get("ignore_zones") or {}
    if not isinstance(data, dict):
        return []
    return data.get(str(index)) or data.get(index) or []


def _center_in_zones(xyxy, zones: list, size) -> bool:
    h, w = int(size[0]), int(size[1])
    if not w or not h:
        return False
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) * 0.5 / w, (y1 + y2) * 0.5 / h
    for z in zones:
        try:
            zx1, zy1, zx2, zy2 = (float(v) for v in z)
        except (TypeError, ValueError):
            continue
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            return True
    return False


def _picture_ok(frame: np.ndarray | None) -> bool:
    """Битый кадр декодера почти чёрный. Тёмная комната всё равно светлее этого порога."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    return float(frame.mean()) >= 8.0


class CameraWorker:
    def __init__(self, index: int, cfg: dict, runtime: "Runtime"):
        self.index = index
        self.cfg = cfg
        self.runtime = runtime
        self.tracker = CameraTracker(index)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fps = 0.0
        self.capture_fps = 0.0
        self.last_jpeg: bytes | None = None
        self.last_jpeg_sub: bytes | None = None
        self.last_meta: dict[str, Any] = {}
        self.error: str | None = None
        self.alive = False
        self._jpeg_lock = threading.Lock()
        self._span: dict[int, list[float]] = {}
        self._overlay_lock = threading.Lock()
        self._overlay: list[dict[str, Any]] = []
        self._recog_lock = threading.Lock()
        self._frame_i = 0
        self._detect_due = False
        self._seen_seq = -1
        self._jpeg_seq = -1
        self._last_det_ok = 0.0
        self._infer_retry_at = 0.0
        self._latest_frame = None
        self._recog_at = 0.0
        self._batch_vecs: dict[tuple[int, str], np.ndarray] = {}
        self._yunet_at = 0.0
        self._det_gen = 0
        self._faces_gen = 0
        self._disp_n = 0
        self._disp_t0 = time.time()
        self._wall_until = 0.0
        self._sub_preview: SubPreview | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.index}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _prefer_gpu_decode(self) -> bool:
        return str(self.runtime.options.get("decode_device", "cuda")).lower() in ("cuda", "gpu", "nvdec", "1", "true")

    def _open_capture(self, cam: dict):
        # По умолчанию main: у части Dahua sub — 12–15 к/с и лицо 20–40 px, 1:N его не берёт.
        # В слот кладётся кадр ≤960, полный 1080p — реже, для детекции. Иначе 35×1080p×25
        # не помещаются в отгрузку с GPU.
        # Детекция и распознавание — только main. Sub используется исключительно для превью.
        main = (cam.get("rtsp_url_main") or cam.get("url") or "").strip()
        url = main
        self._stream_name = "main"
        if not url:
            raise RuntimeError("нет источника камеры")

        # Веб-/USB-камера: "0", "/dev/video0", "webcam", "webcam:0"
        low = url.lower()
        is_webcam = False
        src: Any = url
        if low in ("webcam", "usb", "local", "v4l2"):
            src, is_webcam = 0, True
        elif low.startswith("webcam:"):
            try:
                src = int(low.split(":", 1)[1] or "0")
            except ValueError:
                src = 0
            is_webcam = True
        elif url.isdigit():
            src, is_webcam = int(url), True
        elif url.startswith("/dev/video"):
            src, is_webcam = url, True

        self._is_webcam = is_webcam
        if is_webcam:
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # noqa: BLE001
                pass
            if not cap.isOpened():
                raise RuntimeError(f"веб-камера не открылась: {url}")
            return cap

        cap = self._open_rtsp(url, name=f"cam{self.index}-{self._stream_name}")
        # GpuRtspCapture открывается в фоне (NVDEC): isOpened станет true после первого кадра.
        # Ждём коротко; если тишина — пробуем запасной sub, иначе отдаём захват в цикл.
        deadline = time.time() + 8.0
        while time.time() < deadline and not cap.isOpened() and not self._stop.is_set():
            time.sleep(0.1)
        # даже без кадра возвращаем cap — _loop переоткроет при необходимости
        return cap

    def _open_rtsp(self, url: str, *, name: str = ""):
        return RtspCamera(
            url,
            name=name or f"cam{self.index}",
            device=0,
            use_cuda=self._prefer_gpu_decode(),
        )

    def _publish_sub_jpeg(self, jpg: bytes) -> None:
        with self._jpeg_lock:
            self.last_jpeg_sub = jpg

    def _read_latest(self, cap: Any) -> tuple[bool, Any]:
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, None
        # Один запасной grab, если в буфере уже лежит следующий кадр.
        # Три подряд держали цикл ~200 мс, и детекция на 8 камерах успевала 1–2 раза в секунду.
        if not getattr(self, "_is_webcam", False) and cap.grab():
            ok2, f2 = cap.retrieve()
            if ok2 and f2 is not None:
                frame = f2
        return True, frame

    def _scale_for_display(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        side = max(h, w)
        if side <= _DISPLAY_MAX_SIDE:
            return frame, 1.0
        scale = _DISPLAY_MAX_SIDE / float(side)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return small, scale

    def _draw_overlay(self, frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
        h, w = frame.shape[:2]
        now = time.time()
        with self._overlay_lock:
            boxes = list(self._overlay)
        for item in boxes:
            x1, y1, x2, y2 = (float(v) for v in item["xyxy"])
            age = now - float(item.get("ts") or now)
            if 0.0 < age < 0.25:
                bw, bh = x2 - x1, y2 - y1
                cap = 0.35 * max(bw, bh, 1.0)
                dx = max(-cap, min(cap, float(item.get("vx") or 0) * age))
                dy = max(-cap, min(cap, float(item.get("vy") or 0) * age))
                x1, x2 = x1 + dx, x2 + dx
                y1, y2 = y1 + dy, y2 + dy
            x1, y1, x2, y2 = (int(v * scale) for v in (x1, y1, x2, y2))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            color = item["color"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, item["label"], (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return frame

    def _publish_jpeg(self, frame: np.ndarray) -> None:
        small, scale = self._scale_for_display(frame)
        if small is frame:
            small = frame.copy()
        # Детекция идёт по полному кадру, JPEG — по слоту ≤960. Без этого рамка
        # рисуется в координатах 1080p поверх картинки 960 px и уезжает за край.
        det = self._latest_frame
        if det is not None and frame.shape[1] > 0 and det.shape[1] != frame.shape[1]:
            scale *= frame.shape[1] / float(det.shape[1])
        annotated = self._draw_overlay(small, scale=scale)
        okj, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        if okj:
            with self._jpeg_lock:
                self.last_jpeg = buf.tobytes()

    def _detect(self, frame: np.ndarray, opts: dict[str, Any], conf: float) -> list[dict[str, Any]]:
        # Ниже ползунка тоже ищем: эти рамки только продолжают уже идущий трек (ByteTrack внутри OC-SORT).
        # Полный кадр YuNet дорогой: на каждом проходе его гонять нельзя, иначе 8 камер ждут одну видеокарту.
        # Раз в секунду он добирает мелкие лица, которые YOLO на дальнем плане пропускает.
        floor = min(0.10, max(0.05, float(conf) * 0.45))
        now = time.time()
        # Полный YuNet на 1080p ~35 мс. На всех камерах каждый проход он занимает видеокарту
        # целиком, и YOLO не успевает. На открытой камере чаще, на остальных — контрольный проход.
        preview = self.runtime.preview_index == self.index
        merge = now - self._yunet_at >= (0.8 if preview else 4.0)
        if merge:
            self._yunet_at = now
        # sub-кадр 704 px: вход 640 почти нативный, в 4 раза дешевле 1280; тайлы на нём бессмысленны
        small = max(frame.shape[:2]) <= _SMALL_FRAME_SIDE
        imgsz = _SUB_IMGSZ if small else self.runtime.imgsz_for(self.index)
        # Тайлы 2×2 — только открытая камера: 4 лишних YOLO на 35 потоков не оставляют времени на 1:N.
        if merge and not small and preview:
            faces = detect_tiled(
                frame, cfg=self.cfg, conf=floor, tile_conf=float(conf), imgsz=imgsz, merge_yunet=merge
            )
        else:
            faces = detect_bgr(
                frame, cfg=self.cfg, conf=floor, imgsz=imgsz, with_landmarks=False, merge_yunet=merge
            )
        faces = [
            f for f in faces if not _is_glare_box(frame, f["xyxy"], float(f.get("score") or 0))
        ]
        return faces[: max(int(opts.get("max_faces_per_frame", 10)) * 3, 10)]

    def _track(self, faces: list[dict[str, Any]], high_thresh: float) -> list[dict[str, Any]]:
        linked = self.tracker.update(faces, high_thresh)
        for f in linked:
            tr: Track = f["track"]
            tr.det_score = float(f.get("score") or tr.det_score or 0)
            # validated_ts только после строгого YuNet в _recognize — иначе блик монитора
            # считается «подтверждённым лицом» и висит часами под чужим ФИО
        return linked

    def _hold_faces(self, faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Короткая память рамки: на движении YOLO пропускает часть кадров, но хвост дольше трети секунды не держим."""
        now = time.time()
        seen = {id(f.get("track")) for f in faces}
        held: list[dict[str, Any]] = []
        for tr in self.tracker.active():
            if id(tr) in seen or now - tr.last_ts > 0.55:
                continue
            held.append(
                {
                    "xyxy": tuple(int(v) for v in tr.xyxy),
                    "score": float(tr.det_score or 0),
                    "track": tr,
                    "landmarks": None,
                }
            )
        return faces + held

    def _remember_crop(self, frame: np.ndarray, faces: list[dict[str, Any]]) -> None:
        """Снимок лица для панели, в том числе пока трек ещё не стал известным."""
        now = time.time()
        for f in faces:
            tr = f.get("track")
            if tr is None:
                continue
            if tr.snap and now - float(getattr(tr, "snap_ts", 0.0)) < 1.5:
                continue
            saved = self.runtime._save_snapshot(frame, tr, self.index, now, prefix="pass_")
            if saved:
                tr.snap = saved
                tr.snap_ts = now

    def _publish_state(self, faces: list[dict[str, Any]], cam_name: str, stream: str | None) -> None:
        rt = self.runtime
        overlay: list[dict[str, Any]] = []
        hits: list[dict[str, Any]] = []
        now = time.time()
        seen_ids = {id(f.get("track")) for f in faces}
        for f in faces:
            tr: Track = f["track"]
            kind = tr.kind
            # залипший блик с ФИО: нет свежего YuNet и бокс почти не двигается
            if kind == "known" and not tr.validated(now) and tr.still_hits >= _STALE_STILL_HITS:
                tr.reset_identity()
                kind = tr.kind
            # старые static-zone прятали ВСЕ pending в зоне стола — даже новых людей.
            # прячем только долго сидящий неподтверждённый бокс (затылок/блик), не свежий трек
            age = now - float(getattr(tr, "first_ts", 0) or now)
            stale_pending = (
                kind == "pending"
                and not tr.validated(now)
                and age >= 12.0
                and int(getattr(tr, "still_hits", 0) or 0) >= 8
            )
            if stale_pending and self.tracker.zones.is_static(tr.xyxy, now, True):
                continue
            # Пакет и бутылка дают уверенный бокс YOLO, но YuNet в них лица не находит.
            if kind == "pending" and int(getattr(tr, "reject_n", 0) or 0) >= 2 and not tr.validated(now):
                continue
            # Долго стоящий бокс со слабым максимумом YuNet — блик/лампа/предмет, не человек.
            # Разовый проход 0.5 не делает предмет лицом: настоящие лица дают ~0.9 и не прячутся.
            if (
                kind in ("pending", "unknown")
                and age >= _STATIC_HIDE_S
                and float(getattr(tr, "face_score", 0.0) or 0.0) < _STILL_FACE_SCORE
            ):
                continue
            # Зона-исключения камеры: фиксированные блики/лампы в известном месте.
            # Скрываем только нераспознанные рамки — известные люди показываются всегда.
            if kind != "known" and self._latest_frame is not None:
                zones = _zones_for(rt.options, self.index)
                if zones and _center_in_zones(f["xyxy"], zones, self._latest_frame.shape[:2]):
                    continue
            # Блик/лампа/предмет: рамка часами стоит на одном месте, а максимум YuNet слабый.
            # Человек так не стоит; настоящие лица дают ~0.9 и не прячутся.
            if kind != "known" and self._latest_frame is not None:
                fh, fw = int(self._latest_frame.shape[0]), int(self._latest_frame.shape[1])
                bx1, by1, bx2, by2 = f["xyxy"]
                bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
                sp = self._span.get(id(tr))
                if sp is None:
                    self._span[id(tr)] = [bcx, bcx, bcy, bcy]
                else:
                    sp[0] = min(sp[0], bcx)
                    sp[1] = max(sp[1], bcx)
                    sp[2] = min(sp[2], bcy)
                    sp[3] = max(sp[3], bcy)
                if len(self._span) > 400:
                    self._span = {k: v for k, v in self._span.items() if k in seen_ids}
                if sp is not None and fw and fh and age >= 45.0 and float(getattr(tr, "face_score", 0.0) or 0.0) < 0.85:
                    span_x = (sp[1] - sp[0]) / fw
                    span_y = (sp[3] - sp[2]) / fh
                    if span_x < 0.03 and span_y < 0.03:
                        continue
            if kind == "known":
                person = rt.people.get(tr.name or "", {})
                display = person.get("display") or tr.name
                label = f"{latin(display)} {tr.score or 0:.2f}"
                color = _COLOR_KNOWN
            elif kind == "unknown":
                display = unknown_display(tr.unknown_no or 0)
                label = f"Unknown #{tr.unknown_no}"
                color = _COLOR_UNKNOWN
                person = {}
            else:
                display = "лицо"
                label = f"{float(f.get('score') or 0):.2f}"
                color = _COLOR_PENDING
                person = {}
            near = None
            if kind != "known" and tr.near_name and tr.near_score and tr.near_score >= 0.30:
                near_person = rt.people.get(tr.near_name, {})
                near = {"display": near_person.get("display") or tr.near_name, "score": round(tr.near_score, 3)}
            overlay.append(
                {
                    "xyxy": list(display_box(tr, tr.last_ts)),
                    "vx": float(tr.vx),
                    "vy": float(tr.vy),
                    "ts": float(tr.last_ts),
                    "label": label,
                    "color": color,
                }
            )
            hits.append(
                {
                    "xyxy": [int(v) for v in f["xyxy"]],
                    "track_id": tr.tid,
                    "subject": tr.subject,
                    "kind": kind,
                    "name": tr.name,
                    "display": display,
                    "similarity": tr.score if kind == "known" else (None if kind == "unknown" else round(float(f.get("score") or 0), 3)),
                    "gallery_thumb": person.get("thumb"),
                    "snapshot": tr.snap or None,
                    "thumb": f"/api/snapshot/{tr.snap}" if tr.snap else None,
                    "near": near,
                    "camera_index": self.index,
                    "camera": cam_name,
                    "first_ts": tr.first_ts,
                    "last_ts": tr.last_ts,
                    "live": True,
                }
            )
        self._faces_gen += len(hits)
        with self._overlay_lock:
            self._overlay = overlay
        self.last_meta = {
            "camera": cam_name,
            "index": self.index,
            "faces": len(hits),
            "hits": hits,
            "stream": stream,
        }

    def _backend_thr(self, backend: str, opts: dict[str, Any]) -> float:
        rec = self.cfg.get("recognize") if isinstance(self.cfg.get("recognize"), dict) else {}
        sub = rec.get(backend) if isinstance(rec.get(backend), dict) else {}
        # Ползунок «Порог совпадения» — индивидуальный для камеры.
        # ArcFace в конфиге отставал (0.45 против 0.58 у AdaFace) и сам по себе
        # отсекал лица, которые AdaFace уже признала своими.
        base = self.runtime.match_thr_for(self.index)
        if backend == "adaface":
            return base
        return base

    def _blend_backend(
        self,
        tr: Track,
        backend: str,
        vec: np.ndarray,
        ident: dict[str, Any],
        gal: dict[str, list],
        thr: float,
        mar: float,
    ) -> dict[str, Any]:
        slot = tr.qvecs_arc if backend == "arcface" else tr.qvecs_ada
        slot.append(l2(vec))
        del slot[:-_TRACK_VECS]
        out = dict(ident)
        out["backend"] = backend
        if len(slot) >= 2:
            agg = match(np.mean(np.stack(slot), axis=0), gal, cfg=self.cfg, threshold=thr, margin=mar)
            if agg.get("matched") and float(agg.get("score") or 0) > float(out.get("score") or 0):
                out = dict(agg)
                out["backend"] = backend
        return out

    def _identify(
        self,
        frame: np.ndarray,
        f: dict[str, Any],
        lm,
        side: float,
        tr: Track,
        opts: dict[str, Any],
        mar: float,
        thr_add: float = 0.0,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Оба движка должны назвать одного и того же человека.

        ArcFace и AdaFace — независимые оценки: свои эмбеддинги и свои картотеки.
        При recognize.require_both (по умолчанию) имя принимается, только если оба
        совпали с ОДНИМ человеком; если первый не совпал — второй не считаем, чтобы не
        тратить GPU. Пустая картотека у одного движка означает, что он не голосует,
        и решает оставшийся. require_both=false возвращает прежнее «ИЛИ».

        thr_add — надбавка для трёхчетвертного лица: эмбеддинг считаем, но порог выше,
        чтобы профиль не прилипал к случайной карточке.
        """
        rec = self.cfg.get("recognize") if isinstance(self.cfg.get("recognize"), dict) else {}
        require_both = bool((rec or {}).get("require_both", True))
        if side < self.runtime.min_face_side_for(self.index):
            return None, {"matched": False}
        order = [b for b in ("arcface", "adaface") if (self.runtime.galleries.get(b) or {})]
        if not order:
            return None, {"matched": False}

        def run(backend: str) -> tuple[np.ndarray, dict[str, Any]]:
            gal = self.runtime.galleries.get(backend) or {}
            thr = self._backend_thr(backend, opts) + float(thr_add)
            cached = (self._batch_vecs or {}).get((id(f), backend))
            vec = cached if cached is not None else get_embedder(self.cfg, backend).embed_bgr(frame, landmarks=lm, xyxy=f["xyxy"])
            ident = match(vec, gal, cfg=self.cfg, threshold=thr, margin=mar)
            ident = self._blend_backend(tr, backend, vec, ident, gal, thr, mar)
            ident["backend"] = backend
            return vec, ident

        if not require_both:
            fallback: tuple[np.ndarray, dict[str, Any]] | None = None
            for backend in order:
                vec, ident = run(backend)
                if ident.get("matched"):
                    return vec, ident
                if fallback is None:
                    fallback = (vec, ident)
            return fallback if fallback is not None else (None, {"matched": False})

        if len(order) == 1:
            return run(order[0])

        vec1, ident1 = run(order[0])
        if not ident1.get("matched"):
            # «И» невозможно без первого движка — второй не считаем
            return vec1, ident1
        vec2, ident2 = run(order[1])
        if ident2.get("matched") and ident1.get("name") == ident2.get("name"):
            if float(ident2.get("score") or 0) > float(ident1.get("score") or 0):
                return vec2, ident2
            return vec1, ident1
        # не согласны: не распознан; наружу — лучший по score (для «почти попал»)
        best = (vec2, ident2) if float(ident2.get("score") or 0) > float(ident1.get("score") or 0) else (vec1, ident1)
        bvec, bident = best
        return bvec, {**bident, "matched": False, "name": None}

    def _prefetch_vecs(self, frame: np.ndarray, faces: list[dict[str, Any]]) -> None:
        """Эмбеддинги всех лиц кадра одним батчем на движок (раньше — по одному лицу, по очереди).

        Берём только лица, которые дойдут до 1:N: такой же минимальный размер, те же
        движки. Лица с профилем/бликом отсеиваются ниже, их батч не трогает.
        """
        self._batch_vecs = {}
        min_side = self.runtime.min_face_side_for(self.index)
        todo = []
        for f in faces:
            if f.get("track") is None:
                continue
            x1, y1, x2, y2 = f["xyxy"]
            if min(x2 - x1, y2 - y1) < min_side:
                continue
            try:
                lm = refine_landmarks(frame, f["xyxy"], cfg=self.cfg, min_score=0.42, allow_soft=True)
            except Exception:  # noqa: BLE001
                lm = None
            if lm is None:
                continue
            todo.append((f, lm))
        if not todo:
            return
        for backend in ("arcface", "adaface"):
            if not (self.runtime.galleries.get(backend) or {}):
                continue
            try:
                emb = get_embedder(self.cfg, backend)
                vecs = emb.embed_batch([(frame, lm, f["xyxy"]) for f, lm in todo])
            except Exception as e:  # noqa: BLE001
                log.warning("batch %s cam %s: %s", backend, self.index, e)
                continue
            for (f, _lm), v in zip(todo, vecs):
                self._batch_vecs[(id(f), backend)] = v

    def _recognize(self, frame: np.ndarray, faces: list[dict[str, Any]], cam_name: str, opts: dict[str, Any]) -> None:
        mar = float(opts.get("match_margin", 0.08))
        now = time.time()
        self._prefetch_vecs(frame, faces)
        for f in faces:
            tr: Track = f["track"]
            vec = None
            ident: dict[str, Any] = {"matched": False}
            try:
                # уже узнанный в тишине: 1:N не гоняем, но YuNet/блик всё равно проверяем —
                # иначе монитор навсегда остаётся под чужим ФИО
                skip_embed = (
                    bool(tr.name)
                    and float(getattr(self, "_scene_motion", 0) or 0) < float(getattr(self, "_scene_motion_on", _MOTION_ON))
                    and now - tr.last_embed_ts < 3.0
                )
                score = float(f.get("score") or 0)
                # Один проход YuNet на кропе. Второй такой же проход занимал GPU и резал очередь 1:N.
                # Бокс YOLO уже отсёк большую часть мусора, поэтому порог мягче, чем у YuNet «с нуля».
                yolo_box = f.get("backend") == "yolo" or score >= 0.25
                yunet_need = 0.42 if yolo_box else YUNET_ONLY_SCORE
                sink: list[float] = []
                # Уверенный бокс YOLO у окна и в три четверти часто не проходит строгий YuNet.
                # Мягкий проход только для таких боксов: пакет с низким score глаза не получает.
                lm = refine_landmarks(
                    frame,
                    f["xyxy"],
                    cfg=self.cfg,
                    min_score=yunet_need,
                    allow_soft=bool(yolo_box and score >= 0.50),
                    score_sink=sink,
                )
                yunet_score = float(sink[0]) if sink else 0.0
                f["landmarks"] = lm
                # Совсем в профиль (глаза почти слиплись) не сравниваем. Трёхчетвертной вид
                # идёт в 1:N с надбавкой к порогу: на 1080p это часто тот же человек у двери.
                if lm is None or yunet_score < yunet_need or not landmarks_ok(lm, f["xyxy"], min_eye=0.13):
                    tr.reject_n = int(getattr(tr, "reject_n", 0) or 0) + 1
                    if observe_invalid(tr) and tr.subject is None:
                        log.info("cam %s track %s: лицо не подтвердилось, имя сброшено", self.index, tr.tid)
                    continue
                tr.reject_n = 0
                x1, y1, x2, y2 = f["xyxy"]
                side = min(x2 - x1, y2 - y1)
                side_face = not landmarks_ok(lm, f["xyxy"], min_eye=_MATCH_MIN_EYE)
                if _box_sharpness(frame, f["xyxy"]) >= _OBJECT_SHARPNESS or _is_glare_box(frame, f["xyxy"], score):
                    if observe_invalid(tr) and tr.subject is None:
                        log.info("cam %s track %s: блик/предмет, имя сброшено", self.index, tr.tid)
                    continue
                tr.face_score = max(tr.face_score, yunet_score)
                tr.validated_ts = now
                if skip_embed:
                    continue
                # лучший кроп интервала: смазанный повтор не разбавляет среднее по треку
                sharp = float(side) * max(float(f.get("score") or 0), 0.05)
                if (
                    tr.last_embed_ts > 0
                    and now - tr.last_embed_ts < _EMBED_EVERY_S
                    and sharp < tr.embed_sharp * _EMBED_SHARPER
                ):
                    continue
                vec, ident = self._identify(
                    frame, f, lm, side, tr, opts, mar, thr_add=0.06 if side_face else 0.0
                )
                if vec is None:
                    continue
                tr.last_embed_ts = now
                tr.embed_sharp = sharp
            except Exception as e:  # noqa: BLE001
                ident = {"matched": False, "error": str(e)}
            before = tr.subject
            changed = observe(tr, vec, ident, self.runtime.unknowns, now)
            if tr.subject is None:
                continue
            if (
                tr.name
                and not tr.learned
                and vec is not None
                and ident.get("matched")
                and ident.get("name") == tr.name
                and float(ident.get("score") or 0) >= _LEARN_SCORE
                and float(ident.get("gap") or 0) >= _LEARN_GAP
            ):
                tr.learned = True
                self.runtime.learn_observed(tr.name, vec, str(ident.get("backend") or "arcface"))
            if changed and before and before.startswith("u:") and tr.name:
                self.runtime.relabel_track(self.index, tr, before)
            if tr.name and (changed or now - tr.last_logged_ts >= _SIGHTING_EVERY_S):
                tr.last_logged_ts = now
                self.runtime.log_sighting(self, tr, cam_name, frame)
            role = self.runtime.passage_roles.get(self.index)
            if role and not tr.passage_ts:
                self.runtime.log_passage(self, tr, cam_name, frame, role)
            # журнал — только начало визита: пока человек в кадре, повторных записей нет
            if self.runtime.touch_presence(str(tr.subject), self.index, now):
                self.runtime.alert(self, tr, frame, cam_name, opts)
        self._batch_vecs = {}

    def _recog_preview(self, frame: np.ndarray, faces: list[dict[str, Any]], cam_name: str, opts: dict[str, Any]) -> None:
        try:
            with _INFER_SEM:
                self._recognize(frame, faces, cam_name, opts)
        except Exception as e:  # noqa: BLE001
            log.warning("recog cam %s: %s", self.index, e)
        finally:
            self._recog_lock.release()

    def _bg_full(self, cam_name: str, opts: dict[str, Any], stream: str) -> None:
        """Детект по самому свежему кадру. Рамка публикуется сразу, 1:N её не держит.

        Видеокарта одна на все камеры. _last_det_ok только после реального YOLO —
        иначе 35 камер «думают», что уже детектили, и кадры с людьми пропускаются.
        """
        frame = None
        faces: list[dict[str, Any]] = []
        is_preview = self.runtime.preview_index == self.index
        try:
            # открытая камера ждёт GPU до 0.7 с; остальные не блокируют очередь
            t_sem = time.perf_counter()
            got = _INFER_SEM.acquire(timeout=0.45 if is_preview else 2.0)
            _perf_add("sem_wait", (time.perf_counter() - t_sem) * 1000)
            if not got:
                with _PERF_LOCK:
                    PERF["sem_miss"] += 1
                self._detect_due = True
                self._infer_retry_at = time.time() + (0.04 if is_preview else 0.08)
                return
            try:
                src = self._latest_frame
                if src is None:
                    return
                frame = src
                high = self.runtime.conf_for(self.index)
                t_det = time.perf_counter()
                faces = self._detect(frame, opts, high)
                _perf_add("det", (time.perf_counter() - t_det) * 1000)
                faces = self._track(faces, high)
                self._det_gen += 1
                self._last_det_ok = time.time()
                self._detect_due = False
                # 1:N в том же захвате GPU. Отдельный повторный захват с таймаутом 50 мс
                # почти никогда не доставался фоновой камере: рамка была, имени не было.
                now = time.time()
                if faces and now - self._recog_at >= 0.28:
                    cap_n = max(1, int(opts.get("max_faces_per_frame", 10)))
                    faces_recog = filter_faces_for_recog(faces, frame.shape)
                    faces_recog.sort(key=lambda f: float(f.get("score") or 0), reverse=True)
                    faces_recog = faces_recog[:cap_n]
                    if faces_recog:
                        self._recog_at = now
                        t_rec = time.perf_counter()
                        self._recognize(frame, faces_recog, cam_name, opts)
                        _perf_add("rec", (time.perf_counter() - t_rec) * 1000)
                self._publish_state(self._hold_faces(faces), cam_name, stream)
            finally:
                _INFER_SEM.release()
        except Exception as e:  # noqa: BLE001
            log.warning("bg infer cam %s: %s", self.index, e)
        finally:
            try:
                self._recog_lock.release()
            except RuntimeError:
                pass
        if frame is None or not faces:
            return
        try:
            shown = [
                f for f in faces
                if int(getattr(f.get("track"), "reject_n", 0) or 0) < 2
            ]
            self._remember_crop(frame, shown)
        except Exception as e:  # noqa: BLE001
            log.warning("snap cam %s: %s", self.index, e)

    def _motion_on(self, cap: Any) -> float:
        """Порог движения с поправкой на ползунок «Чувствительность к движению» камеры.

        Меньше значение — раньше срабатывает проверка кадра. Шкала ползунка обратная
        коду: RtspCamera сам умножает шум на MOTION_K, поэтому делим на свой коэффициент.
        """
        base = float(getattr(cap, "motion_threshold", 0.0) or 0.0)
        if base <= 0:
            base = _MOTION_ON
        try:
            k = float(self.runtime.cam_opt(self.index, "motion_k", MOTION_K_REF))
        except (TypeError, ValueError):
            k = MOTION_K_REF
        if k <= 0:
            k = MOTION_K_REF
        return max(0.4, base * (MOTION_K_REF / k))

    def _blk_on(self) -> float:
        try:
            k = float(self.runtime.cam_opt(self.index, "motion_k", MOTION_K_REF))
        except (TypeError, ValueError):
            k = MOTION_K_REF
        if k <= 0:
            k = MOTION_K_REF
        return _BLK_ON * (MOTION_K_REF / k)

    def apply_cam_settings(self, settings: dict[str, Any]) -> None:
        """Настройки читаются из runtime.cam_opt в момент использования, поэтому
        пересоздавать воркер не нужно — метод оставлен как точка расширения."""
        return None

    def _schedule_detect(self, cap: Any, new: bool, is_preview: bool) -> bool:
        """Детекция по движению. Открытая камера — чаще; фоновые — реже, чтобы не забивать GPU."""
        now = time.time()
        if now < self._infer_retry_at:
            return False
        motion = float(getattr(cap, "motion", 0.0) or 0.0)
        motion_on = self._motion_on(cap)
        blk = float(getattr(cap, "motion_blk", 0.0) or 0.0)
        moving = motion >= motion_on or blk >= self._blk_on()
        # «Интервал проверки» — индивидуальный ползунок камеры: сколько секунд ждать
        # между проверками, когда в кадре ничего не происходит.
        still_s = self.runtime.detect_interval_for(self.index)
        track_s = min(0.45, still_s)
        if hasattr(cap, "motion"):
            if self._detect_due and (is_preview or moving or now - self._last_det_ok >= still_s):
                return True
            if new and moving:
                return True
            if is_preview and new and now - self._last_det_ok >= 0.12:
                return True
            try:
                alive = bool(self.tracker.active())
            except Exception:  # noqa: BLE001
                alive = False
            if alive and now - self._last_det_ok >= track_s:
                return True
            return now - self._last_det_ok >= still_s
        return bool(self._detect_due or (new and (is_preview or now - self._last_det_ok >= 0.5)))

    def _kick_detect(self, cap: Any, name: str, opts: dict[str, Any], stream_tag: str) -> None:
        self._scene_motion = float(getattr(cap, "motion", 0.0) or 0.0)
        self._scene_motion_on = float(getattr(cap, "motion_threshold", _MOTION_ON) or _MOTION_ON)
        is_preview = self.runtime.preview_index == self.index
        if not self._recog_lock.acquire(blocking=False):
            self._detect_due = True
            if is_preview:
                self._infer_retry_at = time.time() + 0.05
            return
        # _last_det_ok ставит только _bg_full после успешного YOLO.
        # Общий пул вместо потока на каждую детекцию: иначе при нехватке GPU сотни потоков спорят за CPU.
        _DETECT_POOL.submit(self._bg_full, name, dict(opts), stream_tag)

    def _idle_slot(self, cap: Any, name: str, stream_tag: str, want_detect: bool, is_preview: bool) -> None:
        time.sleep(0.02)
        if not want_detect or self._latest_frame is None:
            return
        if self._schedule_detect(cap, False, is_preview):
            self._kick_detect(cap, name, self.runtime.options, stream_tag)

    def _loop(self) -> None:
        cam = cameras_store.get(self.index, self.cfg) or {}
        name = str(cam.get("name") or f"#{self.index}")
        stream_tag = getattr(self, "_stream_name", None) or ("main" if cam.get("rtsp_url_main") else "sub")
        try:
            get_detector(self.cfg)
            get_embedder(self.cfg, "arcface")
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            log.exception("models load cam=%s", self.index)
            return

        cap = None
        frames = 0
        t0 = time.time()
        while not self._stop.is_set():
            if not self.runtime.enabled:
                # выключение детекции закрывает NVDEC, а не оставляет сессию камеры в фоне
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    cap = None
                self.alive = False
                time.sleep(0.2)
                continue
            is_preview = self.runtime.preview_index == self.index
            want_detect = self.index in self.runtime.detect_on
            if not is_preview and not want_detect:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    cap = None
                self.alive = False
                time.sleep(0.3)
                continue
            try:
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:  # noqa: BLE001
                            pass
                        time.sleep(0.3)
                    cap = self._open_capture(cam)
                    self._cap = cap
                    stream_tag = getattr(self, "_stream_name", stream_tag)
                    self.last_meta = {"camera": name, "index": self.index, "stream": stream_tag}
                    if not cap.isOpened():
                        self.error = "поток открывается"
                        self.alive = False
                        time.sleep(0.5)
                        continue


                seq = getattr(cap, "seq", None)
                if seq is not None and int(seq) == self._seen_seq:
                    ts = float(getattr(cap, "_ts", 0) or 0)
                    if ts > 0 and (time.time() - ts) > 8.0:
                        try:
                            cap.release()
                        except Exception:  # noqa: BLE001
                            pass
                        cap = None
                        time.sleep(0.3)
                        continue
                    if is_preview or time.time() < self._wall_until:
                        watch = getattr(cap, "watch", None)
                        if watch:
                            watch()
                    self._idle_slot(cap, name, stream_tag, want_detect, is_preview)
                    continue
                ok, frame = self._read_latest(cap)
                if not ok or frame is None:
                    self.error = "нет кадра"
                    self.alive = False
                    with self._overlay_lock:
                        self._overlay = []
                    time.sleep(0.15)
                    # _ts==0 — кадр ещё ни разу не пришёл (не путать с «устарел»)
                    ts = float(getattr(cap, "_ts", 0) or 0)
                    if ts > 0 and (time.time() - ts) > 8.0:
                        try:
                            cap.release()
                        except Exception:  # noqa: BLE001
                            pass
                        cap = None
                    elif ts <= 0:
                        # ждём первый кадр до ~20 с после open
                        opened_at = float(getattr(cap, "_opened_at", 0) or 0)
                        if opened_at and (time.time() - opened_at) > 20.0:
                            try:
                                cap.release()
                            except Exception:  # noqa: BLE001
                                pass
                            cap = None
                    continue
                self.alive = True
                self.error = None
                self._cap = cap
                if is_preview or time.time() < self._wall_until:
                    watch = getattr(cap, "watch", None)
                    if watch:
                        watch()
                if seq is not None:
                    self._seen_seq = int(seq)

                self._frame_i += 1
                opts = self.runtime.options
                # Превью — частый кадр ≤960 px. Детекция и 1:N — полный кадр, он обновляется реже.
                read_full = getattr(cap, "read_full", None)
                full_seq = int(getattr(cap, "_full_seq", 0) or 0)
                if read_full and full_seq != getattr(self, "_full_seen", -1):
                    ok_full, full = read_full()
                    if ok_full and full is not None:
                        self._latest_frame = full
                        self._full_seen = full_seq
                if self._latest_frame is None:
                    self._latest_frame = frame
                # JPEG только тем камерам, которые сейчас смотрят. Кодировать 1080p на всех 35
                # потоках не оставляет ядер на 25 к/с.
                if is_preview:
                    self._publish_jpeg(frame)
                rate = float(getattr(cap, "fps", 0.0) or 0.0)
                if rate > 0.0:
                    self.capture_fps = rate
                else:
                    frames += 1
                    dt = time.time() - t0
                    if dt >= 1.0:
                        self.capture_fps = frames / dt
                        frames = 0
                        t0 = time.time()

                skip = max(1, int(opts.get("frame_skip") or 1))
                if want_detect and self._frame_i % skip == 0 and self._schedule_detect(cap, True, is_preview):
                    self._kick_detect(cap, name, opts, stream_tag)
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                self.alive = False
                log.warning("cam %s: %s", self.index, e)
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                time.sleep(1.0)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self.alive = False

    def get_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self.last_jpeg

    def get_jpeg_sub(self) -> bytes | None:
        with self._jpeg_lock:
            return self.last_jpeg_sub

    def stop_sub_preview(self) -> None:
        """Камера не на экране и не детектируется: гасим ffmpeg-превью и кэш кадра."""
        sp = self._sub_preview
        self._sub_preview = None
        self._wall_until = 0.0
        if sp is not None:
            try:
                sp.stop()
            except Exception:  # noqa: BLE001
                pass

    def note_wall(self) -> None:
        """Плитка забрала кадр — держим превью sub через ffmpeg, пока её смотрят."""
        self._wall_until = time.time() + 1.0
        url = self._sub_url_cached()
        if not url:
            return
        if self._sub_preview is None or self._sub_preview.url != url:
            if self._sub_preview is not None:
                self._sub_preview.stop()
            self._sub_preview = SubPreview(url, self._publish_sub_jpeg)
        self._sub_preview.watch()

    def _sub_url_cached(self) -> str:
        cam = cameras_store.get(self.index, self.cfg) or {}
        sub = (cam.get("rtsp_url_sub") or "").strip()
        main = (cam.get("rtsp_url_main") or cam.get("url") or "").strip()
        if not sub.startswith("rtsp://") or sub == main:
            return ""
        return sub

    def note_delivered(self) -> None:
        self._disp_n += 1
        dt = time.time() - self._disp_t0
        if dt >= 1.0:
            self.fps = self._disp_n / dt
            self._disp_n = 0
            self._disp_t0 = time.time()


class Runtime:
    def __init__(self) -> None:
        self.cfg = load_config()
        database.init_db(self.cfg)
        self.enabled = False
        self.detect_on: set[int] = set()
        self.preview_index: int | None = None
        self.wall_set: set[int] = set()
        self.workers: dict[int, CameraWorker] = {}
        self.options: dict[str, Any] = {}
        self.cam_settings: dict[int, dict[str, Any]] = {}
        self.passage_pairs: list[dict[str, Any]] = []
        self.passage_roles: dict[int, str] = self._load_passage_roles()
        self.events: deque[dict[str, Any]] = deque(maxlen=200)
        self.gallery_cache: dict[str, list] = {}
        self.galleries: dict[str, dict[str, list]] = {"arcface": {}, "adaface": {}}
        self.people: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        # (субъект, камера) → когда видели в последний раз; новая строка журнала только после паузы
        self._seen_ts: dict[tuple[str, int], float] = {}
        # карточка → {(камера, трек): ts} за последний час; карточка на десятке разных треков — «магнит»
        self._card_tracks: dict[str, dict[tuple[int, int], float]] = {}
        self.unknowns = UnknownRegistry(int(database.get_setting("unknown_seq", 0) or 0))
        self._unknown_saved = self.unknowns.last_no
        self.snap_dir = resolve(self.cfg, "data/snapshots")
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.reload_options()
        self.reload_gallery()
        self.reload_cam_settings()
        # фактический set_enabled(True) делает api._startup при QARAQU_FACE_ENABLED=1
        try:
            database.set_setting("service_enabled", False)
        except Exception:
            pass

    def reload_options(self) -> dict[str, Any]:
        database.init_db(self.cfg)
        self.options = database.get_options()
        rec = self.cfg.get("recognize", {}) if isinstance(self.cfg.get("recognize"), dict) else {}
        self.options.setdefault("match_threshold", float(rec.get("threshold", 0.58)))
        self.options.setdefault("match_margin", float(rec.get("margin", 0.08)))
        self.options.setdefault("det_conf", _LIVE_DET_CONF)
        self.options.setdefault("frame_skip", 3)
        self.options.setdefault("max_faces_per_frame", 10)
        self.options.setdefault("decode_device", "cuda")
        self.options.setdefault("preview_fps", 15)
        # Старые настройки были под sub 5–10 к/с: порог детекции 0.34 и 1:N 0.47
        # оставляли лица без рамки и без имени. Один раз приводим к рабочим.
        if not self.options.get("live25_migrated"):
            patch: dict[str, Any] = {"live25_migrated": True}
            if float(self.options.get("det_conf") or 0) > 0.28:
                patch["det_conf"] = 0.20
            if float(self.options.get("match_threshold") or 0) > 0.44:
                patch["match_threshold"] = 0.40
            if int(self.options.get("frame_skip") or 3) > 2:
                patch["frame_skip"] = 2
            self.options.update(patch)
            try:
                database.save_options(patch)
            except Exception:  # noqa: BLE001
                pass
        return self.options

    @staticmethod
    def _roles_from_pairs(pairs: list[dict[str, Any]]) -> dict[int, str]:
        roles: dict[int, str] = {}
        for p in pairs:
            if p.get("in") is not None:
                roles[int(p["in"])] = "in"
            if p.get("out") is not None:
                roles[int(p["out"])] = "out"
        return roles

    def _load_passage_roles(self) -> dict[int, str]:
        try:
            pairs = database.get_setting("passage_pairs", []) or []
        except Exception:  # noqa: BLE001
            pairs = []
        self.passage_pairs = [dict(p) for p in pairs if isinstance(p, dict)]
        return self._roles_from_pairs(self.passage_pairs)

    def set_passage_pairs(self, pairs: list[dict[str, Any]]) -> dict[int, str]:
        """Пары «вход + выход» для раздела «Проходы». Камера без пары не считает проходы."""
        clean: list[dict[str, Any]] = []
        for p in pairs:
            row = {
                "in": int(p["in"]) if p.get("in") not in (None, "") else None,
                "out": int(p["out"]) if p.get("out") not in (None, "") else None,
            }
            if row["in"] is not None or row["out"] is not None:
                clean.append(row)
        database.set_setting("passage_pairs", clean)
        self.passage_pairs = clean
        self.passage_roles = self._roles_from_pairs(clean)
        return self.passage_roles

    def reload_cam_settings(self) -> dict[int, dict[str, Any]]:
        try:
            self.cam_settings = database.all_camera_settings()
        except Exception:  # noqa: BLE001
            self.cam_settings = {}
        return self.cam_settings

    def cam_opt(self, index: int, key: str, default: Any = None) -> Any:
        """Настройка конкретной камеры. Берётся в момент использования — ползунок работает без перезапуска."""
        row = self.cam_settings.get(int(index))
        if row and key in row:
            return row[key]
        return database.CAM_SETTING_DEFAULTS.get(key, default)

    def save_cam_settings(self, index: int, patch: dict[str, Any]) -> dict[str, Any]:
        cur = database.save_camera_settings(int(index), patch or {})
        self.cam_settings[int(index)] = cur
        w = self.workers.get(int(index))
        if w is not None:
            w.apply_cam_settings(cur)
        return cur

    def save_options(self, patch: dict[str, Any]) -> dict[str, Any]:
        database.init_db(self.cfg)
        self.options = database.save_options(patch or {})
        return self.options

    def learn_observed(self, name: str, vec: np.ndarray, backend: str = "arcface") -> None:
        """Дописать камерный вектор в наблюдённые карточки той же модели, что дала совпадение."""
        try:
            out = add_observed_vec(name, vec, self.cfg, backend=backend)
        except Exception as e:  # noqa: BLE001
            log.warning("observed %s: %s", name, e)
            return
        if not out.get("added"):
            return
        gal = self.galleries.setdefault(backend, {})
        cur = gal.get(name)
        if cur is not None:
            gal[name] = list(cur) + [l2(vec)]
        if backend == "arcface":
            self.gallery_cache = gal
        log.info("карточка %s: камерный вектор %s (наблюдённых %s)", name, backend, out.get("observed"))

    def reload_gallery(self) -> int:
        self.galleries = {
            "arcface": load_gallery(self.cfg, backend="arcface"),
            "adaface": load_gallery(self.cfg, backend="adaface"),
        }
        self.gallery_cache = self.galleries["arcface"]
        people: dict[str, dict[str, Any]] = {}
        for p in list_people(self.cfg):
            people[p["name"]] = {
                "display": p.get("display") or p["name"],
                "thumb": p.get("thumb"),
                "iin": (p.get("attrs") or {}).get("iin") or "",
            }
        self.people = people
        return len(self.gallery_cache)

    def display_of(self, tr: Track) -> str:
        if tr.name:
            return self.people.get(tr.name, {}).get("display") or tr.name
        return unknown_display(tr.unknown_no or 0)

    def conf_for(self, index: int) -> float:
        # порог YOLO на камере — индивидуальный ползунок «Порог детекции»
        c = self.cam_opt(index, "det_conf")
        if c is None:
            c = self.options.get("det_conf", _LIVE_DET_CONF)
        try:
            return max(0.05, min(0.95, float(c)))
        except (TypeError, ValueError):
            return _LIVE_DET_CONF

    def match_thr_for(self, index: int) -> float:
        # порог 1:N на камере — индивидуальный ползунок «Порог совпадения»
        c = self.cam_opt(index, "match_threshold")
        if c is None:
            c = self.options.get("match_threshold", 0.58)
        try:
            return max(0.10, min(0.90, float(c)))
        except (TypeError, ValueError):
            return 0.58

    def min_face_side_for(self, index: int) -> int:
        # минимальная меньшая сторона бокса, с которой лицо вообще идёт в 1:N
        c = self.cam_opt(index, "min_face_side")
        try:
            return int(max(10, min(300, int(round(float(c))))))
        except (TypeError, ValueError):
            return int(_MATCH_MIN_SIDE)

    def imgsz_for(self, index: int) -> int:
        c = self.cam_opt(index, "imgsz")
        try:
            v = int(round(float(c)))
        except (TypeError, ValueError):
            return int(_LIVE_IMGSZ)
        return int(min(1920, max(640, v)))

    def detect_interval_for(self, index: int) -> float:
        c = self.cam_opt(index, "detect_interval")
        try:
            return max(0.1, min(15.0, float(c)))
        except (TypeError, ValueError):
            return 2.0

    def note_card_track(self, name: str, cam: int, tid: int, now: float) -> None:
        d = self._card_tracks.setdefault(name, {})
        d[(int(cam), int(tid))] = now
        if len(d) > 200:
            for k in sorted(d, key=d.get)[: len(d) - 200]:
                d.pop(k, None)

    def card_track_hits(self, window_s: float = 3600.0) -> dict[str, int]:
        now = time.time()
        out: dict[str, int] = {}
        for name, d in list(self._card_tracks.items()):
            live = {k: t for k, t in d.items() if now - t <= window_s}
            if live:
                self._card_tracks[name] = live
                out[name] = len(live)
            else:
                self._card_tracks.pop(name, None)
        return out

    def log_sighting(self, worker: CameraWorker, tr: Track, cam_name: str, frame: np.ndarray | None = None) -> None:
        if tr.name:
            self.note_card_track(tr.name, worker.index, tr.tid, time.time())
        if self.unknowns.last_no > self._unknown_saved:
            self._unknown_saved = self.unknowns.last_no
            database.set_setting("unknown_seq", self._unknown_saved)
        snap = tr.snap
        if frame is not None:
            saved = self._save_snapshot(frame, tr, worker.index, tr.last_ts, prefix="pass_")
            if saved:
                snap = saved
                tr.snap = saved
        try:
            u = box_h = None
            if frame is not None and tr.xyxy:
                fh, fw = frame.shape[:2]
                x1, y1, x2, y2 = tr.xyxy
                if fw > 0 and fh > 0:
                    u = min(1.0, max(0.0, ((x1 + x2) * 0.5) / fw))
                    box_h = min(1.0, max(0.0, (y2 - y1) / fh))
            database.insert_sighting(
                {
                    "ts": tr.last_ts,
                    "subject": tr.subject,
                    "kind": tr.kind,
                    "name": tr.name,
                    "display": self.display_of(tr),
                    "camera_index": worker.index,
                    "camera": cam_name,
                    "track_id": tr.tid,
                    "snapshot": snap,
                    "u": u,
                    "box_h": box_h,
                }
            )
        except Exception as e:  # noqa: BLE001
            log.warning("sighting persist: %s", e)

    def log_passage(self, worker: CameraWorker, tr: Track, cam_name: str, frame: np.ndarray | None, direction: str) -> None:
        """Проход по камере «вход»/«выход»: одна запись на появление трека в кадре, неизвестные тоже."""
        now = time.time()
        tr.passage_ts = now
        snap = None
        if frame is not None and tr.xyxy:
            snap = self._save_snapshot(frame, tr, worker.index, now, prefix="pass_")
        try:
            database.insert_passage(
                {
                    "ts": now,
                    "subject": tr.subject,
                    "kind": tr.kind,
                    "name": tr.name,
                    "display": self.display_of(tr),
                    "camera_index": worker.index,
                    "camera": cam_name,
                    "direction": direction,
                    "track_id": tr.tid,
                    "snapshot": snap,
                }
            )
        except Exception as e:  # noqa: BLE001
            log.warning("passage persist: %s", e)

    def relabel_track(self, cam_index: int, tr: Track, old_subject: str) -> None:
        try:
            database.relabel_sightings(
                old_subject=old_subject,
                camera_index=cam_index,
                track_id=tr.tid,
                since=tr.first_ts - 1,
                subject=str(tr.subject),
                name=tr.name,
                display=self.display_of(tr),
            )
            database.relabel_passages(
                old_subject=old_subject,
                camera_index=cam_index,
                track_id=tr.tid,
                since=tr.first_ts - 1,
                subject=str(tr.subject),
                kind=tr.kind,
                name=tr.name,
                display=self.display_of(tr),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("sighting relabel: %s", e)

    def _save_snapshot(self, frame: np.ndarray, tr: Track, cam_index: int, ts: float, *, prefix: str = "") -> str | None:
        x1, y1, x2, y2 = tr.xyxy
        h, w = frame.shape[:2]
        side = max(x2 - x1, y2 - y1, 1)
        pad = max(int(0.8 * side), 36)
        crop = frame[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)]
        if crop.size == 0:
            return None
        stem = f"{prefix}{int(ts * 1000)}_{cam_index}_{tr.tid}"
        fname = f"{stem}.jpg"
        if not cv2.imwrite(str(self.snap_dir / fname), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
            return None
        # полный кадр камеры с рамкой лица — показываем по нажатию на событие
        full = frame.copy()
        cv2.rectangle(full, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 3)
        cv2.imwrite(str(self.snap_dir / f"{stem}_full.jpg"), full, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return fname

    def touch_presence(self, subject: str, cam: int, now: float) -> bool:
        """True, если человека не было на этой камере дольше паузы — пора писать журнал."""
        gap = max(0.0, float(self.options.get("alert_cooldown_seconds", 60)))
        key = (subject, int(cam))
        prev = self._seen_ts.get(key)
        self._seen_ts[key] = now
        return prev is None or (now - prev) > gap

    def alert(self, worker: CameraWorker, tr: Track, frame: np.ndarray, cam_name: str, opts: dict[str, Any]) -> None:
        kind = tr.kind
        if kind == "known" and not opts.get("alert_known", True):
            return
        if kind == "unknown" and not opts.get("alert_unknown"):
            return
        now = time.time()
        person = self.people.get(tr.name or "", {})
        ev = {
            "ts": now,
            "camera": cam_name,
            "camera_index": worker.index,
            "name": tr.name,
            "display": self.display_of(tr),
            "score": tr.score,
            "kind": kind,
            "subject": tr.subject,
            "track_id": tr.tid,
            "iin": person.get("iin") or "",
            "gallery_thumb": person.get("thumb"),
        }
        if opts.get("save_snapshots", True):
            snap = self._save_snapshot(frame, tr, worker.index, now)
            if snap:
                ev["snapshot"] = snap
        try:
            ev["id"] = database.insert_alert(ev)
        except Exception as e:  # noqa: BLE001
            log.warning("alert persist: %s", e)
        ev["thumb"] = f"/api/snapshot/{ev['snapshot']}" if ev.get("snapshot") else None
        self.events.appendleft(ev)

    def ensure_worker(self, index: int) -> CameraWorker:
        with self._lock:
            w = self.workers.get(index)
            if w is None:
                w = CameraWorker(index, self.cfg, self)
                self.workers[index] = w
                w.start()
            return w

    def remove_camera(self, index: int) -> None:
        """После удаления из БД индексы сдвигаются: гасим воркер и всех с бóльшим номером.

        Иначе поток продолжает читать старый RTSP, а на том же index уже другая камера —
        в UI кажется, что «удаление не сработало».
        """
        index = int(index)
        with self._lock:
            to_stop: list[CameraWorker] = []
            w = self.workers.pop(index, None)
            if w is not None:
                to_stop.append(w)
            for i in sorted((k for k in self.workers if k > index), reverse=True):
                to_stop.append(self.workers.pop(i))
            self.detect_on.discard(index)
            if self.preview_index == index:
                self.preview_index = None
            self.cam_settings.pop(index, None)
        for w in to_stop:
            try:
                w.stop()
            except Exception:  # noqa: BLE001
                pass

    def stop_all_streams(self) -> int:
        """Закрыть все RTSP/веб-камеры. Сам сайт остаётся."""
        self.enabled = False
        self.preview_index = None
        self.detect_on.clear()
        try:
            database.set_setting("service_enabled", False)
        except Exception:
            pass
        with self._lock:
            workers = list(self.workers.values())
            self.workers.clear()
        for w in workers:
            w.stop()
        return len(workers)

    def set_enabled(self, on: bool) -> None:
        want = bool(on)
        if not want and self.enabled:
            # сначала стоп-флаг, потом дождаться GPU — иначе release NVDEC
            # посреди TensorRT даёт SIGSEGV
            self.enabled = False
            try:
                database.set_setting("service_enabled", False)
            except Exception:
                pass
            try:
                _INFER_SEM.acquire(timeout=3.0)
                _INFER_SEM.release()
            except Exception:  # noqa: BLE001
                pass
            try:
                from qaraqu_face.torch_runtime import gpu_lock

                with gpu_lock():
                    pass
            except Exception:  # noqa: BLE001
                pass
            return
        self.enabled = want
        try:
            database.set_setting("service_enabled", self.enabled)
        except Exception:
            pass
        if self.enabled:
            # модели один раз до потоков: иначе 50+ воркеров одновременно создают TensorRT-контексты
            try:
                get_detector(self.cfg)
                get_embedder(self.cfg, "arcface")
            except Exception as e:  # noqa: BLE001
                log.exception("preload models: %s", e)
                raise
            cams = cameras_store.list_public(self.cfg)
            for i, cam in enumerate(cams):
                w = self.ensure_worker(int(cam["index"]))
                self.detect_on.add(int(cam["index"]))
                try:
                    w.tracker.zones.clear()
                except Exception:  # noqa: BLE001
                    pass
                # не рвать все RTSP разом — камерам и FFmpeg нужна пауза
                if i and i % 8 == 0:
                    time.sleep(0.35)
            # чтобы рамки шли сразу, не только после открытия video_feed
            if self.preview_index is None and cams:
                self.preview_index = int(cams[0]["index"])
            self.reload_gallery()

    def set_wall(self, indices: set[int]) -> None:
        """Стена показывает только эти камеры. Чужие превью гасим сразу, не ждём простоя."""
        self.wall_set = set(indices)
        for i, w in list(self.workers.items()):
            if i in self.wall_set or i in self.detect_on or self.preview_index == i:
                continue
            w.stop_sub_preview()

    def set_preview(self, index: int) -> None:
        self.preview_index = index
        w = self.ensure_worker(index)
        # при открытии камеры сбрасываем залипшие «мёртвые зоны» столов
        try:
            w.tracker.zones.clear()
        except Exception:  # noqa: BLE001
            pass

    def set_detect(self, index: int, on: bool) -> None:
        if on:
            self.detect_on.add(index)
            self.ensure_worker(index)
        else:
            self.detect_on.discard(index)

    def jpeg(self, index: int, *, wall: bool = False) -> bytes | None:
        w = self.workers.get(index)
        if not w:
            return None
        if wall:
            w.note_wall()
            sub = w.get_jpeg_sub()
            if sub:
                return sub
            return w.get_jpeg()
        return w.get_jpeg()

    def board_events(self, limit: int = 30) -> list[dict[str, Any]]:
        """Кто в кадре сейчас и недавние проходы. Журнал неизвестных при этом не пишется."""
        live: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()
        try:
            cut = float(database.get_setting("events_cleared_at", 0) or 0)
        except (TypeError, ValueError):
            cut = 0.0
        for w in self.workers.values():
            for hit in (w.last_meta or {}).get("hits") or []:
                if hit.get("kind") not in ("known", "unknown"):
                    continue
                if cut and float(hit.get("last_ts") or 0) < cut:
                    continue
                row = dict(hit)
                row["live"] = True
                row["ts"] = hit.get("last_ts")
                live.append(row)
                seen.add((hit.get("subject"), hit.get("camera_index")))
        live.sort(key=lambda r: float(r.get("ts") or 0), reverse=True)
        history: list[dict[str, Any]] = []
        try:
            rows = database.recent_sightings(hours=6, limit=80)
        except Exception as e:  # noqa: BLE001
            log.warning("board events: %s", e)
            rows = []
        for row in rows:
            if cut and float(row.get("ts") or row.get("last_ts") or 0) < cut:
                continue
            key = (row.get("subject"), row.get("camera_index"))
            if key in seen:
                continue
            seen.add(key)
            item = dict(row)
            person = self.people.get(item.get("name") or "", {})
            if person.get("thumb"):
                item["gallery_thumb"] = person["thumb"]
            if person.get("display"):
                item["display"] = person["display"]
            if item.get("snapshot"):
                item["thumb"] = f"/api/snapshot/{item['snapshot']}"
            history.append(item)
            if len(live) + len(history) >= limit:
                break
        return (live + history)[:limit]

    def meta(self, index: int | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "enabled": self.enabled,
            "cameras": len(cameras_store.list_public(self.cfg)),
            "gallery": len(self.gallery_cache),
            "detect_on": sorted(self.detect_on),
            "preview_index": self.preview_index,
        }
        if index is not None and index in self.workers:
            w = self.workers[index]
            out.update(
                {
                    "fps": round(w.fps, 1),
                    "capture_fps": round(w.capture_fps, 1),
                    "alive": w.alive,
                    "error": w.error,
                    **(w.last_meta or {}),
                }
            )
        return out


_runtime: Runtime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> Runtime:
    """Один экземпляр на процесс. Без замка два запроса при старте создают два
    Runtime: консоль видит пустой, а чужие воркеры продолжают декодировать."""
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = Runtime()
        return _runtime
