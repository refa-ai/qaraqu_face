"""Трекинг объектов по камерам (PRINCIPLE.md).

Движение рамки на камере — OC-SORT из Ultralytics (Калман и добор слабых детекций).
Личность трека — голосование по результатам 1:N; без совпадения трек
становится «Неизвестный #N». UnknownRegistry узнаёт того же неизвестного
на других камерах по эмбеддингу лица.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.oc_sort import OCSORT
from ultralytics.trackers.utils import matching

from qaraqu_face.recognize import l2

# сколько попыток 1:N без совпадения, прежде чем считать трек неизвестным
UNKNOWN_AFTER = 2
# имя даёт либо два кадра за одного человека, либо один кадр с запасом над порогом.
# Один кадр ровно на пороге давал 73 из 97 «распознаваний» на четыре карточки-магнита.
CONFIRM_HITS = 2
CONFIRM_SINGLE_MARGIN = 0.10
# косинус для «тот же неизвестный» между камерами (ArcFace w600k)
UNKNOWN_REID_SIM = 0.42
# имя на треке держится, пока лицо подтверждается: подряд столько провалов (чужой/нет лица) — сброс.
# Быстрее сбрасываем: блик монитора иначе часами висит под чужим ФИО.
RESET_AFTER_MISSES = 2
# рамка с лицом моложе этого считается подтверждённой (для оверлея)
VALIDATED_TTL_S = 5.0
# неподвижная «рамка лица» без подтверждения (колесо, бухта кабеля): прячем после STATIC_MIN_S
STATIC_IOU = 0.4
STATIC_MIN_S = 20.0
STATIC_MIN_HITS = 8
STATIC_GAP_S = 60.0
STATIC_MAX = 64
UNKNOWN_KEEP_S = 24 * 3600
UNKNOWN_MAX = 2000
# предмет в одном месте не получает номер, пока YuNet не уверен, что это лицо
_STILL_UNKNOWN_HITS = 8
_STILL_FACE_SCORE = 0.72


def _smooth_box(tr: Track, box, ts: float) -> None:
    """Адаптивное EMA: дрожание неподвижной рамки гасится, движение догоняется быстро."""
    if not tr.fbox:
        tr.fbox = tuple(float(v) for v in box)
        return
    delta = max(abs(float(b) - o) for b, o in zip(box, tr.fbox))
    alpha = min(0.6, 0.2 + delta / 40.0)
    new = tuple(alpha * float(b) + (1 - alpha) * o for b, o in zip(box, tr.fbox))
    dt = ts - tr.last_ts
    if dt > 1e-3:
        raw = tuple((n - o) / dt for n, o in zip(new, tr.fbox))
        tr.vel = tuple(0.3 * r + 0.7 * v for r, v in zip(raw, tr.vel))
    tr.fbox = new


def display_box(tr: Track, now: float) -> tuple[int, int, int, int]:
    """Рамка для отрисовки: сглаженная, с экстраполяцией не дольше 0.25 с и не для неподвижного лица."""
    if not tr.fbox:
        return tuple(int(v) for v in tr.xyxy)
    dt = min(max(now - tr.last_ts, 0.0), 0.25)
    out = []
    for f, v in zip(tr.fbox, tr.vel):
        if abs(v) < 20.0:
            v = 0.0
        out.append(int(round(f + v * dt)))
    return tuple(out)


def _advance_track(tr: Track, box, ts: float) -> None:
    """Скорость центра: рамку можно сдвинуть вперёд, пока детектор считает следующий кадр."""
    ox1, oy1, ox2, oy2 = tr.xyxy
    nx1, ny1, nx2, ny2 = box
    dt = ts - tr.last_ts
    if 0.02 < dt < 0.8:
        ivx = ((nx1 + nx2) - (ox1 + ox2)) * 0.5 / dt
        ivy = ((ny1 + ny2) - (oy1 + oy2)) * 0.5 / dt
        tr.vx = tr.vx * 0.4 + ivx * 0.6
        tr.vy = tr.vy * 0.4 + ivy * 0.6
    _smooth_box(tr, box, ts)
    new_box = [int(v) for v in box]
    if iou(tr.xyxy, new_box) >= 0.82:
        tr.still_hits += 1
    else:
        tr.still_hits = 0
    tr.xyxy = new_box
    tr.last_ts = ts


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    tid: int
    xyxy: list[int]
    first_ts: float
    last_ts: float
    votes: dict[str, int] = field(default_factory=dict)
    score_sum: dict[str, float] = field(default_factory=dict)
    name: str | None = None
    score: float | None = None
    unknown_no: int | None = None
    recog_n: int = 0
    emb_sum: np.ndarray | None = None
    best_area: int = 0
    alerted_subject: str | None = None
    near_name: str | None = None
    near_score: float | None = None
    snap: str | None = None
    last_logged_ts: float = 0.0
    # проход по камере с ролью входа/выхода записан: один трек — одно появление в кадре
    passage_ts: float = 0.0
    validated_ts: float = 0.0
    det_score: float = 0.0
    misses: int = 0
    vx: float = 0.0
    vy: float = 0.0
    # сглаженная рамка для отрисовки: без неё шум детектора дёргает рамку на неподвижном лице
    fbox: tuple = ()
    vel: tuple = (0.0, 0.0, 0.0, 0.0)
    # последние качественные векторы трека, отдельно по модели: ArcFace и AdaFace нельзя усреднять
    qvecs: list = field(default_factory=list)
    qvecs_arc: list = field(default_factory=list)
    qvecs_ada: list = field(default_factory=list)
    # лучший кроп, уже отданный в 1:N: повтор внутри интервала не считается
    last_embed_ts: float = 0.0
    embed_sharp: float = 0.0
    # подряд почти неподвижных боксов: телефон и кабель так и висят, голова человека — нет
    still_hits: int = 0
    # уверенность YuNet на кропе; ниже порога неподвижный бокс не становится «Неизвестным»
    face_score: float = 0.0
    # подряд YuNet не нашёл лицо в боксе (пакет, бутылка, затылок). После двух — рамку не показываем.
    reject_n: int = 0
    # камерный вектор этого трека уже дописан в наблюдённые карточки
    learned: bool = False

    @property
    def subject(self) -> str | None:
        if self.name:
            return f"p:{self.name}"
        if self.unknown_no is not None:
            return f"u:{self.unknown_no}"
        return None

    @property
    def kind(self) -> str:
        return "known" if self.name else ("unknown" if self.unknown_no is not None else "pending")

    def mean_emb(self) -> np.ndarray | None:
        return None if self.emb_sum is None else l2(self.emb_sum)

    def validated(self, now: float) -> bool:
        return self.validated_ts > 0 and now - self.validated_ts <= VALIDATED_TTL_S

    def reset_identity(self) -> None:
        self.votes.clear()
        self.score_sum.clear()
        self.name = None
        self.score = None
        self.unknown_no = None
        self.recog_n = 0
        self.emb_sum = None
        self.misses = 0
        self.near_name = None
        self.near_score = None
        self.qvecs.clear()
        self.qvecs_arc.clear()
        self.qvecs_ada.clear()
        self.learned = False
        self.last_embed_ts = 0.0
        self.embed_sharp = 0.0
        self.face_score = 0.0


class UnknownRegistry:
    """Общая на все камеры память неизвестных лиц (номер ↔ эмбеддинг)."""

    def __init__(self, start_no: int = 0) -> None:
        self._lock = threading.Lock()
        self._next = int(start_no) + 1
        self._items: list[dict[str, Any]] = []

    @property
    def last_no(self) -> int:
        return self._next - 1

    def assign(self, emb: np.ndarray, ts: float) -> tuple[int, bool]:
        """Вернуть (номер, новый ли)."""
        q = l2(emb)
        with self._lock:
            self._items = [it for it in self._items if ts - it["ts"] < UNKNOWN_KEEP_S][-UNKNOWN_MAX:]
            best, best_sim = None, UNKNOWN_REID_SIM
            for it in self._items:
                s = float(np.dot(q, it["emb"]))
                if s >= best_sim:
                    best, best_sim = it, s
            if best is not None:
                best["emb"] = l2(best["emb"] * best["n"] + q)
                best["n"] = min(best["n"] + 1, 50)
                best["ts"] = ts
                return int(best["no"]), False
            no = self._next
            self._next += 1
            self._items.append({"no": no, "emb": q, "n": 1, "ts": ts})
            return no, True


class StaticZones:
    """Места, где долго и без движения висит рамка без подтверждённого лица."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._zones: list[dict[str, Any]] = []

    def clear(self) -> None:
        with self._lock:
            self._zones.clear()

    def is_static(self, box, now: float, unvalidated: bool) -> bool:
        with self._lock:
            self._zones = [z for z in self._zones if now - z["last"] <= STATIC_GAP_S]
            best, best_v = None, STATIC_IOU
            for z in self._zones:
                v = iou(box, z["box"])
                if v >= best_v:
                    best, best_v = z, v
            if not unvalidated:
                if best is not None:
                    self._zones.remove(best)
                return False
            if best is None:
                self._zones = (self._zones + [{"box": list(box), "first": now, "last": now, "hits": 1}])[-STATIC_MAX:]
                return False
            best["hits"] += 1
            best["last"] = now
            return now - best["first"] >= STATIC_MIN_S and best["hits"] >= STATIC_MIN_HITS


class _DetBoxes:
    """То, что OCSORT/BYTETracker читает у результата YOLO: conf, xywh и срез по маске."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), dtype=np.float32)
        if len(self.xyxy) == 0:
            self.xywh = np.zeros((0, 4), dtype=np.float32)
            return
        w = self.xyxy[:, 2] - self.xyxy[:, 0]
        h = self.xyxy[:, 3] - self.xyxy[:, 1]
        cx = (self.xyxy[:, 0] + self.xyxy[:, 2]) * 0.5
        cy = (self.xyxy[:, 1] + self.xyxy[:, 3]) * 0.5
        self.xywh = np.stack([cx, cy, w, h], axis=1)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, mask: np.ndarray) -> "_DetBoxes":
        return _DetBoxes(self.xyxy[mask], self.conf[mask])


def _motion_args(high: float) -> SimpleNamespace:
    high = float(high)
    low = min(0.10, max(0.05, high * 0.45))
    if low >= high:
        low = max(0.02, high * 0.5)
    return SimpleNamespace(
        track_high_thresh=high,
        track_low_thresh=low,
        new_track_thresh=high,
        track_buffer=12,
        match_thresh=0.8,
        # на 30 к/с fuse режет слабые рамки; у нас между детектами лицо смещается, IoU важнее процента YOLO
        fuse_score=False,
        delta_t=3,
        inertia=0.2,
        use_byte=True,
    )


class FaceOCSORT(OCSORT):
    """OC-SORT Ultralytics. Второй проход ByteTrack допускает шаг человека между редкими детектами."""

    def _second_association(self, strack_pool, u_track, detections_second, activated, refind, lost) -> None:
        if not self.use_byte:
            return super()._second_association(strack_pool, u_track, detections_second, activated, refind, lost)
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if r_tracked and detections_second:
            dists = matching.iou_distance(r_tracked, detections_second)
            matches, u_left, _ = matching.linear_assignment(dists, thresh=self.args.match_thresh)
            self._apply_matches(matches, r_tracked, detections_second, activated, refind)
        else:
            u_left = list(range(len(r_tracked)))
        for it in u_left:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)


class CameraTracker:
    """Движение рамки — OC-SORT Ultralytics (Калман + добор слабых детекций как в ByteTrack).

    Личность трека (картотека / Неизвестный #N) по-прежнему живёт в Track.
    """

    def __init__(self, camera_index: int) -> None:
        self.camera_index = camera_index
        self.zones = StaticZones()
        self._lock = threading.Lock()
        self._tracks: dict[int, Track] = {}
        self._motion = FaceOCSORT(_motion_args(0.25))

    def update(
        self,
        faces: list[dict[str, Any]],
        high_thresh: float,
        ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """Вернуть только рамки, которые трекер принял. Слабая детекция продолжает трек и не заводит новый."""
        ts = ts or time.time()
        args = _motion_args(high_thresh)
        self._motion.args = args
        self._motion.use_byte = True
        xyxy = [f["xyxy"] for f in faces] or np.zeros((0, 4), dtype=np.float32)
        conf = [float(f.get("score") or 0) for f in faces]
        raw = self._motion.update(_DetBoxes(np.asarray(xyxy, dtype=np.float32), np.asarray(conf, dtype=np.float32)))
        rows = raw if getattr(raw, "ndim", 0) == 2 else np.zeros((0, 8), dtype=np.float32)
        with self._lock:
            out: list[dict[str, Any]] = []
            alive: set[int] = set()
            for row in rows:
                x1, y1, x2, y2, tid, _score, _cls, idx = (float(v) for v in row)
                tid_i, idx_i = int(tid), int(idx)
                src = faces[idx_i] if 0 <= idx_i < len(faces) else None
                box = [int(v) for v in (src["xyxy"] if src else (x1, y1, x2, y2))]
                tr = self._tracks.get(tid_i)
                if tr is None:
                    tr = Track(tid=tid_i, xyxy=box, first_ts=ts, last_ts=ts)
                    self._tracks[tid_i] = tr
                else:
                    _advance_track(tr, box, ts)
                face = dict(src) if src else {"xyxy": tuple(box), "score": float(_score), "landmarks": None, "backend": "yolo"}
                face["xyxy"] = tuple(box)
                face["track"] = tr
                tr.det_score = float(face.get("score") or 0)
                out.append(face)
                alive.add(tid_i)
            removed = {int(t.track_id) for t in self._motion.removed_stracks}
            for tid in [k for k in self._tracks if k in removed and k not in alive]:
                del self._tracks[tid]
            return out

    def active(self) -> list[Track]:
        with self._lock:
            return list(self._tracks.values())


def observe(
    track: Track,
    vec: np.ndarray | None,
    ident: dict[str, Any],
    registry: UnknownRegistry,
    ts: float,
) -> bool:
    """Учесть результат 1:N для трека. True — у трека появился/сменился субъект."""
    before = track.subject
    if track.name:
        confirmed = bool(ident.get("matched")) and str(ident.get("name")) == track.name
        track.misses = 0 if confirmed else track.misses + 1
        if track.misses >= RESET_AFTER_MISSES:
            track.reset_identity()
    track.recog_n += 1
    if ident.get("best") and ident.get("score") is not None:
        track.near_name = str(ident["best"])
        track.near_score = float(ident["score"])
    x1, y1, x2, y2 = track.xyxy
    track.best_area = max(track.best_area, (x2 - x1) * (y2 - y1))
    if ident.get("matched") and ident.get("name"):
        n = str(ident["name"])
        track.votes[n] = track.votes.get(n, 0) + 1
        track.score_sum[n] = track.score_sum.get(n, 0.0) + float(ident.get("score") or 0.0)
        best = max(track.votes, key=lambda k: (track.votes[k], track.score_sum.get(k, 0.0)))
        thr = float(ident.get("threshold") or 0.0)
        strong = best == n and float(ident.get("score") or 0.0) >= thr + CONFIRM_SINGLE_MARGIN
        # большинство попыток за одного человека, не единичный кадр на пороге
        enough = track.votes[best] >= CONFIRM_HITS or strong
        if enough and track.votes[best] * 2 > track.recog_n:
            track.name = best
            track.score = round(track.score_sum[best] / track.votes[best], 4)
            track.unknown_no = None
    elif vec is not None and not track.name:
        v = l2(vec)
        track.emb_sum = v if track.emb_sum is None else track.emb_sum + v
        frozen_thing = track.still_hits >= _STILL_UNKNOWN_HITS and track.face_score < _STILL_FACE_SCORE
        if track.unknown_no is None and track.recog_n >= UNKNOWN_AFTER and not frozen_thing:
            no, _ = registry.assign(track.mean_emb(), ts)
            track.unknown_no = no
    return track.subject != before


def observe_invalid(track: Track) -> bool:
    """Лицо в боксе не подтвердилось (нет глаз/носа/рта). Известный трек без подтверждений теряет имя."""
    before = track.subject
    if track.name:
        track.misses += 1
        if track.misses >= RESET_AFTER_MISSES:
            track.reset_identity()
    return track.subject != before


_TR = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
        "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
        "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
        "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
        "я": "ya", "ә": "a", "ғ": "g", "қ": "k", "ң": "n", "ө": "o", "ұ": "u", "ү": "u",
        "һ": "h", "і": "i", "·": "-",
    }
)


def latin(s: str) -> str:
    """cv2.putText не рисует кириллицу."""
    out = []
    for ch in s:
        low = ch.lower()
        t = low.translate(_TR)
        out.append(t.capitalize() if ch != low and t else t)
    return "".join(c if ord(c) < 128 else "?" for c in "".join(out))
