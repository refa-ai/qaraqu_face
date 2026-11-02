"""Временный разбор неизвестных: кандидаты из картотеки и отдельная полка гостей.

Фото в карточку не копируется — дописывается только эмбеддинг.
Гости лежат в data/guests и в 1:N картотеки не входят.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qaraqu_face.config import load_config, resolve
from qaraqu_face.gallery import (
    MAX_VECS_PER_PERSON,
    commit_embs,
    embed_with_augs,
    list_people,
    load_gallery,
    person_dir,
    read_person_vecs,
)

_lock = threading.Lock()
_CANDIDATES = 5


def _guests_root(cfg: dict | None = None) -> Path:
    root = resolve(cfg or load_config(), "data/guests")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _meta_path(cfg: dict | None = None) -> Path:
    return _guests_root(cfg) / "meta.json"


def _read_meta(cfg: dict | None = None) -> dict[str, Any]:
    p = _meta_path(cfg)
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("next", 1)
                data.setdefault("items", [])
                return data
        except Exception:
            pass
    return {"next": 1, "items": []}


def _write_meta(data: dict[str, Any], cfg: dict | None = None) -> None:
    _meta_path(cfg).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_guests(cfg: dict | None = None) -> list[dict[str, Any]]:
    meta = _read_meta(cfg)
    out = []
    for item in meta.get("items") or []:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "id": str(item.get("id") or ""),
                "label": str(item.get("label") or item.get("id") or "Гость"),
                "embeddings": int(item.get("n") or 0),
            }
        )
    return [g for g in out if g["id"]]


def _guest_dir(guest_id: str, cfg: dict | None = None) -> Path:
    key = os.path.basename(str(guest_id or "").strip())
    if not key.startswith("g") or not key[1:].isdigit():
        raise ValueError("некорректный гость")
    root = os.path.realpath(str(_guests_root(cfg)))
    path = os.path.realpath(os.path.join(root, key))
    if not path.startswith(root + os.sep):
        raise ValueError("некорректный гость")
    return Path(path)


def _snap_file(fname: str, cfg: dict) -> Path:
    name = os.path.basename(str(fname or "").strip())
    if not name or name != str(fname).strip() or not name.lower().endswith((".jpg", ".jpeg", ".png")):
        raise ValueError("нет снимка")
    root = os.path.realpath(str(resolve(cfg, "data/snapshots")))
    path = os.path.realpath(os.path.join(root, name))
    if not path.startswith(root + os.sep):
        raise ValueError("нет снимка")
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(name)
    return file


def _usable_face(face: dict[str, Any]) -> bool:
    """Глаза и нос на месте. На кропе с камеры глаза ближе, чем на паспортном фото."""
    from qaraqu_face.face_align import landmarks_ok

    lm = face.get("landmarks")
    if not lm:
        return False
    return landmarks_ok(lm, face["xyxy"], min_eye=0.11)


def _shift_face(face: dict[str, Any], off: int) -> dict[str, Any]:
    if not off:
        return face
    x1, y1, x2, y2 = face["xyxy"]
    out = dict(face)
    out["xyxy"] = (x1 - off, y1 - off, x2 - off, y2 - off)
    if face.get("landmarks"):
        out["landmarks"] = (np.asarray(face["landmarks"], dtype=np.float32) - off).tolist()
    return out


def _faces_on(bgr: np.ndarray, cfg: dict) -> list[dict[str, Any]]:
    from qaraqu_face.detect import detect_bgr

    faces = detect_bgr(bgr, cfg=cfg, conf=0.05, imgsz=1280, with_landmarks=True, merge_yunet=True)
    good = [f for f in faces if _usable_face(f)]
    if good:
        return good
    # мелкий кроп с камеры: лицо занимает почти весь кадр, YOLO без полей его пропускает
    h, w = bgr.shape[:2]
    off = int(0.5 * max(h, w))
    if off < 8:
        return []
    pad = cv2.copyMakeBorder(bgr, off, off, off, off, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    padded = detect_bgr(pad, cfg=cfg, conf=0.05, imgsz=1280, with_landmarks=True, merge_yunet=True)
    good = []
    for f in padded:
        shifted = _shift_face(f, off)
        if _usable_face(shifted):
            good.append(shifted)
    return good


def _face_vec(bgr: np.ndarray, cfg: dict) -> np.ndarray:
    good = _faces_on(bgr, cfg)
    if not good:
        raise ValueError("на снимке не нашлось лица с глазами и носом")
    face = max(good, key=lambda f: float(f.get("score") or 0))
    vecs = embed_with_augs(bgr, cfg, n_aug=0, landmarks=face["landmarks"], xyxy=face["xyxy"], backend="arcface")
    if not vecs:
        raise ValueError("не удалось посчитать эмбеддинг")
    return vecs[0]


def _gallery_stamp(cfg: dict) -> int:
    root = resolve(cfg, cfg.get("recognize", {}).get("gallery_dir", "data/gallery"))
    stamp = 0
    if not root.is_dir():
        return 0
    for person in root.iterdir():
        emb = person / "embeddings.npy"
        if emb.is_file():
            stamp = max(stamp, int(emb.stat().st_mtime))
    return stamp


def _rank_cache(cfg: dict) -> dict[str, Any]:
    from qaraqu_face import db

    data = db.get_doc("unknown_rank")
    if isinstance(data, dict):
        return data
    return {"stamp": 0, "items": {}}


def _save_rank_cache(cfg: dict, data: dict[str, Any]) -> None:
    from qaraqu_face import db

    db.set_doc("unknown_rank", data)


def _review_path(cfg: dict) -> Path:
    return resolve(cfg, "data/review_queue.json")


def _read_review(cfg: dict) -> list[dict[str, Any]]:
    from qaraqu_face import db

    data = db.get_doc("review_queue")
    if not isinstance(data, list):
        return []
    return [i for i in data if isinstance(i, dict) and str(i.get("subject") or "").startswith("r:")]


def _write_review(items: list[dict[str, Any]], cfg: dict) -> None:
    from qaraqu_face import db

    db.set_doc("review_queue", items)


def enqueue_event(data: dict[str, Any], cfg: dict | None = None) -> dict[str, Any]:
    """Событие из панели «События» — в очередь корректировки. Неизвестный u: уже там."""
    cfg = cfg or load_config()
    subject = str(data.get("subject") or "").strip()
    if subject.startswith("u:"):
        return {"ok": True, "already": True, "subject": subject}
    snap = os.path.basename(str(data.get("snapshot") or "").strip())
    if not snap or snap != str(data.get("snapshot") or "").strip():
        raise ValueError("у события нет снимка")
    _snap_file(snap, cfg)
    key = "r:" + snap
    try:
        camera_index = int(data["camera_index"]) if data.get("camera_index") is not None and data.get("camera_index") != "" else None
    except (TypeError, ValueError):
        camera_index = None
    try:
        ts = float(data["ts"]) if data.get("ts") not in (None, "") else None
    except (TypeError, ValueError):
        ts = None
    with _lock:
        items = _read_review(cfg)
        if any(str(i.get("subject")) == key for i in items):
            return {"ok": True, "already": True, "subject": key}
        items.insert(
            0,
            {
                "subject": key,
                "display": str(data.get("display") or subject or "из события"),
                "camera": str(data.get("camera") or ""),
                "camera_index": camera_index,
                "ts": ts,
                "snapshot": snap,
                "from_subject": subject,
                "from_kind": str(data.get("kind") or ""),
            },
        )
        _write_review(items[:200], cfg)
    return {"ok": True, "subject": key, "already": False}


def _top_people(
    vec: np.ndarray,
    gal: dict,
    people: dict[str, str],
    thumbs: dict[str, str | None],
    k: int = _CANDIDATES,
) -> list[dict[str, Any]]:
    from qaraqu_face.recognize import l2

    q = l2(vec)
    scored: list[tuple[float, str]] = []
    for name, g in gal.items():
        arr = np.atleast_2d(np.asarray(g, np.float32))
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
        scored.append((float(np.max(arr @ q)), name))
    scored.sort(reverse=True)
    out = []
    for score, name in scored[:k]:
        out.append(
            {
                "name": name,
                "display": people.get(name) or name,
                "score": round(score, 4),
                "thumb": thumbs.get(name),
            }
        )
    return out


def _append_vec(folder: Path, vec: np.ndarray) -> int:
    """Ручная правка: камерный вектор идёт в наблюдённые карточки (observed_*.npy), не в фото-векторы.

    Пересборка по фото их не стирает, а карточка после одной правки начинает узнавать человека
    в этом же виде (шапка, очки) с камеры.
    """
    from qaraqu_face.gallery import add_observed_vec, read_observed_vecs

    if (folder / "card.json").is_file():
        out = add_observed_vec(folder.name, vec, backend="arcface")
        return len(read_person_vecs(folder)) + int(out.get("observed") or 0)
    # гость: отдельная папка без card.json — как раньше
    existing = read_person_vecs(folder)
    if len(existing) >= MAX_VECS_PER_PERSON:
        existing = existing[:1] + existing[-(MAX_VECS_PER_PERSON - 2) :]
    commit_embs(folder, existing + [vec])
    return len(existing) + 1


def queue(cfg: dict | None = None, *, limit: int = 40) -> dict[str, Any]:
    """Список неизвестных и до 5 ближайших карточек. Гости в этот список не входят."""
    from qaraqu_face import db as database

    cfg = cfg or load_config()
    listed = list_people(cfg)
    people = {p["name"]: (p.get("display") or p["name"]) for p in listed}
    thumbs = {p["name"]: p.get("thumb") for p in listed}
    review = _read_review(cfg)
    rows = database.list_unknown_sightings(limit=4000)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        subject = str(row.get("subject") or "")
        if subject in grouped:
            continue
        grouped[subject] = row
    unknown_rows = list(grouped.values())
    total = len(review) + len(unknown_rows)
    ordered = list(review) + unknown_rows[: max(1, int(limit))]

    gal = load_gallery(cfg)
    stamp = _gallery_stamp(cfg)
    cache = _rank_cache(cfg)
    if int(cache.get("stamp") or 0) != stamp:
        cache = {"stamp": stamp, "items": {}}
    items_cache: dict[str, Any] = cache.setdefault("items", {})
    dirty = False
    items = []
    for row in ordered:
        snap = str(row.get("snapshot") or "")
        candidates: list[dict[str, Any]] = []
        note = ""
        if snap and snap in items_cache and isinstance(items_cache[snap], list):
            candidates = []
            for c in items_cache[snap]:
                if not isinstance(c, dict):
                    continue
                cand = dict(c)
                cand["thumb"] = thumbs.get(cand.get("name"))
                cand["display"] = people.get(cand.get("name")) or cand.get("display") or cand.get("name")
                candidates.append(cand)
        elif snap:
            try:
                img = cv2.imread(str(_snap_file(snap, cfg)))
                if img is None:
                    raise ValueError("снимок не читается")
                vec = _face_vec(img, cfg)
                candidates = _top_people(vec, gal, people, thumbs)
                items_cache[snap] = candidates
                dirty = True
            except Exception as e:  # noqa: BLE001
                note = str(e)
        else:
            note = "нет снимка"
        items.append(
            {
                "subject": row.get("subject"),
                "display": row.get("display") or row.get("subject"),
                "camera": row.get("camera") or "",
                "camera_index": row.get("camera_index"),
                "ts": row.get("ts"),
                "snapshot": snap or None,
                "candidates": candidates,
                "note": note,
                "from_event": str(row.get("subject") or "").startswith("r:"),
            }
        )
    if dirty:
        _save_rank_cache(cfg, cache)
    return {
        "ok": True,
        "total": total,
        "items": items,
        "people": [
            {"name": p["name"], "display": p.get("display") or p["name"], "thumb": p.get("thumb")}
            for p in sorted(listed, key=lambda p: p.get("display") or p["name"])
        ],
        "guests": list_guests(cfg),
    }


def assign(
    subject: str,
    *,
    person: str | None = None,
    guest: str | None = None,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """Дописать эмбеддинг снимка в выбранную карточку или на полку гостей. Фото не сохраняется."""
    from qaraqu_face import db as database

    cfg = cfg or load_config()
    subject = str(subject or "").strip()
    if subject.startswith("r:"):
        return _assign_review(subject, person=person, guest=guest, cfg=cfg)
    if not subject.startswith("u:"):
        raise ValueError("это не неизвестный")

    with _lock:
        rows = [r for r in database.list_unknown_sightings(limit=4000) if r.get("subject") == subject]
        if not rows:
            raise ValueError("неизвестный уже разобран или не найден")
        snap = next((str(r.get("snapshot") or "") for r in rows if r.get("snapshot")), "")
        if not snap:
            raise ValueError("у записи нет снимка лица")
        img = cv2.imread(str(_snap_file(snap, cfg)))
        if img is None:
            raise ValueError("снимок не читается")
        vec = _face_vec(img, cfg)
        if guest:
            target = _assign_guest(subject, guest, vec, cfg)
            kind, name, display, new_subject = "guest", "", target["label"], target["subject"]
        else:
            key = str(person or "").strip()
            folder = person_dir(key, cfg)
            if not folder.is_dir():
                raise ValueError("карточка не найдена")
            n = _append_vec(folder, vec)
            card = {}
            card_path = folder / "card.json"
            if card_path.is_file():
                try:
                    card = json.loads(card_path.read_text(encoding="utf-8"))
                except Exception:
                    card = {}
            display = ((card.get("attrs") or {}).get("full_name")) or key
            kind, name, new_subject = "known", key, f"p:{key}"
            target = {"person": key, "display": display, "embeddings": n}
        database.relabel_subject(subject, subject=new_subject, kind=kind, name=name, display=display)
    return {"ok": True, "subject": subject, **target}


def _assign_guest(old_subject: str, guest: str, vec: np.ndarray, cfg: dict) -> dict[str, Any]:
    meta = _read_meta(cfg)
    items = [i for i in (meta.get("items") or []) if isinstance(i, dict)]
    guest = str(guest or "new").strip()
    if guest in ("", "new"):
        num = int(meta.get("next") or 1)
        guest_id = f"g{num}"
        meta["next"] = num + 1
        label = f"Гость {num}"
        items.append({"id": guest_id, "label": label, "n": 0, "subjects": []})
    else:
        guest_id = guest
        found = next((i for i in items if i.get("id") == guest_id), None)
        if found is None:
            raise ValueError("гость не найден")
        label = str(found.get("label") or guest_id)
    folder = _guest_dir(guest_id, cfg)
    n = _append_vec(folder, vec)
    for item in items:
        if item.get("id") == guest_id:
            item["n"] = n
            item["label"] = label
            subs = [str(s) for s in (item.get("subjects") or [])]
            if old_subject not in subs:
                subs.append(old_subject)
            item["subjects"] = subs
    meta["items"] = items
    _write_meta(meta, cfg)
    return {"guest": guest_id, "label": label, "embeddings": n, "subject": guest_id}


def _assign_review(
    subject: str,
    *,
    person: str | None,
    guest: str | None,
    cfg: dict,
) -> dict[str, Any]:
    """Эмбеддинг кадра из событий — в выбранную карточку. Меняется только этот снимок."""
    from qaraqu_face import db as database

    with _lock:
        items = _read_review(cfg)
        entry = next((i for i in items if str(i.get("subject")) == subject), None)
        if entry is None:
            raise ValueError("событие уже разобрано или не найдено")
        snap = str(entry.get("snapshot") or "")
        img = cv2.imread(str(_snap_file(snap, cfg)))
        if img is None:
            raise ValueError("снимок не читается")
        vec = _face_vec(img, cfg)
        if guest:
            target = _assign_guest(subject, guest, vec, cfg)
            kind, name, display, new_subject = "guest", "", target["label"], target["subject"]
        else:
            key = str(person or "").strip()
            folder = person_dir(key, cfg)
            if not folder.is_dir():
                raise ValueError("карточка не найдена")
            n = _append_vec(folder, vec)
            card = {}
            card_path = folder / "card.json"
            if card_path.is_file():
                try:
                    card = json.loads(card_path.read_text(encoding="utf-8"))
                except Exception:
                    card = {}
            display = ((card.get("attrs") or {}).get("full_name")) or key
            kind, name, new_subject = "known", key, f"p:{key}"
            target = {"person": key, "display": display, "embeddings": n}
        database.relabel_snapshot(snap, subject=new_subject, kind=kind, name=name, display=display)
        _write_review([i for i in items if str(i.get("subject")) != subject], cfg)
    return {"ok": True, "subject": subject, **target}
