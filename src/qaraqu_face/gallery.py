"""Файловая картотека в формате FINDFACE: embedding.npy + embeddings.npy (несколько векторов)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qaraqu_face.config import load_config, resolve
from qaraqu_face.face_aug import augment_with_landmarks
from qaraqu_face.recognize import get_embedder, l2

# совместимость: runtime ждёт dict[str, ndarray|list]
GalleryVecs = dict[str, list[np.ndarray]]


def gallery_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    d = resolve(cfg, cfg.get("recognize", {}).get("gallery_dir", "data/gallery"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def person_dir(name: str, cfg: dict | None = None) -> Path:
    """Каталог карточки строго внутри галереи; имя с путём/служебное — ValueError."""
    key = os.path.basename(str(name or "").strip())
    if not key or key != str(name).strip() or key.startswith(".") or "\x00" in key or "\\" in key:
        raise ValueError("некорректное имя карточки")
    root = os.path.realpath(str(gallery_dir(cfg)))
    path = os.path.realpath(os.path.join(root, key))
    if not path.startswith(root + os.sep):
        raise ValueError("некорректное имя карточки")
    return Path(path)


def _read_card(person: Path) -> dict[str, Any]:
    from qaraqu_face import db

    card = db.get_card(person.name)
    if card is not None:
        return card
    p = person / "card.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"name": person.name, "attrs": {}}


def _write_card(person: Path, card: dict[str, Any]) -> None:
    from qaraqu_face import db

    db.set_card(person.name, card)


def _norm(v: np.ndarray) -> np.ndarray:
    return l2(np.asarray(v, dtype=np.float32).reshape(-1))


def _emb_names(backend: str) -> tuple[str, str]:
    """ArcFace — исторические embeddings.npy. AdaFace их не затирает."""
    if str(backend).lower() == "adaface":
        return "embedding_ada.npy", "embeddings_ada.npy"
    return "embedding.npy", "embeddings.npy"


def read_person_vecs(person_dir: Path, dim: int = 512, *, backend: str = "arcface") -> list[np.ndarray]:
    """Читает embeddings*.npy (N×D) или fallback embedding*.npy — как FINDFACE."""
    out: list[np.ndarray] = []
    single_name, multi_name = _emb_names(backend)
    multi = person_dir / multi_name
    single = person_dir / single_name
    if multi.is_file():
        try:
            arr = np.load(str(multi)).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for row in arr:
                v = np.asarray(row, dtype=np.float32).reshape(-1)
                if v.size == dim:
                    out.append(_norm(v))
        except Exception:
            out = []
    if not out and single.is_file():
        try:
            v = np.load(str(single)).astype(np.float32).reshape(-1)
            if v.size == dim:
                out.append(_norm(v))
        except Exception:
            pass
    return out


# Наблюдённые векторы: лицо с камеры (шапка, очки, взгляд вниз), которого нет на фото карточки.
# Лежат отдельно от фото-векторов (observed_<backend>.npy), пересборка по фото их не трогает,
# добавляются только при уверенном совпадении (runtime) или ручной правке в «Неизвестных».
MAX_OBSERVED_PER_PERSON = 12
_OBSERVED_DUP_SIM = 0.95
_observed_lock = threading.Lock()


def _backend_tag(cfg: dict | None) -> str:
    rec = (cfg or load_config()).get("recognize", {})
    return str(rec.get("backend") or "adaface").lower() if isinstance(rec, dict) else "adaface"


def observed_path(person_dir: Path, cfg: dict | None = None, backend: str | None = None) -> Path:
    tag = str(backend or _backend_tag(cfg)).lower()
    return person_dir / f"observed_{tag}.npy"


def read_observed_vecs(
    person_dir: Path, cfg: dict | None = None, dim: int = 512, *, backend: str | None = None
) -> list[np.ndarray]:
    p = observed_path(person_dir, cfg, backend)
    if not p.is_file():
        return []
    try:
        arr = np.load(str(p)).astype(np.float32)
    except Exception:
        return []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [_norm(r) for r in arr if r.size == dim]


def add_observed_vec(
    name: str, vec: np.ndarray, cfg: dict | None = None, *, backend: str | None = None
) -> dict[str, Any]:
    """Добавить камерный вектор к карточке. Почти такой же уже есть — не дублируем; старше 12 — вытесняются."""
    cfg = cfg or load_config()
    person = person_dir(name, cfg)
    if not person.is_dir():
        raise FileNotFoundError(name)
    v = _norm(vec)
    with _observed_lock:
        cur = read_observed_vecs(person, cfg, backend=backend)
        if any(float(np.dot(v, c)) >= _OBSERVED_DUP_SIM for c in cur):
            return {"ok": True, "added": False, "observed": len(cur)}
        cur.append(v)
        cur = cur[-MAX_OBSERVED_PER_PERSON:]
        np.save(str(observed_path(person, cfg, backend)), np.stack(cur).astype(np.float32))
    return {"ok": True, "added": True, "observed": len(cur)}


def clear_observed(name: str, cfg: dict | None = None) -> bool:
    p = observed_path(person_dir(name, cfg), cfg)
    if p.is_file():
        p.unlink()
        return True
    return False


def commit_embs(person_dir: Path, vecs: list[np.ndarray], backend: str = "arcface") -> list[np.ndarray]:
    clean = [_norm(v) for v in vecs if v is not None and getattr(v, "size", 0)]
    if not clean:
        return []
    person_dir.mkdir(parents=True, exist_ok=True)
    single_name, multi_name = _emb_names(backend)
    np.save(str(person_dir / single_name), clean[0].astype(np.float32))
    np.save(str(person_dir / multi_name), np.stack(clean, axis=0).astype(np.float32))
    return clean


MAX_VECS_PER_PERSON = 64
# аугментация, ушедшая от оригинала дальше этого, — мусор для картотеки
_AUG_MIN_SIM = 0.5
# фото в одной карточке должны быть одним человеком (ArcFace: разные люди ≲0.3)
_SAME_PERSON_SIM = 0.4


def embed_with_augs(
    bgr: np.ndarray,
    cfg: dict | None = None,
    n_aug: int = 8,
    *,
    landmarks=None,
    xyxy=None,
    backend: str | None = None,
) -> list[np.ndarray]:
    """Оригинал + аугментации с пересчитанными landmarks → несколько эмбеддингов.

    Пустой список, если на фото нет лица с настоящими глазами/носом/ртом: вектор «из всего кадра»
    давал карточки, которые совпадали с подушками и сумками.
    """
    from qaraqu_face.detect import best_valid_face

    emb = get_embedder(cfg, backend)
    lm, box = landmarks, xyxy
    if lm is None:
        face = best_valid_face(bgr, cfg)
        if face is None:
            return []
        lm, box = face["landmarks"], face["xyxy"]
    base = _norm(emb.embed_bgr(bgr, landmarks=lm, xyxy=box))
    vecs: list[np.ndarray] = [base]
    for img, lm_aug in augment_with_landmarks(bgr, lm, n_aug):
        try:
            v = _norm(emb.embed_bgr(img, landmarks=lm_aug, xyxy=box))
        except Exception:
            continue
        if float(np.dot(v, base)) >= _AUG_MIN_SIM:
            vecs.append(v)
    return vecs


def source_photos(person_dir: Path) -> list[Path]:
    """Исходные фото карточки: без кропов, synth_* и точных дублей."""
    seen: set[str] = set()
    out: list[Path] = []
    names = sorted(person_dir.iterdir(), key=lambda p: (p.name != "source.jpg", p.name))
    for p in names:
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or not p.is_file():
            continue
        if p.name.endswith("_crop.jpg") or p.name.startswith("synth_"):
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(p)
    return out


def _cap_vectors(bases: list[np.ndarray], augs: list[np.ndarray], limit: int = MAX_VECS_PER_PERSON) -> list[np.ndarray]:
    room = max(0, limit - len(bases))
    if len(augs) > room:
        idx = np.linspace(0, len(augs) - 1, room).astype(int) if room else []
        augs = [augs[i] for i in idx]
    return bases + augs


def rebuild_person(
    person_dir: Path, cfg: dict | None = None, n_aug: int = 8, *, apply: bool = True, backend: str | None = None
) -> dict[str, Any]:
    """Пересобрать embeddings.npy карточки по её фото.

    Статусы: ok, no_face (ни на одном фото нет валидного лица), no_photos.
    Фото другого человека (низкое сходство с опорным) в вектора не берётся и попадает в dropped.
    """
    name = person_dir.name
    backend = str(backend or _backend_tag(cfg)).lower()
    photos = source_photos(person_dir)
    if not photos:
        return {"name": name, "status": "no_photos", "photos": 0, "dropped": [], "vectors": 0}
    items: list[tuple[Path, list[np.ndarray]]] = []
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        vecs = embed_with_augs(img, cfg, n_aug, backend=backend)
        if vecs:
            items.append((p, vecs))
    if not items:
        return {"name": name, "status": "no_face", "photos": len(photos), "dropped": [p.name for p in photos], "vectors": 0}
    anchor = items[0][1][0]
    kept: list[tuple[Path, list[np.ndarray]]] = []
    dropped: list[str] = [p.name for p in photos if p not in {i[0] for i in items}]
    for p, vecs in items:
        if float(np.dot(vecs[0], anchor)) >= _SAME_PERSON_SIM:
            kept.append((p, vecs))
        else:
            dropped.append(p.name)
    bases = [v[0] for _, v in kept]
    augs = [x for _, v in kept for x in v[1:]]
    final = _cap_vectors(bases, augs)
    if apply:
        commit_embs(person_dir, final, backend)
        card = _read_card(person_dir)
        card["embeddings_rebuilt_ts"] = time.time()
        _write_card(person_dir, card)
    return {
        "name": name,
        "status": "ok",
        "photos": len(photos),
        "used": [p.name for p, _ in kept],
        "dropped": dropped,
        "vectors": len(final),
    }


def backup_gallery(cfg: dict | None = None) -> Path:
    """Копия векторов и card.json (без фото) перед пересборкой."""
    root = gallery_dir(cfg)
    dst = root.parent / "gallery_backup" / time.strftime("%Y%m%d_%H%M%S")
    for person in root.iterdir():
        if not person.is_dir() or person.name.startswith("."):
            continue
        (dst / person.name).mkdir(parents=True, exist_ok=True)
        for fn in (
            "embedding.npy",
            "embeddings.npy",
            "embedding_ada.npy",
            "embeddings_ada.npy",
            "card.json",
        ):
            if (person / fn).is_file():
                shutil.copy2(person / fn, dst / person.name / fn)
    return dst


def quarantine_person(person_dir: Path, cfg: dict | None = None) -> Path:
    """Убрать карточку из распознавания, не удаляя данные."""
    qdir = gallery_dir(cfg) / ".quarantine"
    qdir.mkdir(exist_ok=True)
    dst = qdir / person_dir.name
    shutil.move(str(person_dir), str(dst))
    return dst


def rebuild_gallery(
    cfg: dict | None = None,
    n_aug: int = 8,
    *,
    apply: bool = True,
    quarantine_no_face: bool = False,
    backends: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Пересобрать эмбеддинги по фото. По умолчанию обе модели: ArcFace и AdaFace."""
    root = gallery_dir(cfg)
    tags = [str(b).lower() for b in (backends or ("arcface", "adaface"))]
    report: list[dict[str, Any]] = []
    for person in sorted(root.iterdir()):
        if not person.is_dir() or person.name.startswith("."):
            continue
        by_b: dict[str, dict[str, Any]] = {}
        for tag in tags:
            by_b[tag] = rebuild_person(person, cfg, n_aug, apply=apply, backend=tag)
        statuses = [r.get("status") for r in by_b.values()]
        status = "ok" if "ok" in statuses else (statuses[0] if statuses else "no_photos")
        row: dict[str, Any] = {
            "name": person.name,
            "status": status,
            "photos": max((int(r.get("photos") or 0) for r in by_b.values()), default=0),
            "vectors": sum(int(r.get("vectors") or 0) for r in by_b.values()),
            "vectors_arc": int((by_b.get("arcface") or {}).get("vectors") or 0),
            "vectors_ada": int((by_b.get("adaface") or {}).get("vectors") or 0),
            "backends": by_b,
            "dropped": sorted({d for r in by_b.values() for d in (r.get("dropped") or [])}),
        }
        if apply and quarantine_no_face and status in ("no_face", "no_photos"):
            row["quarantined"] = str(quarantine_person(person, cfg))
        report.append(row)
    return report


_PEOPLE_CACHE: dict[str, tuple[tuple, list[dict[str, Any]]]] = {}


def _gallery_signature(root: Path) -> tuple:
    """Сигнатура картотеки: имя, mtime и размер каждого файла. Любая запись меняет её."""
    sig = []
    for person in sorted(root.iterdir()):
        if not person.is_dir() or person.name.startswith("."):
            continue
        for f in sorted(person.iterdir()):
            try:
                st = f.stat()
            except OSError:
                continue
            sig.append((person.name, f.name, st.st_mtime_ns, st.st_size))
    return tuple(sig)


def list_people(cfg: dict | None = None, q: str = "") -> list[dict[str, Any]]:
    root = gallery_dir(cfg)
    qn = (q or "").strip().lower()
    key = str(root)
    sig = _gallery_signature(root)
    cached = _PEOPLE_CACHE.get(key)
    if cached is not None and cached[0] == sig:
        return _filter_people(cached[1], qn)
    out: list[dict[str, Any]] = []
    for person in sorted(root.iterdir()):
        if not person.is_dir() or person.name.startswith("."):
            continue
        arc = read_person_vecs(person, backend="arcface")
        ada = read_person_vecs(person, backend="adaface")
        vecs = arc + ada
        if not vecs:
            continue
        card = _read_card(person)
        attrs = card.get("attrs") if isinstance(card.get("attrs"), dict) else {}
        display = (
            attrs.get("full_name")
            or " ".join(x for x in (attrs.get("last_name"), attrs.get("first_name"), attrs.get("patronymic")) if x)
            or f"Без ФИО · {person.name[:8]}"
        )
        photos = [p for p in list(person.glob("*.jpg")) + list(person.glob("*.png")) if not p.name.endswith("_crop.jpg")]
        # оригинал раньше synth_* — для превью карточки
        photos_sorted = sorted(
            photos,
            key=lambda p: (1 if p.name.startswith("synth_") else 0, p.name),
        )
        photo_items = []
        for p in photos_sorted:
            kind = "synth" if p.name.startswith("synth_") else "photo"
            tag = p.stem
            if tag.startswith("synth_"):
                tag = tag[len("synth_") :]
                # убрать timestamp хвост _1791...
                parts = tag.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    tag = parts[0]
            photo_items.append(
                {
                    "file": p.name,
                    "url": f"/api/gallery/photo/{person.name}/{p.name}",
                    "kind": kind,
                    "tag": tag,
                }
            )
        thumb = None
        for cand in ("source.jpg",) + tuple(p.name for p in photos_sorted):
            # на карточке — только оригинал/фронт, не synth_*
            if cand.startswith("synth_"):
                continue
            if (person / cand).is_file():
                thumb = f"/api/gallery/photo/{person.name}/{cand}"
                break
        if thumb is None:
            # fallback: synth_front, иначе любое
            for p in photos_sorted:
                if "front" in p.stem.lower() and p.name.startswith("synth_"):
                    thumb = f"/api/gallery/photo/{person.name}/{p.name}"
                    break
            if thumb is None and photos_sorted:
                thumb = f"/api/gallery/photo/{person.name}/{photos_sorted[0].name}"
        row = {
            "name": person.name,
            "display": display,
            "photos": len(photos),
            "photo_items": photo_items,
            "embeddings": len(vecs),
            "embeddings_arc": len(arc),
            "embeddings_ada": len(ada),
            "observed": len(read_observed_vecs(person, cfg)),
            "thumb": thumb,
            "attrs": attrs,
            "excluded": bool(card.get("excluded")),
        }
        out.append(row)
    _PEOPLE_CACHE[key] = (sig, out)
    return _filter_people(out, qn)


def _filter_people(rows: list[dict[str, Any]], qn: str) -> list[dict[str, Any]]:
    keep: list[dict[str, Any]] = []
    for r in rows:
        attrs = r.get("attrs") if isinstance(r.get("attrs"), dict) else {}
        hay = f"{r.get('display', '')} {attrs.get('iin', '')} {r.get('name', '')}".lower()
        if not qn or qn in hay:
            keep.append(dict(r))
    return keep


def load_gallery(cfg: dict | None = None, backend: str = "arcface") -> GalleryVecs:
    """name → список L2-векторов одной модели. Карточки с excluded=true в 1:N не входят."""
    root = gallery_dir(cfg)
    out: GalleryVecs = {}
    if not root.is_dir():
        return out
    for person in root.iterdir():
        if not person.is_dir() or person.name.startswith("."):
            continue
        if _read_card(person).get("excluded"):
            continue
        vecs = read_person_vecs(person, backend=backend)
        if vecs:
            out[person.name] = vecs + read_observed_vecs(person, cfg, backend=backend)
    return out


def _single_vec(bgr: np.ndarray, cfg: dict | None, backend: str | None = None) -> list[np.ndarray]:
    from qaraqu_face.detect import best_valid_face

    face = best_valid_face(bgr, cfg)
    if face is None:
        return []
    return [_norm(get_embedder(cfg, backend).embed_bgr(bgr, landmarks=face["landmarks"], xyxy=face["xyxy"]))]


def enroll(
    name: str,
    bgr: np.ndarray,
    cfg: dict | None = None,
    attrs: dict | None = None,
    *,
    append: bool = True,
    with_augs: bool = True,
) -> dict[str, Any]:
    """
    Добавить фото в карточку.
    append=True — копит эмбеддинги (embeddings.npy), не затирает предыдущие.
    with_augs=True — как FINDFACE: оригинал + аугментации.
    """
    cfg = cfg or load_config()
    person = person_dir(name.strip() or uuid.uuid4().hex, cfg)
    return _enroll_into(person, bgr, cfg, attrs, append=append, with_augs=with_augs)


def create_person(bgr: np.ndarray, attrs: dict | None = None, cfg: dict | None = None) -> dict[str, Any]:
    """Новая карточка: ключ всегда генерируется на сервере."""
    cfg = cfg or load_config()
    return _enroll_into(person_dir(uuid.uuid4().hex, cfg), bgr, cfg, attrs, append=False, with_augs=True)


def _enroll_into(
    person: Path,
    bgr: np.ndarray,
    cfg: dict,
    attrs: dict | None,
    *,
    append: bool,
    with_augs: bool,
) -> dict[str, Any]:
    key = person.name
    person.mkdir(parents=True, exist_ok=True)
    # обе модели: ArcFace узнаёт импортированную картотеку, AdaFace — запас, если Arc не совпал
    committed: list[np.ndarray] = []
    for backend in ("arcface", "adaface"):
        try:
            new_vecs = (
                embed_with_augs(bgr, cfg, backend=backend) if with_augs else _single_vec(bgr, cfg, backend)
            )
        except Exception:
            new_vecs = []
        if not new_vecs:
            continue
        existing: list[np.ndarray] = read_person_vecs(person, backend=backend) if append else []
        all_vecs = existing + new_vecs
        if len(all_vecs) > MAX_VECS_PER_PERSON:
            all_vecs = all_vecs[-MAX_VECS_PER_PERSON:]
        committed = commit_embs(person, all_vecs, backend) or committed
    if not committed:
        return {"ok": False, "error": "на фото не найдено лицо (нужны видны глаза, нос и рот)", "name": key}

    ts = int(time.time() * 1000)
    photo = person / f"{ts}.jpg"
    cv2.imwrite(str(photo), bgr)

    from qaraqu_face import db

    if db.get_card(person.name) is not None or (person / "card.json").is_file():
        card = _read_card(person)
        card["updated_ts"] = time.time()
        if attrs:
            a = card.get("attrs") if isinstance(card.get("attrs"), dict) else {}
            a.update({k: v for k, v in attrs.items() if v})
            full = a.get("full_name") or " ".join(
                x for x in (a.get("last_name"), a.get("first_name"), a.get("patronymic")) if x
            )
            if full:
                a["full_name"] = full
            card["attrs"] = a
    else:
        card = {
            "name": key,
            "created_ts": time.time(),
            "updated_ts": time.time(),
            "sightings": 0,
            "attrs": attrs or {},
        }
        if attrs:
            full = attrs.get("full_name") or " ".join(
                x for x in (attrs.get("last_name"), attrs.get("first_name"), attrs.get("patronymic")) if x
            )
            if full:
                card["attrs"]["full_name"] = full
    _write_card(person, card)
    return {
        "ok": True,
        "name": key,
        "display": (card.get("attrs") or {}).get("full_name") or key,
        "photo": str(photo),
        "dim": int(committed[0].size),
        "embeddings": len(committed),
        "added_vectors": len(new_vecs),
    }


def update_card(name: str, attrs: dict[str, Any], cfg: dict | None = None) -> dict[str, Any]:
    person = person_dir(name, cfg)
    if not person.is_dir():
        raise FileNotFoundError(name)
    card = _read_card(person)
    a = card.get("attrs") if isinstance(card.get("attrs"), dict) else {}
    for k in ("last_name", "first_name", "patronymic", "iin", "dob"):
        if k in attrs:
            a[k] = str(attrs.get(k) or "").strip()
    a["full_name"] = " ".join(x for x in (a.get("last_name"), a.get("first_name"), a.get("patronymic")) if x) or a.get(
        "full_name", ""
    )
    card["attrs"] = a
    if "excluded" in attrs:
        # галочка «исключить из распознавания»: карточка остаётся, но в 1:N не участвует
        card["excluded"] = bool(attrs.get("excluded"))
    card["updated_ts"] = time.time()
    _write_card(person, card)
    return {"ok": True, "name": name, "display": a.get("full_name") or name, "excluded": bool(card.get("excluded"))}


def delete_person(name: str, cfg: dict | None = None) -> None:
    import shutil

    person = person_dir(name, cfg)
    if person.is_dir():
        shutil.rmtree(person)


_DUP_CARD_SIM = 0.6


def _as_vecs(g: Any) -> list[np.ndarray]:
    if isinstance(g, np.ndarray) and g.ndim == 1:
        return [g]
    if isinstance(g, np.ndarray) and g.ndim == 2:
        return [g[i] for i in range(g.shape[0])]
    if isinstance(g, (list, tuple)):
        return [v for v in g if v is not None]
    return []


def _same_person_cards(a: Any, b: Any) -> bool:
    """Две карточки — вероятно один человек (любая пара векторов очень похожа)."""
    va, vb = _as_vecs(a), _as_vecs(b)
    if not va or not vb:
        return False
    ma = np.stack([_norm(v) for v in va])
    mb = np.stack([_norm(v) for v in vb])
    return float((ma @ mb.T).max()) >= _DUP_CARD_SIM


def delete_all_people(cfg: dict | None = None) -> int:
    root = gallery_dir(cfg)
    deleted = 0
    for child in sorted(root.iterdir()):
        if child.name.startswith(".") or not child.is_dir():
            continue
        delete_person(child.name, cfg)
        deleted += 1
    return deleted


RATIO_BAND = 0.08
RATIO_GAP = 0.12


def match(
    emb: np.ndarray,
    gallery: GalleryVecs | dict[str, np.ndarray] | None = None,
    *,
    cfg: dict | None = None,
    threshold: float | None = None,
    margin: float | None = None,
) -> dict[str, Any]:
    """1:N. Оценка карточки — среднее двух лучших её векторов.

    Чистый max по 20–30 векторам превращал карточку в «магнит»: один случайный вектор
    на 0.53 — и чужой человек получает имя. Два вектора сразу так не ошибаются,
    а точный камерный вектор (≈1.0) всё равно вытягивает среднее выше порога.
    """
    cfg = cfg or load_config()
    rec = cfg.get("recognize", {}) if isinstance(cfg.get("recognize"), dict) else {}
    thr = float(threshold if threshold is not None else rec.get("threshold", 0.58))
    mar = float(margin if margin is not None else rec.get("margin", 0.08))
    gal = gallery if gallery is not None else load_gallery(cfg)
    if not gal:
        return {"matched": False, "name": None, "score": None, "threshold": thr, "margin": mar}
    q = _norm(emb)
    scored: list[tuple[float, str]] = []
    for name, g in gal.items():
        vecs = _as_vecs(g)
        if not vecs:
            continue
        sims = sorted((float(np.dot(q, _norm(v))) for v in vecs), reverse=True)
        top = sims[:2]
        scored.append((sum(top) / len(top), name))
    scored.sort(reverse=True)
    best_name, best_score = (scored[0][1], scored[0][0]) if scored else (None, -1.0)
    # дубликаты карточек одного человека не конкуренты: иначе разрыв ≈0 и человек не распознаётся вовсе
    second = -1.0
    for score, name in scored[1:]:
        if best_score - score < mar and _same_person_cards(gal[best_name], gal[name]):
            continue
        second = score
        break
    gap = best_score - second if second >= 0 else best_score
    ok = best_name is not None and best_score >= thr and gap >= mar
    # «явный лидер»: чуть ниже порога, но с большим отрывом от второго. Чужой человек даёт
    # нескольким карточкам близкие 0.15–0.27 (отрыв ≤0.05), свой в шапке/очках — 0.32 при втором 0.17.
    # На 106 камерных кропах с исключённой своей карточкой ни один чужой это правило не прошёл.
    if not ok and best_name is not None and best_score >= thr - RATIO_BAND and gap >= RATIO_GAP:
        ok = True
    return {
        "matched": ok,
        "name": best_name if ok else None,
        "best": best_name,
        "score": round(best_score, 4) if best_name else None,
        "second": round(second, 4) if second >= 0 else None,
        "gap": round(gap, 4) if best_name else None,
        "threshold": thr,
        "margin": mar,
    }


def photo_path(name: str, filename: str, cfg: dict | None = None) -> Path:
    p = person_dir(name, cfg) / Path(filename).name
    if not p.is_file():
        raise FileNotFoundError(filename)
    return p
