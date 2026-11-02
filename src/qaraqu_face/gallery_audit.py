"""Проверка картотеки: дубли, карточки-«магниты», карточки без ФИО. Слияние — только по команде из UI."""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from qaraqu_face.config import load_config
from qaraqu_face.gallery import (
    MAX_VECS_PER_PERSON,
    commit_embs,
    list_people,
    person_dir,
    read_person_vecs,
)

_lock = threading.Lock()
# пара карточек с таким сходством — почти наверняка один человек
DUP_SIM = 0.60
# карточка, у которой столько чужих ближе NEAR_SIM, тянет на себя чужие лица
NEAR_SIM = 0.45
MAGNET_NEIGHBOURS = 4


def _mat(vecs: list[np.ndarray]) -> np.ndarray:
    m = np.stack([np.asarray(v, np.float32) for v in vecs])
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)


def audit(cfg: dict | None = None, *, track_hits: dict[str, int] | None = None) -> dict[str, Any]:
    """Пары-дубли, магниты, без ФИО. Ничего не меняет."""
    cfg = cfg or load_config()
    people = list_people(cfg)
    info = {p["name"]: p for p in people}
    mats: dict[str, np.ndarray] = {}
    for p in people:
        vecs = read_person_vecs(person_dir(p["name"], cfg))
        if vecs:
            mats[p["name"]] = _mat(vecs)
    names = list(mats)
    pairs: list[dict[str, Any]] = []
    near: dict[str, int] = {n: 0 for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            s = float((mats[a] @ mats[b].T).max())
            if s >= NEAR_SIM:
                near[a] += 1
                near[b] += 1
            if s >= DUP_SIM:
                pairs.append({"a": _brief(info[a]), "b": _brief(info[b]), "score": round(s, 3)})
    pairs.sort(key=lambda x: -x["score"])
    magnets = []
    hits = track_hits or {}
    for n in names:
        if near[n] >= MAGNET_NEIGHBOURS or hits.get(n, 0) >= 6:
            row = _brief(info[n])
            row["neighbours"] = near[n]
            row["tracks_hour"] = int(hits.get(n, 0))
            magnets.append(row)
    magnets.sort(key=lambda x: (-x["tracks_hour"], -x["neighbours"]))
    nameless = [_brief(p) for p in people if _nameless(p)]
    return {
        "ok": True,
        "total": len(people),
        "pairs": pairs,
        "magnets": magnets,
        "nameless": nameless,
        "nameless_count": len(nameless),
    }


def _nameless(p: dict[str, Any]) -> bool:
    a = p.get("attrs") or {}
    return not (a.get("full_name") or a.get("last_name") or a.get("first_name"))


def _brief(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": p["name"],
        "display": p.get("display") or p["name"],
        "thumb": p.get("thumb"),
        "embeddings": int(p.get("embeddings") or 0),
        "photos": int(p.get("photos") or 0),
    }


def merge(keep: str, drop: str, cfg: dict | None = None) -> dict[str, Any]:
    """Перенести векторы и фото из drop в keep, карточку drop удалить. Фото keep не трогаются."""
    cfg = cfg or load_config()
    if not keep or not drop or keep == drop:
        raise ValueError("нужны две разные карточки")
    with _lock:
        k_dir, d_dir = person_dir(keep, cfg), person_dir(drop, cfg)
        if not k_dir.is_dir() or not d_dir.is_dir():
            raise ValueError("карточка не найдена")
        k_vecs = read_person_vecs(k_dir)
        d_vecs = read_person_vecs(d_dir)
        merged = list(k_vecs)
        km = _mat(k_vecs) if k_vecs else None
        added = 0
        for v in d_vecs:
            vn = np.asarray(v, np.float32)
            vn = vn / (np.linalg.norm(vn) + 1e-8)
            # точные копии не плодим
            if km is not None and float((km @ vn).max()) >= 0.995:
                continue
            merged.append(v)
            added += 1
        if len(merged) > MAX_VECS_PER_PERSON:
            merged = merged[:MAX_VECS_PER_PERSON]
        commit_embs(k_dir, merged)
        moved = 0
        for p in d_dir.iterdir():
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            target = k_dir / p.name
            if target.exists():
                target = k_dir / f"{p.stem}_{drop[:8]}{p.suffix}"
            shutil.move(str(p), str(target))
            moved += 1
        # имя: если у keep нет ФИО, а у drop есть — берём
        kc, dc = _card(k_dir), _card(d_dir)
        ka = kc.get("attrs") if isinstance(kc.get("attrs"), dict) else {}
        da = dc.get("attrs") if isinstance(dc.get("attrs"), dict) else {}
        if not (ka.get("full_name") or ka.get("last_name")) and (da.get("full_name") or da.get("last_name")):
            ka.update({k: v for k, v in da.items() if v})
        kc["attrs"] = ka
        kc["updated_ts"] = time.time()
        kc.setdefault("merged_from", []).append(drop)
        (k_dir / "card.json").write_text(json.dumps(kc, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(d_dir)
    return {"ok": True, "keep": keep, "dropped": drop, "vectors": len(merged), "added": added, "photos_moved": moved}


def _card(folder: Path) -> dict[str, Any]:
    p = folder / "card.json"
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"name": folder.name, "attrs": {}}
