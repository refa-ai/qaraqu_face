"""SQLite / PostgreSQL: камеры, настройки, тревоги."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from qaraqu_face.config import load_config, resolve

log = logging.getLogger("qaraqu_face.db")

_lock = threading.RLock()
_engine = None
_Session = None
_initialized = False


def _database_url(cfg: dict | None = None) -> str:
    env = (os.environ.get("DATABASE_URL") or os.environ.get("QARAQU_FACE_DATABASE_URL") or "").strip()
    if env:
        return env
    cfg = cfg or load_config()
    db = cfg.get("database") or {}
    url = str(db.get("url") or "").strip()
    if url:
        return url
    path = resolve(cfg, db.get("sqlite_path", "data/qaraqu_face.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def get_engine(cfg: dict | None = None):
    global _engine, _Session
    if _engine is not None:
        return _engine
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    url = _database_url(cfg)
    kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    _engine = create_engine(url, **kwargs)
    _Session = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine


def _is_sqlite() -> bool:
    return str(get_engine().url).startswith("sqlite")


@contextmanager
def session_scope(cfg: dict | None = None) -> Iterator[Any]:
    get_engine(cfg)
    assert _Session is not None
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db(cfg: dict | None = None) -> str:
    """Создать таблицы и при необходимости импортировать JSON."""
    global _initialized
    from sqlalchemy import text

    eng = get_engine(cfg)
    url = str(eng.url)
    pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if _is_sqlite() else "SERIAL PRIMARY KEY"
    with eng.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS cameras (
                    id {pk},
                    sort_index INTEGER NOT NULL DEFAULT 0,
                    name TEXT NOT NULL,
                    rtsp_url_main TEXT NOT NULL DEFAULT '',
                    rtsp_url_sub TEXT NOT NULL DEFAULT '',
                    location TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    roles_json TEXT NOT NULL DEFAULT '["faces"]',
                    lat DOUBLE PRECISION,
                    lon DOUBLE PRECISION,
                    extra_json TEXT NOT NULL DEFAULT '{{}}',
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS alerts (
                    id {pk},
                    ts DOUBLE PRECISION NOT NULL,
                    kind TEXT NOT NULL DEFAULT '',
                    name TEXT,
                    camera TEXT,
                    camera_index INTEGER,
                    score DOUBLE PRECISION,
                    payload_json TEXT NOT NULL DEFAULT '{{}}',
                    guard_action TEXT NOT NULL DEFAULT 'open',
                    created_at DOUBLE PRECISION NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS sightings (
                    id {pk},
                    ts DOUBLE PRECISION NOT NULL,
                    subject TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT,
                    display TEXT NOT NULL DEFAULT '',
                    camera_index INTEGER NOT NULL,
                    camera TEXT NOT NULL DEFAULT '',
                    track_id INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        if _is_sqlite():
            have = {r[1] for r in conn.execute(text("PRAGMA table_info(sightings)"))}
            for col, typ in (("snapshot", "TEXT"), ("u", "DOUBLE PRECISION"), ("box_h", "DOUBLE PRECISION")):
                if col not in have:
                    conn.execute(text(f"ALTER TABLE sightings ADD COLUMN {col} {typ}"))
        else:
            conn.execute(text("ALTER TABLE sightings ADD COLUMN IF NOT EXISTS snapshot TEXT"))
            conn.execute(text("ALTER TABLE sightings ADD COLUMN IF NOT EXISTS u DOUBLE PRECISION"))
            conn.execute(text("ALTER TABLE sightings ADD COLUMN IF NOT EXISTS box_h DOUBLE PRECISION"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS camera_settings (
                    camera_index INTEGER PRIMARY KEY,
                    det_conf DOUBLE PRECISION NOT NULL DEFAULT 0.38,
                    match_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.42,
                    min_face_side INTEGER NOT NULL DEFAULT 64,
                    detect_interval DOUBLE PRECISION NOT NULL DEFAULT 2.0,
                    motion_k DOUBLE PRECISION NOT NULL DEFAULT 3.0,
                    imgsz INTEGER NOT NULL DEFAULT 1280,
                    updated_at DOUBLE PRECISION NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS passages (
                    id {pk},
                    ts DOUBLE PRECISION NOT NULL,
                    subject TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT,
                    display TEXT NOT NULL DEFAULT '',
                    camera_index INTEGER NOT NULL,
                    camera TEXT NOT NULL DEFAULT '',
                    direction TEXT NOT NULL,
                    track_id INTEGER NOT NULL DEFAULT 0,
                    snapshot TEXT
                )
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_passages_ts ON passages(ts)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_sightings_ts ON sightings(ts)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_sightings_subject ON sightings(subject, ts)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_alerts_ts ON alerts(ts)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_alerts_kind ON alerts(kind)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_cameras_sort ON cameras(sort_index)"))
    if not _initialized:
        migrate_from_json(cfg)
        seed_runtime_config(cfg or load_config())
        if get_setting("service_enabled", None) is None:
            set_setting("service_enabled", False)
        _initialized = True
    log.info("db ready: %s", url)
    return url


def get_setting(key: str, default: Any = None) -> Any:
    from sqlalchemy import text

    with session_scope() as s:
        row = s.execute(text("SELECT value_json FROM settings WHERE key=:k"), {"k": key}).fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except Exception:
            return default


def set_setting(key: str, value: Any) -> Any:
    from sqlalchemy import text

    now = time.time()
    payload = json.dumps(value, ensure_ascii=False)
    with session_scope() as s:
        exists = s.execute(text("SELECT 1 FROM settings WHERE key=:k"), {"k": key}).fetchone()
        if exists:
            s.execute(
                text("UPDATE settings SET value_json=:v, updated_at=:t WHERE key=:k"),
                {"k": key, "v": payload, "t": now},
            )
        else:
            s.execute(
                text("INSERT INTO settings(key, value_json, updated_at) VALUES(:k,:v,:t)"),
                {"k": key, "v": payload, "t": now},
            )
    return value


RUNTIME_SECTIONS = ("detect", "recognize", "capture", "api")


CAM_SETTING_DEFAULTS: dict[str, Any] = {
    "det_conf": 0.38,
    "match_threshold": 0.42,
    "min_face_side": 64,
    "detect_interval": 2.0,
    "motion_k": 3.0,
    "imgsz": 1280,
}

CAM_SETTING_BOUNDS: dict[str, tuple[float, float]] = {
    "det_conf": (0.05, 0.95),
    "match_threshold": (0.10, 0.90),
    "min_face_side": (10.0, 300.0),
    "detect_interval": (0.1, 15.0),
    "motion_k": (1.0, 10.0),
}

CAM_IMGSZ_CHOICES = (640, 960, 1280, 1600, 1920)


def _cam_setting_typed(key: str, raw: Any) -> Any:
    if key not in CAM_SETTING_DEFAULTS:
        return None
    if key == "imgsz":
        try:
            v = int(round(float(raw)))
        except (TypeError, ValueError):
            return None
        # ближайшее разрешение из разрешённых — защита от произвольных imgsz из сети
        return min(CAM_IMGSZ_CHOICES, key=lambda c: (abs(c - v), c))
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    lo, hi = CAM_SETTING_BOUNDS[key]
    return min(hi, max(lo, v))


def get_camera_settings(index: int) -> dict[str, Any]:
    """Настройки одной камеры. Нет строки — камера ещё не настроена, отдаём значения по умолчанию."""
    from sqlalchemy import text

    out = dict(CAM_SETTING_DEFAULTS)
    with session_scope() as s:
        row = s.execute(
            text(
                "SELECT det_conf, match_threshold, min_face_side, detect_interval, motion_k, imgsz"
                " FROM camera_settings WHERE camera_index=:i"
            ),
            {"i": int(index)},
        ).fetchone()
    if row:
        for key, raw in zip(CAM_SETTING_DEFAULTS, row):
            typed = _cam_setting_typed(key, raw)
            if typed is not None:
                out[key] = typed
    out["min_face_side"] = int(round(out["min_face_side"]))
    out["imgsz"] = int(out["imgsz"])
    return out


def save_camera_settings(index: int, patch: dict[str, Any]) -> dict[str, Any]:
    """Частичное обновление: переданные ключи меняются, остальные остаются как были."""
    from sqlalchemy import text

    cur = get_camera_settings(index)
    for key in CAM_SETTING_DEFAULTS:
        if key in (patch or {}):
            typed = _cam_setting_typed(key, patch[key])
            if typed is not None:
                cur[key] = typed
    cur["min_face_side"] = int(round(cur["min_face_side"]))
    cur["imgsz"] = int(cur["imgsz"])
    now = time.time()
    with session_scope() as s:
        s.execute(
            text(
                "INSERT INTO camera_settings"
                "(camera_index, det_conf, match_threshold, min_face_side, detect_interval, motion_k, imgsz, updated_at)"
                " VALUES(:i,:d,:m,:f,:v,:k,:z,:t)"
                " ON CONFLICT(camera_index) DO UPDATE SET"
                " det_conf=:d, match_threshold=:m, min_face_side=:f, detect_interval=:v,"
                " motion_k=:k, imgsz=:z, updated_at=:t"
            ),
            {
                "i": int(index),
                "d": float(cur["det_conf"]),
                "m": float(cur["match_threshold"]),
                "f": int(cur["min_face_side"]),
                "v": float(cur["detect_interval"]),
                "k": float(cur["motion_k"]),
                "z": int(cur["imgsz"]),
                "t": now,
            },
        )
    return cur


def all_camera_settings() -> dict[int, dict[str, Any]]:
    from sqlalchemy import text

    out: dict[int, dict[str, Any]] = {}
    with session_scope() as s:
        rows = s.execute(text("SELECT camera_index FROM camera_settings")).fetchall()
    for (idx,) in rows:
        out[int(idx)] = get_camera_settings(int(idx))
    return out


def get_runtime_config() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sec in RUNTIME_SECTIONS:
        val = get_setting(f"config.{sec}")
        if isinstance(val, dict):
            out[sec] = val
    return out


def seed_runtime_config(cfg: dict[str, Any]) -> None:
    for sec in RUNTIME_SECTIONS:
        val = cfg.get(sec)
        if isinstance(val, dict) and get_setting(f"config.{sec}") is None:
            set_setting(f"config.{sec}", val)


def get_doc(name: str, default: Any = None) -> Any:
    return get_setting(f"doc.{name}", default)


def set_doc(name: str, value: Any) -> None:
    set_setting(f"doc.{name}", value)


def get_card(folder: str) -> dict[str, Any] | None:
    val = get_setting(f"card.{folder}")
    return val if isinstance(val, dict) else None


def set_card(folder: str, card: dict[str, Any]) -> None:
    set_setting(f"card.{folder}", card)


def get_options() -> dict[str, Any]:
    val = get_setting("options", {})
    return dict(val) if isinstance(val, dict) else {}


def save_options(patch: dict[str, Any]) -> dict[str, Any]:
    cur = get_options()
    cur.update(patch or {})
    set_setting("options", cur)
    return cur


def _cam_row_to_public(row: Any, index: int) -> dict[str, Any]:
    try:
        roles = json.loads(row.roles_json or "[]")
    except Exception:
        roles = ["faces"]
    try:
        extra = json.loads(row.extra_json or "{}")
    except Exception:
        extra = {}
    out = {
        "id": int(row.id),
        "index": index,
        "name": row.name,
        "location": row.location or "",
        "description": row.description or "",
        "rtsp_url_main": row.rtsp_url_main or "",
        "rtsp_url_sub": row.rtsp_url_sub or "",
        "lat": row.lat,
        "lon": row.lon,
        "roles": roles,
    }
    if isinstance(extra, dict):
        for k, v in extra.items():
            if k not in out:
                out[k] = v
    return out


def list_cameras() -> list[dict[str, Any]]:
    from sqlalchemy import text

    with session_scope() as s:
        rows = s.execute(text("SELECT * FROM cameras ORDER BY sort_index ASC, id ASC")).fetchall()
        return [_cam_row_to_public(r, int(r.id)) for r in rows]


def get_camera(index: int) -> dict[str, Any] | None:
    """Камера по постоянному id (он же index в API). Позиция в списке не влияет."""
    for cam in list_cameras():
        if cam["index"] == int(index):
            return cam
    return None


def add_camera(entry: dict[str, Any]) -> dict[str, Any]:
    from sqlalchemy import text

    now = time.time()
    name = str(entry.get("name") or "Камера").strip() or "Камера"
    main = str(entry.get("rtsp_url_main") or entry.get("url") or "").strip()
    if not main:
        raise ValueError("нужен rtsp_url_main")
    sub = str(entry.get("rtsp_url_sub") or "").strip()
    location = str(entry.get("location") or "").strip()
    description = str(entry.get("description") or "").strip()
    roles = entry.get("roles") or ["faces"]
    skip = {
        "name", "rtsp_url_main", "rtsp_url_sub", "url", "location",
        "description", "roles", "lat", "lon", "index", "id",
    }
    extra = {k: v for k, v in entry.items() if k not in skip}
    with _lock:
        with session_scope() as s:
            n = int(s.execute(text("SELECT COUNT(*) FROM cameras")).scalar() or 0)
            r = s.execute(
                text(
                    """
                    INSERT INTO cameras(
                        sort_index, name, rtsp_url_main, rtsp_url_sub, location, description,
                        roles_json, lat, lon, extra_json, created_at, updated_at
                    ) VALUES (
                        :si,:name,:main,:sub,:loc,:desc,:roles,:lat,:lon,:extra,:c,:u
                    )
                    """ + ("" if _is_sqlite() else " RETURNING id"),
                ),
                {
                    "si": n,
                    "name": name,
                    "main": main,
                    "sub": sub,
                    "loc": location,
                    "desc": description,
                    "roles": json.dumps(roles, ensure_ascii=False),
                    "lat": entry.get("lat"),
                    "lon": entry.get("lon"),
                    "extra": json.dumps(extra, ensure_ascii=False),
                    "c": now,
                    "u": now,
                },
            )
            if _is_sqlite():
                row_id = int(r.lastrowid or 0)
            else:
                row_id = int(r.fetchone()[0])
        return get_camera(row_id) or {"index": row_id, "name": name}


def update_camera(index: int, entry: dict[str, Any]) -> dict[str, Any]:
    from sqlalchemy import text

    cur = get_camera(index)
    if cur is None:
        raise IndexError("камера не найдена")
    cam_id = cur["id"]
    now = time.time()
    fields = {
        "name": str(entry["name"]).strip() if "name" in entry and entry["name"] is not None else cur["name"],
        "rtsp_url_main": (
            str(entry.get("rtsp_url_main") or entry.get("url")).strip()
            if ("rtsp_url_main" in entry or "url" in entry)
            else cur["rtsp_url_main"]
        ),
        "rtsp_url_sub": (
            str(entry["rtsp_url_sub"]).strip()
            if "rtsp_url_sub" in entry and entry["rtsp_url_sub"] is not None
            else cur.get("rtsp_url_sub") or ""
        ),
        "location": (
            str(entry["location"]).strip()
            if "location" in entry and entry["location"] is not None
            else cur.get("location") or ""
        ),
        "description": (
            str(entry["description"]).strip()
            if "description" in entry and entry["description"] is not None
            else cur.get("description") or ""
        ),
    }
    with session_scope() as s:
        s.execute(
            text(
                """
                UPDATE cameras SET
                    name=:name, rtsp_url_main=:rtsp_url_main, rtsp_url_sub=:rtsp_url_sub,
                    location=:location, description=:description, updated_at=:u
                WHERE id=:id
                """
            ),
            {**fields, "u": now, "id": cam_id},
        )
    out = get_camera(index)
    if not out:
        raise IndexError("камера не найдена")
    return out


def delete_camera(index: int) -> None:
    from sqlalchemy import text

    cur = get_camera(index)
    if cur is None:
        raise IndexError("камера не найдена")
    with _lock:
        with session_scope() as s:
            s.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": cur["id"]})
            s.execute(text("DELETE FROM camera_settings WHERE camera_index=:i"), {"i": int(index)})


def insert_alert(ev: dict[str, Any]) -> int:
    from sqlalchemy import text

    now = time.time()
    ts = float(ev.get("ts") or now)
    payload = {
        k: v
        for k, v in ev.items()
        if k not in ("ts", "kind", "name", "camera", "camera_index", "score", "guard_action")
    }
    with session_scope() as s:
        r = s.execute(
            text(
                """
                INSERT INTO alerts(ts, kind, name, camera, camera_index, score, payload_json, guard_action, created_at)
                VALUES(:ts,:kind,:name,:camera,:ci,:score,:payload,:ga,:c)
                RETURNING id
                """
                if not _is_sqlite()
                else """
                INSERT INTO alerts(ts, kind, name, camera, camera_index, score, payload_json, guard_action, created_at)
                VALUES(:ts,:kind,:name,:camera,:ci,:score,:payload,:ga,:c)
                """
            ),
            {
                "ts": ts,
                "kind": str(ev.get("kind") or ""),
                "name": ev.get("name"),
                "camera": ev.get("camera"),
                "ci": ev.get("camera_index"),
                "score": ev.get("score"),
                "payload": json.dumps(payload, ensure_ascii=False),
                "ga": str(ev.get("guard_action") or "open"),
                "c": now,
            },
        )
        if not _is_sqlite():
            row = r.fetchone()
            return int(row[0]) if row else 0
        return int(r.lastrowid or 0)


def list_alerts(
    *,
    hours: int | None = 720,
    kind: str = "",
    q: str = "",
    limit: int = 2000,
) -> list[dict[str, Any]]:
    from sqlalchemy import text

    now = time.time()
    horizon = None if not hours else now - max(0, int(hours)) * 3600
    qn = (q or "").strip().lower()
    sql = "SELECT * FROM alerts WHERE 1=1"
    params: dict[str, Any] = {"lim": int(limit)}
    if horizon is not None:
        sql += " AND ts >= :h"
        params["h"] = horizon
    if kind:
        sql += " AND kind = :kind"
        params["kind"] = kind
    sql += " ORDER BY ts DESC LIMIT :lim"
    with session_scope() as s:
        rows = s.execute(text(sql), params).fetchall()
    out = []
    for r in rows:
        try:
            p = json.loads(r.payload_json or "{}")
        except Exception:
            p = {}
        display = p.get("display") or r.name or "неизвестен"
        hay = f"{display} {r.name or ''} {r.camera or ''} {p.get('iin') or ''}".lower()
        if qn and qn not in hay:
            continue
        out.append(
            {
                "id": int(r.id),
                "ts": r.ts,
                "kind": r.kind,
                "name": r.name,
                "display": display,
                "subject": p.get("subject"),
                "track_id": p.get("track_id"),
                "iin": p.get("iin") or "",
                "camera": r.camera,
                "camera_index": r.camera_index,
                "similarity": r.score,
                "snapshot": p.get("snapshot") or None,
                "thumb": f"/api/snapshot/{p['snapshot']}" if p.get("snapshot") else None,
                "gallery_thumb": p.get("gallery_thumb"),
            }
        )
    return out


def insert_sighting(row: dict[str, Any]) -> None:
    from sqlalchemy import text

    with session_scope() as s:
        s.execute(
            text(
                """
                INSERT INTO sightings(ts, subject, kind, name, display, camera_index, camera, track_id, snapshot, u, box_h)
                VALUES(:ts,:subject,:kind,:name,:display,:ci,:camera,:tid,:snap,:u,:bh)
                """
            ),
            {
                "ts": float(row.get("ts") or time.time()),
                "subject": str(row["subject"]),
                "kind": str(row.get("kind") or ""),
                "name": row.get("name"),
                "display": str(row.get("display") or ""),
                "ci": int(row["camera_index"]),
                "camera": str(row.get("camera") or ""),
                "tid": int(row.get("track_id") or 0),
                "snap": row.get("snapshot"),
                "u": row.get("u"),
                "bh": row.get("box_h"),
            },
        )


def relabel_sightings(
    *,
    old_subject: str,
    camera_index: int,
    track_id: int,
    since: float,
    subject: str,
    name: str | None,
    display: str,
) -> int:
    from sqlalchemy import text

    with session_scope() as s:
        r = s.execute(
            text(
                """
                UPDATE sightings SET subject=:s, kind='known', name=:n, display=:d
                WHERE subject=:old AND camera_index=:ci AND track_id=:tid AND ts>=:since
                """
            ),
            {"s": subject, "n": name, "d": display, "old": old_subject, "ci": int(camera_index), "tid": int(track_id), "since": float(since)},
        )
        return int(r.rowcount or 0)


def insert_passage(row: dict[str, Any]) -> None:
    from sqlalchemy import text

    with session_scope() as s:
        s.execute(
            text(
                """
                INSERT INTO passages(ts, subject, kind, name, display, camera_index, camera, direction, track_id, snapshot)
                VALUES(:ts,:subject,:kind,:name,:display,:ci,:camera,:direction,:tid,:snap)
                """
            ),
            {
                "ts": float(row.get("ts") or time.time()),
                "subject": str(row["subject"]),
                "kind": str(row.get("kind") or ""),
                "name": row.get("name"),
                "display": str(row.get("display") or ""),
                "ci": int(row["camera_index"]),
                "camera": str(row.get("camera") or ""),
                "direction": str(row["direction"]),
                "tid": int(row.get("track_id") or 0),
                "snap": row.get("snapshot"),
            },
        )


def relabel_passages(
    *,
    old_subject: str,
    camera_index: int,
    track_id: int,
    since: float,
    subject: str,
    kind: str,
    name: str | None,
    display: str,
) -> int:
    from sqlalchemy import text

    with session_scope() as s:
        r = s.execute(
            text(
                """
                UPDATE passages SET subject=:s, kind=:k, name=:n, display=:d
                WHERE subject=:old AND camera_index=:ci AND track_id=:tid AND ts>=:since
                """
            ),
            {"s": subject, "k": kind, "n": name, "d": display, "old": old_subject, "ci": int(camera_index), "tid": int(track_id), "since": float(since)},
        )
        return int(r.rowcount or 0)


def list_passages(*, hours: int = 24, q: str = "", limit: int = 2000) -> list[dict[str, Any]]:
    """Журнал проходов, новые сверху. q — поиск по имени, номеру неизвестного или камере."""
    from sqlalchemy import text

    since = time.time() - max(1, int(hours)) * 3600
    sql = (
        "SELECT id, ts, subject, kind, display, camera_index, camera, direction, track_id, snapshot "
        "FROM passages WHERE ts>=:since ORDER BY ts DESC LIMIT :lim"
    )
    with session_scope() as s:
        rows = [dict(r._mapping) for r in s.execute(text(sql), {"since": since, "lim": int(limit)}).fetchall()]
    needle = (q or "").strip().lower()
    if needle:
        rows = [r for r in rows if needle in f"{r['display']} {r['camera']} {r['subject']}".lower()]
    return rows


def passage_summary(*, hours: int = 24) -> list[dict[str, Any]]:
    """По каждому человеку и дню: приход (первый вход), уход (последний выход) и время в объекте.

    Сессия — от входа до следующего выхода. Повторный вход без выхода не открывает новую
    сессию: человек уже внутри. Открытая сессия (вход без выхода) считается до текущего момента.
    Сессия относится к дню своего входа.
    """
    now = time.time()
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for r in list_passages(hours=hours, limit=50000):
        by_subject.setdefault(r["subject"], []).append(r)
    out: list[dict[str, Any]] = []
    for subject, rows in by_subject.items():
        rows.sort(key=lambda x: x["ts"])
        days: dict[str, dict[str, Any]] = {}
        open_in: float | None = None
        for r in rows:
            ts = float(r["ts"])
            day = time.strftime("%Y-%m-%d", time.localtime(ts))
            d = days.setdefault(
                day,
                {"subject": subject, "kind": r["kind"], "display": r["display"], "date": day,
                 "arrival": None, "departure": None, "sessions": 0, "total_s": 0.0, "inside": False},
            )
            d["kind"] = r["kind"]
            d["display"] = r["display"]
            if r["direction"] == "in":
                if open_in is None:
                    open_in = ts
                    d["sessions"] += 1
                    if d["arrival"] is None:
                        d["arrival"] = ts
            else:
                if open_in is not None:
                    d["total_s"] += ts - open_in
                    open_in = None
                d["departure"] = ts
        if open_in is not None:
            day = time.strftime("%Y-%m-%d", time.localtime(open_in))
            days[day]["total_s"] += max(0.0, now - open_in)
            days[day]["inside"] = True
        out.extend(days.values())
    out.sort(key=lambda d: (d["date"], d["display"]), reverse=True)
    return out


def list_unknown_sightings(*, limit: int = 2000) -> list[dict[str, Any]]:
    """Неизвестные u:N, новые сверху. Для временного разбора в админке."""
    from sqlalchemy import text

    sql = (
        "SELECT ts, subject, display, camera, camera_index, snapshot "
        "FROM sightings WHERE kind='unknown' AND subject LIKE 'u:%' "
        "ORDER BY ts DESC LIMIT :lim"
    )
    with session_scope() as s:
        rows = s.execute(text(sql), {"lim": int(limit)}).fetchall()
    return [dict(r._mapping) for r in rows]


def relabel_snapshot(
    snapshot: str,
    *,
    subject: str,
    kind: str,
    name: str | None,
    display: str,
) -> int:
    """Переименовать только строки с этим снимком, не всю историю человека."""
    import os

    from sqlalchemy import text

    snap = os.path.basename(str(snapshot or "").strip())
    if not snap or snap != str(snapshot).strip():
        return 0
    with session_scope() as s:
        r = s.execute(
            text(
                """
                UPDATE sightings SET subject=:s, kind=:k, name=:n, display=:d
                WHERE snapshot=:snap
                """
            ),
            {"s": subject, "k": kind, "n": name or "", "d": display, "snap": snap},
        )
        return int(r.rowcount or 0)


def relabel_subject(
    old_subject: str,
    *,
    subject: str,
    kind: str,
    name: str | None,
    display: str,
) -> int:
    """Переименовать все строки одного неизвестного. Журнал не чистится."""
    from sqlalchemy import text

    with session_scope() as s:
        r = s.execute(
            text(
                """
                UPDATE sightings SET subject=:s, kind=:k, name=:n, display=:d
                WHERE subject=:old
                """
            ),
            {"s": subject, "k": kind, "n": name or "", "d": display, "old": old_subject},
        )
        return int(r.rowcount or 0)


def list_sightings(*, hours: int | None = 24, subject: str = "", limit: int = 50000) -> list[dict[str, Any]]:
    from sqlalchemy import text

    sql = "SELECT ts, subject, kind, name, display, camera_index, camera, track_id, u, box_h FROM sightings WHERE 1=1"
    params: dict[str, Any] = {"lim": int(limit)}
    if hours:
        sql += " AND ts >= :h"
        params["h"] = time.time() - max(0, int(hours)) * 3600
    if subject:
        sql += " AND subject = :s"
        params["s"] = subject
    sql += " ORDER BY ts ASC LIMIT :lim"
    with session_scope() as s:
        rows = s.execute(text(sql), params).fetchall()
    return [dict(r._mapping) for r in rows]


def recent_sightings(*, hours: int = 6, limit: int = 40) -> list[dict[str, Any]]:
    """Последние проходы, новые сверху — для панели «События»."""
    from sqlalchemy import text

    sql = (
        "SELECT ts, subject, kind, name, display, camera_index, camera, track_id, snapshot "
        "FROM sightings WHERE ts >= :h ORDER BY ts DESC LIMIT :lim"
    )
    params = {"h": time.time() - max(0, int(hours)) * 3600, "lim": int(limit)}
    with session_scope() as s:
        rows = s.execute(text(sql), params).fetchall()
    return [dict(r._mapping) for r in rows]


def delete_alerts(ids: list[int | str]) -> int:
    from sqlalchemy import text

    cleaned = []
    for i in ids:
        try:
            cleaned.append(int(i))
        except Exception:
            continue
    if not cleaned:
        return 0
    deleted = 0
    with session_scope() as s:
        for aid in cleaned:
            r = s.execute(text("DELETE FROM alerts WHERE id=:id"), {"id": aid})
            deleted += int(r.rowcount or 0)
    return deleted


def clear_alerts() -> int:
    from sqlalchemy import text

    with session_scope() as s:
        n = int(s.execute(text("SELECT COUNT(*) FROM alerts")).scalar() or 0)
        s.execute(text("DELETE FROM alerts"))
    return n


def migrate_from_json(cfg: dict | None = None) -> None:
    from sqlalchemy import text

    cfg = cfg or load_config()
    with session_scope() as s:
        n = int(s.execute(text("SELECT COUNT(*) FROM cameras")).scalar() or 0)
    if n == 0:
        cam_path = resolve(cfg, cfg.get("cameras", {}).get("store", "data/cameras.json"))
        if cam_path.is_file():
            try:
                data = json.loads(cam_path.read_text(encoding="utf-8"))
                for c in data.get("cameras") or []:
                    if isinstance(c, dict):
                        add_camera(c)
                log.info("migrated cameras from %s", cam_path)
            except Exception as e:  # noqa: BLE001
                log.warning("camera migrate: %s", e)

    if not get_options():
        opt_path = resolve(cfg, "data/options.json")
        if opt_path.is_file():
            try:
                raw = json.loads(opt_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    save_options(raw)
                    log.info("migrated options from %s", opt_path)
            except Exception as e:  # noqa: BLE001
                log.warning("options migrate: %s", e)
