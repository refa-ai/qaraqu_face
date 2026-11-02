"""Хранилище камер — PostgreSQL/SQLite через db.py (JSON только для миграции)."""

from __future__ import annotations

from typing import Any

from qaraqu_face import db as database


def list_public(cfg: dict | None = None) -> list[dict[str, Any]]:
    _ = cfg
    database.init_db()
    return [
        {
            "index": c["index"],
            "name": c["name"],
            "location": c.get("location") or "",
            "description": c.get("description") or "",
            "lat": c.get("lat"),
            "lon": c.get("lon"),
            "rtsp_url_main": c.get("rtsp_url_main") or "",
            "rtsp_url_sub": c.get("rtsp_url_sub") or "",
        }
        for c in database.list_cameras()
    ]


def get(index: int, cfg: dict | None = None) -> dict[str, Any] | None:
    _ = cfg
    database.init_db()
    return database.get_camera(index)


def add(entry: dict[str, Any], cfg: dict | None = None) -> dict[str, Any]:
    _ = cfg
    database.init_db()
    return database.add_camera(entry)


def update(index: int, entry: dict[str, Any], cfg: dict | None = None) -> dict[str, Any]:
    _ = cfg
    database.init_db()
    return database.update_camera(index, entry)


def delete(index: int, cfg: dict | None = None) -> None:
    _ = cfg
    database.init_db()
    database.delete_camera(index)
