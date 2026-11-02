"""План помещения: этажи, позиции камер, картинка плана и маршруты объектов (PRINCIPLE.md)."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from qaraqu_face import db as database
from qaraqu_face.config import load_config, resolve

_LOCK = threading.RLock()
# разрыв дольше — новый визит на той же камере
VISIT_GAP_S = 60.0
# камеры, которые видят человека почти одновременно, — это одно место, не скачок
FUSE_S = 6.0
# дальше по времени или по плану линия не тянется: человек появился заново
PATH_GAP_S = 25.0
PATH_MAX_STEP = 0.38


def _root() -> Path:
    d = resolve(load_config(), "data/floorplans")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json_path() -> Path:
    return resolve(load_config(), "data/floorplan.json")


def _image_path(fid: str) -> Path:
    safe = "".join(c for c in fid if c.isalnum() or c in "-_") or "f1"
    return _root() / f"{safe}.jpg"


def _default() -> dict[str, Any]:
    return {"active": "f1", "floors": [{"id": "f1", "name": "Этаж 1", "cameras": {}}]}


def _read() -> dict[str, Any]:
    from qaraqu_face import db

    data = db.get_doc("floorplan") or {}
    if not isinstance(data, dict):
        data = {}
    floors = data.get("floors") if isinstance(data.get("floors"), list) else []
    if not floors:
        return _default()
    # старый формат: позиции камер лежали в floors_data
    legacy = data.get("floors_data") if isinstance(data.get("floors_data"), dict) else {}
    for f in floors:
        if "cameras" not in f:
            f["cameras"] = (legacy.get(f.get("id")) or {}).get("cameras") or {}
    data = {"active": data.get("active") or floors[0]["id"], "floors": floors}
    return data


def _write(data: dict[str, Any]) -> None:
    from qaraqu_face import db

    db.set_doc("floorplan", data)


def _floor(data: dict[str, Any], fid: str | None) -> dict[str, Any]:
    fid = fid or data.get("active")
    for f in data["floors"]:
        if f.get("id") == fid:
            return f
    raise KeyError("этаж не найден")


def _payload(data: dict[str, Any], fid: str | None = None) -> dict[str, Any]:
    try:
        cur = _floor(data, fid)
    except KeyError:
        cur = data["floors"][0]
    return {
        "ok": True,
        "active": cur["id"],
        "floor": {"id": cur["id"], "name": cur.get("name") or ""},
        "floors": [
            {"id": f["id"], "name": f.get("name") or "", "cameras": len(f.get("cameras") or {})}
            for f in data["floors"]
        ],
        "cameras": cur.get("cameras") or {},
        "has_image": _image_path(cur["id"]).is_file(),
    }


def get(fid: str | None = None) -> dict[str, Any]:
    with _LOCK:
        return _payload(_read(), fid)


def add_floor(name: str = "") -> dict[str, Any]:
    with _LOCK:
        data = _read()
        fid = f"f{int(time.time() * 1000)}"
        data["floors"].append({"id": fid, "name": name.strip() or f"Этаж {len(data['floors']) + 1}", "cameras": {}})
        data["active"] = fid
        _write(data)
        return _payload(data, fid)


def delete_floor(fid: str) -> dict[str, Any]:
    with _LOCK:
        data = _read()
        if len(data["floors"]) <= 1:
            raise ValueError("нельзя удалить единственный этаж")
        _floor(data, fid)
        data["floors"] = [f for f in data["floors"] if f.get("id") != fid]
        if data.get("active") == fid:
            data["active"] = data["floors"][0]["id"]
        _write(data)
        _image_path(fid).unlink(missing_ok=True)
        return _payload(data)


def select_floor(fid: str) -> dict[str, Any]:
    with _LOCK:
        data = _read()
        _floor(data, fid)
        data["active"] = fid
        _write(data)
        return _payload(data, fid)


def rename_floor(fid: str, name: str) -> dict[str, Any]:
    with _LOCK:
        data = _read()
        f = _floor(data, fid)
        if name.strip():
            f["name"] = name.strip()
        _write(data)
        return _payload(data, fid)


def _num(raw: Any, default: float, lo: float, hi: float) -> float:
    try:
        return min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
        return default


def _pose(v: Any, prev: dict[str, Any] | None = None) -> dict[str, float] | None:
    """Позиция на плане и куда смотрит камера. heading: 0° — вверх плана, по часовой."""
    if not isinstance(v, dict):
        return None
    prev = prev or {}
    try:
        x = float(v["x"])
        y = float(v["y"])
    except (KeyError, TypeError, ValueError):
        return None
    pose: dict[str, float] = {"x": min(1.0, max(0.0, x)), "y": min(1.0, max(0.0, y))}
    if "heading" in v or "heading" in prev:
        pose["heading"] = _num(v.get("heading", prev.get("heading", 0)), 0.0, -1e9, 1e9) % 360.0
    if "fov" in v or "fov" in prev:
        pose["fov"] = _num(v.get("fov", prev.get("fov", 80)), 80.0, 20.0, 170.0)
    if "range_m" in v or "range_m" in prev:
        pose["range_m"] = _num(v.get("range_m", prev.get("range_m", 8)), 8.0, 1.0, 80.0)
    return pose


def front_point(x: float, y: float, heading: float, dist: float) -> tuple[float, float]:
    """Точка перед камерой: человек стоит в секторе обзора, не на самом маркере."""
    rad = math.radians(float(heading or 0))
    return (
        min(1.0, max(0.0, x + math.sin(rad) * dist)),
        min(1.0, max(0.0, y - math.cos(rad) * dist)),
    )


def _range_plan(range_m: float) -> float:
    return max(0.05, min(0.45, float(range_m or 8) / 8.0 * 0.14))


def _project(pos: dict[str, Any], u: Any, box_h: Any) -> dict[str, float] | None:
    """Куда на плане попал человек: луч камеры и место лица в кадре, не точка самой камеры."""
    if "heading" not in pos:
        return None
    try:
        heading = float(pos["heading"])
        ox, oy = float(pos["x"]), float(pos["y"])
    except (TypeError, ValueError):
        return None
    fov = float(pos.get("fov") or 80)
    rng = _range_plan(float(pos.get("range_m") or 8))
    if u is None:
        ang = heading
    else:
        ang = heading + (float(u) - 0.5) * fov
    if box_h is not None and float(box_h) > 0.008:
        dist = rng * min(0.92, max(0.15, 0.05 / float(box_h)))
    else:
        dist = rng * 0.55
    fx, fy = front_point(ox, oy, ang, dist)
    rad = math.radians(ang)
    return {"x": fx, "y": fy, "ox": ox, "oy": oy, "dx": math.sin(rad), "dy": -math.cos(rad)}


def _mean_xy(pts: list[tuple[float, float]]) -> tuple[float, float]:
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def _meet(obs: list[dict[str, float]]) -> tuple[float, float]:
    """Одна точка на несколько камер: пересечение лучей, иначе среднее мест на лучах."""
    estimates = [(o["x"], o["y"]) for o in obs]
    if len(obs) < 2:
        return estimates[0]
    spread = 0.0
    for i, a in enumerate(obs):
        for b in obs[i + 1 :]:
            spread = max(spread, abs(a["dx"] * b["dy"] - a["dy"] * b["dx"]))
    if spread < 0.25:
        return _mean_xy(estimates)
    a00 = a01 = a11 = b0 = b1 = 0.0
    for o in obs:
        nx, ny = -o["dy"], o["dx"]
        a00 += nx * nx
        a01 += nx * ny
        a11 += ny * ny
        b0 += nx * nx * o["ox"] + nx * ny * o["oy"]
        b1 += nx * ny * o["ox"] + ny * ny * o["oy"]
    det = a00 * a11 - a01 * a01
    if abs(det) < 1e-8:
        return _mean_xy(estimates)
    px = (b0 * a11 - a01 * b1) / det
    py = (a00 * b1 - a01 * b0) / det
    if not (-0.05 <= px <= 1.05 and -0.05 <= py <= 1.05):
        return _mean_xy(estimates)
    px = min(1.0, max(0.0, px))
    py = min(1.0, max(0.0, py))
    for o in obs:
        along = (px - o["ox"]) * o["dx"] + (py - o["oy"]) * o["dy"]
        if along < -0.02 or along > 0.9:
            return _mean_xy(estimates)
    return px, py


def _clusters(samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Одновременные разные камеры — одна группа. Та же камера позже — уже движение."""
    used = [False] * len(samples)
    groups: list[list[dict[str, Any]]] = []
    for i, s in enumerate(samples):
        if used[i]:
            continue
        group = [s]
        used[i] = True
        seen = {s["camera_index"]}
        for j in range(i + 1, len(samples)):
            if samples[j]["ts"] - s["ts"] > FUSE_S:
                break
            cam = samples[j]["camera_index"]
            if cam in seen:
                continue
            group.append(samples[j])
            seen.add(cam)
            used[j] = True
        groups.append(group)
    return groups


def object_path(rows: list[dict[str, Any]], cam_floor: dict[int, tuple[str, str, dict[str, Any]]], fid: str) -> list[dict[str, Any]]:
    """Маршрут человека по плану. Несколько камер в один момент дают одну точку, без линии между ними."""
    samples: list[dict[str, Any]] = []
    for r in rows:
        place = cam_floor.get(int(r["camera_index"]))
        if not place or place[0] != fid:
            continue
        proj = _project(place[2], r.get("u"), r.get("box_h"))
        if not proj:
            continue
        samples.append({"ts": float(r["ts"]), "camera_index": int(r["camera_index"]), **proj})
    samples.sort(key=lambda s: s["ts"])
    points: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    for group in _clusters(samples):
        x, y = _meet(group)
        ts = group[0]["ts"]
        gap = False
        if prev is not None:
            dt = ts - float(prev["ts"])
            dist = math.hypot(x - float(prev["x"]), y - float(prev["y"]))
            if dt > PATH_GAP_S or dist > PATH_MAX_STEP or (dt > 0.4 and dist / dt > 0.04):
                gap = True
        pt = {
            "x": x,
            "y": y,
            "ts": ts,
            "gap": gap,
            "cameras": [g["camera_index"] for g in group],
            "camera_index": group[0]["camera_index"],
        }
        points.append(pt)
        prev = pt
    return points


def set_cameras(fid: str | None, cams: dict[str, Any]) -> dict[str, Any]:
    """cams: {"<index>": {"x","y","heading","fov","range_m"}} — доли плана, градусы, метры."""
    with _LOCK:
        data = _read()
        f = _floor(data, fid)
        prev_all = f.get("cameras") or {}
        clean: dict[str, dict[str, float]] = {}
        for k, v in (cams or {}).items():
            try:
                key = str(int(k))
            except (TypeError, ValueError):
                continue
            pose = _pose(v, prev_all.get(key) if isinstance(prev_all.get(key), dict) else None)
            if pose:
                clean[key] = pose
        f["cameras"] = clean
        # одна камера — один этаж
        for other in data["floors"]:
            if other is not f:
                other["cameras"] = {k: v for k, v in (other.get("cameras") or {}).items() if k not in clean}
        _write(data)
        return _payload(data, f["id"])


def save_image(fid: str | None, jpeg: bytes) -> dict[str, Any]:
    with _LOCK:
        data = _read()
        f = _floor(data, fid)
        _image_path(f["id"]).write_bytes(jpeg)
        return _payload(data, f["id"])


def image_path(fid: str | None) -> Path | None:
    with _LOCK:
        data = _read()
        try:
            f = _floor(data, fid)
        except KeyError:
            return None
    p = _image_path(f["id"])
    return p if p.is_file() else None


def camera_floor_map() -> dict[int, tuple[str, str, dict[str, float]]]:
    """camera_index → (floor_id, floor_name, {x,y})."""
    out: dict[int, tuple[str, str, dict[str, float]]] = {}
    with _LOCK:
        data = _read()
    for f in data["floors"]:
        for k, pos in (f.get("cameras") or {}).items():
            try:
                out[int(k)] = (f["id"], f.get("name") or "", pos)
            except ValueError:
                continue
    return out


def _visits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visits: list[dict[str, Any]] = []
    for r in rows:
        last = visits[-1] if visits else None
        if last and last["camera_index"] == r["camera_index"] and r["ts"] - last["t_out"] <= VISIT_GAP_S:
            last["t_out"] = r["ts"]
            last["n"] += 1
            continue
        visits.append(
            {"camera_index": int(r["camera_index"]), "camera": r["camera"], "t_in": r["ts"], "t_out": r["ts"], "n": 1}
        )
    return visits


def tracks(
    *,
    hours: int = 24,
    subject: str = "",
    floor: str | None = None,
    unknown: bool = True,
    people: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Маршруты: для каждого субъекта (человек из картотеки или «Неизвестный #N») — визиты по камерам."""
    people = people or {}
    rows = database.list_sightings(hours=hours, subject=subject)
    cam_floor = camera_floor_map()
    cur = get(floor)
    fid = cur["active"]
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if not unknown and r["kind"] != "known":
            continue
        by_subject.setdefault(r["subject"], []).append(r)
    out = []
    for subj, items in by_subject.items():
        visits = _visits(items)
        last = items[-1]
        also: set[str] = set()
        for v in visits:
            place = cam_floor.get(v["camera_index"])
            v["floor"] = place[1] if place else ""
            if place and place[0] != fid:
                also.add(place[1])
        points = object_path(items, cam_floor, fid)
        person = people.get(last.get("name") or "", {})
        out.append(
            {
                "subject": subj,
                "kind": last["kind"],
                "name": last.get("name"),
                "display": last.get("display") or subj,
                "thumb": person.get("thumb"),
                "first_ts": items[0]["ts"],
                "last_ts": last["ts"],
                "last_camera": last["camera"],
                "visits": visits,
                "points": points,
                "also_floors": sorted(also),
            }
        )
    out.sort(key=lambda t: -t["last_ts"])
    return {
        "ok": True,
        "floor": cur["floor"],
        "cameras": cur["cameras"],
        "tracks": out,
        "known": sum(1 for t in out if t["kind"] == "known"),
        "unknown": sum(1 for t in out if t["kind"] != "known"),
    }
