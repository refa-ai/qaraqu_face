"""HTTP API и консоль: камеры, live MJPEG, картотека, журнал, карта движения (PRINCIPLE.md)."""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import hmac
import io
import logging
import os
import subprocess
import threading
import time
from typing import Any, Optional
from urllib.parse import unquote

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

from qaraqu_face import __version__
from qaraqu_face import cameras_store
from qaraqu_face import db as database
from qaraqu_face import floorplan
from qaraqu_face.config import load_config, project_root, resolve
from qaraqu_face.detect import detect_bgr
from qaraqu_face.gallery import (
    backup_gallery,
    create_person,
    delete_all_people,
    delete_person,
    enroll,
    list_people,
    load_gallery,
    person_dir,
    photo_path,
    rebuild_gallery,
    update_card,
)
from qaraqu_face.recognize import embedder_engine, get_embedder, l2
from qaraqu_face.runtime import get_runtime

ROOT = project_root()
TPL = Jinja2Templates(directory=str(ROOT / "web" / "templates"))
app = FastAPI(title="QARAQU FACE", version=__version__)

log = logging.getLogger("qaraqu_face.api")
SESSION_COOKIE = "qface_session"
PRODUCT = "QARAQU FACE"
PRODUCT_HTML = 'QARAQU <span>FACE</span>'
SLOGAN = "Око во тьме · видит скрытое"
ACCENT = "#fb7185"


@app.on_event("startup")
def _startup() -> None:
    import os

    database.init_db()
    # автостарт детекта только если явно QARAQU_FACE_ENABLED=1
    want = os.environ.get("QARAQU_FACE_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
    database.set_setting("service_enabled", want)
    if want:
        get_runtime().set_enabled(True)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000).hex()


def _users() -> list[dict[str, str]]:
    """Логины консоли. Пароли только в виде соли и хеша. Первый запуск берёт пару из конфига."""
    raw = database.get_setting("console_users", None)
    users = [u for u in raw if isinstance(u, dict) and u.get("name") and u.get("salt") and u.get("hash")] if isinstance(raw, list) else []
    if users:
        return users
    api = load_config().get("api", {})
    name = str(api.get("user", "admin")).strip() or "admin"
    password = str(api.get("password", "admin"))
    salt = hashlib.sha256(os.urandom(16)).hexdigest()[:32]
    users = [{"name": name, "salt": salt, "hash": _hash_password(password, salt)}]
    database.set_setting("console_users", users)
    return users


def _save_users(users: list[dict[str, str]]) -> None:
    database.set_setting("console_users", users)


ROLES = ("admin", "operator")


def _role_of(user: dict[str, Any] | None) -> str:
    """Старые записи без роли — администраторы, чтобы не потерять доступ."""
    r = str((user or {}).get("role") or "admin")
    return r if r in ROLES else "admin"


def _is_admin(request: Request) -> bool:
    return _role_of(_user_by_cookie(request)) == "admin"


def _check_user(name: str, password: str) -> dict[str, str] | None:
    name = (name or "").strip()
    for u in _users():
        if u["name"] != name:
            continue
        got = _hash_password(password, u["salt"])
        if hmac.compare_digest(got, u["hash"]):
            return u
    return None


def _session_token(user: dict[str, str]) -> str:
    return hmac.new(b"qaraqu-face-session", f"{user['name']}:{user['hash']}".encode(), hashlib.sha256).hexdigest()


def _user_by_cookie(request: Request) -> dict[str, str] | None:
    got = request.cookies.get(SESSION_COOKIE) or ""
    if not got:
        return None
    for u in _users():
        if hmac.compare_digest(got, _session_token(u)):
            return u
    return None


def _authed(request: Request) -> bool:
    return _user_by_cookie(request) is not None


def _user_name_ok(name: str) -> str:
    name = (name or "").strip()
    if len(name) < 2 or len(name) > 40:
        raise ValueError("логин от 2 до 40 символов")
    if any(c.isspace() for c in name):
        raise ValueError("логин без пробелов")
    return name


ADMIN_ONLY_PREFIXES = (
    "/api/admin/users",
    "/api/admin/restart",
    "/api/admin/streams",
    "/api/options",
)

# оператор может добавлять и редактировать, но не удалять и не сливать карточки
DELETE_POST_PATHS = (
    "/api/alerts/bulk-delete",
    "/api/alerts/clear",
    "/api/floorplan/floors/delete",
    "/api/admin/gallery/merge",
)


def _is_delete_action(request: Request) -> bool:
    if request.method == "DELETE":
        return True
    path = request.url.path
    if path.startswith("/api/cameras/") and path.endswith("/delete"):
        return True
    return request.method == "POST" and path in DELETE_POST_PATHS


@app.middleware("http")
async def _gate(request: Request, call_next):
    path = request.url.path
    if path in ("/login", "/api/health", "/favicon.ico"):
        return await call_next(request)
    user = _user_by_cookie(request)
    if user:
        role = _role_of(user)
        # только администратор: пользователи, перезапуск, остановка потоков, общие настройки
        if path.startswith(ADMIN_ONLY_PREFIXES) and role != "admin":
            return JSONResponse({"ok": False, "error": "нет прав: только администратор"}, status_code=403)
        # оператор не удаляет: записи, камеры, карточки, этажи, слияние
        if role != "admin" and _is_delete_action(request):
            return JSONResponse({"ok": False, "error": "нет прав: удалять может только администратор"}, status_code=403)
        return await call_next(request)
    if path.startswith("/api/") or path.startswith("/video_feed/"):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return RedirectResponse("/login", status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = ""):
    if _authed(request):
        return RedirectResponse("/", status_code=302)
    return TPL.TemplateResponse(request, "login.html", {"product": PRODUCT, "slogan": SLOGAN, "accent": ACCENT, "error": error})


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    user = _check_user(str(form.get("username") or ""), str(form.get("password") or ""))
    if not user:
        return login_page(request, error="неверный логин или пароль")
    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie(SESSION_COOKIE, _session_token(user), httponly=True, samesite="lax")
    return resp


@app.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TPL.TemplateResponse(
        request,
        "console.html",
        {"product_html": PRODUCT_HTML, "slogan": SLOGAN, "cameras_json": cameras_store.list_public()},
    )


@app.get("/api/health")
def health():
    cfg = load_config()
    det = cfg.get("detect", {})
    return {
        "ok": True,
        "version": __version__,
        "detect_weights": str(resolve(cfg, det.get("weights", "data/models/yolov11n-face.pt"))),
        "yunet": str(resolve(cfg, det.get("yunet", "data/models/face_detection_yunet_2023mar.onnx"))),
        "recognize": embedder_engine(cfg)[0],
        "recognize_engine": str(embedder_engine(cfg)[1]),
    }


def _perf_snapshot() -> dict[str, Any]:
    from qaraqu_face import runtime as R

    with R._PERF_LOCK:
        snap = dict(R.PERF)
    out = {"sem_miss": snap["sem_miss"], "det_n": snap["det_n"], "rec_n": snap["rec_n"], "sem_n": snap["sem_wait_n"]}
    for k in ("sem_wait", "det", "rec"):
        n = snap[k + "_n"]
        out[k + "_avg_ms"] = round(snap[k + "_ms"] / n, 1) if n else 0.0
    return out


def _analytics(rt) -> dict[str, Any]:
    return {
        "perf": _perf_snapshot(),
        "motion": {
            str(w.index): {
                "motion": round(float(getattr(w._cap, "motion", 0.0) or 0.0), 2), "blk": round(float(getattr(w._cap, "motion_blk", 0.0) or 0.0), 2) if getattr(w, "_cap", None) is not None else None,
                "thr": round(float(getattr(w._cap, "motion_threshold", 0.0) or 0.0), 2) if getattr(w, "_cap", None) is not None else None,
                "detect": w.index in rt.detect_on,
                "last_det_age": round(time.time() - w._last_det_ok, 1) if w._last_det_ok else None,
            }
            for w in list(rt.workers.values())
        },
        "faces": {
            "enabled": rt.enabled,
            "camera_indices": sorted(rt.detect_on),
            "det_gen": {w.index: w._det_gen for w in list(rt.workers.values())},
            "faces_gen": {w.index: w._faces_gen for w in list(rt.workers.values())},
            # Настройки теперь индивидуальны у каждой камеры; общих det_conf/match_threshold нет.
            "camera_settings": {str(i): s for i, s in rt.cam_settings.items()},
        }
    }


def _thr(v: Any) -> float:
    t = float(v)
    return t / 100.0 if t > 1.0 else t


@app.get("/api/analytics")
@app.post("/api/analytics")
async def api_analytics(request: Request):
    rt = get_runtime()
    if request.method == "POST":
        data = await request.json()
        faces = data.get("faces") if isinstance(data.get("faces"), dict) else {}
        if "enabled" in faces:
            rt.set_enabled(bool(faces["enabled"]))
        if "camera_indices" in faces:
            idxs = {int(x) for x in (faces.get("camera_indices") or [])}
            rt.detect_on = idxs
            for i in idxs:
                rt.ensure_worker(i)
            if idxs and not rt.enabled:
                rt.set_enabled(True)
                rt.detect_on = idxs
        # match_threshold больше не глобальный: порог совпадения задаётся на камеру
        # через POST /api/cameras/{index}/settings.
    return {"ok": True, "analytics": _analytics(rt)}


@app.get("/api/hw")
def api_hw():
    import shutil

    hw: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["gpu"] = round(100.0 * (total - free) / total, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import os

        hw["cpu"] = round(100.0 * os.getloadavg()[0] / (os.cpu_count() or 1), 1)
    except Exception:  # noqa: BLE001
        pass
    du = shutil.disk_usage(str(ROOT))
    hw["disk"] = round(100.0 * du.used / du.total, 1)
    hw["disk_used_gb"] = round(du.used / 1024**3)
    hw["disk_total_gb"] = round(du.total / 1024**3)
    return {"ok": True, "hw": hw}


@app.get("/api/meta")
def api_meta(cam: Optional[int] = None):
    rt = get_runtime()
    m = rt.meta(cam)
    cams = cameras_store.list_public()
    return {
        "ok": True,
        "enabled": rt.enabled,
        "analytics": _analytics(rt),
        "match_threshold": float(rt.options.get("match_threshold", 0.58)),
        "det_conf": float(rt.options.get("det_conf", 0.18)),
        "cameras": cams,
        "slots": [{"index": c["index"], "open": bool((w := rt.workers.get(c["index"])) and w.alive)} for c in cams],
        # capture_fps — уникальные кадры со слота; deliver — отдача MJPEG (раньше врала 25)
        "fps": (
            m.get("capture_fps")
            if m.get("capture_fps") is not None and float(m.get("capture_fps") or 0) > 0
            else m.get("fps")
        ),
        "capture_fps": m.get("capture_fps"),
        "deliver_fps": m.get("fps"),
        "open": m.get("alive"),
        "camera": m.get("camera"),
        "faces": m.get("hits") or [],
        "recent": rt.board_events() if cam is None else list(rt.events)[:30],
        "gallery": m.get("gallery"),
        "error": m.get("error"),
    }


@app.post("/api/preview")
async def api_preview(request: Request):
    data = await request.json()
    idx = int(data.get("camera_index", 0))
    get_runtime().set_preview(idx)
    return {"ok": True, "camera_index": idx}


@app.get("/api/cameras")
def api_cameras():
    return {"ok": True, "cameras": cameras_store.list_public()}


@app.get("/api/cameras/{index}")
def api_camera_get(index: int):
    c = cameras_store.get(index)
    if not c:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    from qaraqu_face import db as database

    return {
        "ok": True,
        "camera": c,
        "settings": database.get_camera_settings(index),
        "defaults": database.CAM_SETTING_DEFAULTS,
    }


@app.get("/api/camera-settings")
def api_camera_settings_all():
    from qaraqu_face import db as database

    rt = get_runtime()
    return {
        "ok": True,
        "settings": {str(i): s for i, s in rt.cam_settings.items()},
        "defaults": database.CAM_SETTING_DEFAULTS,
    }


@app.post("/api/cameras/{index}/settings")
async def api_camera_settings_save(index: int, request: Request):
    from qaraqu_face import db as database

    data = await request.json()
    try:
        rt = get_runtime()
        cur = rt.save_cam_settings(index, data if isinstance(data, dict) else {})
        return {"ok": True, "index": index, "settings": cur}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/cameras")
async def api_cameras_add(request: Request):
    data = await request.json()
    try:
        rec = cameras_store.add(data)
        rt = get_runtime()
        if rt.enabled:
            rt.set_detect(int(rec["index"]), True)
        return {"ok": True, "index": rec["index"], "camera": rec}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/cameras/{index}")
async def api_cameras_upd(index: int, request: Request):
    data = await request.json()
    try:
        return {"ok": True, "camera": cameras_store.update(index, data)}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/admin/streams/stop")
def api_stop_streams():
    n = get_runtime().stop_all_streams()
    return {"ok": True, "stopped": n}


@app.post("/api/admin/streams/start")
def api_start_streams():
    rt = get_runtime()
    rt.set_enabled(True)
    return {"ok": True, "enabled": rt.enabled}


def _password_ok(password: str) -> str:
    password = password or ""
    if len(password) < 4:
        raise ValueError("пароль не короче 4 символов")
    if len(password) > 80:
        raise ValueError("пароль слишком длинный")
    return password


@app.get("/api/me")
def api_me(request: Request):
    u = _user_by_cookie(request)
    return {"ok": True, "name": (u or {}).get("name", ""), "role": _role_of(u)}


@app.get("/api/admin/users")
def api_users():
    return {"ok": True, "users": [{"name": u["name"], "role": _role_of(u)} for u in _users()]}


@app.post("/api/admin/users")
async def api_users_add(request: Request):
    data = await request.json()
    role = str(data.get("role") or "operator")
    try:
        name = _user_name_ok(str(data.get("name") or ""))
        password = _password_ok(str(data.get("password") or ""))
        if role not in ROLES:
            raise ValueError("роль: admin или operator")
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    users = _users()
    if any(u["name"] == name for u in users):
        return JSONResponse({"ok": False, "error": "такой логин уже есть"}, status_code=400)
    salt = hashlib.sha256(os.urandom(16)).hexdigest()[:32]
    users.append({"name": name, "salt": salt, "hash": _hash_password(password, salt), "role": role})
    _save_users(users)
    return {"ok": True, "users": [{"name": u["name"], "role": _role_of(u)} for u in users]}


@app.post("/api/admin/users/role")
async def api_users_role(request: Request):
    data = await request.json()
    name = str(data.get("name") or "")
    role = str(data.get("role") or "")
    if role not in ROLES:
        return JSONResponse({"ok": False, "error": "роль: admin или operator"}, status_code=400)
    users = _users()
    me = _user_by_cookie(request)
    target = next((u for u in users if u["name"] == name), None)
    if not target:
        return JSONResponse({"ok": False, "error": "нет такого пользователя"}, status_code=400)
    # нельзя оставить систему без администратора
    admins = [u for u in users if _role_of(u) == "admin"]
    if _role_of(target) == "admin" and role != "admin" and len(admins) <= 1:
        return JSONResponse({"ok": False, "error": "должен остаться хотя бы один администратор"}, status_code=400)
    if me and me["name"] == name and role != "admin" and len(admins) <= 1:
        return JSONResponse({"ok": False, "error": "нельзя снять роль с последнего администратора"}, status_code=400)
    target["role"] = role
    _save_users(users)
    return {"ok": True, "users": [{"name": u["name"], "role": _role_of(u)} for u in users]}


@app.post("/api/admin/users/password")
async def api_users_password(request: Request):
    data = await request.json()
    try:
        name = _user_name_ok(str(data.get("name") or ""))
        password = _password_ok(str(data.get("password") or ""))
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    users = _users()
    me = _user_by_cookie(request)
    found = None
    for u in users:
        if u["name"] == name:
            salt = hashlib.sha256(os.urandom(16)).hexdigest()[:32]
            u["salt"] = salt
            u["hash"] = _hash_password(password, salt)
            found = u
            break
    if not found:
        return JSONResponse({"ok": False, "error": "нет такого пользователя"}, status_code=400)
    _save_users(users)
    resp = JSONResponse({"ok": True, "users": [{"name": u["name"], "role": _role_of(u)} for u in users]})
    if me and me["name"] == name:
        resp.set_cookie(SESSION_COOKIE, _session_token(found), httponly=True, samesite="lax")
    return resp


@app.get("/api/admin/unknowns")
def api_unknowns(limit: int = 40):
    from qaraqu_face.unknowns import queue

    try:
        return queue(limit=max(1, min(80, int(limit))))
    except Exception as e:  # noqa: BLE001
        log.warning("unknown queue: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/api/admin/gallery/audit")
def api_gallery_audit():
    from qaraqu_face.gallery_audit import audit

    try:
        return audit(track_hits=get_runtime().card_track_hits())
    except Exception as e:  # noqa: BLE001
        log.warning("gallery audit: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/admin/gallery/rebuild")
async def api_gallery_rebuild(request: Request):
    """Пересобрать эмбеддинги всех карточек: ArcFace и AdaFace.

    На время счёта потоки камер гасятся — иначе NVDEC + две сети не влезают в GPU.
    """
    data = {}
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        data = {}
    n_aug = int(data.get("n_aug") or 6)
    n_aug = max(0, min(12, n_aug))
    backends = data.get("backends") or ["arcface", "adaface"]
    if not isinstance(backends, list) or not backends:
        backends = ["arcface", "adaface"]
    backends = [str(b).lower() for b in backends if str(b).lower() in ("arcface", "adaface")]
    if not backends:
        backends = ["arcface", "adaface"]

    rt = get_runtime()
    was_on = bool(rt.enabled)
    if was_on:
        rt.set_enabled(False)
        time.sleep(2.0)
    try:
        for b in backends:
            get_embedder(rt.cfg, b)
        bak = backup_gallery(rt.cfg)
        report = rebuild_gallery(rt.cfg, n_aug=n_aug, apply=True, backends=backends)
        rt.reload_gallery()
    except Exception as e:  # noqa: BLE001
        log.exception("gallery rebuild")
        if was_on:
            try:
                rt.set_enabled(True)
            except Exception:  # noqa: BLE001
                pass
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    if was_on:
        try:
            rt.set_enabled(True)
        except Exception as e:  # noqa: BLE001
            log.warning("streams after rebuild: %s", e)

    by_status: dict[str, int] = {}
    for row in report:
        st = str(row.get("status") or "?")
        by_status[st] = by_status.get(st, 0) + 1
    return {
        "ok": True,
        "backup": str(bak),
        "backends": backends,
        "people": len(report),
        "by_status": by_status,
        "vectors": sum(int(r.get("vectors") or 0) for r in report),
        "vectors_arc": sum(int(r.get("vectors_arc") or 0) for r in report),
        "vectors_ada": sum(int(r.get("vectors_ada") or 0) for r in report),
        "failed": [r["name"] for r in report if r.get("status") != "ok"][:40],
    }


@app.post("/api/admin/gallery/merge")
async def api_gallery_merge(request: Request):
    from qaraqu_face.gallery_audit import merge

    data = await request.json()
    try:
        out = merge(str(data.get("keep") or ""), str(data.get("drop") or ""))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    get_runtime().reload_gallery()
    return out


@app.post("/api/admin/unknowns/from-event")
async def api_unknown_from_event(request: Request):
    from qaraqu_face.unknowns import enqueue_event

    data = await request.json()
    try:
        return enqueue_event(data if isinstance(data, dict) else {})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/admin/unknowns/embed")
async def api_unknown_embed(request: Request):
    from qaraqu_face.unknowns import assign

    data = await request.json()
    try:
        out = assign(
            str(data.get("subject") or ""),
            person=str(data.get("person") or "").strip() or None,
            guest=str(data.get("guest") or "").strip() or None,
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    if out.get("person"):
        get_runtime().reload_gallery()
    return out


@app.post("/api/admin/restart")
def api_restart():
    """Ответ уходит сразу, процесс сменяется через секунду, когда порт свободен."""
    root = project_root()
    log_path = root / "data" / "api.log"

    def _go() -> None:
        time.sleep(0.6)
        log.info("рестарт сервиса")
        subprocess.Popen(
            ["bash", "-c", "sleep 1; exec ./scripts/start.sh >> data/api.log 2>&1"],
            cwd=str(root),
            start_new_session=True,
            env=os.environ.copy(),
        )
        os._exit(0)

    threading.Thread(target=_go, name="restart", daemon=True).start()
    return {"ok": True}


def _cameras_del(index: int):
    cameras_store.delete(index)
    get_runtime().remove_camera(int(index))
    return {"ok": True, "remaining": len(cameras_store.list_public())}


@app.delete("/api/cameras/{index}")
def api_cameras_del(index: int):
    try:
        return _cameras_del(index)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/cameras/{index}/delete")
def api_cameras_del_post(index: int):
    """Запасной путь: некоторые прокси режут HTTP DELETE."""
    try:
        return _cameras_del(index)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/wall")
async def api_wall(request: Request):
    """Камеры, которые сейчас на экране стены. Превью остальных гасятся."""
    data = await request.json()
    idxs = {int(x) for x in (data.get("indices") or []) if str(x).lstrip("-").isdigit()}
    get_runtime().set_wall(idxs)
    return {"ok": True, "indices": sorted(idxs)}


@app.get("/api/preview/{index}")
def api_preview_jpeg(index: int):
    """Один кадр доп. потока для плитки. Короткий запрос, без вечного MJPEG."""
    rt = get_runtime()
    if index not in rt.wall_set:
        return Response(status_code=204)
    if rt.enabled:
        rt.ensure_worker(index)
    jpg = rt.jpeg(index, wall=True)
    if not jpg:
        return Response(status_code=204)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/video_feed/{index}")
async def video_feed(index: int, wall: int = 0, fps: float = 0):
    rt = get_runtime()
    # сетка превью не перехватывает камеру, которую смотрят на весь кадр
    if not int(wall):
        rt.set_preview(index)
    if fps <= 0:
        fps = float(rt.options.get("preview_fps") or 15)
    fps = min(30.0, max(1.0, fps))
    # Опрос источника чаще отдачи: при poll = 1/fps половина кадров проскакивала между проверками.
    poll = min(0.005, 1.0 / fps)
    min_gap = 1.0 / fps
    boundary = b"--frame"

    async def gen():
        last_sent: bytes | None = None
        next_due = 0.0
        # FPS в статистике считаем по новым кадрам, не по повторам
        while True:
            jpg = rt.jpeg(index, wall=bool(int(wall)))
            now = time.monotonic()
            if jpg is not None and jpg is not last_sent and now >= next_due:
                last_sent = jpg
                next_due = now + min_gap
                w = rt.workers.get(index)
                if w:
                    if int(wall):
                        w.note_wall()
                    else:
                        w.note_delivered()
                yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(poll)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot/{fname}")
def api_snapshot(fname: str):
    safe = fname.rsplit("/", 1)[-1]
    p = get_runtime().snap_dir / safe
    if not safe.endswith(".jpg") or not p.is_file():
        return JSONResponse({"ok": False}, status_code=404)
    return Response(p.read_bytes(), media_type="image/jpeg")


# ---- картотека ----


@app.get("/api/gallery")
def api_gallery(q: str = ""):
    people = list_people(q=q)
    return {"ok": True, "people": people, "count": len(people)}


@app.get("/api/gallery/photo/{name}/{filename}")
def api_gallery_photo(name: str, filename: str):
    try:
        return Response(photo_path(name, filename).read_bytes(), media_type="image/jpeg")
    except Exception:
        return JSONResponse({"ok": False}, status_code=404)


def _b64_to_bgr(s: str) -> np.ndarray | None:
    if not s:
        return None
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        arr = np.frombuffer(base64.b64decode(s), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _person_key(name: str) -> str | None:
    key = unquote(name).strip()
    try:
        person_dir(key)
    except ValueError:
        return None
    return key


@app.post("/api/gallery")
async def api_gallery_add(request: Request):
    data = await request.json()
    attrs = {k: str(data.get(k) or "").strip() for k in ("last_name", "first_name", "patronymic", "iin")}
    attrs["full_name"] = " ".join(x for x in (attrs["last_name"], attrs["first_name"], attrs["patronymic"]) if x)
    bgr = _b64_to_bgr(str(data.get("image_b64") or ""))
    if bgr is None:
        return JSONResponse({"ok": False, "error": "нет изображения"}, status_code=400)
    out = create_person(bgr, attrs)
    get_runtime().reload_gallery()
    return out


@app.post("/api/gallery/{name}")
async def api_gallery_edit(name: str, request: Request):
    key = _person_key(name)
    if not key:
        return JSONResponse({"ok": False, "error": "некорректное имя"}, status_code=400)
    data = await request.json()
    try:
        out = update_card(key, data)
    except FileNotFoundError:
        return JSONResponse({"ok": False, "error": "карточка не найдена"}, status_code=404)
    bgr = _b64_to_bgr(str(data.get("image_b64") or ""))
    if bgr is not None:
        res = enroll(key, bgr)
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or "не удалось добавить фото"}
    get_runtime().reload_gallery()
    return out


@app.post("/api/gallery/{name}/photos")
async def api_gallery_photos(name: str, request: Request):
    key = _person_key(name)
    if not key or not person_dir(key).is_dir():
        return JSONResponse({"ok": False, "error": "карточка не найдена"}, status_code=404)
    data = await request.json()
    added = 0
    for s in data.get("images_b64") or []:
        bgr = _b64_to_bgr(str(s))
        if bgr is not None and enroll(key, bgr).get("ok"):
            added += 1
    get_runtime().reload_gallery()
    return {"ok": True, "added": added}


@app.delete("/api/gallery")
def api_gallery_del_all():
    deleted = delete_all_people()
    get_runtime().reload_gallery()
    return {"ok": True, "deleted": deleted}


@app.delete("/api/gallery/{name}")
def api_gallery_del(name: str):
    key = _person_key(name)
    if not key:
        return JSONResponse({"ok": False, "error": "некорректное имя"}, status_code=400)
    if not person_dir(key).is_dir():
        return JSONResponse({"ok": False, "error": "карточка не найдена"}, status_code=404)
    delete_person(key)
    get_runtime().reload_gallery()
    return {"ok": True, "name": key}


@app.post("/api/search")
async def api_search(request: Request):
    data = await request.json()
    bgr = _b64_to_bgr(str(data.get("image_b64") or ""))
    if bgr is None:
        return {"ok": False, "error": "нет изображения"}
    faces = detect_bgr(bgr)
    lm = xy = None
    if faces:
        f0 = max(faces, key=lambda f: (f["xyxy"][2] - f["xyxy"][0]) * (f["xyxy"][3] - f["xyxy"][1]))
        lm, xy = f0.get("landmarks"), f0["xyxy"]
    rt = get_runtime()
    rec = load_config().get("recognize") if isinstance(load_config().get("recognize"), dict) else {}
    arc_thr = float((rec.get("arcface") or {}).get("threshold", 0.45)) if isinstance(rec.get("arcface"), dict) else 0.45
    ada_thr = float(rt.options.get("match_threshold", 0.38))
    matches: list[dict[str, Any]] = []
    used = ""
    thr = arc_thr
    for backend, bthr in (("arcface", arc_thr), ("adaface", ada_thr)):
        gal = (rt.galleries or {}).get(backend) or {}
        if not gal:
            continue
        q = l2(get_embedder(backend=backend).embed_bgr(bgr, landmarks=lm, xyxy=xy))
        rows = []
        for n, vecs in gal.items():
            score = max(float(np.dot(q, l2(v))) for v in vecs)
            rows.append({"name": n, "display": rt.people.get(n, {}).get("display") or n, "similarity": score, "backend": backend})
        rows.sort(key=lambda x: -x["similarity"])
        matches, used, thr = rows, backend, bthr
        if rows and rows[0]["similarity"] >= bthr:
            break
    top = matches[: int(data.get("top_k", 5))]
    return {
        "ok": True,
        "match": bool(top and top[0]["similarity"] >= thr),
        "backend": used,
        "matches": top,
        "face_found": bool(faces),
    }


@app.get("/api/options")
def api_options_get():
    return get_runtime().reload_options()


@app.post("/api/options")
async def api_options_set(request: Request):
    return get_runtime().save_options(await request.json())


@app.get("/api/events")
def api_events():
    return {"ok": True, "events": list(get_runtime().events)}


# ---- журнал и отчёты ----


def _passage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        snap = r.get("snapshot")
        out.append(
            {
                **r,
                "thumb": f"/api/snapshot/{snap}" if snap else None,
            }
        )
    return out


@app.get("/api/passages")
def api_passages(hours: int = 24, q: str = "", limit: int = 2000):
    rows = database.list_passages(hours=hours, q=q, limit=limit)
    return {"ok": True, "passages": _passage_rows(rows)}


@app.get("/api/passages/summary")
def api_passages_summary(hours: int = 24):
    return {"ok": True, "people": database.passage_summary(hours=hours)}


@app.get("/api/passage-pairs")
def api_passage_pairs():
    return {"ok": True, "pairs": get_runtime().passage_pairs}


@app.post("/api/passage-pairs")
async def api_passage_pairs_save(request: Request):
    data = await request.json()
    pairs = data.get("pairs") if isinstance(data, dict) else None
    if not isinstance(pairs, list):
        return JSONResponse({"ok": False, "error": "нужен список pairs"}, status_code=400)
    try:
        get_runtime().set_passage_pairs(pairs)
    except (TypeError, ValueError, KeyError):
        return JSONResponse({"ok": False, "error": "камера в паре указана неверно"}, status_code=400)
    return {"ok": True, "pairs": get_runtime().passage_pairs}


@app.get("/api/analytics/report")
def api_analytics_report(hours: int = 720, interval: str = "day"):
    evs = database.list_alerts(hours=hours, limit=20000)
    known = [e for e in evs if e["kind"] == "known"]
    sims = [float(e["similarity"]) for e in known if e.get("similarity") is not None]
    fmt = {"hour": "%d.%m %H:00", "week": "%Y-W%W"}.get(interval, "%d.%m")
    buckets: dict[str, dict[str, Any]] = {}
    by_hour: dict[str, int] = {}
    persons: dict[str, dict[str, Any]] = {}
    for e in sorted(evs, key=lambda x: x["ts"]):
        lt = time.localtime(float(e["ts"]))
        b = buckets.setdefault(time.strftime(fmt, lt), {"total": 0, "known": 0, "unknown": 0})
        b["total"] += 1
        b["known" if e["kind"] == "known" else "unknown"] += 1
        by_hour[str(lt.tm_hour)] = by_hour.get(str(lt.tm_hour), 0) + 1
        if e["kind"] == "known" and e.get("name"):
            p = persons.setdefault(
                e["name"], {"person_name": e["name"], "display": e["display"], "iin": e.get("iin") or "", "n": 0, "sims": [], "last_seen": 0}
            )
            p["n"] += 1
            if e.get("similarity") is not None:
                p["sims"].append(float(e["similarity"]))
            p["last_seen"] = max(p["last_seen"], float(e["ts"]))
    rows = []
    for p in persons.values():
        sims_p = p.pop("sims")
        p["avg_confidence"] = round(sum(sims_p) / len(sims_p), 3) if sims_p else 0
        rows.append(p)
    rows.sort(key=lambda x: -x["n"])
    total = len(evs)
    return {
        "ok": True,
        "summary": {
            "total": total,
            "known": len(known),
            "unknown": total - len(known),
            "success_rate": round(100.0 * len(known) / total, 1) if total else 0,
            "average_confidence": round(sum(sims) / len(sims), 3) if sims else 0,
            "people": len(persons),
        },
        "series": [{"t": k, **v} for k, v in buckets.items()],
        "by_hour": by_hour,
        "by_type": {"known": len(known), "unknown": total - len(known)},
        "persons": rows[:50],
    }


@app.get("/api/alerts")
def api_alerts(hours: int = 720, kind: str = "", q: str = "", limit: int = 300):
    items = database.list_alerts(hours=hours, kind=kind, q=q, limit=limit)
    return {"ok": True, "alerts": items, "total": len(items)}


def _csv_cell(v: Any) -> str:
    s = str(v or "")
    # Excel исполняет ячейки, начинающиеся с = + - @
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


@app.get("/api/alerts/export.csv")
def api_alerts_csv(hours: int = 720, kind: str = "", q: str = ""):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["время", "тип", "кто", "ИИН", "камера", "сходство", "трек"])
    for e in database.list_alerts(hours=hours, kind=kind, q=q, limit=100000):
        w.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(e["ts"]))),
                "известный" if e["kind"] == "known" else "неизвестный",
                _csv_cell(e["display"]),
                _csv_cell(e.get("iin")),
                _csv_cell(e.get("camera")),
                f"{e['similarity']:.3f}" if e.get("similarity") is not None else "",
                e.get("track_id") or "",
            ]
        )
    return Response(
        "\ufeff" + buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="alerts.csv"'},
    )


@app.post("/api/alerts/bulk-delete")
async def api_alerts_bulk_delete(request: Request):
    data = await request.json()
    return {"ok": True, "deleted": database.delete_alerts(data.get("ids") or [])}


@app.post("/api/alerts/clear")
def api_alerts_clear():
    n = database.clear_alerts()
    database.set_setting("events_cleared_at", time.time())
    for w in get_runtime().workers.values():
        if w.last_meta:
            w.last_meta["hits"] = []
    snap = get_runtime().snap_dir
    if snap.is_dir():
        for p in snap.glob("*.jpg"):
            if p.name.startswith("pass_"):
                continue
            try:
                p.unlink()
            except OSError:
                pass
    get_runtime().events.clear()
    return {"ok": True, "deleted": n}


# ---- карта движения ----


def _fp_call(fn, *args):
    try:
        return fn(*args)
    except (KeyError, ValueError) as e:
        return JSONResponse({"ok": False, "error": str(e).strip("'")}, status_code=400)


@app.get("/api/floorplan")
def api_floorplan(floor: str = ""):
    return floorplan.get(floor or None)


@app.post("/api/floorplan/floors")
async def api_floor_add(request: Request):
    data = await request.json()
    return floorplan.add_floor(str(data.get("name") or ""))


@app.post("/api/floorplan/floors/delete")
async def api_floor_del(request: Request):
    data = await request.json()
    return _fp_call(floorplan.delete_floor, str(data.get("id") or ""))


@app.post("/api/floorplan/floors/select")
async def api_floor_select(request: Request):
    data = await request.json()
    return _fp_call(floorplan.select_floor, str(data.get("id") or ""))


@app.post("/api/floorplan/floors/rename")
async def api_floor_rename(request: Request):
    data = await request.json()
    return _fp_call(floorplan.rename_floor, str(data.get("id") or ""), str(data.get("name") or ""))


@app.post("/api/floorplan/cameras")
async def api_floor_cams(request: Request):
    data = await request.json()
    cams = data.get("cameras") if isinstance(data.get("cameras"), dict) else {}
    return _fp_call(floorplan.set_cameras, str(data.get("floor") or "") or None, cams)


@app.get("/api/floorplan/image")
def api_floor_image(floor: str = ""):
    p = floorplan.image_path(floor or None)
    if not p:
        return JSONResponse({"ok": False}, status_code=404)
    return Response(p.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.post("/api/floorplan/image")
async def api_floor_image_save(request: Request):
    data = await request.json()
    bgr = _b64_to_bgr(str(data.get("image_b64") or ""))
    if bgr is None:
        return JSONResponse({"ok": False, "error": "нужно изображение плана (jpg/png)"}, status_code=400)
    h, w = bgr.shape[:2]
    if max(h, w) > 2400:
        s = 2400 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return JSONResponse({"ok": False, "error": "не удалось сохранить"}, status_code=400)
    return _fp_call(floorplan.save_image, str(data.get("floor") or "") or None, buf.tobytes())


@app.get("/api/floorplan/tracks")
def api_floor_tracks(hours: int = 24, subject: str = "", floor: str = "", unknown: int = 1):
    return floorplan.tracks(
        hours=hours, subject=subject, floor=floor or None, unknown=bool(unknown), people=get_runtime().people
    )
