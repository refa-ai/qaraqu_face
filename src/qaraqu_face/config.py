from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CFG_PATH = _ROOT / "config" / "default.yaml"


def project_root() -> Path:
    return _ROOT


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or _CFG_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    root = Path(data.get("project", {}).get("root") or _ROOT)
    data.setdefault("project", {})["root"] = str(root)
    if path is None:
        _overlay_from_db(data)
    return data


def _overlay_from_db(data: dict[str, Any]) -> None:
    try:
        from qaraqu_face import db

        stored = db.get_runtime_config()
    except Exception:  # noqa: BLE001
        return
    for sec, val in stored.items():
        if isinstance(data.get(sec), dict):
            data[sec] = {**data[sec], **val}
        else:
            data[sec] = val


def resolve(cfg: dict[str, Any], *parts: str | Path) -> Path:
    root = Path(cfg.get("project", {}).get("root") or _ROOT)
    joined = Path(*[str(p) for p in parts if p is not None and str(p) != ""])
    if joined.is_absolute():
        return joined
    return (root / joined).resolve()
