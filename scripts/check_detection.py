#!/usr/bin/env python3
"""Замер детекции и распознавания по живому сервису. Пишет отчёт и копит историю.

Порог приёмки: доля камер с детекцией за окно >= 80 %. Только чтение API и БД.
"""
from __future__ import annotations

import http.cookiejar
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:5020"
WINDOW_S = 60
MIN_CAM_SHARE = 0.80
HISTORY = ROOT.parent / "reports" / "history.jsonl"


def _opener() -> urllib.request.OpenerDirector:
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.open(urllib.request.Request(f"{API}/login", data=b"username=admin&password=admin"), timeout=10)
    return op


def _analytics(op: urllib.request.OpenerDirector) -> dict:
    return json.loads(op.open(f"{API}/api/analytics", timeout=20).read())["analytics"]["faces"]


def main() -> int:
    op = _opener()
    a = _analytics(op)
    db = sqlite3.connect(f"file:{ROOT / 'data' / 'qaraqu_face.db'}?mode=ro", uri=True)
    t0 = time.time()
    time.sleep(WINDOW_S)
    b = _analytics(op)
    dg = {int(k): b["det_gen"][k] - a["det_gen"].get(k, 0) for k in b["det_gen"]}
    fg = {int(k): b["faces_gen"][k] - a["faces_gen"].get(k, 0) for k in b["faces_gen"]}
    cams = len(dg)
    with_det = sum(1 for v in dg.values() if v > 0)
    share = with_det / cams if cams else 0.0
    known = db.execute("select count(*) from sightings where kind='known' and ts>=?", (t0,)).fetchone()[0]
    last_known = db.execute("select max(ts) from sightings where kind='known'").fetchone()[0]
    report = {
        "ts": int(time.time()),
        "window_s": WINDOW_S,
        "cams": cams,
        "cams_with_detection": with_det,
        "detection_share": round(share, 3),
        "detections_total": sum(dg.values()),
        "min_det_per_cam": min(dg.values()) if dg else 0,
        "faces_shown": sum(fg.values()),
        "cams_with_faces": sum(1 for v in fg.values() if v > 0),
        "known_in_window": known,
        "last_known_age_s": round(time.time() - last_known, 1) if last_known else None,
        "pass_detection": share >= MIN_CAM_SHARE,
    }
    out = ROOT.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"detect_{report['ts']}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["pass_detection"] else 1


if __name__ == "__main__":
    sys.exit(main())
