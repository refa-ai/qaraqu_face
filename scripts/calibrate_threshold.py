#!/usr/bin/env python
"""Порог 1:N по данным, а не на глаз (как в NIST FRTE: порог под целевой FPIR).

  scripts/calibrate_threshold.py [--fpir 0.01] [--per-person 8]

Берёт векторы картотеки (embeddings.npy) и, если есть, снимки из журнала sightings с именами:
  • «чужие» — сходство каждой карточки с остальными и каждого снимка с чужими карточками;
  • «свои»  — снимок против своей карточки.
Печатает порог, при котором доля чужих выше порога ≤ FPIR, и сколько своих при нём узнаётся.
Ничего не меняет: значение выставляется ползунком в админке или POST /api/options.
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaraqu_face.config import load_config, resolve  # noqa: E402
from qaraqu_face.detect import best_valid_face  # noqa: E402
from qaraqu_face.gallery import gallery_dir, read_person_vecs  # noqa: E402
from qaraqu_face.recognize import get_embedder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpir", type=float, default=0.01, help="допустимая доля чужих выше порога")
    ap.add_argument("--per-person", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    gal: dict[str, np.ndarray] = {}
    for d in sorted(gallery_dir(cfg).iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            v = read_person_vecs(d)
            if v:
                gal[d.name] = np.stack(v)
    names = list(gal)
    print(f"карточек: {len(names)}")
    imp: list[float] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            imp.append(float((gal[a] @ gal[b].T).max()))
    imp_arr = np.asarray(imp)
    if imp_arr.size:
        print(f"фото↔фото чужие: med={np.median(imp_arr):.3f} p99={np.quantile(imp_arr, .99):.3f} max={imp_arr.max():.3f}")

    db = resolve(cfg, cfg.get("database", {}).get("sqlite_path", "data/qaraqu_face.db"))
    snaps = resolve(cfg, "data/snapshots")
    gen: list[float] = []
    cam_imp: list[float] = []
    if Path(db).is_file():
        c = sqlite3.connect(str(db))
        rows = c.execute(
            "select name, snapshot, track_id from sightings where kind='known' and snapshot is not null order by ts desc"
        ).fetchall()
        emb = get_embedder(cfg)
        cnt: collections.Counter[str] = collections.Counter()
        seen: set[tuple[str, int]] = set()
        for name, snap, tid in rows:
            if name not in gal or (name, tid) in seen or cnt[name] >= args.per_person:
                continue
            fp = Path(snaps) / Path(str(snap)).name
            img = cv2.imread(str(fp)) if fp.is_file() else None
            face = best_valid_face(img, cfg) if img is not None else None
            if face is None:
                continue
            seen.add((name, tid))
            cnt[name] += 1
            v = emb.embed_bgr(img, landmarks=face["landmarks"], xyxy=face["xyxy"])
            sc = {k: float((m @ v).max()) for k, m in gal.items()}
            gen.append(sc[name])
            cam_imp.append(max(s for k, s in sc.items() if k != name))
    all_imp = np.asarray(imp + cam_imp)
    if not all_imp.size:
        print("нет данных для чужих")
        return 1
    thr = float(np.quantile(all_imp, 1.0 - args.fpir))
    print(f"порог при FPIR≤{args.fpir:.3f}: {thr:.3f}  (чужих всего {all_imp.size}, из них с камер {len(cam_imp)})")
    if gen:
        g = np.asarray(gen)
        print(f"свои с камер n={g.size}: med={np.median(g):.3f}; узнаются при этом пороге {np.mean(g >= thr) * 100:.0f}%")
    else:
        print("снимков с именами в журнале нет — порог только по фото; добавьте +0.08 запаса под камеры")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
