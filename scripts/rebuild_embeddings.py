#!/usr/bin/env python
"""Пересобрать мульти-эмбеддинги всех карточек по их фото (только валидные лица, ауги с landmarks).

  scripts/rebuild_embeddings.py --dry-run            # только отчёт
  scripts/rebuild_embeddings.py                      # backup + пересборка
  scripts/rebuild_embeddings.py --quarantine-no-face # карточки без лица уходят в gallery/.quarantine

Сервис после пересборки перезапустить (или дождаться reload_gallery).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaraqu_face.gallery import backup_gallery, rebuild_gallery  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="не записывать, только отчёт")
    ap.add_argument("--n-aug", type=int, default=8)
    ap.add_argument("--quarantine-no-face", action="store_true")
    ap.add_argument("--report", default="data/gallery_rebuild_report.json")
    args = ap.parse_args()

    if not args.dry_run:
        print("backup →", backup_gallery())
    report = rebuild_gallery(n_aug=args.n_aug, apply=not args.dry_run, quarantine_no_face=args.quarantine_no_face)
    by_status: dict[str, int] = {}
    for row in report:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    dropped = [r for r in report if r.get("dropped") and r["status"] == "ok"]
    vec_total = sum(r["vectors"] for r in report)
    print(f"карточек: {len(report)}  статусы: {by_status}  векторов: {vec_total}")
    for r in report:
        if r["status"] != "ok":
            print("  без лица:", r["name"], r.get("quarantined", ""))
    for r in dropped:
        print("  фото другого человека/без лица:", r["name"], r["dropped"])
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("отчёт →", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
