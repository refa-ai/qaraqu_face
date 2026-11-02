#!/usr/bin/env python3
"""Сводка по истории замеров: детекция и распознавание. Пишет summary.md и печатает."""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT.parent / "reports"
HISTORY = REPORTS / "history.jsonl"


def main() -> int:
    rows = []
    if HISTORY.exists():
        for line in HISTORY.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        print("нет данных")
        return 0
    shares = [r["detection_share"] for r in rows]
    dets = [r["detections_total"] for r in rows]
    known = [r["known_in_window"] for r in rows]
    passes = sum(1 for r in rows if r.get("pass_detection"))
    ok_share = sum(1 for s in shares if s >= 0.80)
    md = [
        "# Отчёт qaraqu_face: детекция и распознавание",
        "",
        f"Замеров: {len(rows)}",
        f"Период: {time.strftime('%Y-%m-%d %H:%M', time.localtime(rows[0]['ts']))} — {time.strftime('%Y-%m-%d %H:%M', time.localtime(rows[-1]['ts']))}",
        "",
        "## Детекция",
        f"- Доля камер с детекцией >= 80 %: {ok_share}/{len(rows)} замеров",
        f"- Средняя доля камер с детекцией: {sum(shares)/len(shares):.1%}",
        f"- Минимум камер с детекцией: {min(r['cams_with_detection'] for r in rows)} из {rows[-1]['cams']}",
        f"- Среднее детекций за минуту: {sum(dets)/len(dets):.0f}",
        f"- Минимум детекций на камеру за минуту: {min(r.get('min_det_per_cam', 0) for r in rows)}",
        "",
        "## Распознавание",
        f"- Замеров с распознаванием (known>0): {sum(1 for k in known if k > 0)}/{len(known)}",
        f"- Всего распознаваний за период: {sum(known)}",
        "",
        "## Последние 10 замеров",
        "| время | камер с детекцией | детекций/мин | known |",
        "|---|---|---|---|",
    ]
    for r in rows[-10:]:
        md.append(
            f"| {time.strftime('%H:%M', time.localtime(r['ts']))} | "
            f"{r['cams_with_detection']}/{r['cams']} | {r['detections_total']} | {r['known_in_window']} |"
        )
    text = "\n".join(md)
    (REPORTS / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
