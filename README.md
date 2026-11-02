# qaraqu_face

> **Основной принцип:** [PRINCIPLE.md](PRINCIPLE.md) — распознавание лиц, трекинг известных и неизвестных, карта движения по камерам офиса.

Production-сборка (GPU). Эксперименты и старый мусор — вне этого проекта, в продакшене не используются.

## Стек

| | |
|--|--|
| Детект | YOLO face (`device: "0"`, CUDA) + YuNet landmarks |
| ID | ArcFace ONNX на CUDA (`onnxruntime-gpu`) |
| Трекинг | track-id на камере, голосование 1:N, «Неизвестный #N» с повторным узнаванием между камерами |
| Карта | `data/floorplan.json` + планы `data/floorplans/`, маршруты из таблицы `sightings` |
| Картотека | `data/gallery` |
| БД | SQLite `data/qaraqu_face.db` или PostgreSQL (`DATABASE_URL`) |
| UI/API | порт **5020**, `admin` / `admin` |

## Запуск

```bash
cd /data/nimda/qaraqu_face
./scripts/start.sh
# http://<хост>:5020
```

Python: `/data/nimda/qaraqu_face/.venv` (torch cu128 + GPU).

## Картотека: пересборка эмбеддингов

```bash
/data/nimda/qaraqu_face/.venv/bin/python scripts/rebuild_embeddings.py --dry-run   # отчёт
/data/nimda/qaraqu_face/.venv/bin/python scripts/rebuild_embeddings.py             # backup + пересборка
```

Вектора считаются только по фото с настоящим лицом (глаза, нос, рот), аугментации идут с пересчитанными
landmarks и «камерным» качеством. Фото другого человека в карточке отбрасывается. Бэкап лежит в
`data/gallery_backup/`, сервис после пересборки перезапустить. Новые фото через UI проходят ту же проверку.

## Структура

```
PRINCIPLE.md          # якорь продукта
config/default.yaml   # YOLO GPU + ArcFace GPU
src/qaraqu_face/
  runtime.py          # камеры → YOLO → трекер → ArcFace → события/маршруты
  tracking.py         # CameraTracker, UnknownRegistry
  floorplan.py        # этажи, камеры на плане, построение маршрутов
  api.py db.py gallery.py detect.py recognize.py face_align.py face_aug.py
web/templates/        # console.html, login.html
data/models/          # yolov11n-face.pt, face_detection_yunet, w600k_r50.onnx
data/gallery/         # картотека
data/snapshots/       # снимки лиц для журнала
scripts/start.sh
scripts/snyk_scan.sh  # обязательный прогон Snyk
```

**Монолит:** никаких симлинков на модели/gallery.

## Архив

Архив AI-экспериментов вне проекта (не используется) (UltraFace, 3DDFA, InstantID, Krea…), tmp-конфиги, не для продакшена.
