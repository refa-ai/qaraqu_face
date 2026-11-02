#!/usr/bin/env bash
# Миграция JSON → БД (локально или в контейнере)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
exec python - <<'PY'
from qaraqu_face import db
url = db.init_db()
cams = db.list_cameras()
opts = db.get_options()
print("db:", url)
print("cameras:", len(cams))
print("options keys:", sorted(opts.keys()))
print("service_enabled:", db.get_setting("service_enabled"))
PY
