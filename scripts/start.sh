#!/usr/bin/env bash
# Старт форка: UI/API :5020. Детектор и эмбеддеры — PyTorch, YuNet — ONNX.
set -euo pipefail
cd "$(dirname "$0")/.."
# не тянуть старые tmp-переменные из shell
unset DATABASE_URL QARAQU_FACE_DATABASE_URL QARAQU_FACE_GALLERY 2>/dev/null || true
export QARAQU_FACE_ENABLED="${QARAQU_FACE_ENABLED:-1}"
export QARAQU_FACE_HOST="${QARAQU_FACE_HOST:-0.0.0.0}"
export QARAQU_FACE_PORT="${QARAQU_FACE_PORT:-5020}"
PY="${QARAQU_FACE_PYTHON:-/data/nimda/qaraqu_face/.venv/bin/python}"
exec "$PY" run.py
