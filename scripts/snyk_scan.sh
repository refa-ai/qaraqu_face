#!/usr/bin/env bash
# Обязательный прогон Snyk (см. .cursor/rules/snyk-always.mdc)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"
PY="${QARAQU_FACE_PYTHON:-/data/nimda/qaraqu_face/.venv/bin/python}"

echo "== snyk whoami =="
snyk whoami || { echo "Нужен: snyk auth"; exit 1; }

echo "== snyk code test src/ =="
snyk code test src/ --severity=high || true

echo "== snyk test (venv) =="
snyk test --command="$PY" --severity=high || true
