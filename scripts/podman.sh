#!/usr/bin/env bash
# Обёртка над podman compose для qaraqu_face
set -euo pipefail
cd "$(dirname "$0")/.."

export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"

_ensure_socket() {
  if [[ ! -S "${DOCKER_HOST#unix://}" ]]; then
    systemctl --user start podman.socket 2>/dev/null || true
    sleep 1
  fi
}

_compose() {
  _ensure_socket
  podman compose "$@"
}

case "${1:-}" in
  build)
    _compose build
    ;;
  db-up)
    _compose --profile db up -d db
    ;;
  db-down)
    _compose --profile db stop db 2>/dev/null || true
    podman stop qaraqu_face_db 2>/dev/null || true
    podman rm -f qaraqu_face_db 2>/dev/null || true
    ;;
  up)
    QARAQU_FACE_ENABLED=1 _compose --profile run up -d
    ;;
  down)
    _compose --profile run down 2>/dev/null || true
    _compose --profile db down 2>/dev/null || true
    podman stop qaraqu_face_app qaraqu_face_db 2>/dev/null || true
    podman rm -f qaraqu_face_app qaraqu_face_db 2>/dev/null || true
    ;;
  status)
    podman ps -a --filter name=qaraqu_face --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    ;;
  *)
    echo "usage: $0 {build|db-up|db-down|up|down|status}"
    exit 1
    ;;
esac
