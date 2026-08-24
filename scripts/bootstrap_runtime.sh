#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${EAGLE_EYE_VENV:-$HOME/.venvs/eagle-eye}"
PYTHON_BIN="${EAGLE_EYE_PYTHON:-$(command -v python3.12 || true)}"
UV_BIN="${EAGLE_EYE_UV:-$(command -v uv || true)}"
MODE="runtime"

if [[ "${1:-}" == "--dev" ]]; then
  MODE="dev"
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--dev]" >&2
  exit 2
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.12 is required. Install it or set EAGLE_EYE_PYTHON." >&2
  exit 1
fi
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "uv is required. Install it or set EAGLE_EYE_UV." >&2
  exit 1
fi

LOCK_FILE="$ROOT_DIR/requirements.lock"
if [[ "$MODE" == "dev" ]]; then
  LOCK_FILE="$ROOT_DIR/requirements-dev.lock"
fi
if [[ ! -f "$LOCK_FILE" ]]; then
  echo "Missing dependency lock: $LOCK_FILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$VENV_DIR")"
if [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  rm -rf "$VENV_DIR"
  "$UV_BIN" venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

echo "Synchronizing Eagle Eye $MODE dependencies in $VENV_DIR"
"$UV_BIN" pip sync --python "$VENV_DIR/bin/python" "$LOCK_FILE"
shasum -a 256 "$ROOT_DIR/requirements.lock" | awk '{print $1}' > "$VENV_DIR/.eagle-eye-runtime.sha256"
echo "Runtime ready. Start Eagle Eye with: $ROOT_DIR/run_eagle_eye.sh"
