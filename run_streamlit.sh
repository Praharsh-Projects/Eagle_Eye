#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ "${EAGLE_EYE_SKIP_DOTENV:-false}" != "true" && -f .env ]]; then
  set -a
  source .env
  set +a
fi

VENV_DIR="${EAGLE_EYE_VENV:-$HOME/.venvs/eagle-eye}"
LOCK_FILE="$ROOT_DIR/requirements.lock"
STAMP_FILE="$VENV_DIR/.eagle-eye-runtime.sha256"

if [[ ! -x "$VENV_DIR/bin/python" || ! -x "$VENV_DIR/bin/streamlit" ]]; then
  echo "Eagle Eye runtime is missing at $VENV_DIR." >&2
  echo "Run: $ROOT_DIR/scripts/bootstrap_runtime.sh" >&2
  exit 1
fi

if [[ ! -f "$LOCK_FILE" ]]; then
  echo "Missing dependency lock: $LOCK_FILE" >&2
  exit 1
fi

EXPECTED_LOCK_HASH="$(shasum -a 256 "$LOCK_FILE" | awk '{print $1}')"
INSTALLED_LOCK_HASH="$(cat "$STAMP_FILE" 2>/dev/null || true)"
if [[ "$EXPECTED_LOCK_HASH" != "$INSTALLED_LOCK_HASH" ]]; then
  echo "Eagle Eye dependencies are out of date." >&2
  echo "Run: $ROOT_DIR/scripts/bootstrap_runtime.sh" >&2
  exit 1
fi

export PYTHONPATH=.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$HOME/.cache/eagle-eye/pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX"
PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-127.0.0.1}"
echo "Starting the restored Eagle Eye Streamlit workspace at http://${ADDRESS}:${PORT}"
exec "$VENV_DIR/bin/streamlit" run app/streamlit_app.py \
  --server.address "${ADDRESS}" \
  --server.port "${PORT}" \
  --server.fileWatcherType none
