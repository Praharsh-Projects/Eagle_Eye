#!/usr/bin/env bash
set -euo pipefail

EAGLE_EYE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$EAGLE_EYE_ROOT"

if [[ "${EAGLE_EYE_SKIP_DOTENV:-false}" != "true" && -f .env ]]; then
  set -a
  source .env
  set +a
fi

EAGLE_EYE_RUNTIME="${EAGLE_EYE_VENV:-$HOME/.venvs/eagle-eye}"
if [[ ! -x "$EAGLE_EYE_RUNTIME/bin/python" ]]; then
  echo "Eagle Eye runtime is missing at $EAGLE_EYE_RUNTIME." >&2
  echo "Run: $EAGLE_EYE_ROOT/scripts/bootstrap_runtime.sh" >&2
  exit 1
fi

if [[ ! -f web/dist/index.html ]]; then
  echo "The Eagle Eye React workspace has not been built." >&2
  echo "Run: cd $EAGLE_EYE_ROOT/web && npm install && npm run build" >&2
  exit 1
fi

export PYTHONPATH="$EAGLE_EYE_ROOT"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$HOME/.cache/eagle-eye/pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX"

EAGLE_EYE_ADDRESS="${ADDRESS:-127.0.0.1}"
EAGLE_EYE_PORT="${PORT:-8000}"
echo "Starting the Eagle Eye operations workspace at http://${EAGLE_EYE_ADDRESS}:${EAGLE_EYE_PORT}"
exec "$EAGLE_EYE_RUNTIME/bin/python" -m uvicorn src.api.server:app \
  --host "$EAGLE_EYE_ADDRESS" \
  --port "$EAGLE_EYE_PORT"
