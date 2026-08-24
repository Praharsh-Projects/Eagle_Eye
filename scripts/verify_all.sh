#!/usr/bin/env bash
set -euo pipefail

EAGLE_EYE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EAGLE_EYE_ROOT"

EAGLE_EYE_RUNTIME="${EAGLE_EYE_VENV:-$HOME/.venvs/eagle-eye}"
EAGLE_EYE_PYTHON="$EAGLE_EYE_RUNTIME/bin/python"
if [[ ! -x "$EAGLE_EYE_PYTHON" ]]; then
  echo "Eagle Eye runtime is missing. Run: $EAGLE_EYE_ROOT/scripts/bootstrap_runtime.sh" >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js/npm is required for commercial UI verification." >&2
  exit 1
fi

export PYTHONPATH="$EAGLE_EYE_ROOT"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$HOME/.cache/eagle-eye/pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX"
export EAGLE_EYE_ENABLE_MODEL_RESPONSES=false
export EAGLE_EYE_SKIP_DOTENV=true
unset OPENAI_API_KEY || true
unset AISSTREAM_API_KEY || true

echo "[1/8] Scanning tracked source and configuration for secrets"
"$EAGLE_EYE_ROOT/scripts/check_secrets.sh"

echo "[2/8] Running frozen human-authored regressions and the complete Python suite"
"$EAGLE_EYE_PYTHON" -m pytest -q -p no:cacheprovider

echo "[3/8] Checking canonical and compatibility API contracts"
"$EAGLE_EYE_PYTHON" "$EAGLE_EYE_ROOT/scripts/verify_api.py"

echo "[4/8] Building the commercial web application"
npm --prefix "$EAGLE_EYE_ROOT/web" run build

echo "[5/8] Auditing all web dependencies"
npm --prefix "$EAGLE_EYE_ROOT/web" audit --audit-level=moderate

echo "[6/8] Running Playwright visual, responsive, and axe accessibility gates"
npm --prefix "$EAGLE_EYE_ROOT/web" run test:e2e

EAGLE_EYE_VERIFY_PORT="${EAGLE_EYE_VERIFY_PORT:-18080}"
EAGLE_EYE_VERIFY_LOG="${TMPDIR:-/tmp}/eagle-eye-fastapi-verify.log"
EAGLE_EYE_SERVER_PID=""

cleanup() {
  if [[ -n "$EAGLE_EYE_SERVER_PID" ]] && kill -0 "$EAGLE_EYE_SERVER_PID" 2>/dev/null; then
    kill "$EAGLE_EYE_SERVER_PID" 2>/dev/null || true
    wait "$EAGLE_EYE_SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[7/8] Starting the offline production path"
: > "$EAGLE_EYE_VERIFY_LOG"
EAGLE_EYE_SKIP_DOTENV=true \
OPENAI_API_KEY= \
EAGLE_EYE_ENABLE_MODEL_RESPONSES=false \
PORT="$EAGLE_EYE_VERIFY_PORT" \
ADDRESS="127.0.0.1" \
"$EAGLE_EYE_ROOT/run_fastapi.sh" >"$EAGLE_EYE_VERIFY_LOG" 2>&1 &
EAGLE_EYE_SERVER_PID=$!

for _ in {1..60}; do
  if curl --silent --fail "http://127.0.0.1:${EAGLE_EYE_VERIFY_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$EAGLE_EYE_SERVER_PID" 2>/dev/null; then
    echo "FastAPI exited before becoming healthy." >&2
    tail -n 80 "$EAGLE_EYE_VERIFY_LOG" >&2
    exit 1
  fi
  sleep 0.25
done

echo "[8/8] Exercising the served UI and v2 capabilities"
curl --silent --fail "http://127.0.0.1:${EAGLE_EYE_VERIFY_PORT}/" | grep -q '<div id="root"></div>'
curl --silent --fail "http://127.0.0.1:${EAGLE_EYE_VERIFY_PORT}/api/v2/capabilities" \
  | "$EAGLE_EYE_PYTHON" -c 'import json,sys; payload=json.load(sys.stdin); assert payload["api_version"] == "2.0"; assert payload["freshness"]["historical"] is True'

echo "Eagle Eye verification passed. No live model calls were made."
