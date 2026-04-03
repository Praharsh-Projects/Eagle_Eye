#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

BASE_URL="${BASE_URL:-http://127.0.0.1:8501}"
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
SCENARIOS_PATH="${SCENARIOS_PATH:-review/scenarios.json}"
OUTPUT_DIR="${OUTPUT_DIR:-review/latest}"
LOG_DIR="${LOG_DIR:-review/logs}"
TIMEOUT_MS="${TIMEOUT_MS:-45000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
NO_API_CHECKS="${NO_API_CHECKS:-0}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  STREAMLIT_BIN="$ROOT_DIR/.venv/bin/streamlit"
else
  PYTHON_BIN="$(command -v python3 || true)"
  STREAMLIT_BIN="$(command -v streamlit || true)"
fi

if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python not found. Create .venv or install python3." >&2
  exit 1
fi
if [[ -z "${STREAMLIT_BIN:-}" || ! -x "$STREAMLIT_BIN" ]]; then
  echo "Error: streamlit binary not found. Install dependencies first." >&2
  exit 1
fi
if [[ ! -f "$SCENARIOS_PATH" ]]; then
  echo "Error: scenario file not found: $SCENARIOS_PATH" >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found." >&2
    exit 1
  fi
}

require_cmd curl

port_in_use() {
  local port="$1"
  lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

wait_http() {
  local url="$1"
  local seconds="$2"
  local name="$3"
  local deadline=$((SECONDS + seconds))
  while (( SECONDS < deadline )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[ok] $name ready: $url"
      return 0
    fi
    sleep 1
  done
  echo "Error: timed out waiting for $name at $url" >&2
  return 1
}

cleanup() {
  set +e
  if [[ -n "${UI_PID:-}" ]]; then kill "$UI_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${API_PID:-}" ]]; then kill "$API_PID" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

if [[ "$SKIP_BOOTSTRAP" != "1" ]]; then
  echo "[1/7] Installing UI review dependencies (playwright if missing)"
  "$PYTHON_BIN" -c "import playwright" >/dev/null 2>&1 || "$PYTHON_BIN" -m pip install playwright
  "$PYTHON_BIN" -m playwright install chromium >/dev/null 2>&1 || true
else
  echo "[1/7] Skipping dependency bootstrap (SKIP_BOOTSTRAP=1)"
fi

if port_in_use 8501; then
  echo "Error: port 8501 already in use. Stop existing Streamlit process first." >&2
  exit 1
fi
if [[ "$NO_API_CHECKS" != "1" ]] && port_in_use 8000; then
  echo "Error: port 8000 already in use. Stop existing API process first or set NO_API_CHECKS=1." >&2
  exit 1
fi

echo "[2/7] Starting Streamlit"
PYTHONPATH=. "$STREAMLIT_BIN" run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true > "$LOG_DIR/streamlit.log" 2>&1 &
UI_PID=$!
wait_http "$BASE_URL" 120 "streamlit"

if [[ "$NO_API_CHECKS" != "1" ]]; then
  echo "[3/7] Starting FastAPI"
  PYTHONPATH=. "$PYTHON_BIN" -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 > "$LOG_DIR/api.log" 2>&1 &
  API_PID=$!
  wait_http "$API_BASE_URL/health" 120 "api"
else
  echo "[3/7] API checks disabled (NO_API_CHECKS=1)"
fi

echo "[4/7] Running UI audit scenarios"
CMD=("$PYTHON_BIN" -m src.review.ui_audit --base-url "$BASE_URL" --scenarios "$SCENARIOS_PATH" --output-dir "$OUTPUT_DIR" --timeout-ms "$TIMEOUT_MS" --max-attempts "$MAX_ATTEMPTS")
if [[ "$NO_API_CHECKS" == "1" ]]; then
  CMD+=(--no-api-checks)
else
  CMD+=(--api-base-url "$API_BASE_URL")
fi

set +e
"${CMD[@]}"
AUDIT_EXIT=$?
set -e

echo "[5/7] Writing run history copy"
RUN_ID="$("$PYTHON_BIN" - << 'PY'
import json
from pathlib import Path
p = Path('review/latest/review_index.json')
if not p.exists():
    print('missing')
else:
    data = json.loads(p.read_text(encoding='utf-8'))
    print(data.get('run_id','unknown'))
PY
)"
if [[ "$RUN_ID" != "missing" ]]; then
  mkdir -p "review/runs/$RUN_ID"
  rm -rf "review/runs/$RUN_ID"/*
  cp -R "$OUTPUT_DIR"/* "review/runs/$RUN_ID"/
fi

echo "[6/7] Local artifacts"
echo "- JSON: $ROOT_DIR/$OUTPUT_DIR/review_index.json"
echo "- Summary: $ROOT_DIR/$OUTPUT_DIR/review_summary.md"
echo "- Screenshots: $ROOT_DIR/$OUTPUT_DIR/screenshots"
if [[ "$RUN_ID" != "missing" ]]; then
  echo "- Run archive: $ROOT_DIR/review/runs/$RUN_ID"
fi

echo "[7/7] Done (exit code=$AUDIT_EXIT)"
if [[ $AUDIT_EXIT -ne 0 ]]; then
  echo "UI audit reported failures. Open review_summary.md for scenario-level details." >&2
fi

exit $AUDIT_EXIT
