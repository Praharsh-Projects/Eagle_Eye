#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

exec python3 "$ROOT_DIR/scripts/secret_scan.py" \
  --root "$ROOT_DIR" \
  --allowlist "$ROOT_DIR/scripts/secret_scan_allowlist.json"
