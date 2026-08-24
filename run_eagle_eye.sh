#!/usr/bin/env bash
set -euo pipefail

EAGLE_EYE_ROOT="$(cd "$(dirname "$0")" && pwd)"

# The custom React operations workspace is the public default. FastAPI serves
# the production build and the unchanged canonical query service together.
export PORT="${PORT:-8000}"
export ADDRESS="${ADDRESS:-127.0.0.1}"
exec "$EAGLE_EYE_ROOT/run_fastapi.sh"
