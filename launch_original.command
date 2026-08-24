#!/bin/zsh
set -e

EAGLE_EYE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$EAGLE_EYE_ROOT"
PORT=8000 ADDRESS=127.0.0.1 ./run_eagle_eye.sh
