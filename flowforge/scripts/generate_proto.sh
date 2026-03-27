#!/usr/bin/env bash
set -euo pipefail

python -m grpc_tools.protoc \
  -I ./proto \
  --python_out=./api/grpc/generated \
  --grpc_python_out=./api/grpc/generated \
  ./proto/flowforge.proto
