#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(cd -- "$HERE/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$ROOT/bench/test_v7_cuda_prototype.py" "$@"
