#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
PY=${PY:-$ROOT/venv/bin/python}
SITE=${VLLM_SITE:-}
MODE=--dry-run

usage() {
  cat <<'EOF'
Usage: install.sh [--dry-run|--apply|--check|--reverse] [--site PATH]

Installs only the verifier-only sparse-read oracle prototype.

Prerequisite:
  1. normal repository vLLM 0.27.1 patch stack
  2. experimental/cmp170hx-mixed-fp8/install.sh --apply

The prototype is intentionally separate from M7.  It can be reversed without
touching the qualified mixed-FP8 series.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) MODE=--dry-run ;;
    --apply) MODE=--apply ;;
    --check) MODE=--check ;;
    --reverse) MODE=--reverse ;;
    --site)
      shift
      (($#)) || { echo "--site requires a path" >&2; exit 2; }
      SITE=$1
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ -z "$SITE" ]]; then
  [[ -x "$PY" ]] || { echo "Python not executable: $PY" >&2; exit 2; }
  SITE=$("$PY" - <<'PY'
import inspect
import pathlib
import vllm
print(pathlib.Path(inspect.getfile(vllm)).resolve().parent)
PY
  )
fi

TARGET="$SITE/v1/attention/ops/spec_decode_attn.py"
[[ -f "$TARGET" ]] || { echo "Not a vLLM package: $SITE" >&2; exit 2; }

# Fail closed if somebody tries to install over the pre-M7 verifier.
grep -q '_spec_attn_partial_q8_g6_fp8' "$TARGET"
grep -q '_e4m3fn_to_bf16_ldg_nc' "$TARGET"
grep -q 'SEG_TILE=triton.next_power_of_2(self.nseg)' "$TARGET"

exec "$PY" "$HERE/apply_sparse_read.py" "$MODE" --site "$SITE"
