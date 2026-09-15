#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
PY=${PY:-$ROOT/venv/bin/python}
MODE=dry-run
SITE=${VLLM_SITE:-}

usage() {
  cat <<'EOF'
Usage: install.sh [--dry-run|--apply|--check|--reverse] [--site PATH]

Applies only the CMP 170HX mixed-FP8 experimental series. The repository's
normal patches must already be installed in the target vLLM package.

PY=/path/to/python may be used instead of --site. --reverse is intended only
for disposable test trees; rebuilding a clean environment is safer.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) MODE=dry-run ;;
    --apply) MODE=apply ;;
    --check) MODE=check ;;
    --reverse) MODE=reverse ;;
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
  SITE=$($PY - <<'PY'
import inspect
import pathlib
import vllm
print(pathlib.Path(inspect.getfile(vllm)).resolve().parent)
PY
  )
fi

[[ -f "$SITE/__init__.py" ]] || {
  echo "Not a vLLM package directory: $SITE" >&2
  exit 2
}

mapfile -t SERIES < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$HERE/series")
((${#SERIES[@]})) || { echo "Empty series: $HERE/series" >&2; exit 2; }

require_base() {
  grep -q 'class SpecDecodeAttention' "$SITE/v1/attention/ops/spec_decode_attn.py"
  grep -q 'k_scale_cache' "$SITE/v1/attention/ops/spec_decode_attn.py"
  grep -q 'to(tl.int64)' "$SITE/v1/attention/ops/spec_decode_attn.py"
}

if ! require_base; then
  echo "The normal split-KV and INT8 compatibility patches are not installed in $SITE" >&2
  exit 1
fi

run_forward() {
  local name=$1 patch_file="$HERE/patches/$1"
  [[ -f "$patch_file" ]] || { echo "Missing patch: $patch_file" >&2; exit 1; }
  case "$MODE" in
    dry-run)
      echo "DRY-RUN $name"
      patch -p1 --dry-run --forward --batch -d "$SITE" < "$patch_file"
      ;;
    apply)
      echo "APPLY $name"
      patch -p1 --forward --batch -d "$SITE" < "$patch_file"
      ;;
    check)
      echo "CHECK $name"
      patch -p1 -R --dry-run --batch -d "$SITE" < "$patch_file"
      ;;
  esac
}

if [[ "$MODE" == reverse ]]; then
  for ((i=${#SERIES[@]}-1; i>=0; i--)); do
    name=${SERIES[$i]}
    patch_file="$HERE/patches/$name"
    echo "REVERSE $name"
    patch -p1 -R --batch -d "$SITE" < "$patch_file"
  done
else
  for name in "${SERIES[@]}"; do
    run_forward "$name"
  done
fi

echo "mixed-FP8 experimental series: $MODE OK ($SITE)"

