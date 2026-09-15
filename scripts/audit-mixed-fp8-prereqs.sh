#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-$ROOT/venv/bin/python}

fail=0
ok() { printf 'OK   %s\n' "$*"; }
warn() { printf 'WARN %s\n' "$*"; }
bad() { printf 'FAIL %s\n' "$*"; fail=1; }

need_file() {
  local path=$1
  if [[ -f "$ROOT/$path" ]]; then ok "$path"; else bad "missing $path"; fi
}

need_grep() {
  local pattern=$1 path=$2 label=${3:-$2}
  if grep -Eq "$pattern" "$ROOT/$path" 2>/dev/null; then
    ok "$label"
  else
    bad "$label (pattern not found in $path)"
  fi
}

echo '== repository prerequisites =='
need_file patches/dflash2-backport.patch
need_file patches/spec-decode-attn.patch
need_file patches/spec-decode-int8-kv.patch
need_file patches/hybrid-kv-groups-v2-cudagraph.patch
need_file patches/hybrid-sw-block-promote.patch
need_file patches/marlin-repack-staged-sm80.patch
need_grep 'to\(tl\.int64\)' patches/spec-decode-attn.patch 'split-KV block ids use int64'
need_grep 'k_scale_cache' patches/spec-decode-int8-kv.patch 'quantized verifier passes scale metadata'
need_grep 'SPEC_CFG=' single-user/start_qwen.sh 'DFlash speculative config is launcher-owned'

echo
echo '== installed vLLM prerequisites =='
if [[ ! -x "$PY" ]]; then
  warn "$PY not found; repository audit complete, installed-package audit skipped"
  exit "$fail"
fi

SITE=$($PY - <<'PY'
import inspect, pathlib, vllm
print(pathlib.Path(inspect.getfile(vllm)).resolve().parent)
PY
)

printf 'vLLM package: %s\n' "$SITE"

check_installed() {
  local rel=$1 pattern=$2 label=$3
  local f="$SITE/$rel"
  if [[ ! -f "$f" ]]; then
    bad "$label (missing $rel)"
  elif grep -Eq "$pattern" "$f"; then
    ok "$label"
  else
    bad "$label (pattern not found in $rel)"
  fi
}

check_installed 'config/speculative.py' 'attention_backend' 'draft-specific attention backend field'
check_installed 'config/speculative.py' 'kv_cache_dtype' 'draft-specific KV dtype field'
check_installed 'v1/worker/gpu/spec_decode/dflash/utils.py' 'backend=speculative_config\.attention_backend' 'DFlash loader consumes draft backend override'
check_installed 'v1/worker/gpu/spec_decode/dflash/utils.py' 'cache_dtype=speculative_config\.kv_cache_dtype' 'DFlash loader consumes draft KV dtype override'
check_installed 'v1/attention/ops/spec_decode_attn.py' 'to\(tl\.int64\)' 'installed split-KV high-block-id fix'

if grep -Rqs 'INT8_PER_TOKEN_HEAD' "$SITE/v1/attention"; then
  ok 'installed quantized split-KV support detected'
else
  warn 'INT8 per-token-head split-KV support not detected; patch application may be incomplete'
fi

if grep -Rqs 'VLLM_ALIGN_HETEROGENEOUS_ATTN_PAGES' "$SITE"; then
  ok 'private-style heterogeneous-page environment hook already exists'
else
  warn 'VLLM_ALIGN_HETEROGENEOUS_ATTN_PAGES not present (expected before reconstruction)'
fi

if grep -Rqs 'VLLM_FP8_SPEC_FULL_CG' "$SITE"; then
  ok 'private-style FP8 spec CUDA-graph hook already exists'
else
  warn 'VLLM_FP8_SPEC_FULL_CG not present (expected before reconstruction)'
fi

exit "$fail"
