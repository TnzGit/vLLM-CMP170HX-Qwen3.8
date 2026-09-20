#!/usr/bin/env bash
set -euo pipefail

# Start one research-only sparse-read gate arm on port 8002.
#
# Usage:
#   bench/sparse-read/start_gate_server.sh dense
#   bench/sparse-read/start_gate_server.sh 32000
#   bench/sparse-read/start_gate_server.sh 48000
#   bench/sparse-read/start_gate_server.sh 65000
#
# This wrapper freezes the production long-profile knobs so the executor does
# not tune while running the gate.  Override MODEL/DRAFT/VENV/PYTHONPATH only if
# the host paths differ.

ARM=${1:-}
case "$ARM" in
  dense|32000|48000|65000) ;;
  *) echo "arm must be dense, 32000, 48000, or 65000" >&2; exit 2 ;;
esac

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VENV=${VENV:-/home/base-node/.codex_tasks/pixelml-cmp170hx/runtime-v0271/venv}
export PYTHONPATH=${PYTHONPATH:-/home/base-node/.codex_tasks/pixelml-cmp170hx/mixed-fp8-test-site}
export MODEL=${MODEL:-/home/base-node/models/Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4}
export DRAFT=${DRAFT:-/home/base-node/models/Qwen3.8-27B-DFlash2-W4A16}

export SPEC=dflash2
export CTX=cmp-mixed-fp8
export MAX_LEN=262144
export DFLASH_MAX_LEN=262144
export MAX_SEQS=1
export DFLASH_TOKENS=3
export LOOKUP=0
export PREFIX_CACHE=0
export VISION=0
export SPEC_ATTN=1
export VLLM_SPEC_DECODE_ATTN_SEGMENTS=35
export VLLM_FP8_SPEC_VERIFY=1
export VLLM_FP8_SPEC_FULL_CG=1
export VLLM_ALIGN_HETEROGENEOUS_ATTN_PAGES=1
export CUDAGRAPH_MODE=FULL
export CMP_TARGET_ATTN_BACKEND=FLASHINFER
export CMP_TARGET_KV_DTYPE=fp8
export DFLASH_ATTN_BACKEND=FLASH_ATTN
export DFLASH_KV_CACHE_DTYPE=bfloat16
export PORT=8002
export GPU_UTIL=0.90
export KV_MEM=
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export EXTRA_ARGS=${EXTRA_ARGS:---enable-prompt-tokens-details}

if [[ "$ARM" == dense ]]; then
  export VLLM_SPARSE_KV_READ=0
  unset VLLM_SPARSE_KV_READ_TOKENS || true
else
  export VLLM_SPARSE_KV_READ=1
  export VLLM_SPARSE_KV_READ_TOKENS="$ARM"
  export VLLM_SPARSE_KV_READ_SINK_TOKENS=${VLLM_SPARSE_KV_READ_SINK_TOKENS:-1024}
  export VLLM_SPARSE_KV_READ_MAX_BLOCKS=${VLLM_SPARSE_KV_READ_MAX_BLOCKS:-8192}
fi

[[ "$PORT" == 8002 ]] || { echo "research wrapper refuses PORT=$PORT (must be 8002)" >&2; exit 2; }
[[ -x "$VENV/bin/python" ]] || { echo "missing VENV: $VENV" >&2; exit 2; }

RUNTIME_SITE="$VENV/lib/python3.12/site-packages/vllm"
SHADOW_SITE="$PYTHONPATH/vllm"
for site in "$RUNTIME_SITE" "$SHADOW_SITE"; do
  [[ -f "$site/v1/attention/ops/spec_decode_attn.py" ]] || {
    echo "missing vLLM site: $site" >&2; exit 2;
  }
  grep -q '_spec_attn_partial_q8_g6_fp8' "$site/v1/attention/ops/spec_decode_attn.py"
  if [[ "$ARM" != dense ]]; then
    grep -q 'BEGIN cmp170hx sparse-read oracle' "$site/v1/attention/ops/spec_decode_attn.py" || {
      echo "sparse-read patch missing from $site" >&2; exit 2;
    }
  fi
done

echo "[sparse-gate] arm=$ARM port=$PORT k=$DFLASH_TOKENS max_seqs=$MAX_SEQS nseg=$VLLM_SPEC_DECODE_ATTN_SEGMENTS"
echo "[sparse-gate] model=$MODEL"
echo "[sparse-gate] draft=$DRAFT"
echo "[sparse-gate] PYTHONPATH=$PYTHONPATH"

exec "$REPO/single-user/start_qwen.sh"
