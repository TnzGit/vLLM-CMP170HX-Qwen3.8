# V7 standalone CUDA verifier prototype

This directory contains an isolated CUDA C++ candidate for the CMP 170HX
SM80 verifier.  It is intentionally not imported by vLLM, FlashInfer, the
active `series`, or any production dispatch.

## Frozen geometry

```text
SM       80
Hq/Hkv   24/4 (GQA group = 6)
D        256
QMAX     8
page     896 tokens
tile     32 tokens
NSEG     35
CTA      4 warps / 128 threads
```

`partial` launches one CTA per `(request, KV head, segment)` and writes the
existing flattened split-KV workspace:

```text
part_o[((req * Hq + h) * QMAX + qi) * NSEG + seg, d]  FP32
part_m[((req * Hq + h) * QMAX + qi) * NSEG + seg]      FP32
part_l[((req * Hq + h) * QMAX + qi) * NSEG + seg]      FP32
```

`combine` consumes those three tensors and writes the caller's `[tokens, Hq,
D]` BF16/FP16 output.  The raw K/V tensors are contiguous `uint8` views of
E4M3FN storage with NHD layout `[physical_block, 896, 4, 256]`; `fp8_lut` is
the same 256-entry BF16 decode table used by the experimental Triton path.
Static K and V scales are applied separately to scores and values.

The partial kernel keeps all four warps active on every tile: each warp owns
12 query/group rows, each lane owns one score token and eight output columns.
K/V are copied into two shared-memory stages with SM80 `cp.async`; the next
stage is issued before the current stage is computed.  Empty/causal tails are
zero-filled, and physical block IDs are loaded/promoted as `int64_t` before
multiplication by cache strides.

## E1 on-chip accumulator result (rejected for throughput)

The current source is the E1 structural candidate.
Shared memory is fixed at 81,920 bytes: scaled Q is a BF16 raw-`uint16` tile
(48 x 256 x 2 = 24,576 B), matching the Triton conversion for both BF16 and
FP16 callers; the persistent value accumulator is an FP16
raw-`uint16` tile of the same size, and the two K/V stages use 32,768 B.  The
online `m/l` state is retained in per-warp lane-zero register arrays and
broadcast for each row.  `part_o/m/l` remain the existing FP32 workspace ABI,
but are written only once per segment after the tile loop; there is no
per-tile global workspace traffic.

E1 compiled and passed the complete correctness gate on the CMP 170HX.  It used
126 registers/thread, zero local spill and two CTAs/SM, but scalar QK/PV math
made it about 19x slower at 4K and 28-29x slower at 70K-250K than the qualified
Triton verifier.  E1 is therefore a documented structural scaffold, not a
throughput candidate.  The next implementation must use SM80 BF16 tensor-core
MMA while retaining the workspace/addressing contract.

`resources()` reports `cudaFuncGetAttributes` and an occupancy estimate for the
canonical BF16/int32 specialization after applying the dynamic shared-memory
carveout.  It is diagnostic only; it does not alter dispatch.

## Build and smoke test

Run on an SM80 host with the vLLM/PyTorch CUDA environment (CUDA 12/13 and
`nvcc` available):

```bash
python bench/test_v7_cuda_prototype.py --build-only
python bench/test_v7_cuda_prototype.py
# equivalent wrapper from this directory:
experimental/cmp170hx-mixed-fp8/cuda_prototype/build_and_smoke.sh
```

Use `PYTHON_BIN=/path/to/the/cuda-python` with the wrapper when the host has
more than one Python installation.

The bench JIT-builds `v7_verifier.cu` with `-gencode=arch=compute_80,code=sm_80`
and tests 895/896/897-token boundaries, a 4097-token request, mixed request
lengths, and int64 index/block-table dispatch.  Set `TORCH_EXTENSIONS_DIR` to
a writable build cache if the default PyTorch extension cache is unsuitable.

On a non-CUDA or non-SM80 workstation the bench exits with an explicit `SKIP`;
that is expected and is not a kernel qualification result.

## E0 qualification result (historical baseline)

The out-of-tree extension was compiled on the CMP 170HX using Torch
2.13.0+cu130, CUDA 13.0.88 and G++ 13.3.  The canonical BF16/int32 partial
specialization reported:

```text
threads/CTA              128
registers/thread          48
dynamic shared        81,920 B
static shared             16 B
local bytes                 0
active CTAs/SM              2
```

The standalone smoke passed 895/q5, 896/q8 and mixed 897/q5 + 4097/q8 cases,
including the int64 index/block-table dispatch.  Maximum absolute error was
0.000977 in all three cases.  This validates the build, ABI, page-boundary and
basic numerical contract only; it is not an E1 throughput result.

The expanded command below adds q=6/7, mixed q lengths, 8K/32K/65K KV and a
physical block ID above the signed-int32 element-offset boundary:

```bash
python bench/test_v7_cuda_prototype.py --full --high-block-id
```

It passed on the CMP 170HX; the high-ID case allocated 4.00 GiB across the two
raw caches and produced maximum absolute error 0.031250 (<0.08).

For isolated latency against the qualified Triton module:

```bash
python bench/spec_attn_fp8_ctx_scan.py \
  --contexts 4096,70000,126000,200000,250000 \
  --queries 8 --segments 35 --warmup 10 --iters 50 \
  --module experimental/cmp170hx-mixed-fp8/cuda_prototype/spec_decode_attn_v7.py
```

## Scope and limitations

This is a correctness-first prototype, not a production performance claim.
It requires contiguous NHD raw-byte caches and BF16/FP16 Q/output, assumes
`0 <= query_len <= 8`, and does not implement per-token INT8 KV, sliding
windows, softcaps, ALiBi, sinks, DCP, CUDA graph integration, or FlashInfer
dispatch.  The high-block-ID arithmetic is present and int64-dispatched, but a
true sparse high-ID GPU run needs a physically large/specially allocated KV
pool and is not fabricated by the local bench.
