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

The V7-E3b partial candidate keeps the same four-warp CTA and fixed geometry,
but maps QK/PV to SM80 BF16 WMMA tensor cores.  Each tile is first decoded by
the full CTA from raw FP8 bytes through the supplied BF16 LUT into one shared
K/V tile; no double buffer is required.  The 48 query/group rows are processed
as three sequential 16-row packs.  Q/K/V use a padded physical leading
dimension of 264 elements while their logical head dimension remains 256:
Q is `[16,264]`, and K/V are `[32,264]` token-major.  QK therefore uses two
N16 WMMA tiles (`A` row-major logical 16x256, `B` col-major logical 256x32,
both `ld=264`) and stores scores in the first 2 KiB of the unchanged FP32
temporary tile.  Warp 0 lanes 0..15 run the 32-item causal online softmax,
write dense BF16 P with `ld=32`, and preserve FP32 alpha.  PV is split into a
224-column main phase and a 32-column tail phase: warps 0..2 each compute four
N16 tiles for d=0..223, warp 3 computes two, then warp 3 computes the two tail
tiles d=224,240 and stores them through the scratch prefix as a dense `ld=32`
tile.  Each phase merges `previous * alpha + tmp` into the unchanged all-row
FP16 accumulator.  Empty/causal tails remain masked, and physical block IDs
are loaded/promoted as `int64_t` before multiplication by cache strides.

## V7-E3b padded-WMMA scaffold (correct; not production-fast)

The current source is the E3b tensor-core candidate.  Its fixed 81,152-byte
dynamic shared-memory layout is:

```text
all-row FP16 accumulator [48, 256]       24,576 B
one decoded BF16 K/V tile [2, 32, 264]    33,792 B
one BF16 Q/P pack [16, 264]                8,448 B
one FP32 score/PV temporary [16, 224]     14,336 B
                                           ------
                                           81,152 B
```

The 264-element leading dimensions add eight BF16 padding elements per
physical WMMA row only; the logical D remains 256.  P is repacked densely as
`[16,32]` with `ld=32`, and the Q/P pack's unused tail stores 16 FP32 alpha
values between softmax and PV fusion.  The FP32 scratch's main-phase rows use
`ld=224`; the tail reuses its first 16x32 entries with `ld=32`.  Accumulator,
logical tmp/output indexing, launcher, current-stream selection, int64 index
and block-table dispatch, page arithmetic, and flattened FP32 partial
workspace ABI are unchanged.

On hardware E3b compiled to 79 registers/thread, zero spill, 81,152 B dynamic
shared and two CTAs/SM.  Full correctness/high-block-ID passed.  The five-tier
scan at 4K/70K/126K/200K/250K measured
502.7/8,093.1/14,481.3/23,061.9/28,746.7 us per layer.  This is about 5%
faster than E2 at long contexts but 3% slower at 4K and still 18-19x behind
Triton.  NCU retained 52.26% long-scoreboard stalls, so E4 starts with the
FP8 decode/LUT feed path rather than more WMMA changes.

### E3a baseline (historical, not an E3b qualification)

The padded E3a source used the same Q/K/V physical leading dimension of 264
but retained a 16x256 FP32 temporary, for 83,200 B total.  On hardware it
compiled to 79 registers/thread with zero spill and passed the complete
correctness/high-block-ID gate.  Padding reduced shared-load bank conflicts
from 190.54 M to 63.51 M, but 83,200 B reduced occupancy from two CTAs/SM to
one.  It regressed at every tier: 4K/70K/126K/200K/250K measured
882.3/13,220.8/23,600.3/37,307.3/46,681.1 us per layer.  E3b recovered 2,048 B
of temporary storage while preserving padding and restored two CTAs/SM.

### E2 baseline (historical, not an E3b qualification)

The unpadded E2 source used 81,920 B of dynamic shared memory and passed the
complete correctness/high-block-ID gate after correcting P's compact stride
from 256 to 32.  On the CMP 170HX it compiled to 88 registers/thread, zero
spill, and two CTAs/SM, but was rejected for throughput: 4K/126K/250K measured
488.2/15,269.0/30,208.5 us per layer, roughly 9-20x slower than the qualified
Triton path.

NCU confirmed that both paths execute 6,048,768 tensor-pipe instructions at
126K, but E2 had 190.54 M shared-load bank conflicts versus Triton's 2.02 M,
51.20% versus 12.09% long-scoreboard stalls, and only 0.82% versus 17.21%
tensor-pipe active time.  E3a changed only those BF16 shared leading
dimensions; it reduced conflicts by about 3x but lost the second resident CTA.

## E1 on-chip accumulator result (rejected for throughput)

The preceding E1 source was the structural candidate.
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
throughput candidate.  E2 is the follow-on SM80 BF16 tensor-core candidate and
retains the same workspace/addressing contract.

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
basic numerical contract only; it is not an E2 qualification result.

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
