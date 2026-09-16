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

## V7-E11 persistent-Q structural candidate (accepted isolated scaffold)

E11 starts from E6 semantics rather than mixing in E10a's page staging.  It
keeps the exact pure-bit E4M3FN-to-BF16 decoder (`0x7f/0xff` fail closed to
zero), four-phase compact raw staging (K stage/decode, then V stage/decode),
the current stream, int64 block addressing, and the unchanged partial/combine
ABI.  Every tile takes the E6 path of one current-page `block_table` load;
there is no segment-local page-base prefetch.

The structural factor is query reuse.  Warps 0, 1, and 2 each own one 16-row
group.  Before the first KV tile, the owner converts its logical `[16,256]`
Q rows once into the padded `[16,264]` BF16 shared view and loads sixteen K16
row-major WMMA A fragments into registers.  Those fragments remain resident
across the entire segment tile loop.  For each tile, each owner warp then
sequentially computes the two N16 QK tiles (N=0..15 and N=16..31), using the
same Q fragments and the decoded K.  The three groups' 48x32 FP32 scores are
stored in the post-decode alias of the existing Q/P shared allocation.

After the score barrier, groups are handled in order: the owner warp performs
the causal online softmax, writes dense BF16 P `[16,32]` and FP32 alpha, and
the four warps cooperate on the unchanged 224-column PV main phase plus
warp-3's two-tile 32-column tail.  `tmp_shared` remains the `[16,224]` FP32
PV scratch; the Q/P allocation is only 8,448 B and aliases 8,192 B of raw
stage, 1,024 B of P, 6,144 B of three score packs, and alpha in disjoint
post-barrier lifetimes.  The persistent all-row FP16 accumulator and final
FP32 `part_o/m/l` publication remain unchanged.

The compile-time shared layout is still 81,664 B, under the SM80 98,304-B
dynamic limit and the prior two-CTA/SM target.  The main new resource risk is
the sixteen resident BF16 A fragments per participating thread.  Assuming
the SM80 BF16 fragments lower to four 32-bit registers each, the source-level
estimate was roughly 140--160 registers/thread (versus E6/E10a's 79/86).
Hardware measured 166 registers/thread, zero local bytes/spill and retained
two CTAs/SM because shared memory remains the occupancy limiter. Exhaustive
decode, full correctness and the 4-GiB high-block-ID gate passed.

The first 4K/70K/126K/200K/250K run measured
346.5/3,397.3/5,020.4/7,841.9/9,791.5 us per layer; repeated long tiers were
stable at about 5.01/7.85/9.79 ms. This improves E6 by about 38-39% from 126K
through 250K. NCU at 126K measured 447.44 M instructions, 9.28% long
scoreboard and 2.41% tensor activity versus E6's 711.7 M, 28.48% and 1.51%.
Barrier/short-scoreboard stalls are now 29.52%/13.97%, defining the next
optimization target. E11 is accepted only as the isolated CUDA scaffold; it
is not wired to production dispatch.

## V7-E10a split-local physical page/base staging (correct; rejected)

The historical source was the E10a tensor-core candidate.  Its fixed 81,664-byte
dynamic shared-memory layout is unchanged:

```text
all-row FP16 accumulator [48, 256]       24,576 B
one decoded BF16 K/V tile [2, 32, 264]    33,792 B
one BF16 Q/P pack [16, 264]                8,448 B
one FP32 score/PV temporary [16, 224]     14,336 B
reused LUT/page-base allocation [256]        512 B
                                           ------
                                           81,664 B
```

The 264-element leading dimensions add eight BF16 padding elements per
physical WMMA row only; the logical D remains 256.  P is repacked densely as
`[16,32]` with `ld=32`, and the Q/P pack's unused tail stores 16 FP32 alpha
values between softmax and PV fusion.  The FP32 scratch's main-phase rows use
`ld=224`; the tail reuses its first 16x32 entries with `ld=32`.  Before Q/P is
live, E10a aliases its first 8,192 B for one raw matrix at a time.  Each K/V
pass coalesced-stages aligned 16-byte `uint4` chunks, with a safe scalar
fallback for an unaligned external base and zero-fill for invalid token tails.
The four stage/decode barriers are unchanged from E6.  At segment start, the
same initial accumulator barrier publishes up to sixteen K/V page bases stored
in the former LUT allocation (256 B of the 512 B region).  Tile loads use the
local page entry and only add `tile_slot * stride_s`; an oversized segment
falls back to the old per-tile block-table load/barrier path without truncating
its page index.  The pure bit decoder maps `0x7f/0xff` to BF16 zero, and the
exported helper/test retains the 254-finite-bit-exact plus two-zero exhaustive
contract.  Accumulator, logical tmp/output indexing, launcher, current-stream
selection, int64 index and block-table dispatch, page arithmetic, and flattened
FP32 partial workspace ABI are unchanged. On CMP 170HX, E10a used 86
registers/thread, zero local bytes, 81,664 B shared and two active CTAs/SM.
All exhaustive, full correctness and 4-GiB high-block-ID gates passed. Three
performance runs showed only about 0.7-0.9% stable improvement at 126K-250K
versus E6, below admission; NCU at 126K measured 710.90 M instructions, 19.02%
barrier stalls, 28.08% long scoreboard and 1.52% tensor activity. It is
retained as a measured factorial, not a production candidate.

### E9b baseline (historical, hardware qualified but rejected)

E9b was the preceding FP8x2-to-half intrinsic factorial.  It passed the
exhaustive/correctness/resource gates but regressed the long-context tiers;
its measured result is retained below for comparison.

For finite E4M3FN code `(s,e,m)`, normal values use `(8+m)*2^(e-10)` and map
to BF16 exponent `e+120` with fraction `m<<4`; `e=0` subnormals are normalized
from the highest set mantissa bit.  Both NaN encodings (`0x7f`, `0xff`) follow
the qualified LUT's fail-closed policy and decode to BF16 bits `0x0000`, rather
than propagating PyTorch NaNs.  The bench's exported CUDA helper exhaustively
checks all 256 codes: 254 finite codes must match PyTorch BF16 bits exactly,
and the two NaN codes must return zero.  Accumulator, logical
tmp/output indexing, launcher, current-stream selection, int64 index and
block-table dispatch, page arithmetic, and flattened FP32 partial workspace
ABI are unchanged.

E9b compiled and passed the exhaustive, complete and high-block-ID gates on
the CMP 170HX. Resources remained 79 registers/thread, zero local bytes,
81,664 B shared and two CTAs/SM. Its 4K/70K/126K/200K/250K latency was
346.9/4,829.9/8,354.1/13,228.3/16,511.7 us. Relative to E6 this is
-8.3%/-2.6%/+2.8%/+3.0%/+2.9%: the paired intrinsic helps short context but
regresses every long tier. NCU at 126K measured 783.4 M instructions, 63.51 M
shared-load conflicts, 18.73% barrier stalls, 27.36% long-scoreboard stalls
and 1.46% tensor activity. It is rejected; E6 remains the performance base.

### E8 baseline (historical, correct but rejected)

The corrected exhaustive test passed all 254 finite encodings bit-exactly and
required `0x7f/0xff` to fail closed to zero. Resources and complete correctness
also passed. Five-tier latency was
407.8/4,924.6/7,979.2/12,692.6/15,781.1 us: 0.7-1.9% faster than E6 from
70K-250K but 7.7% slower at 4K. NCU instructions fell to 675.1 M, yet barrier
stalls remained 19.31%. E8 missed the admission thresholds and is retained as
a correct measured factorial, not the next performance base.

### E7 baseline (historical, hardware qualified but rejected)

All exhaustive/correctness/resource gates passed, but five-tier latency was
484.6/7,899.3/13,663.6/21,651.6/27,041.6 us, 28-69% slower than E6. NCU showed
the trade directly: barriers fell 19.23% -> 11.19%, while long-scoreboard
stalls rose 28.48% -> 54.34%, instructions rose 711.7 M -> 845.6 M and tensor
activity fell 1.51% -> 0.92%. Direct global loading is rejected; E8 preserves
the E6 bit decoder and reduces staged phases with separate K/V aliases.

### E6 baseline (historical, hardware qualified)

The E6 device helper passed an exhaustive all-256-code comparison against
PyTorch for all 254 finite codes; the two NaN encodings followed the qualified
LUT's fail-closed BF16-zero policy. Hardware retained 79 registers/thread, zero spill, 81,664 B shared
and two CTAs/SM; complete correctness/high-block-ID passed. Five-tier latency
was 378.5/4,957.3/8,130.3/12,845.7/16,041.2 us, improving E5 by
20.0%/37.9%/43.4%/43.8%/43.7%. NCU instructions fell from 1.093 B to 711.7 M,
long-scoreboard stalls from 51.99% to 28.48%, and tensor activity rose from
0.88% to 1.51%. Barrier stalls rose to 19.23%, motivating this direct-load
versus staged-load factorial. E6 remains an isolated hardware-qualified
scaffold, not a production integration.

### E5 baseline (historical, not an E10a qualification)

Hardware E5 retained 79 registers/thread, zero spill, 81,664 B shared and two
CTAs/SM; full correctness/high-block-ID passed. Five-tier latency was
473.4/7,984.3/14,355.3/22,863.6/28,509.2 us. This improved E4a at 4K/70K by
1.2%/2.9% but was flat at 126K-250K. NCU also stayed flat at 1.093 B
instructions, 51.99% long-scoreboard stalls, 0.88% tensor activity and
72.22 M shared-load conflicts. It is rejected for long-context admission.

### E4b baseline (historical, not an E10a qualification)

Hardware E4b retained 79 registers/thread, zero local spill, 81,664 B shared and
two CTAs/SM. Full correctness/high-block-ID passed. However,
4K/70K/126K/200K/250K measured
541.5/8,240.1/14,349.7/22,862.3/28,517.7 us, a 13.0% short-context regression
and no meaningful long-context gain over E4a. NCU still measured 1.093 B
instructions, 52.00% long-scoreboard stalls, 0.88% tensor activity and
72.22 M shared-load conflicts. Direct vector load plus register unpack did not
shorten the feed dependency chain; E5 replaced that path with shared raw
staging, and E6 replaces its shared-LUT hot read with bitwise decode.

### E4a baseline (historical, not an E10a qualification)

On CMP 170HX hardware E4a compiled to 79 registers/thread, zero local spill,
81,664 B dynamic shared and two active CTAs/SM. The complete correctness and
4-GiB high-block-ID gate passed (largest high-block error 0.0625). Its
4K/70K/126K/200K/250K latency was
479.3/8,222.1/14,338.5/22,858.5/28,522.1 us per layer. This was a 6.1%/5.6%
improvement over E2 at 126K/250K but still about 18-19x slower than Triton.
NCU retained 51.97% long-scoreboard stalls, only 0.88% tensor-pipe activity,
and 72.22 M shared-load conflicts. E4b changed only raw-load vectorization
and per-tile address-base preparation; the measured result above rejected it.

### E3b baseline (historical, not an E10a qualification)

The E3b padded source used 81,152 B dynamic shared memory and two CTAs/SM.
Full correctness/high-block-ID passed.  The five-tier scan at
4K/70K/126K/200K/250K measured
502.7/8,093.1/14,481.3/23,061.9/28,746.7 us per layer.  This was about 5%
faster than E2 at long contexts but 3% slower at 4K and still 18-19x behind
Triton.  NCU retained 52.26% long-scoreboard stalls, motivating E4a's shared
LUT feed change.

### E3a baseline (historical, not an E10a qualification)

The padded E3a source used the same Q/K/V physical leading dimension of 264
but retained a 16x256 FP32 temporary, for 83,200 B total.  On hardware it
compiled to 79 registers/thread with zero spill and passed the complete
correctness/high-block-ID gate.  Padding reduced shared-load bank conflicts
from 190.54 M to 63.51 M, but 83,200 B reduced occupancy from two CTAs/SM to
one.  It regressed at every tier: 4K/70K/126K/200K/250K measured
882.3/13,220.8/23,600.3/37,307.3/46,681.1 us per layer.  E3b recovered 2,048 B
of temporary storage while preserving padding and restored two CTAs/SM.

### E2 baseline (historical, not an E10a qualification)

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
Every non-build invocation first runs the exported CUDA E4M3FN decoder helper
over all 256 encodings, requiring finite BF16 bit equality with PyTorch and
explicit NaN-code checks requiring BF16-zero fail-closed bits.

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
