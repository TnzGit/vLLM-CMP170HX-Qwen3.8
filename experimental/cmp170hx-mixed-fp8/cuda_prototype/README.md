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

## V7-E13b padded score rows (accepted isolated scaffold)

E13b restores E12's dense FP32 PV scratch and changes only each owner's score
pack from logical/physical `[16,32]` to logical `[16,32]`, physical
`[16,36]`. The three packs remain inside the 8,192-byte post-staging alias.
Resources stayed at 164 registers/thread, zero local bytes/spills, 81,664 B
shared and two CTAs/SM; full correctness and the 4-GiB high-block-ID gate
passed.

With graphics clocks locked to 1350MHz, three high-iteration runs measured
the following E13b medians (us/layer): 4K 245.7, 10K 460.4, 20K 778.5,
40K 1,406.6, 60K 2,032.1, 64K 2,169.7, 70K 2,343.3, 90K 2,964.6,
126K 4,075.0, 200K 6,367.0 and 250K 7,927.3. Against matched E12 lock
baselines, the gains are 3.7-6.8% at every tier. An earlier unlocked run was
discarded because Boost-state drift reversed the short-tier comparison.

At 126K, NCU measured 447.13 M instructions, 6.049 M tensor instructions,
3.24% tensor activity, 14.83% barrier, 11.46% long scoreboard, 15.85% short
scoreboard, 0.10% MIO throttle, 27.230 M/23.571 M shared load/store conflicts
and 258.22 MB DRAM read. E13b is accepted as the new isolated scaffold, but
remains disconnected from production dispatch.

## V7-E14 padded BF16 P rows (accepted secondary scaffold)

E14 keeps E13b's FP32 score `ld=36` and pads each logical BF16 P row from 32 to
40 elements (`ld=40`) for the PV WMMA loads. The three physical P packs occupy
3,840 B within the existing 14,336-B temporary allocation. Resources remain
164 registers/thread, zero local bytes/spills, 81,664 B shared and two CTAs/SM;
all correctness and high-block-ID gates pass.

With graphics clocks locked at 1350MHz, the E14 medians (us/layer) were 240.1
(4K), 751.8 (20K), 1,966.0 (60K), 3,945.8 (126K), 6,172.1 (200K) and 7,689.2
(250K), a further 2.3-3.4% over E13b. NCU at 126K measured 9.081 M shared
loads and 14.486 M stores, 3.33% tensor activity, 14.62% barrier and 14.80%
short-scoreboard stalls. E14 is accepted as an isolated secondary scaffold,
not production dispatch.

## V7-E15 persistent accumulator padding (rejected)

E15 changed only the persistent FP16 accumulator's physical row stride from
logical `ld=256` to `ld=258`; logical output indexing, K/V and P/score layouts,
the ABI and barriers were unchanged. The resource-safe version compiled with
81,856B dynamic shared (81,920B allocator round), 164 registers/thread, zero
local bytes/spills and two active CTAs/SM. Exhaustive correctness, int32/int64
and high-block-ID tests passed.

At locked 1350MHz, three 300-iteration scans over 4K/20K/60K/126K/200K/250K
were stable but indistinguishable from E14 (query-8 medians
239.7/751.5/1,965.2/3,949.6/6,175.2/7,683.7 us/layer; all differences about
±0.1%). NCU at 126K measured 9.079M shared-load and 14.481M shared-store
conflicts, 3.33% tensor activity, 14.63% barrier, 10.84% long-scoreboard and
14.49% short-scoreboard stalls—effectively the same profile as E14. E15 is
retained only as a negative control and is not an accepted optimization.

## V7-E13a padded PV scratch (correct; rejected)

E13a changed only each owner warp's FP32 WMMA scratch from physical
`[16,16]`, `ld=16` to `[16,20]`, `ld=20`. Float accumulator stores permit a
leading dimension divisible by four, and all three padded slices still fit in
the retained `tmp_shared` allocation. Resources remained 164 registers/thread,
zero local bytes/spills, 81,664 B shared and two CTAs/SM; the full correctness
and 4-GiB high-block-ID gates passed.

Three 4K/70K/126K/200K/250K runs measured
267.7/2,623.5/4,053.8/6,330.9/7,907.6 us,
302.1/2,902.9/4,056.3/6,344.1/7,900.9 us and
252.9/2,398.2/4,057.5/6,335.7/7,901.5 us per layer. Stable long-context tiers
were 0.2-0.5% slower than E12, so E13a is rejected before NCU. The source in
this milestone preserves the exact rejected factor for reproducibility; the
next experiment must restore E12 before changing score layout.

## V7-E12 owner-local softmax/PV candidate (independent prototype)

E12 is a single-variable follow-up to E11.  It preserves the exact pure-bit
E4M3FN-to-BF16 decoder (`0x7f/0xff` fail closed to zero), E6's four-phase
compact raw staging (K stage/decode, then V stage/decode), current-stream
selection, int64 index/block addressing, fixed geometry, and the unchanged
partial/combine ABI.  Every tile still takes one current-page `block_table`
load; there is no dual staging, page prefetch, or segment-local page table.

Warps 0, 1, and 2 continue to own one 16-row Q group and retain the sixteen
persistent K16 row-major WMMA A fragments loaded before the tile loop.  Each
owner computes its two N16 QK tiles into its own FP32 score pack, performs its
causal online softmax independently, and writes dense BF16 P `[16,32]` to its
own disjoint tmp sub-allocation. The owner then computes all sixteen
D16 PV output tiles and merges them into only its own sixteen rows of the
persistent FP16 accumulator.  Warp 3 is idle during QK/softmax/PV and only
participates in the existing stage/decode collectives.

The three group computations have no inter-group `__syncthreads()`.  Each
WMMA output tile stores to a private 16x16 FP32 scratch slice, executes
`__syncwarp()`, then the same owner warp merges the slice. The retained
14,336 B `tmp_shared` allocation holds three disjoint 1,024 B P packs followed
by three disjoint 1,024 B scratch slices; the rest stays reserved to keep the
E11 81,664 B geometry. Alpha is kept in
the owner lane registers and fetched with warp shuffle while merging, so the
owners never read/write another group's P, score, scratch, or accumulator
rows.  The persistent all-row FP16 accumulator and final FP32 `part_o/m/l`
publication remain unchanged.

### E12 shared layout and race contract

All offsets are CTA-shared byte offsets and are unchanged from E11 unless
noted as a sub-allocation:

```text
offset       size       allocation
0            24,576 B   acc_shared [48,256] FP16
24,576       33,792 B   kv_shared [2,32,264] BF16 (logical D=256)
58,368        8,448 B   q_shared/Q/scores/raw alias [16,264] BF16-sized
66,816       14,336 B   tmp_shared (reserved E11 allocation)
81,152          512 B   copied fp8_lut [256] BF16
total        81,664 B
```

Inside `q_shared`, `raw_stage` is bytes `[0,8192)` only between the four
stage/decode barriers. After the fourth decode barrier, the three disjoint
FP32 score packs are `[1024,3072)`, `[3072,5120)`, and `[5120,7168)` and stay
intact during softmax. The frozen alpha slot `[7168,7232)` remains reserved
but is not used by E12. Inside `tmp_shared`, `[0,3072)` holds the three BF16 P
packs; `[3072,6144)` holds the three 16x16 FP32 scratch slices. No owner touches
another group's P or scratch region. Keeping P separate avoids the cross-lane
race that would occur if compact BF16 rows overwrote unread FP32 score rows.

Per tile, the block-id load has its existing CTA barrier, and
`load_kv_bf16` retains all four E6 K-stage/decode/V-stage/decode CTA barriers.
Owners use only warp barriers after their score store, P publication, and each
scratch store/merge.  A single CTA barrier at the bottom of the tile is
required before the next tile can stage raw K into `q_shared`, preventing a
fast owner from overwriting a slower owner's score/P data. This is the only
new cross-group ordering point; Q preparation's existing barriers and the
partial/combine publication are unchanged.

E12 is hardware-qualified as an isolated prototype.  It compiled to 164
registers/thread, zero local bytes/spill, 81,664 B dynamic shared memory and
two active CTAs/SM.  The exhaustive 256-code E4M3FN decoder, full reference
suite and 4-GiB high-block-ID test all passed.  Three 4K/70K/126K/200K/250K
runs measured 239.4/2,325.9/4,040.9/6,319.0/7,852.2 us initially,
241.7/2,382.8/4,056.4/6,304.6/7,885.7 us and
260.1/2,548.8/4,046.5/6,303.0/7,876.7 us.  Against E11, stable long-context
latency therefore falls about 19.2%/19.7%/19.5% at 126K/200K/250K.

NCU at 126K measured 436.54 M instructions, 6.049 M tensor instructions,
3.01% tensor activity, 15.47% barrier stalls, 10.74% long-scoreboard stalls,
21.01% short-scoreboard stalls, 0.25% MIO throttle, 69.558 M/25.074 M shared
load/store conflicts and 258.22 MB DRAM read.  Relative to E11, the intended
CTA-barrier bottleneck was nearly halved (29.52% -> 15.47%) and tensor-pipe
activity rose 2.41% -> 3.01%; the higher short-scoreboard and shared-load
conflict counts identify the next optimization boundary.  E12 is accepted as
the new isolated V7 scaffold but remains disconnected from production
dispatch.

## V7-E11 persistent-Q structural baseline (accepted isolated scaffold)

E11 is retained as the measured baseline for E12.  Its persistent query reuse
and exact E6 decode/staging semantics are unchanged in the E12 source.  E11
processed groups in order with CTA barriers and used cooperative four-warp PV;
the E12 change is limited to owner-local score/P/16-tile PV and scratch
ownership described above.

E11 measured 166 registers/thread, zero local bytes/spill, 81,664 B dynamic
shared, and two CTAs/SM.  Exhaustive decode, full correctness, and the 4-GiB
high-block-ID gate passed.  The first 4K/70K/126K/200K/250K run measured
346.5/3,397.3/5,020.4/7,841.9/9,791.5 us per layer; repeated long tiers were
stable at about 5.01/7.85/9.79 ms.  At 126K, NCU measured 447.44 M
instructions, 9.28% long scoreboard, 2.41% tensor activity, 29.52% barrier
stalls, and 13.97% short-scoreboard stalls.  E11 remains isolated and is not
wired to production dispatch.

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
