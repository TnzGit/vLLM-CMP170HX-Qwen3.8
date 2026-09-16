# CMP 170HX mixed-FP8 verifier redesign log

This is the append-only handover log for structural verifier work on
`work/cmp170hx-mixed-fp8`.  Every milestone must leave enough evidence for a
new agent to reproduce, reject, or continue the experiment without relying on
chat history.

## Milestone V0 — freeze the qualified baseline

**Date:** 2026-09-16

**Parent commit:** `333f651`

**Runtime:** vLLM 0.27.1, PyTorch 2.13.0, CUDA 13, SM80 CMP 170HX 64 GiB

**Service:** `cmp170hx-mixed-fp8-full-256k-8002.service`

**Power:** 180 W peak profile; 175 W separately qualified

**Model:** Qwen3.8-27B W4A16 target + DFlash2 W4A16, seven draft tokens

**Cache:** FP8 E4M3 target KV + BF16 draft KV, 896/448-token logical pages

**Graph:** FULL CUDA Graph

**Concurrency/capacity:** C4 / 262144 tokens
**Verifier:** q<=8, GQA=6 specialization; 35 segments; 32-token KV tiles

### Objective

Freeze the exact code and measurement boundary before structural changes.  V0
does not change code generation or runtime behavior.

### Actual runtime entry points

The deployed test-site source is:

```text
vllm/v1/attention/backends/flash_attn.py
  _spec_attn_run()

vllm/v1/attention/ops/spec_decode_attn.py
  _spec_attn_partial_q8_g6_fp8()
  _spec_attn_combine()
  SpecDecodeAttention.run()
```

The repository representation of the active extension is the ordered series in
`experimental/cmp170hx-mixed-fp8/series`.  Do not patch only the remote
test-site and leave the series behind.

### Qualified kernel structure

For the production q<=8/GQA=6 case, one CTA owns one
`(request, kv_head, segment)` and computes all 48 useful query rows as 32+16.
Each 32-token K/V tile is loaded once and reused by both row groups.  Splitting
query rows across CTAs would duplicate the dominant long-context K/V traffic and
is not an acceptable first redesign.

The kernel retains FP32 online-softmax maxima/normalizers, FP16 running output,
read-only-cache E4M3FN LUT loads, page-local block-table carry, and int64
physical block IDs.  Partial outputs use graph-stable `part_o/part_m/part_l`
buffers and a separate segment combine kernel.

### Frozen performance and profiler boundary

The deterministic exact-token harness reports the following 180 W step
latencies.  Decode tok/s is acceptance-sensitive; `ms/step` is the primary
structural metric.

| input | qualified ms/step | representative decode tok/s |
|---:|---:|---:|
| 4K | 22.315 | ~161 |
| 126K | 35.230 | ~98 |
| 250K | 46.941 | ~70 |

At 126K, a shape-aware CUDA profile assigned 30.426 ms of GPU kernel time per
step as follows:

| component | ms/step | share |
|---|---:|---:|
| FP8 verifier partials | 12.847 | 42.2% |
| target Marlin GEMMs | 13.219 | 43.4% |
| GatedDeltaNet | 1.120 | 3.7% |
| combine | 0.123 | 0.4% |
| other | ~3.12 | 10.3% |

The verifier partial kernel used about 250 registers/thread, no local spill,
43,008 bytes of shared memory, about 284 GB/s measured memory throughput, and
54.75% no-eligible-warp cycles.  The grid is 35 segments x 4 KV heads = 140
CTAs, matching 70 SMs x two resident CTAs.

### Correctness gates carried forward

Every candidate must preserve:

- exact 256-code E4M3FN semantics, with the two NaN encodings fail-closed to 0;
- explicit-dequantization agreement for q=5/8/16/64;
- 895/896/897 boundaries with `--block-size 896`;
- int64 addressing before physical-block stride multiplication;
- mixed KV lengths and high physical block IDs;
- graph-stable workspace addresses;
- zero preemptions, CUDA errors and new Xid/NVRM events in qualified runtime A/B.

Known test gaps before claiming production readiness are mixed query lengths,
q=6/7 special-path edges, a production-896 high-block-ID case, and automated
CUDA Graph capture/replay parity.  These should be closed alongside the first
candidate that survives isolated performance testing.

### V1 decision

The first experiment is a source-controlled q8/GQA6 pipeline variant with
Triton `num_stages=2` and, only if viable, `num_stages=3`.  It must keep cache
layout, tile size, grid, workspace, accumulation and global K/V byte traffic
unchanged.  Merely setting `num_stages` is not evidence of pipelining: generated
code must be inspected for `cp.async` or an equivalent overlapped load schedule.

V1 stops and is rejected if any of the following occurs:

- K/V global bytes increase;
- local spill appears or registers/shared memory prevent two resident CTAs;
- 126K and 250K isolated-kernel gain is below about 5%;
- 4K whole-model step latency regresses by more than 2%;
- correctness, graph replay, preemption or kernel-log gates fail.

### Handover command boundary

Before changing runtime code, confirm:

```bash
git switch work/cmp170hx-mixed-fp8
git status --short --branch
systemctl --user is-active cmp170hx-mixed-fp8-full-256k-8002.service
curl -fsS http://127.0.0.1:8002/health
```

The active 8002 service is the reference baseline.  Candidate code must be
installed in a separate test-site or guarded by a candidate-only environment
variable, and the baseline path must remain recoverable without rebuilding.

## Milestone V1 — Triton automatic K/V pipeline (rejected)

**Date:** 2026-09-16

**Parent commit:** `1231ccf`
**Candidate patches:**

```text
experimental/cmp170hx-mixed-fp8/candidates/
  spec-decode-fp8-q8-pipeline-stage2.patch
  spec-decode-fp8-q8-pipeline-stage3.patch
```

### Hypothesis

The baseline q8/GQA6 verifier launches with `num_stages=1` and reports 54.75%
no-eligible-warp cycles.  Increasing Triton's software-pipeline depth might
overlap the next K/V tile load with the current dot/softmax work without
changing global workspace, page geometry, the NSEG35 grid, or K/V read count.

### Isolation

The qualified baseline test-site remained unchanged.  Two complete copies were
created on the test host, and each candidate changed only the q8/GQA6 launch's
`num_stages`.  The API service was stopped so no second CUDA context could
pollute timings.

The isolated production-shape scan used:

```text
Hq/Hkv/D       24/4/256
page/tile      896/32
query length   8
segments       35
warmup/iters   10/50
contexts       4096,70000,126000,200000,250000
```

### Result

| context | stage1 us/layer | stage2 | stage2 delta | stage3 | stage3 delta |
|---:|---:|---:|---:|---:|---:|
| 4K | 52.4 | 54.6 | +4.2% | 59.2 | +13.0% |
| 70K | 450.0 | 445.2 | -1.1% | 613.9 | +36.4% |
| 126K | 794.0 | 788.8 | -0.7% | 1100.1 | +38.6% |
| 200K | 1244.6 | 1199.4 | -3.6% | 1690.0 | +35.8% |
| 250K | 1549.2 | 1581.5 | +2.1% | 2170.3 | +40.1% |

Positive delta means slower.  Stage2's isolated gains were small and did not
hold at either end of the range; it failed the predeclared >=5% improvement at
both 126K and 250K and exceeded the 2% 4K regression limit.  Stage3 was a large
regression at every useful long-context tier.

Generated-code inspection proved that this was a real pipeline experiment, not
an ignored launch hint:

| variant | registers/thread | local/stack | dynamic shared | PTX `cp.async` count |
|---|---:|---:|---:|---:|
| stage1 | 250 | 0 | 43,008 B | 0 |
| stage2 | 248 | 0 | 73,728 B | 22 |
| stage3 | 252 | 0 | 90,112 B | 32 |

Stage2 preserved two-CTA shared-memory feasibility on SM80 but spent much more
shared memory for inconsistent overlap.  Stage3's 90,112-byte footprint cannot
keep two such CTAs resident in the available SM shared memory, which is
consistent with its severe second-wave regression.  No whole-model or FULL
Graph test was run because both candidates failed the isolated admission gate.

### Decision and rollback

**Rejected.**  Neither candidate is added to
`experimental/cmp170hx-mixed-fp8/series`; the baseline test-site was never
modified.  Restoring the service therefore requires no code rollback, only
starting the unchanged baseline unit.

### Next milestone

V2 must address the persistent `acc0[32,256] + acc1[16,256]` live state rather
than adding automatic pipeline buffers.  The preferred design target is a CTA
internal producer/consumer or row strip-mining scheme that continues to load
each K/V tile once.  Before writing that larger kernel, close the cheap test
gaps for q=6/7, production 896-page boundaries, and mixed query lengths so V2
has a stronger admission harness.

## Milestone V2 — strengthen the isolated correctness gate

**Date:** 2026-09-16

**Parent commit:** `958cd6d`

**Changed file:** `bench/test_spec_decode_fp8_sm80.py`

### Objective

Close the low-cost correctness gaps identified during V0 before accepting a
larger kernel redesign.  This milestone changes only the test harness; it does
not modify or restart the qualified verifier implementation.

### Added coverage

- q=6 and q=7, covering both sides of the q8/GQA6 32+16 row split;
- per-request mixed query lengths through explicit cumulative query offsets;
- production page-boundary mode that defaults to and enforces block size 896;
- a production-page mixed batch: KV lengths 895/896/897/4097 with query
  lengths 5/8/6/1;
- a high physical block ID derived from the actual block stride instead of a
  block-size-64 constant;
- VRAM estimation and a clear skip result if the high-ID allocation cannot fit.

The enhanced high-ID test computes the first block whose element offset is
above signed int32.  With the production geometry this is block 2341 at a
917,504-element stride and about 4.00 GiB for the synthetic K/V allocations.

### GPU result

The baseline verifier passed on the CMP 170HX with:

```bash
python bench/test_spec_decode_fp8_sm80.py \
  --production-page-boundaries --high-block-id
```

Key results:

| case | maximum absolute error | result |
|---|---:|---|
| 895 / q5 | 0.00069 | pass |
| 896 / q8 | 0.00078 | pass |
| 897 / q8 | 0.00079 | pass |
| mixed KV 895/896/897/4097, q 5/8/6/1 | 0.00066 | pass |
| q6 | 0.00066 | pass |
| q7 | 0.00054 | pass |
| mixed KV 4097/1300/8192/64, q 5/8/6/1 | 0.00200 | pass |
| 65,536 / q16 | 0.00011 | pass |
| high block 2341, page 896 / q5 | 0.00096 | pass |

All cases were below the existing 0.08 admission threshold.  The process
completed without CUDA OOM or illegal access.  This milestone does not yet add
automated Xid collection or true CUDA Graph capture/replay parity; those remain
runtime gates for a candidate that survives isolated performance testing.

### Handover

This enhanced harness is now the minimum isolated correctness command for V3
and later candidates.  A candidate must run from an isolated module/test-site;
the production 896-page flag must not be omitted.

## Milestone V3 — eight-warp / segment-count factorial (rejected)

**Date:** 2026-09-16

**Parent commit:** `030f941`

**Candidate patch:**

```text
experimental/cmp170hx-mixed-fp8/candidates/spec-decode-fp8-q8-warp8.patch
```

### Hypothesis

The four-warp q8 kernel uses 250 registers/thread.  An eight-warp compilation
uses only 167 registers/thread with no local spill.  Pairing the larger CTA with
17 segments gives 68 CTAs, close to one resident wave on 70 SMs, and might
improve instruction-level scheduling without duplicating K/V reads.

### Resource model

| launch | threads/CTA | registers/thread | registers/CTA | likely CTA/SM |
|---|---:|---:|---:|---:|
| 4 warps | 128 | 250 | 32,000 | 2 |
| 8 warps | 256 | 167 | 42,752 | 1 |

Both cases expose about eight resident warps per SM.  The candidate therefore
does not increase theoretical warp occupancy; it trades two independent CTAs
for one larger CTA.  NSEG17 was tested because 4 KV heads x 17 segments = 68
CTAs.  NSEG18 produces 72 CTAs and a two-CTA second-wave tail.

### Isolated factorial

All runs used the V1 production geometry, q=8, 10 warmups and 50 measured
iterations.  Values are microseconds per layer.

| context | 4w/35 baseline | 4w/17 | 4w/18 | 8w/35 | 8w/17 | 8w/18 |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 53.9 | 60.8 | 60.6 | 78.9 | 73.2 | 71.1 |
| 70K | 433.5 | 772.2 | 920.9 | 914.7 | 916.2 | 1600.0 |
| 126K | 760.1 | 1305.5 | 1338.4 | 1289.7 | 1332.0 | 2403.6 |
| 200K | 1193.9 | 1719.6 | 2065.6 | 2043.7 | 2077.3 | 3697.7 |
| 250K | 1476.1 | 2202.1 | 2588.3 | 2575.9 | 2644.8 | 4636.2 |

Against 4w/NSEG35, the intended 8w/NSEG17 candidate was 35.8% slower at 4K,
75.2% slower at 126K, and 79.2% slower at 250K.  Keeping NSEG35 with eight
warps was also 69.7%/74.5% slower at 126K/250K.  NSEG18 confirmed the expected
tail-wave pathology.

The segment variants agreed within small FP16 partial-reduction ordering error
(`maxdiff <= 0.000244` in the scan).  No full-model or Graph test was justified
after the isolated regression.

### Decision

**Rejected.**  Lower registers/thread is not a useful objective when total
registers per CTA remove the second resident CTA.  The eight-warp patch remains
under `candidates/` for audit and is not added to the active series.

### Next milestone

Do not continue adjusting warp count or segment count.  The next throughput
candidate must preserve the 4w/NSEG35 grid and attack accumulator live ranges
inside the CTA.  FP16 `part_o` is a separate low-risk memory-headroom experiment
but is expected to provide less than 1% whole-model throughput; it should not be
confused with the main structural redesign.

## Milestone V4 — FP16 partial-output workspace (rejected)

**Date:** 2026-09-16

**Parent commit:** `f75a362`

**Candidate patch:**

```text
experimental/cmp170hx-mixed-fp8/candidates/
  spec-decode-fp16-partial-workspace.patch
```

### Hypothesis

The static-FP8 q8 kernel already rounds its running accumulator to FP16 after
every tile, but writes the segment result into an FP32 `part_o` workspace.
Changing only that workspace to FP16 would halve its memory footprint and might
reduce partial-store/combine-load traffic without adding a new numerical round.

### Result

The first production-shape scan was slower at every context.  Three additional
interleaved 126K/250K rounds confirmed that this was not a single-run outlier:

| round | 126K baseline us | FP16 | regression | 250K baseline us | FP16 | regression |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 765.6 | 814.7 | 6.4% | 1543.2 | 1636.4 | 6.0% |
| 2 | 823.9 | 949.4 | 15.2% | 1535.8 | 1665.0 | 8.4% |
| 3 | 759.2 | 836.1 | 10.1% | 1544.5 | 1627.3 | 5.4% |

The dtype change also altered Triton's lowering: recent four-warp candidate
variants used about 57,344 bytes of dynamic shared memory rather than the
qualified 43,008-byte footprint, with 252-255 registers/thread.  Thus the
apparently smaller global workspace did not translate to a lighter partial
kernel.

### Decision

**Rejected for the throughput branch.**  The memory saving is real, but current
capacity already fits and a repeatable 5-15% verifier regression is not an
acceptable trade.  The patch remains an audit artifact and is not added to the
active series.  No full-model or CUDA Graph run was performed after the
isolated admission failure.

### Next milestone

All low-risk launch/workspace variants are now exhausted.  V5 is the first true
kernel-structure milestone: preserve 4w/NSEG35 and one K/V load per tile while
shortening the lifetime of the 48x256 running output.  Its design must state
where accumulator state lives, how producer/consumer synchronization works,
and why two resident CTAs remain possible before implementation begins.

## Milestone V5 — disable loop-invariant hoisting (rejected)

**Date:** 2026-09-16

**Parent commit:** `e327b48`

**Candidate patch:**

```text
experimental/cmp170hx-mixed-fp8/candidates/
  spec-decode-fp8-q8-disable-licm.patch
```

### Hypothesis

Triton documents `tl.range(..., disable_licm=True)` as a way to avoid long live
ranges caused by loop-invariant-code motion.  Replacing only the q8/GQA6 KV
loop iterator might reduce the 250-register footprint while preserving
4w/NSEG35, one K/V load per tile, workspace layout, and attention arithmetic.

Explicit Triton `warp_specialize` was not used: Triton 3.7.1 documents that
feature as Blackwell-only, while this host is SM80.

### Generated resources

The candidate reduced registers/thread from 250 to 239 and kept local/stack at
zero, but dynamic shared memory increased from 43,008 to 57,344 bytes.  It did
not reach a qualitatively different occupancy or scheduling regime.

### Performance

The initial scan was slower by about 20% at 70K/126K and 3% at 250K.  Three
interleaved 100-iteration 126K/250K rounds were noisy but showed no repeatable
gain:

| round | 126K baseline us | candidate | 250K baseline us | candidate |
|---:|---:|---:|---:|---:|
| 1 | 833.5 | 953.4 | 1549.5 | 1592.1 |
| 2 | 761.5 | 841.8 | 1533.9 | 1615.9 |
| 3 | 901.3 | 806.0 | 1668.7 | 1605.8 |

Across the three runs the mean candidate latency was about 4.2% worse at 126K
and 1.3% worse at 250K.  It failed the >=5% admission threshold and increased
shared memory while leaving the persistent 48x256 accumulator intact.

### Decision

**Rejected.**  Preventing LICM changes scheduling but does not solve accumulator
lifetime.  The patch remains outside the active series.  V6 tests an explicit
16+16+16 row organization in one Triton program; acceptance depends entirely
on generated resources and isolated latency, not source-level appearance.

## Milestone V6 — explicit 16+16+16 row split (rejected)

**Date:** 2026-09-16

**Parent commit:** `dbb30ef`

**Candidate patch:**

```text
experimental/cmp170hx-mixed-fp8/candidates/
  spec-decode-fp8-q8-3x16.patch
```

### Hypothesis

The qualified q8/GQA6 verifier keeps a 32-row and a 16-row accumulator alive
through the KV loop.  Re-expressing the same 48 rows as three explicit 16-row
groups might let Triton schedule shorter score/softmax temporaries while still
loading each K/V tile exactly once.  The launch stayed at 4 warps, NSEG35 and
`num_stages=1`; page carry, int64 block IDs and the external workspace were
unchanged.

### Correctness gate

The candidate passed the strengthened isolated suite on the CMP 170HX:

- q=5/6/7/8 and the generic q=16 path;
- KV lengths 895/896/897 around the production 896-token page boundary;
- mixed request/query batches;
- 65,536-token KV and q=16;
- physical block ID 2341, which places the synthetic cache above a signed
  int32 element offset and requires about 4.00 GiB.

Maximum absolute error remained <=0.00200 and no CUDA illegal access or OOM
occurred.

### Generated resources

| kernel | registers/thread | local/stack | dynamic shared |
|---|---:|---:|---:|
| qualified 32+16 | 250 | 0 | 43,008 B |
| candidate 16+16+16 | 248 | 0 | 57,344 B |

The source-level grouping did not shorten the persistent accumulator lifetime:
all three 16x256 accumulators coexist for the full KV loop.  It saved only two
registers/thread and caused Triton to allocate 14,336 additional shared bytes.

### Interleaved performance

Values are microseconds per layer, 10 warmups and 100 measured iterations.

| round | 126K baseline | candidate | 250K baseline | candidate |
|---:|---:|---:|---:|---:|
| 1 | 759.1 | 833.1 | 1529.4 | 1662.1 |
| 2 | 814.6 | 1010.7 | 1540.2 | 1848.9 |
| 3 | 842.9 | 981.8 | 1540.5 | 1781.9 |

The three-round mean regressed by about 16.9% at 126K and 14.8% at 250K.
An initial five-context scan also lost at every tier: 56.2/477.4/842.2/
1291.9/1690.2 us at 4K/70K/126K/200K/250K.

### Decision

**Rejected before full-model or CUDA Graph testing.**  Merely changing the
source grouping does not alter the lifetime of the 48x256 running state.  The
candidate is retained as an auditable negative result and remains outside the
active series.

### Next milestone

The remaining material path is no longer another Triton launch hint.  V7 is a
standalone fixed-geometry CUDA C++ prototype that explicitly separates FP8 K/V
staging from the 48-row computation and exposes the real resource budget.  It
must first reproduce the enhanced correctness suite and report registers,
shared memory, local spill and eligible-warps behavior before any vLLM dispatch
integration.  The qualified Triton verifier remains the service baseline.

## Milestone V7-E0 — out-of-tree CUDA build and contract smoke (accepted)

**Date:** 2026-09-16

**Parent commit:** `d2f42c9`

**Files:**

```text
experimental/cmp170hx-mixed-fp8/cuda_prototype/
bench/test_v7_cuda_prototype.py
```

### Scope

E0 deliberately does not modify vLLM, the active experimental series or the
qualified 8002 test-site.  A JIT-built PyTorch CUDA extension fixes the geometry
to SM80, q<=8, Hq/Hkv/D=24/4/256, page896, tile32 and NSEG35.  It exposes the
existing `part_o/m/l` workspace ABI and a standalone combine kernel.

K/V use a two-stage SM80 `cp.async` shared buffer.  All four warps participate
in the row computation.  For this correctness scaffold, online accumulator
state is intentionally read from and written to the global partial workspace
for every tile; that choice bounds registers but is not expected to be fast.

### Build and resource result

The isolated extension compiled with Torch 2.13.0+cu130, CUDA 13.0.88 and G++
13.3 on the CMP 170HX.  Runtime attributes for the canonical specialization:

| metric | result |
|---|---:|
| threads/CTA | 128 |
| registers/thread | 48 |
| dynamic shared | 81,920 B |
| static shared | 16 B |
| local bytes | 0 |
| active CTAs/SM | 2 |

This exactly consumes the two-CTA dynamic-shared budget on an SM80 SM.  Future
E1 changes cannot add shared memory without either shrinking another region or
losing the required second resident CTA.

### Correctness smoke

| case | max absolute error | result |
|---|---:|---|
| 895 / q5 / int32 | 0.000977 | pass |
| 896 / q8 / int32 | 0.000977 | pass |
| mixed 897/q5 + 4097/q8 / int64 | 0.000977 | pass |

No illegal access, Xid or OOM occurred.  The 256-code E4M3FN LUT test also
passes and maps the two NaN encodings to zero.

The follow-up full standalone gate also passed q=6/7, mixed q=5/8/6/1,
8K/32K/65K KV, all 895/896/897 boundaries, and physical block ID 2341.  The
largest ordinary-case error was 0.003906; the synthetic 4.00-GiB high-ID cache
case was 0.031250, still below the established 0.08 gate.  This closes the E0
integer-addressing and mixed-request coverage without changing the kernel.

### Decision and next gate

**E0 accepted as an isolated scaffold, not as a performance candidate.**
V7-E1 must move the repeated global accumulator traffic on chip without
exceeding 81,920 dynamic shared bytes, retain zero local spill and two CTAs/SM,
then pass the complete V2 correctness/high-block-ID suite.  Only after that
does it earn an interleaved 4K/126K/250K throughput comparison.

## Milestone V7-E1 — shared accumulator with scalar CUDA math (rejected)

**Date:** 2026-09-16

**Parent commit:** `36bff72`

E1 replaced per-tile global accumulator checkpoints without increasing the
81,920-byte shared allocation: scaled BF16 Q and the persistent FP16
accumulator use 24,576 bytes each, while raw FP8 K/V double buffering uses
32,768 bytes.  Each warp owns 12 rows; per-row `m/l` remains in registers and
`part_o/m/l` is written once after the segment.

The canonical specialization compiled to 126 registers/thread, zero stack and
spill, 16 static shared bytes, 81,920 dynamic shared bytes and two active
CTAs/SM.  The complete standalone correctness/high-block-ID gate passed; the
largest error was 0.0625 at physical block 2341 (<0.08).

| context | V7-E1 us/layer | qualified Triton reference us/layer | ratio |
|---:|---:|---:|---:|
| 4K | 997.9 | about 52-54 | about 19x slower |
| 70K | 12503.7 | about 434-450 | about 28x slower |
| 126K | 22347.6 | about 760-805 | about 28x slower |
| 200K | 35337.5 | about 1194-1245 | about 29x slower |
| 250K | 44185.7 | about 1530-1545 | about 29x slower |

**Rejected as a throughput implementation.**  Shared-state lifetime and CTA
residency are solved, but QK and PV remain scalar CUDA loops.  Triton's
`tl.dot` uses tensor cores, so launch or memory tuning cannot close this
19-29x arithmetic gap.

The next candidate must map QK and PV to SM80 BF16 tensor cores.  A feasible
two-CTA layout is: one expanded BF16 K/V tile (32 KiB), all-row FP16
accumulator (24 KiB), one 16-row BF16 Q/P buffer (8 KiB), and one 16x256 FP32
WMMA workspace (16 KiB), totaling 80 KiB.  Process the three 16-row groups
sequentially, reuse the FP32 region for scores/output, and reuse Q for P after
scores are formed.

## Milestone V7-E2 — BF16 WMMA QK/PV (correct, rejected for throughput)

**Date:** 2026-09-16

**Parent commit:** `73ed877`

E2 mapped both QK and PV to SM80 BF16 WMMA while retaining the fixed V7
workspace/addressing contract.  It decodes one 32-token K/V tile to BF16,
processes the 48 GQA/query rows as three sequential 16-row packs, keeps the
all-row value accumulator in FP16 shared memory, and publishes `part_o/m/l`
once per segment.

The first real-device correctness run exposed a concrete layout bug: P was
written with Q's 256-column stride but consumed by PV as compact `[16, 32]`.
Only the first row of each pack was therefore valid.  A zero-Q/constant-V
diagnostic localized the failure to row 1 and later; changing the P store to
stride 32 closed the fault without changing the external ABI.

### Resource and correctness gate

| metric | E1 scalar | E2 WMMA |
|---|---:|---:|
| registers/thread | 126 | 88 |
| dynamic shared | 81,920 B | 81,920 B |
| local/spill | 0 | 0 |
| active CTAs/SM | 2 | 2 |

After the P-stride fix, E2 passed q=5/6/7/8, 895/896/897-token boundaries,
mixed requests, 8K/32K/65K contexts, int32/int64 dispatch and physical block
ID 2341.  The largest ordinary error was 0.003906; the 4.00-GiB high-ID case
was 0.031250 (<0.08).

### Isolated latency

| context | qualified Triton us/layer | E2 WMMA us/layer | ratio |
|---:|---:|---:|---:|
| 4K | about 52-54 | 488.2 | about 9.2x slower |
| 70K | about 434-450 | 8,546.7 | about 19.3x slower |
| 126K | about 760-805 | 15,269.0 | about 19.5x slower |
| 200K | about 1,194-1,245 | 24,167.6 | about 19.8x slower |
| 250K | about 1,530-1,545 | 30,208.5 | about 19.6x slower |

E2 is faster than scalar E1 at 250K (30.2 ms versus 44.2 ms), but still far
outside the admission threshold.

### Counter-guided diagnosis at 126K

The baseline and E2 execute the same 6,048,768 tensor-pipe instructions and
read essentially the same 258 MB from DRAM.  The difference is feed/schedule
efficiency rather than missing tensor-core lowering:

| NCU metric | qualified Triton | E2 WMMA |
|---|---:|---:|
| tensor-pipe active | 17.21% | 0.82% |
| shared-load bank conflicts | 2.02 M | 190.54 M |
| shared-store bank conflicts | 4.44 M | 31.21 M |
| long-scoreboard stall | 12.09% | 51.20% |
| barrier stall | 6.58% | 11.63% |
| executed instructions | 126.0 M | 1,079.1 M |
| measured memory throughput | 289.0 GB/s | 13.6 GB/s |

This establishes two primary causes: unswizzled 256-column BF16 shared layouts
create extreme bank conflicts, and scalar per-element LUT/address preparation
creates long dependencies and about 8.6x the instruction count.  The WMMA
instructions are present and numerically correct, but spend almost all their
time starved.

### Decision and next gate

**Rejected for production dispatch; retained as the E3 base.**  E3 must first
remove shared-bank conflicts with padded/swizzled Q/K/V layouts while retaining
two CTAs/SM.  Only then should it optimize page carry, vectorized raw loads,
shared/read-only LUT access and redundant Q/P fragment loads.  The next
milestone must publish the same NCU A/B counters, not only wall-clock latency.

## Milestone V7-E3a — padded WMMA leading dimensions (rejected)

**Date:** 2026-09-16

**Parent commit:** `e81c5fd`

E3a changed only the physical BF16 shared leading dimensions for Q/K/V from
256 to 264.  Logical D=256, compact P `ld=32`, accumulator, temporary, grid,
workspace ABI and attention arithmetic were unchanged.  This adds a 16-byte
bank phase shift to consecutive physical rows.

The layout grew from 81,920 to 83,200 B.  It compiled to 79 registers/thread
with zero spill and passed the complete ordinary/high-block-ID correctness
gate, but runtime occupancy fell from two CTAs/SM to one.

| context | E2 unpadded us/layer | E3a padded us/layer | change |
|---:|---:|---:|---:|
| 4K | 488.2 | 882.3 | +80.7% |
| 70K | 8,546.7 | 13,220.8 | +54.7% |
| 126K | 15,269.0 | 23,600.3 | +54.6% |
| 200K | 24,167.6 | 37,307.3 | +54.4% |
| 250K | 30,208.5 | 46,681.1 | +54.5% |

NCU at 126K confirmed that the padding did reduce shared-load bank conflicts
from 190.54 M to 63.51 M.  It did not address the scalar feed chain:
long-scoreboard stalls remained 49.01%, shared-store conflicts remained
31.00 M, and tensor-pipe activity fell from 0.82% to 0.53% with half the CTA
residency.

**Rejected.**  Padding is directionally correct but cannot consume extra
shared memory past the two-CTA cliff.  E3b must retain padded Q/K/V while
recovering at least 1,280 B from the FP32 PV temporary.  A testable option is
a 224-column main PV phase plus a 32-column tail phase; it must prove that the
additional barriers cost less than the recovered occupancy and conflict
reduction.

## Milestone V7-E3b — two-phase PV scratch recovery (accepted scaffold)

**Date:** 2026-09-16

**Parent commit:** `62d0375`

E3b retained padded Q/K/V `ld=264` and compact P `ld=32`, but reduced FP32 PV
scratch from 16x256 to 16x224.  The main phase computes d=0..223; after its
result is merged, warp 3 reuses the first 16x32 entries for d=224..255.  Total
dynamic shared fell to 81,152 B.

Resources returned to 79 registers/thread, zero spill and **two CTAs/SM**.
The full ordinary/high-block-ID correctness suite passed with the same error
bounds as E3a.

| context | E2 unpadded | E3a one CTA | E3b two CTA | E3b vs E2 |
|---:|---:|---:|---:|---:|
| 4K | 488.2 | 882.3 | 502.7 | +3.0% |
| 70K | 8,546.7 | 13,220.8 | 8,093.1 | -5.3% |
| 126K | 15,269.0 | 23,600.3 | 14,481.3 | -5.2% |
| 200K | 24,167.6 | 37,307.3 | 23,061.9 | -4.6% |
| 250K | 30,208.5 | 46,681.1 | 28,746.7 | -4.8% |

NCU at 126K reported 63.51 M shared-load conflicts, 31.14 M shared-store
conflicts, 52.26% long-scoreboard stalls, 10.99% barrier stalls and 0.87%
tensor-pipe activity.  Thus recovered occupancy converted padding into a real
~5% long-context gain, but the remaining 18-19x gap is no longer primarily an
occupancy question.  Scalar FP8 decode, random global LUT lookup and repeated
64-bit address preparation now dominate the feed path.

**Accepted only as the next isolated scaffold, not for production dispatch.**
It misses the production admission gate (4K regression >2%, 250K gain <5%,
and absolute latency remains far above Triton).  E4a should spend the remaining
768-B two-CTA shared budget on a 512-B BF16 decode LUT and remeasure the same
long-scoreboard/conflict counters before combining vectorized raw loads or
page-carry changes.

## Milestone V7-E4a — shared BF16 decode LUT (small accepted scaffold gain)

**Date:** 2026-09-16

**Parent commit:** `6e0d464`

E4a copied the 256-entry BF16 FP8 decode table to 512 B of shared memory once
per CTA.  It reused the existing accumulator-init barrier; all WMMA, padded
layout, two-phase PV and external ABI were unchanged.  Total shared became
81,664 B and hardware retained 79 registers/thread, zero spill and two
CTAs/SM.  Full correctness/high-block-ID passed; the high-ID error was 0.0625
(<0.08).

| context | E3b global LUT | E4a shared LUT | change |
|---:|---:|---:|---:|
| 4K | 502.7 | 479.3 | -4.7% |
| 70K | 8,093.1 | 8,222.1 | +1.6% |
| 126K | 14,481.3 | 14,338.5 | -1.0% |
| 200K | 23,061.9 | 22,858.5 | -0.9% |
| 250K | 28,746.7 | 28,522.1 | -0.8% |

Against unpadded E2, E4a is 1.8% faster at 4K and 6.1%/5.6% faster at
126K/250K.  The counter change is much smaller than the wall-clock improvement
from restoring occupancy: long-scoreboard stalls moved only 52.26% -> 51.97%,
while shared-load bank conflicts increased 63.51 M -> 72.22 M because random
LUT indices now contend in shared memory.  Tensor activity was essentially
unchanged at 0.88%.

**Accepted as the next isolated scaffold, not production.**  The LUT copy is
cheap and the overall E2-relative admission numbers are positive, but absolute
latency remains 18-19x Triton.  E4b must reduce the scalar feed instruction
chain using aligned 16-byte raw K/V loads and tile-level block/head base
calculation.  Page carry should remain a later independent change.

## Milestone V7-E4b — direct 16-byte K/V loads (rejected)

**Date:** 2026-09-16

**Parent commit:** `0e09a1d`

E4b mapped each K/V tile to 1,024 aligned 16-byte chunks, assigned eight
chunks to each of 128 threads, and hoisted block/head bases to tile scope.
Unaligned external cache bases retained a safe scalar fallback. Geometry,
shared allocation, WMMA, two-phase PV and ABI were unchanged.

The implementation passed every resource and correctness gate: 79
registers/thread, zero spill, 81,664 B shared, two CTAs/SM, all q5-q8/mixed/
65K cases, and the 4-GiB high-block-ID case at 0.0625 max error.

| context | E4a scalar raw load | E4b direct vector load | change |
|---:|---:|---:|---:|
| 4K | 479.3 | 541.5 | +13.0% |
| 70K | 8,222.1 | 8,240.1 | +0.2% |
| 126K | 14,338.5 | 14,349.7 | +0.1% |
| 200K | 22,858.5 | 22,862.3 | +0.02% |
| 250K | 28,522.1 | 28,517.7 | -0.02% |

NCU remained essentially identical: 1.093 B executed instructions, 52.00%
long-scoreboard stalls, 0.88% tensor-pipe activity, 72.22 M shared-load and
31.15 M shared-store conflicts, and 259.35 MB DRAM reads. `uint4` global loads
were offset by scalar register unpack and per-byte shared LUT lookups.

**Rejected.** The next isolated experiment should stage one compact 8-KiB raw
matrix in the otherwise-not-yet-used Q/P buffer, then decode from shared to
the padded BF16 K/V matrix. This separates/coalesces global fetch from decode
without increasing the 81,664-B allocation. It must be measured because the
extra phase barriers may erase the latency benefit.

## Milestone V7-E5 — compact raw shared staging (short-only gain; rejected)

**Date:** 2026-09-16

**Parent commit:** `4c83b37`

E5 reused the 8,448-B Q/P buffer before Q loading as an 8,192-B compact raw
matrix. K and V each follow an explicit stage -> barrier -> shared-LUT decode
sequence. This removes register-side `uint4` unpack and cleanly separates the
global fetch dependency from decode without increasing shared allocation.

Resources remained 79 registers/thread, zero spill, 81,664 B and two CTAs/SM.
The complete correctness/high-block-ID gate passed at 0.0625 max error.

| context | E4a direct scalar | E5 compact staging | change |
|---:|---:|---:|---:|
| 4K | 479.3 | 473.4 | -1.2% |
| 70K | 8,222.1 | 7,984.3 | -2.9% |
| 126K | 14,338.5 | 14,355.3 | +0.1% |
| 200K | 22,858.5 | 22,863.6 | +0.02% |
| 250K | 28,522.1 | 28,509.2 | -0.05% |

NCU remained 1.093 B executed instructions, 51.99% long-scoreboard stalls,
0.88% tensor activity, 72.22 M shared-load and 31.15 M shared-store conflicts.
The dependency separation helps short/mid scheduling slightly, but every raw
byte still performs a random shared LUT access.

**Rejected for the long-context objective.** The next experiment should
exhaustively validate and then substitute exact E4M3FN-to-BF16 bit conversion
for the per-byte LUT lookup. This directly targets the unchanged conflict and
instruction counters.

## Milestone V7-E6 — exact bitwise E4M3FN decode (major accepted scaffold)

**Date:** 2026-09-16

**Parent commit:** `cc04e4e`

E6 replaced every hot-loop shared LUT lookup with pure integer synthesis of
the BF16 bits. Its initial helper checked all finite encodings bit-exactly but
incorrectly canonicalized `0x7f/0xff` as NaN instead of preserving the
qualified LUT's fail-closed zero contract. Cross-review caught this before any
production integration; normal finite-cache performance numbers are unaffected
and E8 corrected both device behavior and exhaustive test. The 512-B LUT
allocation stayed in place for a controlled occupancy comparison.

Resources remained 79 registers/thread, zero spill, 81,664 B and two CTAs/SM.
The complete correctness/high-block-ID gate passed at 0.0625 max error.

| context | E5 shared LUT | E6 bit synthesis | change |
|---:|---:|---:|---:|
| 4K | 473.4 | 378.5 | -20.0% |
| 70K | 7,984.3 | 4,957.3 | -37.9% |
| 126K | 14,355.3 | 8,130.3 | -43.4% |
| 200K | 22,863.6 | 12,845.7 | -43.8% |
| 250K | 28,509.2 | 16,041.2 | -43.7% |

NCU confirms the mechanism: executed instructions fell 1.093 B -> 711.7 M,
shared-load conflicts 72.22 M -> 63.52 M, long-scoreboard stalls 51.99% ->
28.48%, and tensor activity rose 0.88% -> 1.51%. Barrier stalls became the
largest newly exposed structural cost at 19.23%.

**Accepted as the new isolated scaffold, not production.** Next perform a
controlled factorial that removes compact raw staging and its extra barriers
while keeping E6 bit synthesis. That will distinguish global-load scoreboard
cost from phase-barrier cost before attempting page carry or overlap.

## Milestone V7-E7 — direct global load factorial (rejected)

**Date:** 2026-09-16

**Parent commit:** `c732fe9`

E7 retained the exact E6 bit decoder and identical 81,664-B resource geometry,
but removed compact raw staging and reduced K/V publication to one final CTA
barrier. Every exhaustive/correctness/resource gate remained green.

| context | E6 staged bit decode | E7 direct global | regression |
|---:|---:|---:|---:|
| 4K | 378.5 | 484.6 | +28.0% |
| 70K | 4,957.3 | 7,899.3 | +59.4% |
| 126K | 8,130.3 | 13,663.6 | +68.1% |
| 200K | 12,845.7 | 21,651.6 | +68.6% |
| 250K | 16,041.2 | 27,041.6 | +68.6% |

NCU proved the trade: barrier stalls fell 19.23% -> 11.19%, but long-scoreboard
stalls jumped 28.48% -> 54.34%, tensor activity fell 1.51% -> 0.92%, and
instructions rose 711.7 M -> 845.6 M.

**Rejected.** Return to E6 staging. The next experiment should stage raw K and
V concurrently in two otherwise-idle aliases (Q/P and tmp), then decode both
after one publication barrier and use one final publication barrier. This
targets barrier count without giving up the scoreboard benefit.

## Milestone V7-E8 — dual-alias two-phase staging (correct; rejected)

**Date:** 2026-09-16

**Parent commit:** `b6980c6`

E8 corrected NaN fail-closed semantics, retained E6 bit synthesis, staged raw
K and V concurrently in Q/P and tmp aliases, and reduced four stage/decode
phases to two. All 256-code, resource, correctness and high-block-ID gates
passed.

| context | E6 four-phase | E8 dual two-phase | change |
|---:|---:|---:|---:|
| 4K | 378.5 | 407.8 | +7.7% |
| 70K | 4,957.3 | 4,924.6 | -0.7% |
| 126K | 8,130.3 | 7,979.2 | -1.9% |
| 200K | 12,845.7 | 12,692.6 | -1.2% |
| 250K | 16,041.2 | 15,781.1 | -1.6% |

Instructions fell 711.7 M -> 675.1 M, but barrier stalls stayed 19.31%; long
scoreboard was 29.11% and tensor activity 1.53%. Fewer explicit barriers did
not reduce measured barrier waiting because phase arrival imbalance dominates.

**Rejected by admission thresholds.** Keep E6 as the performance baseline and
E8's corrected fail-closed semantics in all future tests. Next compare CUDA's
two-code FP8 conversion intrinsic with bit synthesis, then investigate
split-local page metadata/base staging.

## Milestone V7-E9b — CUDA FP8x2-to-half bridge (correct; rejected)

**Date:** 2026-09-16

**Parent commit:** `0445323`

The target CUDA 13.0 header does not expose the proposed direct
`__nv_cvt_fp8x2_to_bf162raw` symbol. E9b therefore used the actually available
`__nv_cvt_fp8x2_to_halfraw2`, converted both raw half lanes through FP32 to
BF16, and explicitly fail-closed `0x7f/0xff` to zero. It retained E6's
four-phase compact staging, resource geometry and external ABI.

The extension compiled on the CMP 170HX to 79 registers/thread, zero local
bytes, 81,664 B dynamic shared and two CTAs/SM. The exhaustive device test
passed all 254 finite encodings bit-exactly and required both invalid encodings
to return zero. All q=5/6/7/8, 895/896/897, mixed-request, 8K/32K/65K,
int32/int64 and 4-GiB high-block-ID correctness gates passed.

| context | E6 pure bit synthesis | E9b FP8x2-half bridge | change |
|---:|---:|---:|---:|
| 4K | 378.5 | 346.9 | -8.3% |
| 70K | 4,957.3 | 4,829.9 | -2.6% |
| 126K | 8,130.3 | 8,354.1 | +2.8% |
| 200K | 12,845.7 | 13,228.3 | +3.0% |
| 250K | 16,041.2 | 16,511.7 | +2.9% |

NCU at 126K measured 783.4 M instructions versus E6's 711.7 M, while shared
load conflicts remained 63.51 M and DRAM reads remained about 258 MB. Barrier
stalls improved slightly (19.23% -> 18.73%) and long-scoreboard stalls moved
28.48% -> 27.36%, but tensor activity fell 1.51% -> 1.46%. The extra
half-to-float-to-BF16 bridge therefore wins at short context but accumulates
too many instructions over long scans.

**Rejected.** CUDA FP8x2 is not automatically a faster SM80 BF16 feed path
when the public API stops at half2. Preserve E6 as the performance base and
E8's corrected invalid-code semantics. The next independent experiment is
split-local physical-page/base staging; do not combine it with this rejected
conversion bridge.

## Reference audit — `Ithrial/ninfer-cmp170hx`

**Date:** 2026-09-16

This repository is a genuine SM80 build/runtime port, but it is not an FP8
verifier source. Its Qwen3.8 path uses groupwise integer weights and BF16 or
INT8-G64 KV attention. The source contains the useful independent topology of
one CTA per KV head, all six GQA query heads, split-KV partial/reduce, 32/64-key
tiles, online softmax and segment-local physical-page ID staging.

The port must not be copied as a performance implementation. Its current
geometry still defines `DecodeSplits = 85` and retains comments/policies tuned
for a 170-SM parent GPU rather than this 70-SM CMP. Public controlled A/B data
also reports about 38.16 tok/s with MTP versus 138.6 tok/s for the vLLM+DFlash2
control. It has no SM80 FP8 KV/verifier kernel; later native FP8 MMA paths in
the NInfer lineage are Blackwell-only.

Use it as a correctness and dataflow reference for CTA ownership, page-list
staging and an optional INT8 control. Do not transplant its cache ABI, 64-token
page geometry, groupwise quantization, split constants or native-FP8 path into
the 896-token static-FP8 verifier.

## Milestone V7-E10a — split-local physical page/base staging (correct; rejected)

**Date:** 2026-09-16

**Parent commit:** `297e00e`

E10a applied the safest useful idea from `Ithrial/ninfer-cmp170hx` without
copying its incompatible cache ABI. It reuses the retained 512-byte LUT
allocation to prefetch up to sixteen K/V physical page bases once per segment.
The tile loop then adds only `tile_slot * stride_s`; a segment wider than the
fixed table takes the original per-tile block-table path without truncation.
E10a otherwise restores E6's four-phase staging and exact fail-closed bitwise
E4M3FN decode.

The extension compiled to 86 registers/thread, zero local bytes, 81,664 bytes
dynamic shared memory and two active CTAs/SM. The exhaustive 256-code test,
q=5/6/7/8, int32/int64, page boundaries, mixed requests, 8K/32K/65K and the
4-GiB physical block-ID 2341 case all passed. The largest high-ID error was
0.0625 (<0.08).

| context | E6 per-tile block lookup | E10a page-base prefetch | change |
|---:|---:|---:|---:|
| 4K | 378.5 | 387.8 first run; 342.2/356.6 repeats | noisy |
| 70K | 4,957.3 | 4,938.3; 4,719.1; 4,577.3 | noisy improvement |
| 126K | 8,130.3 | 8,055.0; 8,056.6; 8,065.3 | about -0.9% |
| 200K | 12,845.7 | 12,762.6; 12,757.5; 12,756.9 | about -0.7% |
| 250K | 16,041.2 | 15,915.4; 15,913.9; 15,913.0 | about -0.8% |

NCU at 126K measured 710.90 M executed instructions, 6.049 M tensor
instructions, 1.52% tensor-pipe activity, 63.515 M shared-load conflicts,
31.213 M shared-store conflicts and 258.26 MB DRAM reads. Relative to E6,
barrier stalls moved 19.23% -> 19.02% and long-scoreboard stalls 28.48% ->
28.08%. The saved block-table/address work is therefore real, but it is a
small fraction of the remaining feed cost; register use also rose 79 -> 86.

**Rejected by the production admission thresholds.** The stable long-context
gain is below 1%, not the required 5%, and short tiers are noisy. Preserve
E10a as a correct measured factorial and NInfer-derived design check, but keep
E6 as the base for the next structural experiment. The next high-value target
is repeated Q conversion/loading: each tile currently rebuilds all three
16-row Q groups. A persistent-Q fragment design should be isolated from page
staging so its register/residency trade is measurable.

## Milestone V7-E11 — persistent Q WMMA fragments (major accepted scaffold)

**Date:** 2026-09-16

**Parent commit:** `b35f4db`

E11 returned to E6's per-tile page lookup, four-phase raw K/V staging and
exact fail-closed bitwise FP8 decode, isolating one factor: Q preparation.
Warps 0..2 each own one 16-row query group, convert Q once, load sixteen K16
BF16 WMMA A fragments once, and keep those fragments in registers across the
entire KV scan. Each owner warp computes both N16 score tiles; the three
`16x32` score packs share the former raw-stage lifetime before per-group
softmax and PV.

The resource risk proved acceptable on CMP 170HX: 166 registers/thread, zero
local bytes/spills, 81,664 bytes dynamic shared and two active CTAs/SM. The
exhaustive decoder, q=5/6/7/8, page boundaries, mixed requests, int32/int64,
8K/32K/65K and 4-GiB high-block-ID tests all passed; high-ID max error remained
0.0625 (<0.08).

| context | E6 repeated Q prep | E11 persistent Q | change |
|---:|---:|---:|---:|
| 4K | 378.5 | 346.5 first run; 272.4/326.4 repeats | no regression; noisy gain |
| 70K | 4,957.3 | 3,397.3; 2,854.1; 3,129.6 | about -37% median |
| 126K | 8,130.3 | 5,020.4; 5,012.3; 5,011.8 | about -38.3% |
| 200K | 12,845.7 | 7,841.9; 7,859.0; 7,854.6 | about -38.9% |
| 250K | 16,041.2 | 9,791.5; 9,787.3; 9,795.7 | about -38.9% |

NCU at 126K confirmed the expected mechanism. Executed instructions fell
711.7 M -> 447.44 M (-37.1%); long-scoreboard stalls fell 28.48% -> 9.28%;
tensor-pipe activity rose 1.51% -> 2.41%. Tensor instruction count remained
6.049 M and DRAM reads stayed about 258.24 MB, so the result did not come from
skipping arithmetic or cache data. The newly exposed costs are barrier stalls
at 29.52% and short-scoreboard stalls at 13.97%; shared-load/store conflicts
remain about 63.514 M/31.097 M.

**Accepted as the new isolated CUDA scaffold, not production dispatch.** E11
passes the isolated admission threshold by a large margin, but remains roughly
6.3x slower than the qualified Triton verifier at 126K/250K. The next
scientific target is no longer page metadata or Q conversion: it is the
serialized group softmax/PV/barrier schedule and short shared-memory
dependency chain. Any E12 change must preserve E11's persistent fragments and
measure barrier, short-scoreboard, tensor-active and conflict counters before
production integration is considered.

## Milestone V7-E12 — owner-local softmax/PV (major accepted scaffold)

**Date:** 2026-09-16

**Parent commit:** `1c0856c`

E12 preserves E11's persistent Q fragments and E6's exact four-phase FP8
decode, but removes the three serialized CTA-wide group softmax/PV phases.
Warps 0..2 retain ownership of one 16-row group apiece. Each computes its two
QK tiles, online softmax and all sixteen D16 PV tiles independently, using a
disjoint BF16 P pack and FP32 `16x16` scratch slice. Group-local ordering uses
`__syncwarp()`; one CTA barrier remains at the tile tail before raw K/V staging
can reuse the shared alias. Scores and compact P use separate storage, avoiding
the cross-lane overwrite race caught during source audit.

The kernel compiled to 164 registers/thread, zero local bytes/spills, 81,664
bytes dynamic shared and two active CTAs/SM. Exhaustive E4M3FN decode, all
reference/boundary/mixed-length cases and the 4-GiB high-block-ID test passed.

| context | E11 persistent Q | E12 owner-local PV | change |
|---:|---:|---:|---:|
| 4K | 346.5 first; 272.4/326.4 repeats | 239.4; 241.7; 260.1 | faster, short-tier noise |
| 70K | 3,397.3; 2,854.1; 3,129.6 | 2,325.9; 2,382.8; 2,548.8 | about -20% median |
| 126K | 5,020.4; 5,012.3; 5,011.8 | 4,040.9; 4,056.4; 4,046.5 | about -19.2% |
| 200K | 7,841.9; 7,859.0; 7,854.6 | 6,319.0; 6,304.6; 6,303.0 | about -19.7% |
| 250K | 9,791.5; 9,787.3; 9,795.7 | 7,852.2; 7,885.7; 7,876.7 | about -19.5% |

NCU at 126K measured 436.54 M instructions, 6.049 M tensor instructions,
3.01% tensor activity, 15.47% barrier, 10.74% long scoreboard, 21.01% short
scoreboard, 0.25% MIO throttle, 69.558 M/25.074 M shared load/store conflicts
and 258.22 MB DRAM read. Against E11, barrier stalls nearly halved from 29.52%
and tensor activity rose from 2.41%, directly validating the experimental
hypothesis. Short-scoreboard stalls and shared-load conflicts increased, so
the next factorial should target owner-warp P/V dependency and shared access,
not reintroduce CTA group serialization.

**Accepted as the new isolated CUDA scaffold, not production dispatch.** The
change is correct and repeatably clears the 5% admission threshold. Production
integration remains a separate milestone because the standalone V7 kernel is
still materially slower than the qualified Triton path.

## Milestone V7-E13a — padded FP32 PV scratch (correct; rejected)

**Date:** 2026-09-16

**Parent commit:** `04576bb`

E13a isolated one bank-layout hypothesis from E12: each owner warp's FP32
`16x16` WMMA scratch changed from `ld=16` to the legal float-accumulator
`ld=20`. The three physical slices grew from 1,024 to 1,280 bytes each but
remained inside the retained 14,336-byte allocation; ABI, total dynamic shared
memory and logical merge remained unchanged.

The extension still compiled to 164 registers/thread, zero local bytes/spills,
81,664 bytes dynamic shared and two active CTAs/SM. Exhaustive decode, full
correctness and the 4-GiB high-block-ID test passed.

| context | E12 median us/layer | E13a median us/layer | change |
|---:|---:|---:|---:|
| 4K | 241.7 | 267.7 | noisy regression |
| 70K | 2,382.8 | 2,623.5 | noisy regression |
| 126K | 4,046.5 | 4,056.3 | +0.24% |
| 200K | 6,304.6 | 6,335.7 | +0.49% |
| 250K | 7,876.7 | 7,901.5 | +0.31% |

**Rejected before NCU.** The stable long tiers are flat-to-slower and fail the
5% admission gate. Padding this WMMA store cannot pay for the scalar merge's
less favorable bank phase. Restore E12 and target the FP32 score-pack read
layout independently; do not combine score and scratch padding in one factor.

## Milestone V7-E13b — padded FP32 score rows (accepted isolated scaffold)

**Date:** 2026-09-16

**Parent commit:** `a2681c3`

E13b restores E12's dense `ld=16` PV scratch and changes only the FP32 score
pack. Each logical `[16,32]` group uses physical `ld=36`; WMMA stores and the
scalar online-softmax reads share that stride. The four-column pad rotates
successive rows across shared-memory banks. Three packs occupy 6,912 bytes and
still end at byte 7,936 of the 8,192-byte raw-stage alias. The final A/B was
run with graphics clocks locked to 1350MHz so automatic Boost could not bias
the comparison; the lock was removed after testing.

Resources remained 164 registers/thread, zero local bytes/spills, 81,664
bytes shared and two active CTAs/SM. Exhaustive decode, full correctness and
the 4-GiB high-block-ID test passed.

| context | E12 @1350 median us/layer | E13b @1350 median us/layer | change |
|---:|---:|---:|---:|
| 4K | 255.2 | 245.7 | -3.7% |
| 10K | 482.1 | 460.4 | -4.5% |
| 20K | 823.2 | 778.5 | -5.4% |
| 40K | 1,495.2 | 1,406.6 | -5.9% |
| 60K | 2,168.3 | 2,032.1 | -6.3% |
| 64K | 2,316.5 | 2,169.7 | -6.3% |
| 70K | 2,502.9 | 2,343.3 | -6.4% |
| 90K | 3,175.2 | 2,964.6 | -6.6% |
| 126K | 4,371.4 | 4,075.0 | -6.8% |
| 200K | 6,824.2 | 6,367.0 | -6.7% |
| 250K | 8,499.7 | 7,927.3 | -6.7% |

NCU at 126K validates the mechanism. Shared-load conflicts fell
69.558 M -> 27.230 M (-60.9%), short-scoreboard stalls fell
21.01% -> 15.85%, barrier stalls fell 15.47% -> 14.83%, MIO throttle fell
0.25% -> 0.10%, and tensor activity rose 3.01% -> 3.24%. DRAM remained
258.22 MB and tensor instructions remained 6.049 M. Executed instructions rose
436.54 M -> 447.13 M, but the dependency reduction more than paid for the
extra padded addressing at long context.

**Accepted as the new isolated scaffold.** Under controlled clocks it clears
the 5% gate from 20K through 250K and does not regress 4K/10K. The earlier
unlocked run that appeared to regress 4K/70K was discarded as a clock-state
confounder. Production integration is still a separate milestone because the
standalone V7 kernel remains disconnected from dispatch.

## Milestone V7-E14 — padded BF16 P rows (accepted secondary scaffold)

**Date:** 2026-09-16

**Parent commit:** `7fefb58`

E14 preserves E13b's FP32 score `ld=36` and changes only the logical BF16 P
operand used by the PV WMMA loads. Each `[16,32]` P group is physically
`[16,40]`, with `ld=40` valid for BF16 WMMA. The three padded packs consume
3,840 bytes inside the retained 14,336-byte temporary allocation; ABI,
logical output and total dynamic shared memory are unchanged.

Resources remained 164 registers/thread, zero local bytes/spills, 81,664
bytes shared and two active CTAs/SM. Exhaustive decode, full correctness and
the 4-GiB high-block-ID test passed.

The final A/B locked graphics clocks at 1350MHz and used 300 timed iterations:

| context | E13b median us/layer | E14 median us/layer | change |
|---:|---:|---:|---:|
| 4K | 245.7 | 240.1 | -2.3% |
| 20K | 778.5 | 751.8 | -3.4% |
| 60K | 2,032.1 | 1,966.0 | -3.3% |
| 126K | 4,075.0 | 3,945.8 | -3.2% |
| 200K | 6,367.0 | 6,172.1 | -3.1% |
| 250K | 7,927.3 | 7,689.2 | -3.0% |

NCU at 126K measured 447.13 M instructions, 6.049 M tensor instructions,
3.33% tensor activity, 14.62% barrier, 10.83% long scoreboard, 14.80% short
scoreboard, 0.09% MIO throttle, 9.081 M/14.486 M shared load/store conflicts
and 258.53 MB DRAM read. Relative to E13b, shared-load conflicts fell about
67% and store conflicts about 39%; the runtime gain is smaller because the
remaining time is dominated by global K/V feed and WMMA work.

**Accepted as a secondary isolated scaffold.** The incremental gain is below
the original 5% single-factor gate, but it is repeatable, has no correctness
or occupancy cost, and compounds with E13b to roughly 9-10% over E12. Keep it
isolated until the remaining accumulator-store dependency is measured.

## Milestone V7-E15 — persistent accumulator row padding (rejected)

**Date:** 2026-09-16

**Parent commit:** `3e42331`

E15 changed only the persistent FP16 accumulator's physical row stride from
logical `ld=256` to `ld=258`; logical output indexing, the partial/combine ABI,
P/score layouts, K/V traffic and synchronization were unchanged. The smaller
two-element pad was chosen after rejecting `ld=264`, whose 82,432-byte layout
would cross the allocator-rounded two-CTA residency budget. E15 uses 81,856
bytes of dynamic shared memory (81,920-byte allocator round), 164 registers per
thread, zero local bytes/spills and two active CTAs/SM.

The complete exhaustive decoder, ordinary/mixed boundary suite, int32/int64
indices and 4-GiB high-block-ID test all passed. Three locked-1350MHz runs
with 300 iterations, contexts 4K/20K/60K/126K/200K/250K and query lengths 4/8
were stable. Query-8 medians were 239.7/751.5/1,965.2/3,949.6/6,175.2/7,683.7
us per layer (the corresponding E14 medians were 240.1/751.8/1,966.0/3,945.8/
6,172.1/7,689.2). Differences stayed within about ±0.1% and did not form a
repeatable gain.

NCU at 126K confirmed why: E15 measured 9.079M shared-load conflicts and
14.481M shared-store conflicts, 3.33% tensor activity, 14.63% barrier,
10.84% long-scoreboard and 14.49% short-scoreboard stalls. These are
effectively identical to E14 (9.081M/14.486M conflicts and 3.33%/14.62%/
10.83%/14.80% stalls), so accumulator row padding did not move the dominant
conflict/dependency boundary.

**Rejected.** Keep E14 as the qualified isolated source of truth; retain E15
only as a documented negative control. The next optimization must first
attribute the remaining shared stores (for example with source-level NCU or
SASS classification) before changing another layout factor.

## Milestone V7-E16 — half2 accumulator merge (accepted isolated scaffold)

**Date:** 2026-09-16

**Parent commit:** `f0f2ec4`

E16 changes only the owner-local accumulator merge. Instead of eight scalar
FP16 load/convert/round/store operations per lane, each lane handles four
adjacent values as `half2`: the pair is converted to `float2`, multiplied by
the FP32 row alpha with FMA, and stored back as `half2`. The 128-pair lane map
covers every logical `[16,256]` group exactly once; `ld=258` remains only a
physical stride and publication still uses the logical `D=256` ABI.

The kernel keeps 81,856B dynamic shared memory (81,920B allocator round), but
registers fall from 164 to 134 per thread; local bytes/spills remain zero and
the driver reports two active CTAs/SM. Accumulator and scratch bases have
compile-time alignment assertions. Exhaustive E4M3FN decode, ordinary and
mixed boundaries, int32/int64 indices and the 4-GiB high-block-ID test all
passed; max errors stayed within the existing FP16 accumulation tolerance.

Three locked-1350MHz scans (300 iterations, query lengths 4/8) were stable.
Query-8 medians below are us/layer:

| context | E14 | E16 | change |
|---:|---:|---:|---:|
| 4K | 240.1 | 225.3 | -6.2% |
| 20K | 751.8 | 684.5 | -9.0% |
| 60K | 1,966.0 | 1,751.1 | -10.9% |
| 126K | 3,945.8 | 3,485.7 | -11.7% |
| 200K | 6,172.1 | 5,439.8 | -11.9% |
| 250K | 7,689.2 | 6,757.3 | -12.1% |

At 126K NCU measured 6.049M tensor instructions, 12.098M shared-load
conflicts, 17.556M shared-store conflicts, 258.21MB DRAM reads, 12.86%
barrier stalls, 13.11% long-scoreboard and 14.78% short-scoreboard stalls.
Compared with E14, conflict counters rose (the half2 transaction pattern is
not bank-conflict-free), yet wall latency fell substantially because the
scalar accumulator merge's instruction and dependency chain was halved and
register pressure dropped by 30. E16 demonstrates that conflict counters must
be interpreted together with transaction count, register pressure and end to
end latency.

**Accepted as the current isolated V7 scaffold.** It is still not wired into
vLLM or any production dispatch; integration requires a separate adapter,
multi-request correctness gate and end-to-end A/B before deployment.

## Milestone V7-E17 — disjoint score tail and tile-barrier reduction (accepted secondary scaffold)

**Date:** 2026-09-16

**Parent commit:** `123a6fe`

E17 changes only the lifetime of the FP32 score packs and the corresponding
CTA synchronization. E16 kept scores in the first 6,912 bytes of `q_shared`,
so every tile needed a tail `__syncthreads()` before a faster owner could let
the next tile overwrite that raw staging alias. E17 puts the three score packs
in the unused tail of the existing `tmp_shared` allocation (offset 6,912,
total 13,824 of 14,336 bytes). `q_shared` is therefore raw-only after the
fixed load/decode barriers. The per-tile tail barrier is removed; one
end-of-loop CTA barrier remains before publication so warp 3 cannot read the
accumulator while an owner warp is finishing its final merge.

The logical cache/block geometry, partial/combine ABI, K/V traffic, and total
dynamic shared allocation are unchanged. The candidate compiled with 132
registers/thread, zero local bytes/spills, 81,856 B dynamic shared and two
active CTAs/SM. Exhaustive E4M3FN decoding, ordinary/mixed boundary cases,
int32/int64 indices and the 4-GiB high-block-ID case all passed with the same
existing numerical tolerance.

Three locked-1350MHz runs (query length 8, 300 timed iterations) produced the
following medians in us/layer:

| context | E16 | E17 | change |
|---:|---:|---:|---:|
| 4K | 225.3 | 225.0 | -0.1% |
| 20K | 684.5 | 674.2 | -1.5% |
| 60K | 1,751.1 | 1,726.0 | -1.4% |
| 126K | 3,485.7 | 3,442.7 | -1.2% |
| 200K | 5,439.8 | 5,372.7 | -1.2% |
| 250K | 6,757.3 | 6,689.1 | -1.0% |

The gains are small but repeatable and monotonic with context; there was no
short-tier regression or occupancy change. A 126K Nsight Compute run measured
12.50% barrier, 12.54% long-scoreboard and 13.08% short-scoreboard stalls,
versus E16's 12.86%/13.11%/14.78%; aggregate shared-bank conflicts remained
about 29.65 M and DRAM reads 258.21 MB. This supports the intended diagnosis:
the removed barriers reduce synchronization/dependency wait, while unchanged
memory traffic explains why the improvement is only about 1%.

**Accepted as a low-risk secondary isolated scaffold.** It is below the
original 5% single-factor gate, but it is stable, resource-neutral and
composes with E16 for roughly 21% lower verifier latency than E12 at long
context. Keep the final publication barrier; removing that last barrier is
not safe. Production integration remains a separate adapter and A/B gate.

## Milestone V7-E18 — shared-LUT FP8 decode (accepted)

**Date:** 2026-09-16

**Parent commit:** `301d840`

E18 changes one kernel factor: the FP8-to-BF16 conversion in the K/V tile
loader now reads the exact 256-entry BF16 LUT already copied into shared
memory, instead of recomputing the E4M3FN exponent/mantissa conversion with
integer arithmetic and `clz` for every element. The LUT contains the same
fail-closed `0x7f/0xff` semantics, so cache bytes, scales, shared layout,
block table, synchronization and the E17 score lifetime are unchanged.

Build resources remain 132 registers/thread, zero local bytes/spills, 81,856 B
dynamic shared and two active CTAs/SM. Exhaustive device LUT decoding,
ordinary/mixed boundaries, int32/int64 indices and the 4-GiB high-block-ID
case all passed with max error within the existing tolerance.

Three locked-1350MHz scans (query length 8, 300 timed iterations) gave these
medians in us/layer:

| context | E17 | E18 | change |
|---:|---:|---:|---:|
| 4K | 225.0 | 198.9 | -11.6% |
| 20K | 674.2 | 558.1 | -17.2% |
| 60K | 1,726.0 | 1,378.2 | -20.2% |
| 126K | 3,442.7 | 2,709.1 | -21.3% |
| 200K | 5,372.7 | 4,202.8 | -21.8% |
| 250K | 6,689.1 | 5,239.6 | -21.7% |

At 126K, NCU reported 12.50% barrier, 15.61% long-scoreboard and 16.74%
short-scoreboard stalls, about 38.42 M aggregate shared-bank conflicts and
258.21 MB DRAM reads. The LUT adds shared loads (and therefore more bank and
scoreboard activity) but removes the much larger per-element integer decode
path; wall latency falls 17-22% at medium/long context with no occupancy or
correctness cost. This is a positive result, not a contradiction: the
source-level transaction counters must be read together with instruction
count and end-to-end time.

**Accepted as the new isolated V7 scaffold.** The cumulative query-8 gain is
about 21% over E17 and about 38% over the pre-E16 E14 baseline at 250K. It is
still disconnected from vLLM; next gates are multi-request stress, CUDA Graph
capture compatibility and an end-to-end dispatch A/B.

## Milestone V7-E19 — paired shared-LUT decode (accepted)

**Date:** 2026-09-16

**Parent commit:** `a2dba6b`

E19 keeps E18's exact shared LUT but processes two adjacent FP8 bytes per
 iteration: one aligned 16-bit raw-stage load, two LUT reads and one aligned
 32-bit BF16 store. The element mapping is even-aligned by construction, so
 no cache bytes, LUT values, scales, block geometry, synchronization or ABI
 change. This isolates loop/address overhead from E18's lookup choice.

The candidate compiles with 132 registers/thread, zero local bytes/spills,
81,856 B dynamic shared and two active CTAs/SM. Exhaustive E4M3FN decoding,
ordinary/mixed boundaries, int32/int64 indices and the 4-GiB high-block-ID
case all pass; maximum errors remain within the existing FP16 accumulation
tolerance.

Three locked-1350MHz scans (query length 8, 300 timed iterations) produced
these medians in us/layer:

| context | E18 | E19 | change |
|---:|---:|---:|---:|
| 4K | 198.9 | 190.2 | -4.4% |
| 20K | 558.1 | 516.5 | -7.5% |
| 60K | 1,378.2 | 1,260.3 | -8.6% |
| 126K | 2,709.1 | 2,464.7 | -9.0% |
| 200K | 4,202.8 | 3,822.9 | -9.0% |
| 250K | 5,239.6 | 4,745.7 | -9.4% |

At 126K, NCU reports 12.50% barrier, 17.70% long-scoreboard and 18.75%
short-scoreboard stalls, about 38.40 M aggregate shared-bank conflicts,
258.21 MB DRAM reads and the same 132-register/81,856-B launch geometry as
E18. Shared conflicts and scoreboard percentages rise slightly because each
pair adds two LUT accesses, but the halved decode-loop iterations and address
work reduce wall latency. This is another case where end-to-end timing and
resource/traffic invariants are more informative than one counter.

**Accepted as the current isolated V7 scaffold.** E19 is about 29% faster than
E17 and about 38% faster than the E14 baseline at 250K in this microbenchmark.
It remains disconnected from vLLM; multi-request stress, CUDA Graph capture
and end-to-end dispatch A/B are still required before integration.

## Milestone V7-E20 — global read-only LUT negative control (rejected)

**Date:** 2026-09-16

**Parent commit:** `79ef3fc`

E20 kept E19's paired two-byte decode and changed only the LUT address space:
the same 256-entry device table was read through `__ldg` instead of the
CTA-local shared copy. Full correctness, mixed-boundary, int64 and high-ID
checks passed, and resources stayed at 132 registers/thread, 81,856 B shared
and two active CTAs/SM.

Three locked-1350MHz scans (query length 8, 300 iterations) were consistently
slower than E19. Median us/layer at 4K/20K/60K/126K/200K/250K were
193.5/543.3/1,335.8/2,615.0/4,015.0/5,031.7 versus E19's
190.2/516.5/1,260.3/2,464.7/3,822.9/4,745.7, i.e. regressions of 1.7-6.1%
(about 6% through long context). The read-only cache did not hide the random
per-lane LUT latency; E19's shared lookup remains faster despite its higher
shared-conflict count.

**Rejected.** The source and current scaffold remain E19; this negative
control is retained in the log only to prevent retrying the global-LUT route
without a different access pattern.

## Milestone V7-E21 — warp-3 next-K prefetch (accepted isolated scaffold)

**Date:** 2026-09-16

**Parent source:** E19 (`79ef3fc` lineage; E20 remains a documentation-only
negative control)

E21 changes only the tile-feed schedule. Warp 3, which is idle during the
owner-local QK/softmax/PV phase, loads the next tile's block id and raw K bytes
into the existing `q_shared` raw-staging alias. The following iteration begins
with the retained CTA barrier, decodes the prefetched K and then stages/decodes
V through the unchanged all-thread path. No KV storage, block-table format,
shared-memory allocation, scheduler state or public ABI changes. The final
end-of-loop publication barrier is retained.

The source builds with 133 registers/thread, 81,856 B dynamic shared, zero
local bytes/spills and two active CTAs/SM. Exhaustive E4M3FN decoding,
mixed-boundary lengths, int32/int64 indices, 4-GiB high-block-ID and existing
numerical-tolerance checks all pass.

Three locked-1350MHz scans (query length 8, 300 timed iterations) produced:

| context | E19 | E21 | change |
|---:|---:|---:|---:|
| 4K | 190.2 | 186.1 | -2.2% |
| 20K | 516.5 | 479.0 | -7.3% |
| 60K | 1,260.3 | 1,136.3 | -9.8% |
| 126K | 2,464.7 | 2,205.6 | -10.5% |
| 200K | 3,822.9 | 3,403.6 | -11.0% |
| 250K | 4,745.7 | 4,218.6 | -11.1% |

At 126K NCU measured 12.50% barrier, ~7.6% long-scoreboard and ~19.55%
short-scoreboard stalls, ~38.45M aggregate shared-bank conflicts and 258.21
MB DRAM reads. E19 reported 17.70%/18.75% long/short scoreboard, so the
prefetch hides most exposed K-load latency while slightly increasing short
dependency pressure; traffic and occupancy are unchanged.

**Accepted as the current isolated V7 scaffold.** The result is strong and
monotonic at medium/long context, but it is not a production claim. Required
next gates are multi-request/request-switch correctness, CUDA Graph capture
compatibility and end-to-end vLLM dispatch A/B. Production remains untouched.

## Milestone V7-E22 — two-request CUDA Graph capture (qualified validation)

**Date:** 2026-09-16

**Parent:** E21 source (`018e7db`)

The final E21 extension was warmed up and captured with a fixed two-request
shape (`lengths=[895,896]`, `q_lens=[5,8]`) across the existing `partial` and
`combine` bindings, then replayed on the same static tensor addresses. Capture
and replay completed without CUDA errors; the replayed output matched the
reference with `max_abs=0.001953`. This qualifies E21's retained barriers and
warp-3 K prefetch for CUDA Graph execution at that shape.

The result is intentionally limited: CUDA Graphs freeze tensor addresses and
launch geometry. A vLLM adapter must maintain graph variants keyed by capture
shape, refresh request-local block tables before replay, and use eager fallback
for unsupported scheduler shapes. This is a validation gate, not a production
integration or throughput claim.

## Milestone V7-E23 — aligned 32-bit LUT container (rejected negative control)

**Date:** 2026-09-16

E23 changed only the shared LUT representation: each BF16 value was stored in
an aligned 32-bit slot, while 512 bytes were reclaimed from the reserved
temporary tail so the total dynamic shared allocation remained 81,856 B and
two CTAs/SM. Exhaustive decoding, mixed requests and the high-ID case passed
(maximum error 0.0625, below the 0.08 threshold); resources remained 133
registers/thread, zero local bytes/spills and two active CTAs/SM.

Three locked-1350MHz scans (query=8, 300 iterations) produced medians in
us/layer:

| context | E21 | E23 | change |
|---:|---:|---:|---:|
| 4K | 186.1 | 192.2 | +3.3% |
| 20K | 479.0 | 475.0 | -0.8% |
| 60K | 1,136.3 | 1,128.5 | -0.7% |
| 126K | 2,205.6 | 2,188.2 | -0.8% |
| 200K | 3,403.6 | 3,375.9 | -0.8% |
| 250K | 4,218.6 | 4,182.8 | -0.8% |

The small long-context difference is below the single-factor acceptance gate,
while the short-tier regression is clear enough to reject the candidate. The
source and accepted isolated scaffold remain E21; no production code changed.

## Milestone V7-E24 — warp-3 V prefetch (rejected negative control)

**Date:** 2026-09-16

E24 moved V staging after K decode and used idle warp 3 to copy the current
tile's V bytes while owner warps computed QK/softmax, followed by a barrier and
V decode before PV. The candidate passed exhaustive decoding, mixed-request,
int64/high-ID and numerical checks with unchanged 133 registers/thread,
81,856 B shared and two CTAs/SM.

Two locked-1350MHz scans nevertheless showed a large, repeatable regression:

| context | E21 | E24 | change |
|---:|---:|---:|---:|
| 4K | 186.1 | 192.2 | +3.3% |
| 20K | 479.0 | 556.8 | +16.2% |
| 60K | 1,136.3 | 1,389.5 | +22.3% |
| 126K | 2,205.6 | 2,740.4 | +24.2% |
| 200K | 3,403.6 | 4,252.4 | +24.9% |
| 250K | 4,218.6 | 5,288.1 | +25.3% |

The single-warp V copy loses the original all-thread staging bandwidth; the
overlap does not compensate. **Rejected.** The source was restored to E21 and
no production code changed.

## Milestone V7-E26 — vLLM API dispatch wrapper (rejected)

**Date:** 2026-09-16

Before changing the installed vLLM tree, a disposable wrapper routed the
existing SpecDecodeAttention API to E21 only for the exact qualified shape:
SM80, Hq/Hkv/D=24/4/256, qmax=8, NSEG=35, block=896, and contiguous
static-FP8 NHD cache. It used identical cache bytes, query, block table and
scales for the original Triton q8 path and E21, and fell back to Triton for
all other shapes.

The wrapper was numerically clean, with max_abs 0.000008-0.000015, but the
actual vLLM-facing A/B was:

| context | Triton us/call | E21 us/call | change |
|---:|---:|---:|---:|
| 4K | 71.8 | 180.8 | +151.7% |
| 126K | 746.7 | 2,109.1 | +182.5% |
| 250K | 1,495.6 | 4,111.0 | +174.9% |

This rejects E21 for direct integration: the standalone E21 scan was not an
apples-to-apples comparison with the current Triton q8 specialization. The
test-site dispatch and remote production tree were left unchanged; no
production service was started or modified.

## Milestone V7-E27 — q8 global K/V cache policy (`.cg`) (rejected)

**Date:** 2026-09-16

NCU showed the existing Triton q8 specialization has high L1 reuse (87.6%),
so this isolated test changed only the K/V raw-byte loads to use the CUDA
`.cg` cache modifier. The global FP8/LUT path, page-boundary block-table
reload, TILE=32, NSEG=35, query layout and output ABI were unchanged.

At locked 1350MHz, q=8 and NSEG=35 (30 warmups, 100 iterations), the current
path versus `.cg` was:

| context | current us/layer | `.cg` us/layer | change |
|---:|---:|---:|---:|
| 4K | 53.7 | 53.6 | -0.2% |
| 126K | 774.6 | 765.2 | -1.2% |
| 250K | 1,529.6 | 1,537.2 | +0.5% |

Outputs were numerically identical. The long-context change is below the
2% acceptance gate and the direction is not consistent across lengths, so
the candidate is rejected; the qualified q8 source remains unchanged.

## Milestone V7-E28 — q8 warp count 4→8 (rejected)

**Date:** 2026-09-16

This isolated test changed only the q8 Triton launch from four warps to eight;
the kernel code, global FP8/LUT loads, TILE=32 and NSEG=35 were unchanged.
At locked 1350MHz (q=8, 30 warmups, 100 iterations), results in us/layer
were:

| context | 4 warps | 8 warps | change |
|---:|---:|---:|---:|
| 4K | 54.6 | 66.4 | +21.6% |
| 126K | 776.1 | 1,379.2 | +77.7% |
| 250K | 1,548.1 | 2,708.1 | +74.9% |

Numerical output was identical. The larger launch is a clear regression and
was rejected; the qualified source remains four warps.

An NCU default-set sample of the same partial launch (126K, q=8, NSEG=35)
explains the gap. Triton q8 took 895 us under profiling with 73.53% memory
throughput, 16.30% DRAM throughput, 252 registers/thread and 57.34 KiB dynamic
shared memory. E21 took 2.52 ms with 54.65% memory throughput, 5.79% DRAM
throughput, 133 registers/thread and 81.86 KiB dynamic shared memory. Both
used 128 threads and grid 140 with the same 12.5% theoretical occupancy; the
regression is therefore not an occupancy win left on the table. E21's shared
FP8 staging/LUT decode is doing more work while achieving less effective L2/
DRAM traffic than the existing q8 global-load specialization.

## Milestone V7-E29 — q8 TILE 32→64 (rejected)

**Date:** 2026-09-16

The q8 specialization was tested with only its K/V tile width changed from
32 to 64 (896-token pages are divisible by both). Four warps, global FP8/LUT
loads, NSEG=35 and all arithmetic were unchanged. At locked 1350MHz
(q=8, 30 warmups, 100 iterations), us/layer was:

| context | TILE=32 | TILE=64 | change |
|---:|---:|---:|---:|
| 4K | 53.6 | 59.5 | +11.0% |
| 126K | 775.7 | 1,278.5 | +64.8% |
| 250K | 1,533.3 | 2,502.7 | +63.3% |

Outputs were identical, but the wider tile increased register/live-state
pressure and was much slower. The candidate is rejected; TILE=32 remains
qualified.

## Milestone V7-E30 — q8 arithmetic E4M3 decode (rejected)

**Date:** 2026-09-16

To reduce LUT traffic, an isolated module replaced the q8 kernel's 256-entry
E4M3FN LUT lookup with integer sign/exponent/mantissa extraction and `exp2`.
Global K/V loads, four-warps launch, TILE=32, NSEG=35 and the attention
algorithm were unchanged. At locked 1350MHz (q=8, 30 warmups, 100
iterations), latency in us/layer was:

| context | LUT | arithmetic | change |
|---:|---:|---:|---:|
| 4K | 57.1 | 64.2 | +12.4% |
| 126K | 772.2 | 1,349.6 | +74.8% |
| 250K | 1,532.7 | 2,658.2 | +73.4% |

The arithmetic module produced finite, internally consistent outputs, but the
extra integer/exp2 instructions overwhelm any LUT-load reduction. It is
rejected and no qualified source changed.

## Milestone V7-E25 — cp.async K prefetch (rejected negative control)

**Date:** 2026-09-16

E25 changed only E21's warp-3 next-K copy to use SM80 `cp.async` groups, with
synchronous fallback for unaligned/tail chunks. Exhaustive decoding,
mixed-request, int64/high-ID and numerical checks passed. Resources increased
to 134 registers/thread while dynamic shared remained 81,856 B and occupancy
remained two CTAs/SM.

Three locked-1350MHz scans (query=8, 300 iterations) produced medians in
us/layer:

| context | E21 | E25 | change |
|---:|---:|---:|---:|
| 4K | 186.1 | 194.8 | +4.7% |
| 20K | 479.0 | 482.2 | +0.7% |
| 60K | 1,136.3 | 1,149.8 | +1.2% |
| 126K | 2,205.6 | 2,233.6 | +1.3% |
| 200K | 3,403.6 | 3,448.8 | +1.3% |
| 250K | 4,218.6 | 4,272.2 | +1.3% |

Immediate commit/wait-group synchronization outweighed additional copy
overlap. **Rejected.** The source remains E21 and no production code changed.

## Milestone V7-E30 — q8 arithmetic E4M3 decode (rejected)

**Date:** 2026-09-16

An isolated module replaced the q8 kernel's 256-entry E4M3FN LUT with
integer sign/exponent/mantissa extraction plus `exp2`; global K/V loads,
four-warps launch, TILE=32, NSEG=35 and the attention algorithm were
unchanged. At locked 1350MHz (q=8, 30 warmups, 100 iterations), us/layer was
57.1/64.2 at 4K, 772.2/1,349.6 at 126K and 1,532.7/2,658.2 at 250K
(LUT/arithmetic). Extra instructions overwhelmed LUT-load savings; the
candidate is rejected.

## Milestone V7-E31 — q8 block-ID int32 fast path (rejected)

**Date:** 2026-09-16

Removing only the two explicit q8 block-ID `int64` conversions changed long
latency by under 0.5%; repeated 500-iteration 4K pairs varied from -0.9% to
+4.6% to +2.0%. The short result was noise, so the qualified source remains
unchanged.

## Milestone V7-E32 — q8 next-page block-table prefetch (rejected)

**Date:** 2026-09-16

The isolated candidate prefetched the next page's block-table ID one tile
before the boundary, then consumed that value at the boundary. K/V loads,
E4M3 LUT decode, four-warps launch, `TILE=32` and `NSEG=35` were unchanged.
At locked 1350MHz the baseline/candidate latency was 55.9/55.3 us at 4K,
775.8/770.4 us at 126K and 1533.7/1543.2 us at 250K. Changes were -0.8%,
-0.7% and +0.6%, with identical output. The candidate is below the 2% gate
and is rejected; no qualified or production source changed.

## Milestone V7-E33 — full register-resident persistent accumulator (rejected)

**Date:** 2026-09-16

This isolated candidate removed the 24,768-byte shared FP16 accumulator and
kept four `__half2` pairs per owner lane for each of the sixteen D16 output
tiles. KV staging, LUT decode, WMMA QK/PV, `TILE=32`, `NSEG=35` and the
workspace ABI were unchanged. Exhaustive smoke output was identical. At
locked 1350MHz, baseline/candidate latency was 186.3/179.5 us at 4K,
2209.9/2116.6 us at 126K and 4225.2/4035.9 us at 250K (about 3.7%, 4.2%
and 4.5% faster).

The resource gate failed: ptxas used 255 registers/thread and reported
124--172 bytes of spill stores/loads for the partial-kernel variants. The
candidate is rejected despite the modest timing gain; no source change was
kept and no production dispatch changed.

## Milestone V7-E34 — partial register accumulator (rejected)

**Date:** 2026-09-16

Only the first four D16 output tiles were kept in owner-lane `__half2`
registers; the other twelve retained the E21 shared accumulator. The
candidate passed numerical smoke and ptxas resource checks (168
registers/thread, zero spills, unchanged shared geometry). Interleaved
locked-1350MHz scans measured 186.5/183.2 us at 4K, 2207.5/2183.3 us at
126K and 4224.0/4174.7 us at 250K (baseline/candidate): 1.8%, 1.1% and 1.2%
lower latency. These deltas are below the 2% gate, so the candidate is
rejected and the qualified E21 source remains unchanged.

## Milestone V7-E35 — cooperative half-warp softmax (accepted isolated candidate)

**Date:** 2026-09-16

E35 split each 32-column owner-row softmax into two 16-column halves. The
correct mapping is `local_row = lane & 15`, `half = lane >> 4`, with the peer
exchanged by `__shfl_xor_sync(..., 16)`; the lower half owns the final
alpha/m/l publication. Reference-based correctness passed for standard,
mixed int32/int64 and high block-ID cases (max absolute error 0.000977 in
standard cases and 0.062500 for the high-ID case). ptxas used 164
registers/thread with zero spills; shared-memory and two-CTA geometry were
unchanged.

At locked 1350MHz, q=8/NSEG=35 interleaved medians (E21/E35, us/layer) were
185.8/179.8 at 4K, 2,209.5/2,064.0 at 126K and 4,220.9/3,952.6 at 250K,
corresponding to 3.2%, 6.6% and 6.4% lower latency. E35 clears the 2%
isolated gate and is retained as a candidate, not yet as vLLM or production
dispatch. Reduction-order differences require reference-based tolerances.
The next gate is matched NCU attribution plus real API, two-request and
CUDA-Graph A/B validation.
