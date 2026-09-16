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
