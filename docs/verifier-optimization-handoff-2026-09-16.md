# CMP170HX verifier optimization handoff (2026-09-16)

## Objective and hard boundaries

The goal is to improve long-context speculative-verification attention on a
single CMP 170HX (SM80) without sacrificing correctness, CUDA-graph safety,
or the currently qualified production path. The active production service is
not part of this experiment. No production vLLM process, port 8000, Guardian,
model file, or production configuration has been changed.

All measurements below were run in an isolated directory on 206:

- Test site: `/home/base-node/.codex_tasks/pixelml-cmp170hx/mixed-fp8-test-site`
- Benchmark repo: `/home/base-node/.codex_tasks/pixelml-cmp170hx/mixed-fp8-repo`
- E21 CUDA prototype: `/home/base-node/.codex_tasks/pixelml-cmp170hx/v7-cuda-prototype/experimental/cmp170hx-mixed-fp8/cuda_prototype/v7_verifier.cu`
- E21 Python adapter: `/home/base-node/.codex_tasks/pixelml-cmp170hx/v7-cuda-prototype/experimental/cmp170hx-mixed-fp8/cuda_prototype/spec_decode_attn_v7.py`
- Runtime: `/home/base-node/.codex_tasks/pixelml-cmp170hx/runtime-v0271/venv/bin/python`
- CUDA toolkit: `.../site-packages/nvidia/cu13`
- Extension cache: `/home/base-node/.codex_tasks/pixelml-cmp170hx/v7-cuda-prototype/build`

The GitHub documentation branch is `work/cmp170hx-mixed-fp8` in
`https://github.com/TnzGit/vLLM-CMP170HX-Qwen3.8.git`. The qualified source
remains E21; rejected experiments were only temporary copies or restored
source.

## Real call path and target shape

The actual vLLM route is:

`flashinfer.py → _spec_attn_run → SpecDecodeAttention`

The production q8 specialization is `_spec_attn_partial_q8_g6_fp8`:

- static FP8 E4M3 K/V cache, represented as raw uint8 in Triton;
- Hq/Hkv/D = 24/4/256, therefore G = 6;
- q length normally 8 for the current MTP configuration;
- physical block size 896 tokens;
- `TILE=32`, `NSEG=35`, four warps, one stage;
- page-boundary block-table reload;
- global FP8 loads plus `ld.global.nc.u16` LUT decode;
- a small combine kernel follows the partial kernel.

The E21 prototype is a separate CUDA kernel with the same logical shape. It
stages raw FP8 K/V into shared memory, decodes through a shared LUT, uses a
warp-3 next-K prefetch, and maintains an FP16 accumulator with `kAccLd=258`.
E21 resources are 133 registers/thread, 81,856 B dynamic shared memory, 128
threads, and two CTAs/SM.

## Baseline measurements

The isolated benchmark is
`bench/spec_attn_fp8_ctx_scan.py`, with identical cache/query/block-table
inputs, `q=8`, `NSEG=35`, 30 warmups and 100 timed iterations for the locked
clock scans. Values are microseconds per attention layer.

The existing Triton q8 path is currently qualified:

| context | Triton q8 us/layer |
|---:|---:|
| 4K | 53.6-57.1 |
| 126K | 772-776 |
| 250K | 1,529-1,533 |

The earlier real-API wrapper A/B (same inputs, unlocked clock) measured E21 at
2.52x/2.83x/2.75x Triton latency at 4K/126K/250K (180.8/2,109.1/4,111.0 us
versus 71.8/746.7/1,495.6). Therefore the attractive standalone E21 scans
were not an apples-to-apples vLLM baseline and must not be used to justify
dispatch integration.

## NCU evidence (same 126K, q=8, NSEG=35 partial launch)

Performance counters were collected with NCU `default`/`detailed` sections
using root-only counter access. Both kernels launch 128 threads with grid 140
and have 12.5% theoretical occupancy.

| metric | Triton q8 | E21 |
|---|---:|---:|
| profiled duration | 895 us | 2.52 ms |
| memory throughput | 73.53% | 54.65% |
| DRAM throughput | 16.30% | 5.79% |
| L1/TEX throughput | 74.84% | 55.16% |
| L1 hit rate | 87.59% | 0.16% |
| L2 hit rate | 26.32% | 26.06% |
| compute throughput | 44.08% | 30.75% |
| registers/thread | 252 | 133 |
| dynamic shared | 57.34 KiB | 81.86 KiB |
| achieved occupancy | 12.21% | 12.44% |

The key inference is that E21 is not losing because it has fewer active
warps. The existing Triton path gets substantial L1 reuse from repeated K/V
loads across the six query heads. E21 adds shared K/V staging and LUT decode,
but its global reads have almost no L1 hit and its total instruction/traffic
pattern is less efficient. Any new candidate must beat the q8 specialization,
not merely beat an older generic Triton path.

## Experiments already completed

### E21 accepted prototype baseline

Correctness passed exhaustive E4M3 decoding, mixed request lengths, int64/high
physical block IDs, and a two-request CUDA-graph capture/replay. Locked 1350MHz
standalone medians (us/layer) were 186.1/479.0/1,136.3/2,205.6/3,403.6/4,218.6
at 4K/20K/60K/126K/200K/250K. This is accepted as a correctness prototype,
not as a production dispatch candidate.

### E23 aligned LUT container — rejected

Preserved 81,856 B and two CTAs/SM; long-context change was under 1% while 4K
regressed 3.3%. Source restored to E21.

### E24 warp-3 V prefetch — rejected

Correctness passed but one warp copied V at one quarter of the original copy
parallelism. Long-context latency regressed about 25% (250K 5.29 ms versus
4.22 ms). Source restored to E21.

### E25 cp.async K prefetch — rejected

Correctness passed, but registers rose from 133 to 134 and locked-clock scans
regressed 1.2-1.3% at 20K-250K and 4.6% at 4K. Immediate commit/wait overhead
outweighed overlap. Source restored to E21.

### E26 direct vLLM API wrapper — rejected

A disposable wrapper routed only the exact static-FP8 shape to E21 and used
Triton fallback elsewhere. Numerical max_abs was 0.000008-0.000015. Actual
API A/B rejected direct integration because E21 was 1.52-1.83x slower. This
also explains why a standalone prototype timing is misleading.

### E27 q8 `.cg` K/V cache modifier — rejected

Only the q8 raw K/V loads changed to `cache_modifier=".cg"`. Locked 1350MHz
results (Triton current versus `.cg`) were 53.7/53.6, 774.6/765.2 and
1,529.6/1,537.2 us/layer at 4K/126K/250K. Outputs were identical; the mixed
±1.2% effect is below the gate and not directionally consistent.

### E28 q8 warp count 4→8 — rejected

Only the launch warp count changed. Locked results were 54.6/66.4,
776.1/1,379.2 and 1,548.1/2,708.1 us/layer at 4K/126K/250K. Eight warps
regressed 22-78% with identical output.

### E29 q8 TILE 32→64 — rejected

Only the q8 K/V tile width changed. Locked results were 53.6/59.5,
775.7/1,278.5 and 1,533.3/2,502.7 us/layer at 4K/126K/250K. The wider tile
regressed 11% at 4K and 63-65% long-context, with identical output.

### E30 arithmetic E4M3 decode — rejected

Only LUT decode changed to integer sign/exponent/mantissa extraction plus
`exp2`. Locked results were 57.1/64.2, 772.2/1,349.6 and 1,532.7/2,658.2
us/layer at 4K/126K/250K. Extra instructions overwhelmed any LUT-load saving.

### NSEG scan

The service profile uses NSEG=35. An isolated scan found (us/layer):

| context | NSEG=16 | NSEG=32 | NSEG=35 | NSEG=64 |
|---:|---:|---:|---:|---:|
| 4K | 57.0 | 54.5 | 56.7 | 57.7 |
| 126K | 1,347 | 913 | **769** | 858 |
| 250K | 2,346 | 1,586 | **1,533** | 1,663 |

NSEG=35 is already the best tested long-context setting; increasing segment
count is not a free scaling knob.

## What is still unresolved

1. Why the current Triton q8 kernel has such high L1 reuse while E21's shared
   staging lowers effective traffic. The likely explanation is that six query
   heads repeatedly reuse identical K/V tiles inside the q8 CTA; E21 pays
   staging/decode/barrier cost without eliminating the dominant cache-resident
   work.
2. Whether a new kernel can preserve the q8 specialization's global FP8 loads
   and hard-coded 48 query rows while borrowing only one E21 idea (for example
   a cheaper page-base/prefetch mechanism). This must be implemented as a
   q8-specific kernel, not by dispatching the generic E21 path.
3. Whether the long-context decay is predominantly verifier attention or
   another engine component (MTP acceptance, GDN, or scheduling). The
   attention microbench only isolates one layer; end-to-end conclusions need
   prefill/decode telemetry with the same model and MTP setting.

## Recommended next experiments (one factor at a time)

The next implementation should keep the qualified q8 kernel as the default
and add an opt-in candidate selected by an environment variable. Recommended
order:

1. **q8-specific page-base/prefetch micro-change:** preserve direct global
   K/V/LUT loads and hard-coded q8 row layout; change only block-table/page
   address handling. Compare NSEG=35 at 4K/126K/250K, then inspect NCU L1
   hit, instruction count and duration.
2. **Fuse or vectorize the existing LUT path without changing cache policy:**
   arithmetic decode and shared-LUT staging are already negative controls;
   only a true vectorized table access that lowers instruction count is worth
   trying.
3. **MTP-shape specialization:** q=6 or q=8 should use separate static
   layouts; do not pay generic padding for q=8. Validate q=4/6/8, two-request
   batching and graph capture before timing.
4. **End-to-end telemetry:** instrument proposal/verify acceptance, q8 kernel
   time, GDN time and scheduler gaps at 4K/70K/120K/200K. This distinguishes
   verifier attention from non-attention long-context decay.

Reject any candidate that fails correctness, introduces spills/local memory,
reduces graph safety, or improves only one noisy length by less than 2%.

## Reproduction snippets

Qualified Triton q8:

```bash
cd /home/base-node/.codex_tasks/pixelml-cmp170hx/mixed-fp8-repo
PYTHONPATH=/home/base-node/.codex_tasks/pixelml-cmp170hx/mixed-fp8-test-site \
CUDA_HOME=/home/base-node/.codex_tasks/pixelml-cmp170hx/runtime-v0271/venv/lib/python3.12/site-packages/nvidia/cu13 \
TORCH_EXTENSIONS_DIR=/home/base-node/.codex_tasks/pixelml-cmp170hx/v7-cuda-prototype/build \
/home/base-node/.codex_tasks/pixelml-cmp170hx/runtime-v0271/venv/bin/python \
bench/spec_attn_fp8_ctx_scan.py --contexts 4096,126000,250000 \
--queries 8 --segments 35 --warmup 30 --iters 100
```

The E21 adapter is selected by adding:

```bash
--module /home/base-node/.codex_tasks/pixelml-cmp170hx/v7-cuda-prototype/experimental/cmp170hx-mixed-fp8/cuda_prototype/spec_decode_attn_v7.py
```

For NCU, run as root on the isolated host and use launch skip 20 to select
the partial attention kernel; the first 20 launches are cache construction.

## Current decision

Do not merge E21 into vLLM dispatch. The qualified production q8 Triton
specialization is faster and numerically correct. Continue only with a
q8-specific design that preserves its global-load/reuse behavior, and use
this document as the handoff boundary for independent review.

## Post-handoff milestones

### E31 — q8 block-ID int32 fast path (rejected)

Removing the two explicit q8 block-ID `int64` conversions produced no
repeatable long-context gain (under 0.5%); interleaved 500-iteration 4K
repeats ranged from -0.9% to +4.6% to +2.0%. The qualified path was restored.

### E32 — q8 next-page block-table prefetch (rejected)

Prefetching the next page block ID one tile early gave 55.9/55.3 us at 4K,
775.8/770.4 us at 126K and 1533.7/1543.2 us at 250K (baseline/candidate,
locked 1350MHz, q=8, NSEG=35). Output was identical, but the deltas (-0.8%,
-0.7%, +0.6%) are below the 2% gate. No source change was kept.

### Current handoff decision

E27–E32 found no durable one-knob improvement. NCU attributes the gap to
execution/resource behavior rather than an obvious cache-policy or address
conversion issue: the qualified Triton q8 path has high L1 reuse and better
effective memory behavior than the E21 shared-staging candidate, while both
have similarly low achieved occupancy. The next justified work is either a
q8-specific structural kernel design that preserves global FP8/LUT reuse, or
end-to-end telemetry to prove that verifier attention is the dominant wall
time before changing it. Production dispatch remains untouched.

### E33 — full register-resident persistent accumulator (rejected)

The isolated full-register candidate was numerically exact and reduced
locked-clock verifier latency by about 3.7%/4.2%/4.5% at 4K/126K/250K.
However, ptxas used 255 registers/thread and emitted 124--172B spill
stores/loads. This fails the zero-spill gate; the source was discarded.

### Updated next step

E33 confirms that shared accumulator traffic is measurable but that all-at-once
registerization exceeds the compiler's safe register budget. Any follow-up
must keep the live register set bounded (for example, a small tile-group
register cache with explicit shared checkpoints) and must pass ptxas before
timing. In parallel, end-to-end telemetry should establish whether this
partial-kernel gain can matter to whole-model latency.

### E34 — partial register-resident accumulator (rejected)

Registerizing four of sixteen D16 tiles passed ptxas (168 registers/thread,
zero spills) but reduced locked-clock latency only 1.8%/1.1%/1.2% at
4K/126K/250K, below the 2% gate. The qualified source was restored.

### Next candidate: cooperative two-lane softmax

The remaining WMMA path has owner-local softmax where only lanes 0--15 each
scan 32 score columns serially. A two-lane-per-row candidate will split each
row into two 16-column halves and combine max/sum with warp shuffles. It must
preserve BF16 P writes, causal masking and FP16-per-tile rounding, then pass
the same numerical, ptxas and locked-clock gates.

### E35 — cooperative half-warp softmax (accepted isolated candidate)

The candidate uses the WMMA-compatible mapping `local_row = lane & 15`,
`half = lane >> 4`, and exchanges the peer half with
`__shfl_xor_sync(..., 16)`. Each row's 32 score columns are processed as two
16-column halves; the lower half owns the final alpha/m/l publication. This
avoids the incorrect adjacent-lane mapping (`lane >> 1`, `lane & 1`) and
leaves cache, block table, QK/PV, BF16 P writes and the workspace ABI intact.

Reference-based correctness passed for single, mixed, int32/int64 and high
block-ID cases: max absolute error was 0.000977 for standard cases and
0.062500 for the high-ID case. ptxas reported 164 registers/thread, zero
spills and unchanged shared-memory/two-CTA geometry.

Interleaved locked-1350MHz q=8/NSEG=35 medians (E21/E35, us/layer) were
185.8/179.8 at 4K (3.2% faster), 2,209.5/2,064.0 at 126K (6.6% faster),
and 4,220.9/3,952.6 at 250K (6.4% faster). E35 clears the 2% isolated gate
and is retained as a candidate, but is not yet vLLM dispatch-ready: matched
API, two-request and CUDA-Graph A/B checks remain required. Reduction-order
differences are why acceptance is reference-based rather than bit-identical
to E21.

Current next gate: obtain matched NCU attribution if the installed legacy
Nsight Compute accepts an unambiguous option form, then measure E35 through
the real SpecDecodeAttention API. Do not merge it into production from
standalone timing alone.

### E35-A1 NCU attribution (matched 4K/q=8 partial launch)

The old Nsight Compute 2022.4 CLI works when the application name precedes
its arguments; the earlier failures were option-parser misuse. Matched
sampling of the same 128-thread/grid-140 partial kernel at locked clocks
reported E21/E35 duration 202.912/195.136 us (3.8% lower for E35),
compute-memory throughput 23.98%/24.85%, DRAM throughput 2.38%/2.47%,
L1/TEX throughput 24.62%/25.50%, L1 hit 87.86%/87.93%, and L2 hit
55.19%/54.96%. Both had the same 200-block, 2-CTA launch limit and
81.856 KiB dynamic shared; E21/E35 used 133/164 registers per thread.
Thus E35's isolated gain is not an occupancy change or a cache-policy
change; it is consistent with doing less serial softmax work while retaining
the same memory geometry. This is still not an end-to-end vLLM result.

### E35-Q1 — two-request API-shaped qualification (accepted)

Using the same standalone `SpecDecodeAttention` ABI with two requests
(`q=5+8`) and request-private block-table rows, E35 remained faster than
the E21 standalone adapter: 377.5/352.6 us at 4K (6.60%), 4,349.5/4,059.5
us at 126K (6.67%) and 8,357.2/7,793.0 us at 250K (6.75%). Reference
max-error was 0.000000, 0.000015 and 0.000000 respectively. This confirms
the half-warp mapping is request-safe in the fixed ABI; it does not yet
prove scheduler or vLLM integration safety.

### E35-Q2 — two-request CUDA Graph capture/replay qualification (accepted)

With fixed cache, query, block-table and workspace addresses, both E21 and
E35 captured and replayed the two-request (`q=5+8`, 4K) sequence. Each
replay matched its eager output with max absolute difference 0.000000.
This is a capture-safety result for the standalone adapter only; arbitrary
scheduler shapes still require a graph pool or eager fallback.

### E35-A2 long-context NCU attribution (126K)

Matched legacy Nsight Compute sampling of the same partial launch measured
2.5264 ms for E21 and 2.3367 ms for E35 (7.5% lower). Compute-memory
throughput was 54.66%/59.11%, DRAM throughput 5.79%/6.25%, L1/TEX
throughput 55.17%/59.84%, L1 hit 16.40%/16.38% and L2 hit 26.05%/26.06%
(E21/E35). Launch geometry/shared memory stayed identical and registers
were 133/164. The long-context delta therefore remains consistent with
reduced softmax serialization rather than changed cache residency.

## E35 disposition and E36 design gate

E35 is now a **qualified research baseline**, not a production dispatch
candidate. The qualification and attribution records are intentionally named
`E35-Q1` (two-request eager/API-shaped), `E35-Q2` (two-request CUDA Graph
capture/replay), `E35-A1` (matched 4K NCU) and `E35-A2` (matched 126K NCU).
This keeps `E36` available for a real data-flow change rather than another
qualification pass. No production vLLM service or port is changed by this
status update.

The next candidate is **E36: direct register-fed K for QK**. It must retain
E35's cooperative half-warp softmax, `TILE=32`, `NSEG=35`, accumulator,
V/PV path, output/workspace ABI and request mapping. The intended data flow
is:

```text
global FP8 K load -> per-lane LUT decode -> BF16 register operands
                                          -> SM80 mma.sync QK
```

The current code cannot safely substitute a register value for the
`wmma::fragment<matrix_b>` argument: WMMA fragment register layout is
implementation-defined. Therefore E36 must introduce an explicit operand
mapping (and, if required by the PTX instruction shape, a matching Q-register
mapping) before emitting `mma.sync.aligned.m16n8k16`. V/PV remains on the
existing WMMA/shared path for this experiment. Keeping the existing shared
allocation during the first prototype is preferred so any gain is attributable
to the K feed rather than an occupancy/layout change; reclaiming shared bytes
is a separate follow-up.

### E36 hard gates

1. Compare against E35 with identical inputs, q=8, `NSEG=35`, and locked
   clocks at 4K, 126K and 250K. The 126K median must improve by at least 10%
   to justify extending the idea to V/PV; a 2--5% gain is a rejection.
2. Exhaustive FP8 decode, mixed query lengths, int32/int64 and high block IDs
   must match the reference within the existing tolerance; no illegal access,
   NaN/Inf divergence or stale request state is allowed.
3. `ptxas` must report zero spills/local bytes, no worse than two CTAs/SM,
   and no hidden workspace/ABI change. Record registers, shared bytes and
   launch geometry before timing.
4. Run the same two-request eager and fixed-address CUDA Graph checks used by
   E35-Q1/Q2 before any end-to-end integration. Use a fresh user-owned build
   cache for every candidate; do not reuse root-owned NCU/Ninja directories.

If the explicit SM80 operand mapping cannot be implemented without increasing
registers enough to lose two-CTA residency, E36 should be marked blocked and
the next work should return to the production Triton q8 path rather than
force a WMMA/inline-PTX hybrid.

### E36-A0 result — explicit `mma.sync` score path (rejected)

**Date:** 2026-09-17

The SM80 BF16 fragment ABI was first verified in isolation: WMMA-loaded
operands fed to explicit `mma.sync.aligned.m16n8k16` produced identical
accumulator values and the documented four-register output mapping. E36-A0
then replaced each E35 m16n16 QK operation with two explicit m16n8 operations,
while retaining E35's shared FP8 decode, cooperative softmax, PV path, shared
geometry, `TILE=32`, `NSEG=35` and workspace ABI. The first implementation
performed a shared read-modify-write for each K sub-fragment and was rejected
after it regressed long-context latency. It was corrected to keep both 8-column
accumulators in registers across all 16 K sub-fragments and write scores once.

With clocks locked at 1350 MHz and identical `q=8`, `NSEG=35` inputs, the
corrected E36-A0 measured E35/E36 latency (us/layer) of 180.5/182.7 at 4K,
2064.7/2077.6 at 126K and 3947.1/3976.4 at 250K. The deltas are -1.2%,
-0.6% and -0.7% for E36, respectively; all runs had `maxdiff=0.000000`.
`ptxas` reported 165 registers/thread, zero spill stores/loads, 81,856 B
dynamic shared and the same two-CTA residency target. Thus the explicit MMA
ABI is correct but this decomposition has no performance value and is
rejected. No production or qualified E35 source changed.

The remaining E36 data-flow target is the actual NInfer-style change: remove
decoded-K shared staging and feed BF16 K operands from a per-lane FP8 decode
into registers, with an explicit operand mapping. That is a new candidate and
must be benchmarked independently; E36-A0 must not be presented as evidence
for or against that register-fed K design.
