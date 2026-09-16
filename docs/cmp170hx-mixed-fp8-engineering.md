# CMP 170HX mixed-FP8 engineering map

This document is the implementation ledger for reproducing the 64 GB CMP 170HX Qwen3.8-27B W4A16 + DFlash2 mixed-KV configuration discussed in the LocalLLM experiments.

The working branch starts from `c003cc73b4c34e418b29e074735107a0771c6ca1`, whose upstream recipe base is syv-ai commit `69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c` and vLLM 0.27.1.

## Target runtime

The target configuration is deliberately asymmetric:

| Component | Target model | DFlash2 draft |
|---|---|---|
| weights | W4A16 | W4A16 |
| attention backend | FlashInfer | FlashAttention-2 |
| KV dtype | E4M3 FP8 | BF16 |
| logical block | 896 tokens | 448 tokens |
| physical page | 1,835,008 bytes | 1,835,008 bytes |

The equal physical page size is important. The scheduler, cache allocator, prefix cache and attention backends must agree on logical token geometry without pretending the two cache groups use the same tokens-per-block.

## Audit: already present / reusable

Do not duplicate these pieces when implementing the final path.

### Present in this repository

- DFlash2 backport for vLLM 0.27.1.
- Split-KV speculative verify attention (`patches/spec-decode-attn.patch`).
- Quantized split-KV verifier template for per-token-head int8 (`patches/spec-decode-int8-kv.patch`).
- Hybrid KV group / V2 CUDA-graph support.
- Sliding-window block promotion / mixed physical page handling.
- SM80 staged Marlin repack path.
- KVarN long-context route and its V2 runner integration.

### Upstream vLLM capabilities to reuse

vLLM 0.27.1 already models draft-specific `attention_backend` and `kv_cache_dtype` in `SpeculativeConfig`, and the DFlash loader can build the draft model with those overrides. The new CMP profile should wire those fields into the speculative JSON rather than adding a second configuration mechanism.

Mixed-page support also exists upstream: `unify_kv_cache_spec_page_size()` can enlarge a smaller logical block when its natural page divides the maximum page and can pad supported attention/Mamba pages when necessary. Later vLLM PR #45181 is the reference design for DFlash mixed page sizes and block-stride indexed views.

## Missing implementation

### 1. Launcher wiring

Add an opt-in mixed profile that produces the equivalent of:

```json
{
  "method": "dflash",
  "model": "...",
  "num_speculative_tokens": 7,
  "attention_backend": "FLASH_ATTN",
  "kv_cache_dtype": "bfloat16"
}
```

while the target server is launched with FlashInfer + FP8 KV.

The profile must remain opt-in until the FP8 verify kernel and geometry tests pass.

### 2. SM80 FP8 split-KV speculative verifier

The current split-KV verifier handles BF16 and, through `spec-decode-int8-kv.patch`, per-token-head int8. Add a static/per-layer FP8 mode for the target FlashInfer cache:

- load raw E4M3 K/V from the target cache;
- convert to BF16/FP32 on SM80;
- apply the layer K/V scales used by vLLM's normal FP8 cache path;
- run the existing split-KV online softmax / partial reduction;
- preserve the current int64 physical block-id addressing;
- keep the BF16 and int8 paths unchanged.

Do not invent a new cache format. vLLM's ordinary FP8 cache stores E4M3 values with layer-level K/V scales; the verifier should consume the same representation.

### 3. FlashInfer backend hook

The target FlashInfer attention backend must call the split-KV verifier only for the multi-query speculative verify case and only when all kernel restrictions are satisfied. Normal prefill/decode must continue through FlashInfer.

Required inputs are:

- query/output slices;
- raw paged target K/V views;
- block table and sequence lengths;
- target layer `_k_scale` / `_v_scale` (or their float equivalents);
- max query length / group geometry.

### 4. 896/448 logical geometry

Implement this through KV-cache specs / group geometry, not by globally forcing `--block-size 896`.

Acceptance criteria:

- target FP8 group resolves to 896 tokens per logical block;
- DFlash BF16 group resolves to 448;
- both report exactly the same physical bytes per page;
- existing backend kernel block-size constraints are still respected;
- scheduler/hash/prefix-cache units are derived from group geometry rather than a single hard-coded block size.

### 5. Mamba / prefix-cache alignment

The final profile should align Mamba checkpoint geometry to the 896-token target unit and reconcile complete 448-token DFlash pages. This must be tested independently from checkpoint lifetime/eviction-order fixes.

The important regression is a repeated ~9-10K prefix: the second request must show a real common prefix hit and a large TTFT reduction, not just report that prefix caching is enabled.

## Correctness gates

The implementation is not considered usable merely because the server boots.

1. BF16 DFlash2 control path remains bit/greedy-correct.
2. FP8 target with speculation off works through FlashInfer.
3. FP8 target + BF16 DFlash2 gives correct greedy output versus a non-spec target reference.
4. High physical block IDs exercise the split-KV verifier without 32-bit overflow or Xid.
5. Cold and cached layouts are both tested.
6. Exact repeated prefix produces a non-zero reconciled hit.
7. 32K -> 65K -> 85,514 context qualification before larger contexts.
8. Eager correctness before CUDA Graph qualification.
9. At least 512 output tokens per stress request; short 16/32-token smoke tests are insufficient.
10. Kernel/NVRM logs are checked after each stress tier.

## Performance qualification

Performance work starts only after correctness gates pass. Record at least:

- decode 256 and 900 tokens;
- prefill around 6.6K and at a long-context tier;
- TTFT cold and repeated-prefix warm;
- accepted tokens / draft acceptance;
- VRAM, power, SM clock and temperature;
- eager versus CUDA Graph.

The current 180 W BF16 baseline in this repository is the first negative-control target: ~133.8 tok/s decode256, ~126.8 tok/s decode900 and ~1,869 tok/s prefill on the recorded host.

### Qualified split-KV tuning (CMP 170HX, 180 W)

The static-FP8 verifier was measured with the production geometry (24 query heads,
4 KV heads, head size 256 and 896-token target pages).  Increasing the split count
from 16 to 32 is the best balanced setting.  The 64-split kernel saved less than 2%
at 126K/250K in the isolated kernel scan and regressed short-context latency, so the
FULL mixed-FP8 service profiles explicitly select 32 while the generic code default
remains 16.

The whole-model A/B used the same fixed prompts, DFlash2 `k=7`, 512 generated tokens,
FULL CUDA Graph, FP8 target KV, BF16 draft KV and no preemptions:

| input | segments | decode tok/s | accepted tok/step | verifier ms/pass |
|---:|---:|---:|---:|---:|
| 4K | 16 | 144.8 | 3.37 | 23.1 |
| 4K | 32 | 158-161 | 3.66-3.70 | 23.0 |
| 126K | 16 | 54.2 | 3.42 | 62.8 |
| 126K | 32 | 68.2 | 3.41 | 50.0 |
| 250K | 16 | 32.5 | 3.38 | 103.3 |
| 250K | 32 | 42.2 | 3.28 | 77.5 |

The long-context gain is not an acceptance artifact: acceptance stayed effectively
flat while verifier latency fell 20-25%.  TTFT was unchanged within noise because
ordinary target prefill does not use this verifier kernel.  Prefix-cache reuse and
FULL-graph concurrency were also checked at C1/C2/C4; a 70K C2 run reached 112.6
aggregate decode tok/s with zero preemptions.

`VLLM_SPEC_DECODE_ATTN_SEGMENTS` is immutable after the first verifier workspace is
created.  Changing it requires a process restart because CUDA Graphs capture those
workspace addresses.  `bench/spec_attn_fp8_ctx_scan.py` is the matching isolated
kernel scan; run it only with the API service stopped so another CUDA context cannot
pollute timings.

### Exact BF16 E4M3FN decode table

On SM80, Triton cannot lower a native E4M3FN load.  Reconstructing every cache
byte with masks and `tl.exp2` inside the long-context loop was therefore a major
hidden cost.  The mixed-FP8 series now builds a 256-entry BF16 table once per
verifier workspace.  Every finite E4M3FN value is exactly representable in BF16;
the two NaN encodings retain the old fail-closed mapping to zero.  The 512-byte
table has a CUDA-Graph-stable address and remains hot in cache.

The controlled whole-model A/B used the same prompt salts and settings as the
segment scan: C1, DFlash2 `k=7`, 512 output tokens, 32 segments, FULL CUDA Graph,
180 W, FP8 target KV and BF16 draft KV.  Acceptance and TTFT stayed unchanged:

| input | bitwise decode tok/s | BF16 LUT tok/s | bitwise ms/pass | BF16 LUT ms/pass |
|---:|---:|---:|---:|---:|
| 4K | 158-161 | 164.1 | 23.0 | 22.5 |
| 126K | 68.2 | 81.8 | 50.0 | 41.7 |
| 250K | 42.2 | 53.8 | 77.5 | 61.2 |

The standalone kernel/reference suite passed contexts through 65K, mixed request
lengths and verify lengths through 64 tokens with the same maximum error as the
old decoder.  `bench/test_spec_decode_fp8_lut.py` exhaustively checks all 256
codes; `bench/spec_attn_fp8_ctx_scan.py --module ...` supports isolated candidate
A/B without modifying the installed runtime.

### Reduced verifier spill traffic

CUDA Driver resource queries identified the next bottleneck instead of inferring it
from throughput: the production q=8 partial kernel used 255 registers/thread and a
96-byte local frame, which held occupancy to 12.5%.  Artificial register caps raised
local traffic further and were 37-50% slower, so this is not an occupancy problem that
can be fixed by spilling more aggressively.

For the static-FP8 path only, scores, running maxima and normalizers remain FP32 while
the per-tile running output is rounded to FP16.  The production 896-token page is also
an exact multiple of the 32-token kernel tile, so its block ID is loaded once per tile
instead of materializing 32 identical IDs.  Other KV modes and page geometries retain
the original path.  The q=8 kernel's local frame fell from 96 to 32 bytes without
changing its 12.5% occupancy.

The explicit dequantized-reference suite covered q=5/8/16/64, mixed request lengths,
895/896/897-token page boundaries and a high physical block ID.  Maximum absolute
error was 0.00541 against the existing 0.08 budget.  In the same FULL-graph C1 model
test used above, stable forward-pass latency changed as follows (decode tok/s is also
shown, but varies with DFlash2 acceptance):

| input | BF16 LUT ms/pass | reduced-spill ms/pass | reduced-spill decode tok/s |
|---:|---:|---:|---:|
| 4K | 22.5 | 22.5-22.8 | 163-167 |
| 126K | 41.7 | 38.2-38.6 | 89-90 |
| 250K | 61.2 | 54.2 | 61-65 |

The long-context forward step is therefore 8-11% faster with no preemption and no
acceptance regression.  BF16 accumulation, FP16 partial scratch, hoisted scale loads,
and `maxnreg` 160/168/192 variants were all slower and remain rejected experiments.

### Production q=8 / GQA=6 row specialization

The production DFlash2 verify window has at most eight query positions and six query
heads per KV head: 48 useful rows.  The generic power-of-two `BLOCK_M=64` path still
updated 16 masked accumulator rows for every KV tile.  A shape-gated SM80 kernel keeps
one CTA per request/KV-head/segment and therefore still loads each K/V tile only once,
but computes the 48 rows as 32+16.  It is enabled only for static FP8, `G=6`, `D=256`,
`q<=8`, and page sizes divisible by 32; every other geometry uses the generic kernel.

Driver resource inspection reports 250 registers/thread, zero local memory, 43,008
bytes shared memory and the same 12.5% theoretical occupancy.  The isolated scan was
8.7-11.5% faster from 4K through 250K and bit-identical to the reduced-spill candidate.
Explicit-reference and 895/896/897 page-boundary suites retained the same maximum
error.  Stable FULL-graph model results were:

| input | reduced-spill ms/pass | q8/G6 ms/pass | q8/G6 decode tok/s |
|---:|---:|---:|---:|
| 4K | 22.5-22.8 | 22.5 | 167.6 |
| 126K | 38.2-38.6 | 37.1-37.3 | 93-95 |
| 250K | 54.2 | 51.4-51.5 | 64-66 |

Thus the removed padded rows produce another repeatable 3% at 126K and 5% at 250K
at the whole-model step boundary.  `maxnreg=168/192` variants again regressed and are
not shipped.

### Page-local block-table carry

The production 896-token page contains exactly 28 verifier tiles of 32 tokens.
The q8 specialization used to reload the same block-table entry for every tile.
It now carries the physical block ID through the loop and refreshes it only when
crossing a page boundary.  This is arithmetic- and layout-neutral: five isolated
rounds were bit-identical and improved the 250K kernel by 0.7-1.3% every time.

Paired FULL-graph tests used identical prompts and acceptance in both runs:

| input | baseline tok/s | page-carry tok/s | baseline ms/pass | page-carry ms/pass |
|---:|---:|---:|---:|---:|
| 4K | 156.1 | 156.9 | 22.6 | 22.5 |
| 126K | 87.7 | 88.0 | 37.4 | 37.2 |
| 250K | 65.0 | 65.2 | 51.8 | 51.6 |

The gain is deliberately small, but it is repeatable, has no new allocation or
graph node, and removes redundant metadata traffic rather than trading accuracy.

### Read-only cache for the immutable FP8 decode table

Nsight Compute identified the remaining LUT access as the dominant pathological
memory pattern: ordinary scalar loads generated 85,998,528 excessive global
sectors, 64% of the kernel's total.  The table contains only 256 immutable BF16
entries, so the SM80-only q8/GQA6 specialization now loads it with NVIDIA PTX
`ld.global.nc.u16`.  Portable Triton loads remain in the generic path.

The full explicit-dequantization suite passed 895/896/897-token boundaries,
mixed request lengths, q=5/8/16/64, 65K context and a high physical block ID;
maximum absolute error remained 0.00541.  At 250K, NCU measured:

| metric | ordinary LUT load | read-only LUT load |
|---|---:|---:|
| excessive global sectors | 85,998,528 (64%) | 76,288 (~0%) |
| partial-kernel duration | ~2.13 ms | 1.82 ms |
| no eligible warp cycles | ~63% | 54.75% |
| memory throughput | ~241 GB/s | 284 GB/s |

Paired FULL-graph model tests used identical prompt salts and acceptance:

| input | page-carry tok/s | read-only tok/s | page-carry ms/pass | read-only ms/pass |
|---:|---:|---:|---:|---:|
| 4K | 156.9 | 157.1 | 22.5 | 22.5 |
| 126K | 88.0 | 90.0 | 37.2 | 36.3 |
| 250K | 65.2 | 67.3 | 51.6 | 50.0 |

This is a cache-routing improvement rather than an approximation: the LUT bits,
attention arithmetic and cache addressing are unchanged.

### One-wave segment alignment for 70 SMs / 140 resident CTAs

NCU reported only 0.91 waves for the q8 verifier: `32 segments × 4 KV heads =
128 CTAs`.  The CMP 170HX has 70 SMs, and this kernel can keep two CTAs resident
per SM, so one full resident wave is 140 CTAs.  Raising the split count
indiscriminately is harmful—40, 48 and 64 create a partially occupied second
wave—but 35 produces exactly 140 CTAs.  The partial kernel keeps the same
arithmetic; only the small final segment reduction is padded to a legal
power-of-two with masked loads.

Interleaved isolated A/B against NSEG32 measured:

| input | NSEG32 us/layer | NSEG35 us/layer | speedup |
|---:|---:|---:|---:|
| 4K | 49.2 | 51.2 | 0.96× |
| 70K | 502.2 | 458.7 | 1.095× |
| 126K | 846.9 | 780.6 | 1.085× |
| 200K | 1312.7 | 1228.8 | 1.068× |
| 250K | 1667.8 | 1575.8 | 1.058× |

The short-context micro regression is hidden by fixed whole-model work: FULL-graph
4K remained 22.4 ms/pass.  Repeated model results were 34.7-35.4 ms/pass at
126K and 47.5-48.1 at 250K, versus 36.3 and 50.0 for NSEG32.  All tests had zero
preemptions; the full reference suite, including high block IDs, passed.

### Post-verifier profiler boundary

After the verifier changes above, a shape-aware CUDA profile at 126K measured
30.426 ms of GPU kernels per decode step.  The two remaining large components are
now nearly equal:

| component | time / step | share |
|---|---:|---:|
| speculative FP8 verifier partials | 12.847 ms | 42.2% |
| target Marlin GEMMs | 13.219 ms | 43.4% |
| GatedDeltaNet kernels | 1.120 ms | 3.7% |
| combine reduction | 0.123 ms | 0.4% |
| all other kernels | ~3.12 ms | 10.3% |

The dominant target GEMMs were `M=16,N=34816,K=5120` (64 calls, 6.111 ms)
and `M=16,N=5120,K=17408` (64 calls, 3.139 ms).  The selected Marlin kernel
launches exactly 70 CTAs on the 70-SM GPU and uses `(thread_k, thread_n,
threads)=(128,128,256)`.  Two source-built SM80-only alternatives were tested
with the exact-token harness.  `(128,64,128)` increased step latency by 18% at
4K and 12.4% at 126K; `(64,128,128)` increased it by 12.3% and 8.7%.
Therefore the stock selector remains qualified.  Do not raise its grid to 140:
the one-wave/140-CTA argument belongs to the two-resident-CTA verifier, while
this Marlin kernel consumes about 163 KiB dynamic shared memory and sustains one
CTA per SM.

This profile also rules out recurrent-state work as the principal long-context
bottleneck.  Future material gains must reduce verifier KV-scan cost without
duplicating reads, or change Marlin implementation/resource use more
fundamentally than reordering its existing tile candidates.

## Long-context policy

Do not jump directly to the advertised 1M capacity profile. Qualify in stages:

`64K -> 128K/256K -> 350K -> 500K -> 700K -> 1M`

For each tier validate memory fit, TTFT, decode, prefix reuse, semantic retrieval and Xid-free stress. YaRN/VRAM capacity alone is not evidence that a 1M context profile is production-ready.
