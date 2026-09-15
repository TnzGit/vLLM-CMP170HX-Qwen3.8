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

## Long-context policy

Do not jump directly to the advertised 1M capacity profile. Qualify in stages:

`64K -> 128K/256K -> 350K -> 500K -> 700K -> 1M`

For each tier validate memory fit, TTFT, decode, prefix reuse, semantic retrieval and Xid-free stress. YaRN/VRAM capacity alone is not evidence that a 1M context profile is production-ready.
