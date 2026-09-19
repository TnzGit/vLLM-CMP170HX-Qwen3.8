# M7 historical measurement contract (recovered from git history)

Recovered by reading the M0–M7 commits and the freeze document, **not** by inferring from
current doc names. This is the contract the historical numbers were taken under, and it
is what any "M7 replay" must reproduce before its numbers may be compared with them.

## Where the numbers and the contract actually live

| artifact | commit | what it is |
| --- | --- | --- |
| `docs/verifier-redesign-log.md` "Frozen performance and profiler boundary" | `1231ccf` "docs: freeze verifier redesign baseline" | the authoritative table, the kernel inventory, the correctness gates, the V1 stop conditions |
| `bench/context_ab.py` | present by `1231ccf`; unchanged in this repo since 2026‑09‑16 | **the harness that produced `ms/step`** |
| `bench/make_long_corpus.py`, `bench/labd_bench.py` | same era | the long-corpus builders |
| M0–M7 | `54ac927`, `9b2b463`, `9906e6f`, `85677cd`, `e7afec8`, `63dd0d8`, `8a5ceba`, `7f15950` | M7 = `7f15950` "perf: align verifier grid to CMP 170HX SM count" (NSEG 35) |
| `2e03605` | "docs: correct CMP170HX resident CTA geometry" | the CTA-geometry correction after M7 |

All nine SHAs are present locally and **M7 is an ancestor of the current branch**, so the
M7 tree is fully available.

## The metric definitions (verbatim from `bench/context_ab.py`)

```python
steps       = max(delta["vllm:spec_decode_num_drafts_total"], 1.0)
accepted    = delta["vllm:spec_decode_num_accepted_tokens_total"]
"tokens_per_step": 1.0 + accepted / steps,
"ms_per_step":     decode_s * 1000.0 / steps,
"decode_tok_s":    max(generated - 1, 0) / decode_s,
```

**`ms/step` is millisecond per actual speculative iteration.** The step count is the
server's own `vllm:spec_decode_num_drafts_total` Prometheus counter delta — **not** the
output-token count. This is the question the review asked to settle, and the answer is
that the historical number is a genuine per-speculative-step figure.

`decode_s` is a **direct decode interval**, not a subtraction:

```python
stream=True, stream_options={"include_usage": True}
first = time.perf_counter() when the first SSE choice with non-empty text arrives
end   = time.perf_counter() at stream end
ttft     = first - start
decode_s = end - first
```

So TTFT is measured directly from the first content chunk and decode is measured from
that chunk to completion. **The historical protocol never subtracted an independent
prefill wall**, which is exactly the unstable method this project must now retire for
126K/250K (per the review's point D).

## Full contract

| dimension | M7 value |
| --- | --- |
| target | the mixed-FP8 W4A16 target used by `cmp170hx-mixed-fp8` (AutoRound W4A16, Marlin) |
| drafter | DFlash2 W4A16, `method=dflash` |
| DFlash k | **7** |
| target KV | **static FP8 (E4M3FN)** via FlashInfer, block size **896** |
| draft KV | BF16 |
| attention backend | the FP8 spec-verify path (`FlashInfer` target attention + the split-KV q8 verifier), **not** `TRITON_ATTN` |
| NSEG | **35** (M7's change: 35 segments × 4 KV heads = 140 CTAs ≈ 70 SMs × 2 resident CTAs) |
| graph mode | **FULL** CUDA Graph |
| MAX_SEQS | single-stream for the frozen numbers (C1) |
| clocks / power | **180 W** power limit; the freeze doc says "180 W step latencies" (see the caveat below on 1350 MHz) |
| prompt source | `--corpus-root <dir of .py/.md/.txt/.cu/.cuh/.h>`, walked in `sorted(rglob)` order, each file prefixed `\n\nFILE {name}\n`, concatenated until 8,000,000 chars |
| exact token length | `exact_prompt_ids()`: header + corpus repeated to `body_len` + query, **asserted `len(ids) == length`** |
| prefix-cache defeat | `--salt`, which changes the header text between A/B samples |
| output length | `--output`, default 256 |
| acceptance | reported as `tokens_per_step = 1 + accepted/steps` |
| sampling | `temperature=0.0`, `ignore_eos=True` |
| metrics source | the server's `/metrics` Prometheus endpoint, delta across the request |

Historical values (180 W, C1):

| input | qualified ms/step | representative decode tok/s |
| --- | --- | --- |
| 4K | 22.315 | ~161 |
| 126K | 35.230 | ~98 |
| 250K | 46.941 | ~70 |

At 126K the shape-aware profile assigned 30.426 ms of GPU kernel time per step:

| component | ms/step | share |
| --- | --- | --- |
| FP8 verifier partials | 12.847 | **42.2%** |
| target Marlin GEMMs | 13.219 | **43.4%** |
| GatedDeltaNet | 1.120 | 3.7% |
| combine | 0.123 | 0.4% |
| other | ~3.12 | 10.3% |

## Correctness gates M7 already carried

The freeze document lists the gates every candidate must preserve, and one of them is
directly relevant to the defect this project just fixed:

> int64 addressing before physical-block stride multiplication

**The FP8 verifier already widens the physical block id before the stride multiply**,
and this is verifiable in the M7 patch itself. From
`experimental/cmp170hx-mixed-fp8/patches/spec-decode-fp8-page-carry-sm80.patch` at `7f15950`:

```python
-        blk = tl.load(bt_ptr + req * stride_bt + tok0 // BLOCK_SIZE).to(tl.int64)
+        next_page = tok0 // BLOCK_SIZE
+        if next_page != page_idx:
+            page_idx = next_page
+            blk = tl.load(bt_ptr + req * stride_bt + page_idx).to(tl.int64)
+        slot = (tok0 % BLOCK_SIZE) + tl.arange(0, TILE)
+        kptr = k_ptr + blk * stride_kb + slot[:, None] * stride_ks + ...
+        vptr = v_ptr + blk * stride_vb + slot[:, None] * stride_vs + ...
```

Note the **`-` line**: even the pre-M7 form already carried `.to(tl.int64)` before
`blk * stride_kb`. The sibling kernel had the widening from the start; the
`TRITON_ATTN` + `int8_per_token_head` path (`_spec_attn_partial`) did not, and that
omission was the long-context illegal memory access this project fixed. The fix brings the
int8 path to the same contract the FP8 path already met -- which is corroboration from the
repository's own history that `blk * stride` overflow was a recognised hazard on this
hardware, not a speculative theory.

Also listed: exact 256-code E4M3FN semantics, explicit-dequantization agreement for
q=5/8/16/64, 895/896/897 boundaries at `--block-size 896`, mixed KV lengths and high
physical block IDs, graph-stable workspace addresses, and zero new Xid/NVRM events.

## What is NOT reproducible, and must be stated rather than papered over

**The frozen corpus no longer exists.** `bench/make_long_corpus.py` builds
`~/bench/labd_corpus_long.txt` from `~/bench/labd_corpus.txt` (the repo's own docs) plus
vLLM's source; on the lab host **`~/bench/` is empty** — both files are missing. So the
exact byte sequence that produced 22.315 / 35.230 / 46.941 is gone.

Consequences for a replay, in order of preference:

1. `build_corpus(--corpus-root)` is **deterministic given a tree**: `sorted(rglob)`,
   fixed suffixes, fixed prefix format, 8M-char cap. So a replay can rebuild a corpus
   reproducibly *from a named tree* — but it will not be the same corpus unless the tree
   matches, and the 8M-char cap makes the result sensitive to exactly which files sort
   first;
2. therefore a replay must (a) name its tree explicitly, (b) hash the resulting corpus
   and the prompt token IDs, and (c) report acceptance alongside `ms/step`, because
   `ms/step` is acceptance-insensitive by construction while `decode tok/s` is not;
3. a replay that reproduces `ms/step` within a few percent on a *different* corpus is
   evidence the kernel timing is reproduced; it is **not** a token-for-token replay, and
   must not be described as one.

## Clock discrepancy to resolve before the replay

The freeze doc says the latencies are **180 W** step latencies. This project's recent work
has pinned **1350 MHz** and 180 W. Those are different operating points, and the review
asks for 1350 MHz locked. The replay should therefore record which it used and, if the
numbers disagree with history, try the other before blaming the protocol.

## Source of these numbers

`docs/verifier-redesign-log.md` as introduced by `1231ccf`; `bench/context_ab.py`,
`bench/make_long_corpus.py` as present at `1231ccf`; the M7 diff `7f15950`.
