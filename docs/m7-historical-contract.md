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

## Corroboration of the `step` definition from the M7 patch itself

The `ms/step` meaning is confirmed independently by the M7 patch's own prose
(`experimental/cmp170hx-mixed-fp8/patches/spec-decode-nseg35-sm80.patch` at `7f15950`):

> Align the CMP 170HX verifier grid to its 140 streaming multiprocessors. ... With four
> KV heads, NSEG=32 launches only 128 CTAs and leaves twelve of the 140 SMs idle.
> NSEG=35 launches exactly 140 CTAs, while NSEG=40/48/64 spill into a second, poorly
> occupied wave. ...
> **Repeated FULL-graph model passes fell from 36.3 to 34.7-35.4 ms at 126K and from
> 50.0 to 47.5-48.1 ms at 250K; 4K stayed at 22.4 ms** with zero preemptions.

Two things follow:

1. `ms/step` is the **whole speculative model pass** under **FULL CUDA Graph** — the
   patch describes exactly that quantity, and its post-M7 values (34.7-35.4 at 126K,
   47.5-48.1 at 250K, 22.4 at 4K) bracket the frozen table's 35.230 / 46.941 / 22.315.
   So the frozen numbers are M7's repeated-pass latencies, not isolated-kernel times and
   not per-output-token times;
2. the geometry is **NSEG 35 x 4 KV heads = 140 CTAs on 140 SMs**. Note this is the SM
   count as the kernel sees it; earlier notes in this project described "70 SMs x two
   resident CTAs", and `2e03605` ("docs: correct CMP170HX resident CTA geometry") is the
   correction. The replay must use **NSEG=35**, not the deployed service's 32.

## Replay feasibility: my first conclusion was WRONG, corrected here

I first concluded "the deployed baseline service is not M7" and was about to write that
up as a blocker. That conclusion was wrong, and the checks that refute it are recorded
here so the mistake is not repeated.

**What I checked and what it actually means:**

| check | value | what it does *not* prove |
| --- | --- | --- |
| `mixed-fp8-repo/handover.md` header | "Last updated: 2026-09-15" | the *launcher* tree's date; the vLLM code lives elsewhere |
| `mixed-fp8-repo/deploy/cmp170hx-mixed-fp8.env.example` | `VLLM_SPEC_DECODE_ATTN_SEGMENTS=32` | an **example file**, overridden by the unit |
| `mixed-fp8-repo` patches dir | no `spec-decode-nseg35-sm80.patch` | the patch is applied to the **test-site**, not kept in that tree |

**What is actually true**, from the service unit and the test-site:

```
cmp170hx-mixed-fp8-full-256k-8002.service
  CTX=cmp-mixed-fp8  MAX_LEN=262144  MAX_SEQS=4  SPEC_ATTN=1
  VLLM_FP8_SPEC_VERIFY=1
  VLLM_FP8_SPEC_FULL_CG=1                 <- FULL CUDA Graph, as M7 requires
  VLLM_SPEC_DECODE_ATTN_SEGMENTS=35       <- M7's NSEG, set by the UNIT
  PYTHONPATH=/…/mixed-fp8-test-site       <- the M7 code lives here, shadowing the runtime
```

and the test-site itself is M7:

- `test-site/vllm/v1/attention/backends/flash_attn.py:1776` —
  `if value not in (8, 16, 32, 35, 64)` — **35 is allowed**, i.e. the nseg35 patch is applied;
- `test-site/vllm/v1/attention/ops/spec_decode_attn.py` contains `SEG_TILE` (3 occurrences)
  — the nseg35 combine padding is applied;
- dated 2026-09-16, matching M7.

**So M7 is installed and the replay can proceed by starting that service.** The runtime
this project has been using for the int8 work (`runtime-v0271`) contains **no** FP8
verifier stack at all — no `VLLM_FP8_SPEC_VERIFY`, no `_SPEC_ATTN_NSEG` — which is why the
patches did not apply to it: they belong to the test-site, not to that runtime. The two
paths are separate installs, which is the correct arrangement and also explains why the
int8 fix and the FP8 baseline never interfered.

The lesson, and it is the same one this project keeps relearning: **I read a configuration
artifact (an `.env.example`) and a directory date as evidence about the running system.**
The authoritative source is the unit's effective environment and the code actually on
`PYTHONPATH`.

## Checkpoint mismatch to carry into any Path A / Path B A/B

The M7 service targets `Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4`, while the repaired
`int8_per_token_head` path measured in phase 1 targets
`Qwen3.8-27B-W4A16-AutoRound-fast`. Those are **different checkpoints**, from different
quantization pipelines, so an A/B across them would confound path with checkpoint. Either:

- run both paths on the **same** checkpoint, or
- report any cross-checkpoint comparison explicitly as confounded.

The review's point about not calling different stacks the same "production control"
applies directly here.

## Replay plan (grounded in the checks above)

### What installing M7 requires

1. **Patch chain into the test-site**, in order (the nseg35 patch says "apply after
   `spec-decode-segments-sm80.patch`"):
   - `spec-decode-segments-sm80.patch` — makes the segment count explicit and adds it to
     the workspace cache key; allows `8, 16, 32, 64`, default 16;
   - `spec-decode-nseg35-sm80.patch` — adds **35** to the allowed set and pads the combine
     reduction to `next_power_of_2(nseg)` via `SEG_TILE`.
2. **Environment**: `VLLM_SPEC_DECODE_ATTN_SEGMENTS=35` (the deployed env example says
   **32**, which is pre-M7), plus the FP8 flags the service already sets
   (`VLLM_FP8_SPEC_VERIFY=1`, `VLLM_FP8_SPEC_FULL_CG=1`, `SPEC_ATTN=1`), `MAX_LEN=262144`,
   `MAX_SEQS=4`, `DFLASH_TOKENS=7`, `LOOKUP=0`, `PREFIX_CACHE=1`.
3. **FULL graph** (`VLLM_FP8_SPEC_FULL_CG=1`), because M7's number is a repeated
   FULL-graph pass.
4. **180 W** power limit (the freeze doc's operating point). Decide explicitly whether to
   also pin 1350 MHz, and record which was used.
5. **Corpus**: rebuild via `--corpus-root <tree>`, then **hash both the corpus and the
   prompt token IDs** and publish those hashes, because the original corpus is gone.

### The one decision that changes what "apples-to-apples" can mean

The M7 service targets `Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4`; the repaired
`int8_per_token_head` path in phase 1 targets `Qwen3.8-27B-W4A16-AutoRound-fast`. Both are
present on the host (15 GB each), so there are two defensible scopes:

- **A. Same-checkpoint Path A/B (clean A/B, loses M7-history comparability).** Run both
  paths on `AutoRound-fast` (or both on `Uncensored`), so the only difference is the
  path. The M7 *historical* numbers were taken on `Uncensored`, so a same-checkpoint A/B
  on `AutoRound-fast` compares Path A vs Path B cleanly but cannot be placed next to the
  frozen 22.315/35.230/46.941 without a checkpoint caveat.
- **B. M7-faithful replay first, then a same-checkpoint A/B (more GPU time).** Install M7
  on `Uncensored`, replay 4K/65K/126K/250K C1, and see whether 22.315/35.230/46.941
  reproduce. Then run the repaired int8 path on the **same** `Uncensored` checkpoint for
  the A/B, and separately note that phase 1's numbers were on `AutoRound-fast`.

Both are honest; they answer different questions. B answers "has the baseline moved?" and
"is the repaired path competitive with M7 on M7's own terms", which is what the review's
question 2 asks. A answers "is path A or path B faster" without the history question.

The review's question 2 ("is M7 mixed-FP8 still the fastest long-context production
candidate?") requires **B**. Question 3 ("how much faster/slower is repaired int8 at
4K/65K/126K/250K?") requires a **same-checkpoint** comparison, which B also provides on
`Uncensored`.

### Order of work, cheapest-first

1. paired 4K int64-widening regression test (review item 6) — short context, ~30 min,
   and it decides whether the widening cost anything;
2. install the M7 patch chain + `SEGMENTS=35` into the test-site, verify the kernel
   compiles and the service is healthy;
3. rebuild + hash the corpus, then replay 4K C1 → 65K C1 → 126K C1 → 250K C1 with the
   historical harness (`bench/context_ab.py`), 3 rounds each, recording `ms/step`,
   `tokens_per_step`, TTFT, decode tok/s and Xid delta;
4. run the repaired int8 path on the same corpus and checkpoint for the A/B;
5. only then answer the phase-2 questions and pick the next optimisation target.

## Corpus: exactly what is and is not reproducible

There are **two** corpus builders in the M7 era, and they are not the same rule:

**(a) `bench/make_long_corpus.py`** — used to build the long document that the LABD runs
consumed. It is a *concatenation*, not a tree walk:

```
head  = ~/bench/labd_corpus.txt          (frozen: this repo's own docs, ~84k tokens)
tail  = <venv>/vllm/v1/**/*.py sorted by glob, skipping files < 2000 chars,
        each prefixed "\n\n### {basename}\n\n", accumulated until 900,000 chars
assert long[:len(base)] == base          # the head must stay byte-identical
```

**(b) `bench/context_ab.py::build_corpus`** — the harness that produced `ms/step`. It takes
`--corpus-root <tree>` and walks it:
`sorted(root.rglob("*"))`, keeping `.py/.md/.txt/.cu/.cuh/.h`, each prefixed
`\n\nFILE {name}\n`, concatenated until **8,000,000** chars.

**What is reproducible:** the vLLM-source tail of (a) is on the host, so that half can be
rebuilt byte-identically given the same venv and the same 900,000-char cap. Tree (b) is
fully deterministic given a named tree.

**What is not:** `~/bench/labd_corpus.txt` (the frozen repo-docs head) **is gone** —
`~/bench/` is empty on the host. So the exact byte sequence that produced
22.315 / 35.230 / 46.941 cannot be regenerated, and the 8,000,000-char cap in (b) makes
the result sensitive to which files sort first.

**Therefore the replay's corpus is a reconstruction, and every number from it must carry:**

- the corpus tree used, named explicitly;
- `sha256(corpus)` and `sha256(prompt_token_ids)`;
- the statement that it is **not** token-identical to the historical corpus;
- the consequence: the replay is **strong evidence about kernel/iteration timing** and
  **weak evidence about historical output tok/s**, because acceptance depends on the text.

The tree chosen for the replay is the repository itself
(`/…/mixed-fp8-repo`, 9.0 MB, 86 `.py`/`.md` files), because `make_long_corpus.py`'s head
was this repo's docs and its tail was vLLM source — so the repo tree is the closest
single-tree stand-in for "docs + source", and it is reproducible and hashable.
