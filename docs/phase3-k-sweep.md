# Phase 3 — DFlash k = 3 / 5 / 7 whole-step sweep on the production M7 path

**k = 7 is not optimal.** The best k depends on context: **k=5 at 4K, k=7 at 126K, k=3 at
250K**, with **+8.7% at 250K** for k=3 — the context where latency actually hurts.

## Conditions

| dimension | value |
| --- | --- |
| path | M7 mixed-FP8, NSEG 35, FULL graph, FP8 target KV |
| checkpoint | `Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4`, DFlash2 W4A16 drafter |
| clocks / power | **1350 MHz locked, 180 W** |
| prompts | exact-token, unique text, same corpus (`sha256 ebf41c9d…`) |
| engine | **fresh per (k, context)**; C1; 3 rounds |
| profiling | **none** (the six metrics do not need one; attribution was Phase 2) |
| Xid | **delta 0 on all 9 cells** |
| stability | per-round spread 1.006–1.17x, all under the 3.0x gate |

Block size is recorded per cell from the running engine: **800 at k=3, 816 at k=5, 832 at
k=7** — it is derived from page-size arithmetic, not fixed at 896.

## The six required metrics

| k | ctx | ms/output-token | ms/spec-iteration | accepted/pass | passes/100 out tok | output tok/s | spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 4K | 8.098 | 21.152 | 2.598 | 38.82 | 123.49 | 1.087 |
| 3 | 126K | 11.167 | 30.294 | 2.667 | 37.65 | 89.55 | 1.047 |
| 3 | 250K | 15.409 | 38.380 | 2.594 | 38.95 | **64.90** | 1.170 |
| 5 | 4K | 7.425 | 21.963 | 2.970 | 33.99 | **134.68** | 1.006 |
| 5 | 126K | 11.341 | 34.376 | 3.037 | 33.20 | 88.18 | 1.028 |
| 5 | 250K | 17.042 | 46.231 | 2.705 | 37.12 | 58.68 | 1.030 |
| 7 | 4K | 7.712 | 22.825 | 2.963 | 34.12 | 129.67 | 1.045 |
| 7 | 126K | 11.085 | 34.898 | 3.178 | 31.63 | 90.21 | 1.029 |
| 7 | 250K | 16.753 | 46.945 | 2.738 | 36.86 | 59.69 | 1.092 |

`ms/output-token` and `ms/spec-iteration` are separate quantities throughout and are never
called each other.

## Winners, and the mechanism

| ctx | winner | tok/s | vs k=7 |
| --- | --- | --- | --- |
| 4K | **k=5** | 134.68 | **+3.9%** |
| 126K | k=7 | 90.21 | — (k=3 is −0.7%, inside the spread) |
| 250K | **k=3** | 64.90 | **+8.7%** |

Relative to k=7, decomposed into the two terms the review named
(`throughput ≈ accepted work per pass / (verify + draft cost)`):

| ctx | k | ms/spec-iteration | accepted/pass | output tok/s |
| --- | --- | --- | --- | --- |
| 4K | 3 | **−7.3%** | **−12.3%** | −4.8% |
| 4K | 5 | −3.8% | +0.2% | **+3.9%** |
| 126K | 3 | **−13.2%** | **−16.1%** | −0.7% |
| 126K | 5 | −1.5% | −4.4% | −2.3% |
| 250K | 3 | **−18.2%** | **−5.3%** | **+8.7%** |
| 250K | 5 | −1.5% | −1.2% | −1.7% |

The mechanism is exactly the predicted tradeoff:

- **at 4K the pass is cheap** (21–23 ms), so acceptance dominates and cutting k costs more
  acceptance (−12.3%) than it saves in pass cost (−7.3%) → low k loses;
- **at 250K the pass is expensive** (38–47 ms), so pass cost dominates: k=3 buys −18.2% per
  pass for only −5.3% acceptance → low k wins;
- **k=5 is a near-neutral middle** everywhere, and the best compromise at 4K.

So the optimum shifts from high-k at short context to low-k at long context, and the
crossover sits between 4K and 126K.

## Correctness gates (all pass)

**Greedy cross-k identity — the strongest available test.** Greedy speculative decoding is
*exact*: with temperature 0 the accepted sequence must equal plain greedy decoding for any
k. Therefore k=3, 5 and 7 must produce **byte-identical** greedy output on the same prompt.

```
k=3 GREEDY sha256=bbea525b9cd42edcb573f30119b3dc8834b18efce521fca599c6125dfaa692f6 len=443
k=5 GREEDY sha256=bbea525b9cd42edcb573f30119b3dc8834b18efce521fca599c6125dfaa692f6 len=443
k=7 GREEDY sha256=bbea525b9cd42edcb573f30119b3dc8834b18efce521fca599c6125dfaa692f6 len=443
```

**Identical.** The expected result, and it rules out any k being subtly wrong rather than
merely different in cost.

`temperature = 0.8` produces valid, differing text for each k — expected, since each k has a
different draft/verification step pattern and therefore a different random stream. This also
exercises the Phase 1B random-stream fix under sampling.

`Xid delta = 0` on every cell, and no preemptions.

## A corrupted first attempt, recorded so it is not repeated

The first sweep reported k=3 4K at **2–6 tok/s** (expected ~115–137). It was not a k=3
property:

- a clean re-measure of the same k=3 engine gives **111.7 tok/s**;
- `dmesg` shows **no Xid**;
- the engine **died mid-run** (`stop_profile` lost the connection, then every later cell got
  `ConnectionRefusedError`), so the harness recorded crash-adjacent timings as data.

The trigger was `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` left set from the Phase 2 diagnosis,
combined with a profiler run across whole rounds. Controlled comparisons (4K, k=7) give
22.48 ms/iter with neither, 22.15 with the profiler alone, 22.15 with both — so **the
mechanism is not fully isolated** and is recorded as such rather than explained away. What
is established is that the cell was not a measurement.

The real defect was **the absence of a sanity gate**, the same class of error already fixed
in the ABBA and B2 drivers. v2 adds `MAX_ROUND_SPREAD = 3.0` (exit rc=4 and abort the sweep
rather than bank a broken number), a fresh engine per cell, and a liveness check before each
cell. See `docs/phase3-k-sweep-protocol-fix.md`.

## Decision gate

250K's **+8.7%** clears the "clearly wins" bar, so per instruction the next step is C2/C4 at
250K before any production default change. 126K does **not** clear it (k=7 wins by 0.7%,
inside the spread), and 4K's +3.9% for k=5 is real but small.

A production default is therefore **not** changed by this phase. The data supports a
context-dependent policy rather than a single k, and that is a larger change than a default
flip — it needs the C2/C4 evidence first.
