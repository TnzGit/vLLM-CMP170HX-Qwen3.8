# Phase 4B — tensor-core softmax denominator: NEGATIVE RESULT

**The variant is 11–14% SLOWER, not 2–3% faster. The gate fails and this line of work
stops.** No further tuning was attempted, per the review's instruction.

## Why this was the first experiment

Phase 2 measured the verifier at **52.2% of the 250K step**, growing with context. Phase 4A
found one structural idea in the production kernel worth testing: the softmax denominator is
reduced on the generic FP32 path (`tl.sum(p, 1)`) while the numerator already uses tensor
cores (`tl.dot`). Expressing the row-sum as `p @ ones` moves it onto the tensor cores without
changing the KV representation or the algorithm.

## Method

An **isolated kernel microbenchmark** on the production geometry (SM80, D=256, Hq/Hkv=24/4,
GQA=6, q=4/8, block 832, NSEG=35), comparing the production kernel against a generated
variant that differs in **exactly** the intended way.

The variant is produced by `ph4b_generate_variant.py`, which asserts each anchor matches
exactly once and fails otherwise, rather than hand-copying a 120-line Triton kernel. The
verified diff is: the `_rowsum_tc` helper plus **two** substitutions.

```
    l0 = l0 * a0 + _rowsum_tc(p0, TILE)
    l1 = l1 * a1 + _rowsum_tc(p1, TILE)
```

Every load, mask, `exp`, scale, store and the combine kernel are untouched.

**Baseline cross-validates Phase 2.** The isolated kernel gives 12.357 ms/16-layers at 126K
against Phase 2's 12.653 ms/pass from the in-server trace — **2.3% agreement** — and 24.323
against 24.888 at 250K (2.3%). Two independent methods, same number.

## Result

| ctx | q | production µs/layer | TC-denominator µs/layer | delta | max abs diff |
| --- | --- | --- | --- | --- | --- |
| 126K | 4 | 773.24 | 875.07 | **+13.17%** | 1.630e-02 |
| 126K | 8 | 777.99 | 883.28 | **+13.53%** | 1.660e-02 |
| 250K | 4 | 1520.66 | 1713.25 | **+12.67%** | 1.114e-02 |
| 250K | 8 | 1544.68 | 1713.95 | **+10.96%** | 1.349e-02 |

## Gate evaluation

| gate | result |
| --- | --- |
| 1. oracle correctness within tolerance | **PASS** — max abs diff 1.66e-02 on an O(1) output (threshold 5e-2) |
| 2. **≥2–3% latency gain at BOTH 126K and 250K** | **FAIL** — +13.5% and +11.0%, i.e. slower |
| 3. no occupancy/register/spill regression | not reached (gate 2 already failed) |
| 4. CUDA-graph fixed-address replay clean | not reached |

**VERDICT: negative. Stop.**

## Why it is slower, not faster

The substitution trades a cheap reduction for an expensive MMA. `tl.dot(p, ones)` with `p`
shaped `[32, TILE]` and `ones` shaped `[TILE, 16]` performs a `32 x TILE x 16` matrix
multiply — on SM80 that is a full tensor-core instruction sequence, and **N=16 is the minimum
`tl.dot` allows**, so 15 of the 16 computed columns are discarded. It also requires casting
`p` from FP32 to BF16.

So the variant pays an MMA plus a dtype conversion to replace a 32-wide FP32 row-sum that the
generic reduction path already does efficiently. The row-sum was never the bottleneck: Phase
2's profile shows the verifier's cost is dominated by the **KV load and decode** (the
`_e4m3fn_to_bf16_ldg_nc` LUT path over a context-length scan), not by the softmax reduction.

This is a useful negative result precisely because it rules out a whole family: **moving the
softmax reduction to the tensor cores does not help, because the reduction is not where the
verifier's time goes.** Any further idea in this direction inherits the same refutation.

## Artifacts

| file | contents |
| --- | --- |
| `bench/int8-g64/ph4b_tc_denominator.py` | isolated baseline harness (also establishes the oracle) |
| `bench/int8-g64/ph4b_generate_variant.py` | verified textual generator for the variant |
| `/tmp/ph4b_ab.py` (host) | the gated A/B driver |

The production file was **never modified**: the variant lives in a separate module and is
imported only by the experiment.
