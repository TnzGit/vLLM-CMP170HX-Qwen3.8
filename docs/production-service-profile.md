# Production service profile for the M7 long-context path

The deliverable: turn the fastest known M7 configuration into a small number of deployable
profiles with explicit context/concurrency boundaries, rather than accumulating experimental
patches. Everything below is measured on this box, not modelled.

## 1. The k crossover (C1, same protocol throughout)

Fresh engine per cell, 1350 MHz / 180 W, exact-token unique corpus, greedy, 3 rounds, spread
gate, no profiler. Block size read from the running engine.

| ctx | k=3 | k=5 | k=7 | winner | 2nd-best gap | block (k=3/5/7) |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | 123.49 | **134.68** | 129.67 | **k=5** | −3.7% | 800 / 816 / 832 |
| 32K | 134.64 | **147.02** | 137.62 | **k=5** | −6.4% | 800 / 816 / 832 |
| **48K** | **91.68** | 91.57 | 88.58 | **k=3** | **−0.1%** | 800 / 816 / 832 |
| 65K | **100.05** | 90.64 | 94.56 | **k=3** | −5.5% | 800 / 816 / 832 |
| 126K | 89.55 | 88.18 | **90.21** | k=7 | −0.7% | 800 / 816 / 832 |
| 250K | **64.90** | 58.68 | 59.69 | **k=3** | −8.0% | 800 / 816 / 832 |

**The crossover is at 48K**, where k=3 and k=5 are identical to within 0.1%. **k=7 is never a
meaningful winner**: its only lead is 0.7% at 126K, inside the measured round spread
(1.07–1.19), so it is dropped.

### The structural metric disagrees, and that matters

By **`ms/spec-iteration`** — acceptance-insensitive, and the metric the freeze document calls
primary — **k=3 wins at every context**:

| ctx | k=3 | k=5 | k=7 |
| --- | --- | --- | --- |
| 4K | **21.152** | 21.963 | 22.825 |
| 32K | **23.364** | 25.367 | 25.962 |
| 48K | **24.756** | 27.036 | 27.626 |
| 65K | **26.122** | 28.707 | 29.312 |
| 126K | **30.294** | 34.376 | 34.898 |
| 250K | **38.380** | 46.231 | 46.945 |

Lower k verifies fewer query rows per pass, so its pass is always cheaper. **The k=5 wins at
short context come entirely from higher acceptance** (e.g. at 32K, 3.76 accepted/pass vs 3.08),
which more than pays for the more expensive pass.

**Consequence for deployability:** the throughput-optimal short-context k depends on acceptance,
and acceptance depends on prompt content. So:

- if the production workload resembles this corpus, **k=5 is the short-context choice**;
- if acceptance on the real workload is materially lower (making the pass-cost term dominate),
  **k=3 everywhere is the safe single choice** and costs nothing structurally.

This is a genuine workload-dependence, not measurement noise, and it is why the recommendation
below offers both a 2-profile and a 1-profile option.

## 2. Long-context concurrency policy: `MAX_SEQS=1`

Service-policy A/B, fresh engine per arm, ABBA order, block 832 throughout, `Xid delta = 0`.

| arm | makespan | useful agg | first-completion | slowest request | KV max |
| --- | --- | --- | --- | --- | --- |
| **MAX_SEQS=1** @126K | **215.8 s** | **4.75 tok/s** | **107.6 s** | **84.2 tok/s** | 0.231 |
| MAX_SEQS=2 @126K | 219.8 s | 4.66 tok/s | 218.4 s | **4.4 tok/s** | 0.459 |
| **MAX_SEQS=1** @250K | **561.9 s** | **1.82 tok/s** | **278.1 s** | **68.7 tok/s** | 0.411 |
| MAX_SEQS=2 @250K | 589.8 s | 1.74 tok/s | 589.8 s | **1.7 tok/s** | **0.823** |

`MAX_SEQS=2` is **worse on every axis** at both contexts, including total makespan (+1.8% at
126K, +5.0% at 250K) — the axis it was supposed to improve. A single long-context request
already saturates the GPU, so a second one adds interference without throughput and only
redistributes latency (one request starved to 1.7–4.4 tok/s). It also consumes the KV margin
(82% at 250K).

**Long-context profile: `MAX_SEQS=1`.** The prefill budget does not change this (ITL flat within
0.4% across a 4x budget range), so this is a policy decision, not a tuning one.

## 3. The profile table

| profile | context range | k | MAX_SEQS | block/page geometry | rationale |
| --- | --- | --- | --- | --- | --- |
| **short** | ≤ 32K | **5** | 4 (default) | block 816 | measured winner at 4K (+3.7%) and 32K (+6.4%); higher acceptance pays for the pricier pass |
| **long** | ≥ 48K | **3** | **1** | block 800 | k=3 ties at 48K and wins at 65K (+5.5%) and 250K (+8.0%); MAX_SEQS=1 dominates at 126K and 250K |

Two profiles are sufficient. A third is **not** justified: k=7 wins nothing outside noise, and
250K needs no separate profile beyond `MAX_SEQS=1`, which the long profile already sets.

Optional single-profile fallback if a deployment prefers one configuration: **k=3, MAX_SEQS=1** —
structurally fastest at every context, and it forfeits only the short-context acceptance gain
(4K 134.68 → 123.49 tok/s, −8.3%).

## 4. Measured throughput and latency per profile

| profile | C1 decode tok/s | C1 ms/spec-iteration | TTFT (prefill) | accept/pass | notes |
| --- | --- | --- | --- | --- | --- |
| short @4K | 134.68 | 21.963 | ~2.5 s | 2.97 | |
| short @32K | 147.02 | 25.367 | ~19 s | 3.76 | |
| long @65K | 100.05 | 26.122 | ~40 s | 2.61 | |
| long @126K | 89.55 | 30.294 | ~103 s | 2.67 | |
| long @250K | 64.90 | 38.380 | ~272 s | 2.59 | single resident sequence only |

## 5. Capacity constraints

| constraint | value | evidence |
| --- | --- | --- |
| KV pool | ~1,149,000 tokens | engine startup |
| max **resident** sequences at 250K | **2** | `running_max = 2.0` with C=4 and `waiting_capacity = 3` |
| 250K C4 as a 4-way batch | **not achievable** | steady window negative (−602.8 s) |
| 250K C2 steady decode window | **window-limited** (1.40 s) | unique-prompt workload cannot form one |

## 6. Exact runtime contract (the profile is only valid with these)

| item | value |
| --- | --- |
| target | `Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4` |
| drafter | DFlash2 W4A16, `method=dflash` |
| attention backend | mixed-FP8 split-KV verifier, `VLLM_FP8_SPEC_VERIFY=1` |
| NSEG | `VLLM_SPEC_DECODE_ATTN_SEGMENTS=35` |
| FP8 spec FULL graph | `VLLM_FP8_SPEC_FULL_CG=1` |
| clocks / power | **1350 MHz locked, 180 W** |
| `max_model_len` | 262144 |
| `max_num_batched_tokens` | default 2048 → **2020 scheduled** |
| chunked prefill | enabled (parsed) |
| scheduler policy | `fcfs` |
| block size | **derived**, not fixed: 800 (k=3) / 816 (k=5) / 832 (k=7) |
| two vLLM installs | runtime **and** `mixed-fp8-test-site` (PYTHONPATH) must both be patched |

## 7. Xid and correctness status

- **Xid 31 delta = 0** in every experiment in this phase (k sweep, crossover scan, MAX_SEQS A/B,
  concurrency telemetry, budget sweep). Machine total: **0**.
- `preemptions = 0` throughout.
- Phase 1 patches installed in **both** vLLM trees: `#51812` (P0 correctness) and `#54282`
  (RNG isolation / reproducibility).
- Greedy cross-k identity verified earlier: k=3/5/7 produce **byte-identical** greedy output.

## 8. What this profile deliberately does not claim

- It does not claim 250K C2/C4 throughput: that regime is capacity/window-limited and is
  reported as such rather than measured badly.
- It does not propose an in-engine dynamic k. k participates in engine-level cache geometry
  (block 800/816/832 via page-size arithmetic), so a per-request switch is not available
  without restructuring the cache layout — out of scope.
- It does not claim the short-context k=5 choice is workload-independent; acceptance drives it,
  and acceptance is content-dependent (see §1).

## 9. DFlash2 greedy-equivalence caveat

The RNG-isolation patch in PR #6 improves reproducibility; it does **not** prove that a
block-shaped speculative verifier is token-exact with ordinary target-only `q_len=1` greedy
decoding.

Upstream vLLM issue #54928 reports Qwen3.8 cases where target-only and DFlash2 remain
individually deterministic but first diverge at a near-tie because the target's multi-position
verification forward ranks a different token than the single-token forward. Instrumented reports
show the emitted speculative token following the verifier argmax (`E == V != A`), including cases
with `--enforce-eager`; this is distinct from process-global RNG contamination.

This project has **not** established that the current W4A16 + M7 CMP170HX stack exhibits that
same behavior. Therefore:

- do not describe PR #6 as a token-equivalence fix;
- byte-identical output across k=3/5/7 proves cross-k consistency inside the speculative path,
  not target-only equivalence;
- before any future claim of target-only greedy equivalence, run a dedicated target-only vs
  DFlash2 gate on the exact production checkpoint/runtime and capture token IDs plus target
  top-logprobs around the first divergence;
- if a divergence appears, distinguish verifier numerical-path differences from recurrent/KV
  state corruption before changing state-management code.

Upstream reference: <https://github.com/vllm-project/vllm/issues/54928>.
