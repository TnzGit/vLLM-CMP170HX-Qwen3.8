# Phase 2 — component attribution on the M7 production path (current head)

Answers: *on the final M7 path, where does a whole step's time go?* Only Path A is
profiled. This is a fresh measurement at the current head, not a restatement of the
historical profile.

## Method, and the two bugs it took to get right

Attribution comes from vLLM's **in-server** torch profiler, driven through
`POST /start_profile` / `POST /stop_profile`. No client-side profiler is involved, so the
CUDA kernels are genuinely server-side.

Two defects in my first analysis, both caught because the numbers were impossible:

1. **the trace glob missed gzipped traces.** vLLM writes `*.pt.trace.json.gz` by default,
   so a `*.json` glob found nothing: the latency table looked perfect while the
   attribution silently failed;
2. **the first parse summed the whole trace and double-counted.** It included the ~277 s
   prefill and added `gpu_user_annotation` events, which *wrap* kernels. Shares summed to
   **77,705%**, which is how the bug announced itself.

Correct method (`bench/int8-g64/parse_trace_attribution.py`): locate the decode passes from
the annotations, take the decode window, and count **only `cat == "kernel"`** events inside
it.

**Cross-validation.** The trace gives 48.21 ms/pass at 250K while `/metrics` independently
gives 47.14 ms/spec-iteration — two independent methods within **1.1%**. The method is
sound.

## Conditions

| dimension | value |
| --- | --- |
| path | M7 mixed-FP8: FP8 target KV, BF16 draft KV, NSEG 35, `VLLM_FP8_SPEC_FULL_CG=1` (FULL graph) |
| checkpoint | `Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4` |
| drafter | DFlash2 W4A16, k=7 |
| clocks / power | **1350 MHz locked, 180 W** |
| prompts | exact-token, unique text, same corpus (`sha256 ebf41c9d…`) |
| engine | fresh per context, C1, 3 rounds (last profiled) |
| Xid | **delta 0** at both contexts |

## Three quantities, never conflated

| ctx | ms/output-token | ms/spec-iteration | accepted/pass | passes/100 out tok | output tok/s | TTFT |
| --- | --- | --- | --- | --- | --- | --- |
| 126K | **10.935** | **34.637** | 3.187 | 31.57 | 91.48 | 102.4 s |
| 250K | **16.916** | **47.143** | 2.809 | 35.88 | 59.12 | 276.6 s |

`ms/output-token` is the throughput view; `ms/spec-iteration` is the historical `ms/step`
(denominator = `vllm:spec_decode_num_drafts_total`). They differ by the accepted tokens per
pass, and neither is called the other.

## Component split (ms per speculative pass)

**126K**

| component | ms/pass | share |
| --- | --- | --- |
| **target Marlin GEMMs** | **17.203** | **48.8%** |
| **verifier partial** (`_spec_attn_partial`) | **12.653** | **35.9%** |
| GDN | 2.220 | 6.3% |
| unclassified | 1.368 | 3.9% |
| elementwise | 0.935 | 2.7% |
| rmsnorm / silu | 0.374 | 1.1% |
| flash attention | 0.364 | 1.0% |
| verifier combine | 0.134 | 0.4% |
| **total kernel time** | **35.252** | |

**250K**

| component | ms/pass | share |
| --- | --- | --- |
| **verifier partial** | **24.888** | **52.2%** |
| **target Marlin GEMMs** | **17.330** | **36.4%** |
| GDN | 2.238 | 4.7% |
| unclassified | 1.379 | 2.9% |
| elementwise | 0.945 | 2.0% |
| rmsnorm / silu | 0.378 | 0.8% |
| flash attention | 0.367 | 0.8% |
| verifier combine | 0.139 | 0.3% |
| **total kernel time** | **47.664** | |

## Comparison with the historical 126K profile

| component | historical (freeze doc) | now | change |
| --- | --- | --- | --- |
| verifier partials | 42.2% | **35.9%** | −6.3 pts |
| target Marlin GEMMs | 43.4% | **48.8%** | +5.4 pts |
| GDN | 3.7% | 6.3% | +2.6 pts |
| combine | 0.4% | 0.4% | — |
| other | 10.3% | ~8.6% | −1.7 pts |

The ordering is unchanged — **verifier and Marlin remain the two dominant blocks** — but
the balance has shifted toward Marlin by roughly 5–6 points. The historical figure was
taken on a different corpus and a different head, so this is a re-measurement rather than a
regression: the split is now established for the current head, which is what Phase 4 needs.
The verifier's share also *falls* with context at 250K relative to 126K? No — it rises
(35.9% → 52.2%), and Marlin's falls (48.8% → 36.4%), because the verifier scans KV and so
grows with context while the GEMMs do not. That is the single most important structural
fact for Phase 4.

## What could NOT be separated, stated rather than guessed

The review asks for `draft ms/pass` separately from target verify ms/pass. **The trace
cannot split them**: the drafter and target share one Marlin template signature
(`void marlin::Marlin<…>`, 501.5 calls/pass), and `gpu_model_runner: draft` phase
annotations are null contexts unless `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`, which the runs
did not set.

What can be bounded from the data: Marlin call durations are bimodal. Calls **< 20 µs**
number 227.8/pass and total only **0.9 ms/pass (5.2% of Marlin time)**; calls ≥ 20 µs total
16.3 ms/pass. The drafter is 5 layers against the target's 64, so its GEMMs are the small
population. **Therefore the draft contributes at most ~5% of Marlin time**, and the
"target Marlin 48.8%" block is at least ~95% genuinely target-side. That bound is enough to
keep Phase 4 aimed at the verifier, but a true draft/target split needs
`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` plus the phase annotations, and is not claimed here.

Draft-adjacent kernels that *were* identifiable are negligible: `_cache_draft_logits_kernel`
0.002 ms/pass, `_combine_sampled_and_draft_tokens_kernel` 0.005, `_compute_local_logits_stats`
0.014, `reshape_and_cache_flash` 0.090 — together 0.112 ms/pass (**0.32%** of decode kernel
time). So draft *bookkeeping* is not a cost centre either.

## Answer to the Phase 2 question

> On the current M7 path, is a unit of engineering time better spent on the verifier, on
> DFlash proposal efficiency, or on target Marlin?

**The verifier.** It is 52.2% of the 250K step and **grows with context**, while Marlin is
36.4% and context-independent. At 250K the verifier is the single largest block and the
only one whose share increases exactly where latency hurts most. Marlin is second and worth
a structural attempt later (paired gate/up + SwiGLU), but the verifier dominates the
long-context regression that is the actual production complaint.
