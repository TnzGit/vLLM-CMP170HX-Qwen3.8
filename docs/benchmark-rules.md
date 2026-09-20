# Benchmark rules — permanent checklist

Written after one defect class appeared **three times in a single session**, each time
producing a plausible-looking but wrong number. These are binding for any future measurement
in this repository.

## R1 — Every ratio must state the interval its numerator and denominator cover

The defect that recurred three times was always the same: **a ratio whose numerator and
denominator cover different intervals.** It never announced itself as an error; it produced a
number that looked reasonable.

| # | ratio | what went wrong |
| --- | --- | --- |
| 1 | `it / max_i(end_i − first_i)` as "ms/spec-iteration" at C>1 | the window counted another request's **prefill** as decode time — inflated by ~14x |
| 2 | `request_total_tokens / steady_window` | the numerator included tokens emitted **before** the window opened — inflated 4K C2 from 159 to 224 tok/s |
| 3 | `counter_delta / steady_window` | the delta spans first-sample..last-sample, **shorter** than the window — under-reported by 35%, which looked like "throughput collapsed" |

**Required for any ratio:**

```
numerator   start timestamp, end timestamp
denominator start timestamp, end timestamp
```

Emit the ratio **only if the two intervals are the same**. Otherwise:

- report `null`, or
- report an explicit **bounded range**,

and never a "reasonable estimate". A metric whose name and computation disagree is a bug even
when the number looks plausible.

**Corollary for counters.** A Prometheus counter delta covers first-sample..last-sample, not the
nominal window. Either sample at the window edges or divide by the measured sample span.

**Corollary for concurrency.** Server counters are **global**, with no per-request attribution.
Never divide a global counter delta by a per-request interval.

## R2 — Concurrent decode must use a steady-state window

```
steady_start = max_i(first_output_i)     # every request is past prefill
steady_end   = min_i(finish_i)           # no request has finished
analyse only [steady_start, steady_end]
```

The window is valid only if **all** of:

- `prompt_tokens` delta inside the window is **≈ 0**;
- every intended request is `running` and `waiting == 0`;
- no request has finished;
- the token-counter sample span equals the rate denominator (R1).

If the window cannot be made long enough, **lengthen the output** — never fall back to
full-request wall time to hide prefill. If it is still too short, report **`window-limited`** and
do not compute the metric.

Measured example of why: at 250K C2 the requests' first tokens are **302.5 s apart**, so the
steady window is **1.40 s**, and at 250K C4 it is **negative** (all four never decode together).
Reporting a number there would be fabrication.

## R3 — Distinguish the three kinds of "waiting"

| kind | how to identify | meaning |
| --- | --- | --- |
| scheduler **token-budget** waiting | `num_requests_waiting_by_reason{reason="capacity"}` with `running < max_num_seqs` and low KV | the chunked-prefill budget could not schedule two long prefills in one step |
| **KV / residency capacity** | KV usage near the limit and `running` pinned at a ceiling | the engine cannot physically hold more sequences |
| **true preemption** | `vllm:num_preemptions_total` delta > 0 | a running request was evicted |

**Do not** automatically read the generic `capacity` reason as "GPU KV OOM". At 250K C4 the
engine reported `running_max = 2.0` with `waiting_capacity = 3` at **82% KV usage** — that is a
residency ceiling, and it was established by watching `running`, not by interpreting a label.

## R4 — Config facts come from the parsed runtime, never from the launcher

- Read `max_num_batched_tokens` / `max_num_scheduled_tokens`, `enable_chunked_prefill`,
  `long_prefill_token_threshold`, `policy` from the engine's parsed config or startup log.
- **A flag's absence from a launcher is not evidence about the value the engine uses.** An
  earlier conclusion in this project was invalidated by exactly that mistake.
- The effective **block size is derived**, not fixed: 800 at k=3, 816 at k=5, 832 at k=7, from
  page-size arithmetic. Read it per run (`Setting attention block size to N`); never assume the
  kernel comment's 896.
- Read the engine's mode from `/proc/<engine-pid>/environ`, not from the launcher.

## R5 — Separate the measurement categories, and label them

| category | may be used for |
| --- | --- |
| isolated kernel microbenchmark | deciding whether an e2e A/B is worth running — **not** as an e2e win |
| service policy (makespan, TTFT, completion) | admission/concurrency decisions |
| steady-state concurrent decode | concurrency scaling |
| C1 decode throughput | single-stream throughput and k selection |

Never mix them in one cell, and never present a modelled number as a measured one.

## R6 — Negative results are results

Record every killed direction with the evidence that killed it, so it is not re-attempted. This
session alone produced: tensor-core softmax denominator (slower), two-level PV (register
ceiling), fused DFlash conv (contribution too small), prefill-budget tuning (no effect),
`MAX_SEQS=2` (worse on every axis), and the register-release audit (no headroom).

## R7 — Standing operational rules

- Read the effective config from the **running** engine.
- One context per **fresh engine** for any cell that matters.
- Unique text per request; exact token lengths verified against `usage.prompt_tokens`.
- Report **Xid delta**, never an absolute count (dmesg is a ring buffer holding other runs).
- A crashed engine is **poisoned** — restart before the next cell.
- Cleanup by **PID** from `nvidia-smi --query-compute-apps`, never `pkill -f` (it self-matches
  and has killed the invoking session).
- Add a **fail-fast** to every driver: a cell that produces no measurement must abort the run,
  and a spread gate must refuse a cell whose rounds disagree by more than ~3x.
- **Port 8000 is production and is never touched**; research runs on 8002.
- Distinguish measurer from measured: fix the *tool* and say so when the tool was wrong.

## R8 — Reproducibility is not target-only speculative equivalence

A dedicated RNG stream can make the speculative path reproducible without making its output
token-exact with a target-only `q_len=1` greedy forward. For Qwen3.8, upstream issue #54928 has
instrumented examples where the block verifier's target logits choose a different argmax near a
tie and the emitted token follows that verifier argmax.

For any equivalence claim, compare on the **same checkpoint and runtime**:

- target-only emitted token `A`;
- verifier target-logit argmax `V`;
- speculative emitted token `E`;
- target-only top-1/top-2 margin at the first divergence.

`E == V != A` is evidence of a target numerical/execution-path mismatch, not by itself evidence
of draft RNG or cache rollback corruption. RNG-repeatability tests and target-equivalence tests
are separate gates.
