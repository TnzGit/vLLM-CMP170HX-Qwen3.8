# Concurrency root cause — classification of the 250K C2/C4 anomaly

Answers the review's taxonomy with telemetry rather than inference. **Case B is the
mechanism that corrupted the metric; Case A makes 250K C4 structurally unmeasurable; Case C
is ruled out.**

## Two corrections to my own earlier reasoning

1. **The "refutation" was invalid.** I argued that because neither run set
   `--max-num-batched-tokens`, concurrent prefill/decode interference was disproved. That does
   not follow: C1 has no competing long prefill, so an identical scheduler config can behave
   completely differently once C2/C4 introduces one. And a CLI flag's absence is not a fact
   about the parsed config.
2. **The "measurement-definition only" conclusion was premature.** The server's own log did
   report single-digit generation throughput, which is real wall-clock behaviour, not an
   artifact of my arithmetic.

## Actual parsed runtime configuration (not inferred from the launcher)

| item | value | source |
| --- | --- | --- |
| `max_num_scheduled_tokens` | **2020** | engine startup log (derived from the default `max_num_batched_tokens = 2048` minus draft-token slots) |
| `enable_chunked_prefill` | **True** | parsed config dump |
| `long_prefill_token_threshold` | 0 (default) | `SchedulerConfig` fields |
| `policy` | `fcfs` | `SchedulerConfig` fields |
| `scheduler_reserve_full_isl` | True | `SchedulerConfig` fields |
| `num_requests_waiting_by_reason` | `capacity` / `deferred` | live `/metrics` |

## Telemetry (1 Hz: running, waiting-by-reason, prompt/generation token deltas, KV usage, preemptions, per-request first/finish)

output = 512 tokens per request, k=7, 1350 MHz / 180 W, exact-token unique prompts.

| ctx | C | batch wall | **steady window** | running max | waiting (capacity) | prompt Δ in window | **gen Δ in window** | **aggregate tok/s** | ITL p50 | ITL p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4K | 2 | 11.7 s | 4.56 s | **2.0** | 1 | 0.0 | 726 | **159.2** | 28.0 ms | 56.3 ms |
| 32K | 2 | 43.6 s | 4.17 s | **2.0** | 1 | 0.0 | 707 | **169.5** | 34.2 ms | 34.7 ms |
| 126K | 2 | 220.0 s | 5.19 s | **2.0** | 1 | 0.0 | 495 | **95.4** | 53.3 ms | 54.0 ms |
| 250K | 2 | 587.0 s | **1.40 s** | 2.0 | 1 | 0.0 | **0** | n/a | 77.8 ms | 81.1 ms |
| 250K | 4 | 1193.7 s | **−602.8 s** | **2.0** | **3** | n/a | 0 | n/a | n/a | n/a |

`Xid 31 delta = 0`; `preemptions = 0` throughout.

### First-token skew — the interference signature

| ctx | C | first content at (s from batch start) | skew |
| --- | --- | --- | --- |
| 4K | 2 | 3.78, 4.25 | **0.47 s** |
| 32K | 2 | 20.66, 39.05 | **18.4 s** |
| 126K | 2 | 103.83, 212.61 | **108.8 s** |
| 250K | 2 | 277.08, 579.62 | **302.5 s** |
| 250K | 4 | 278.10, 581.16, 887.07, 1185.32 | **907.2 s** |

## Classification

| case | verdict | evidence |
| --- | --- | --- |
| **A — capacity / admission serialization** | **CONFIRMED (250K C4; partial elsewhere)** | at 250K C4, `running_max = 2.0` with C=4 and `waiting_capacity = 3` at 82% KV usage — **4-way residency never occurred**. `waiting_capacity = 1` at every C2 cell: the 2020-token budget cannot schedule two long prefills in one step, so prefills serialize. |
| **B — prefill/decode scheduler interference** | **CONFIRMED — the mechanism that corrupted the metric** | skew grows 0.47 s → 907 s with context. A request whose prefill finishes first decodes while the others still prefill, so its first-token→finish interval is mostly *other requests' prefill*. My old aggregate took `max_i(end_i − first_i)` as "the decode window" and divided the global iteration counters by it. |
| **C — genuine decode-concurrency collapse** | **RULED OUT** | inside the steady window, `prompt_tokens_delta = 0.0`, `running = 2.0`, `waiting = 0`, and aggregate throughput is **159 / 170 / 95 tok/s** at 4K/32K/126K — *higher* than the same engine's C1 (129.7 / 152.8 / 90.2). Concurrency works. |
| **D — measurement-only artifact** | partially | the old `ms/spec-iteration` was mis-defined for C>1, but the cause is A+B, not a definitional quirk: real work (prefill) genuinely occupied the interval. |

### The C2 < C4 inversion is explained

The review flagged 757 ms/iter (C2) vs 229 ms/iter (C4) as a clue. It is a **first-token
synchronization artifact**: at C4 the four prefills happen to land such that the
slowest-to-first-token request is not the one whose window is taken, whereas at C2 one request
decodes alone for ~300 s. It is not a verifier property, and at C4 the cell is not even 4-way.

## The steady-state decode metric (definition adopted)

For any concurrency comparison:

```
steady_start = max_i(first_output_i)      # every request is past prefill
steady_end   = min_i(finish_i)            # no request has finished
analyse only [steady_start, steady_end]
```

Within that window the global counter deltas *are* attributable to concurrent decode, so:

- **numerator must cover the same interval as the denominator.** In-window tokens come from
  the server's own `vllm:generation_tokens_total` delta, **not** from the request's total
  output. My first version of this harness repeated the original defect here (4K C2 reported
  224 tok/s instead of 159); it is now fixed and the wrong value is retained in the output as
  `steady_tok_s_naive_wrong` so the error cannot silently return.
- report aggregate tok/s, per-request ITL p50/p95, and the in-window prompt-token delta, which
  must be ~0 for the window to be valid.

## Production findings that outrank a verifier micro-optimisation

1. **Concurrent decode scaling degrades with context**: 2x concurrency buys **1.23x** at 4K,
   **1.11x** at 32K, and only **1.06x** at 126K. Adding a second long-context request is nearly
   worthless for aggregate throughput on this stack.
2. **250K cannot host C4 at all** on this configuration: residency caps at 2 sequences.
3. **The 2020-token scheduling budget serializes long prefills**, producing first-token skews
   of 100–900 s. This is the *dominant* effect at long context and is a scheduler/admission
   property, not a kernel property.

## Consequences

- **250K C2/C4 are marked capacity/window-limited and are NOT a k production gate.** Their
  earlier 757 ms/iter figure is withdrawn as a verifier measurement.
- The concurrency k gate should use **126K**, the longest context with a usable steady window
  (5.19 s).
- Phase 3's **C1 conclusions stand unchanged** (4K k=5 +3.9%, 126K k=7, 250K k=3 +8.7%) —
  fresh engine, exact corpus, 1350 MHz/180 W, clean Xid, stable spread.
- **No in-engine dynamic k policy**: k changes the effective KV/page geometry (block 800 at
  k=3, 816 at k=5, 832 at k=7), so k participates in engine-level cache layout. Any future
  policy must be a **service-profile static k**, not a per-request switch.
