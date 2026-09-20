# Phase 3 extension — the concurrency cell was not measurable, and why

The 250K C2/C4 extension ran to completion (4/4 cells, `rc=0`, `Xid delta=0`) and produced
numbers that were **impossible**: 250K C2 at 2.19 tok/s and 757 ms/spec-iteration, against
38 ms/spec-iteration at C1. A 20x per-pass regression cannot come from 2x concurrency.

This records what was wrong, because the failure was in the **metric definition**, not in
the engine.

## The defect

My concurrency aggregate used `decode_window = max_i(end_i - first_i)`. Under **chunked
prefill** that is not a decode window at all. Instrumented at 32K C2:

| request | first content | end | window |
| --- | --- | --- | --- |
| req0 | 20.6 s | 40.1 s | 19.5 s |
| req1 | **39.0 s** | 40.4 s | 1.4 s |

req1 was **prefilling until 39.0 s**, so req0's 19.5 s "decode window" is **93% req1's
prefill**. Dividing the global iteration counters by that interval inflated
`ms/spec-iteration` by ~14x.

Confirmed by a controlled contrast — the same harness on the same engine:

| ctx | C1 decode | C2 decode | inflation |
| --- | --- | --- | --- |
| 4K | 1.0 s | 1.6 s | 1.6x (prefill negligible) |
| 32K | 0.8 s | **19.1 s** | **24x** |
| 250K | ~1-2 s | ~233 s | ~150x |

The inflation scales with prefill length, which is exactly the signature of prefill leaking
into the decode window. At 4K, where prefill is ~2 s, there is no inflation.

Two independent checks rule out a harness-only artifact: the **server's own log** reports
`Avg generation throughput: 0.6-3.2 tokens/s` during those cells, and `preemptions = 0`, so
the engine really was slow **in wall-clock terms** — because it was prefilling, not because
decode had collapsed.

## The correction

`ms per speculative iteration` is **not well-defined for C>1 with this design**. The
`/metrics` counters are global across the batch, so `iterations` accumulates over the whole
batch, while any single window covers a different interval:

- `max_i(end_i - first_i)` counts a slower request's prefill as decode time (the bug above);
- `wall - max_i(first_i)` excludes the iterations that occurred before the last request
  finished prefilling, which would inflate the *rate* instead.

Neither is a decode window, so the harness now **reports `null`** for `ms/spec-iteration` and
`ms/output-token` when concurrency > 1, and reports the quantities that *are* well-defined:

| quantity | definition | valid at C>1? |
| --- | --- | --- |
| `accepted_tokens_per_pass` | `1 + total_accepted / total_iterations` | **yes** — a global ratio |
| `tokens_per_iteration` | `total_output / total_iterations` | **yes** |
| `passes_per_100_output_tokens` | `100 * total_iterations / total_output` | **yes** |
| `output_tok_s` | `total_output / batch_wall` | **yes** — but includes prefill |
| `ms/spec-iteration` | needs a decode-only window | **no — reported null** |

## What survives from the run

`accepted_tokens_per_pass` is a global ratio and remains valid:

| k | C | accepted/pass | passes/100 output tok | output tok/s (incl. prefill) |
| --- | --- | --- | --- | --- |
| 3 | 2 | 2.526 | 59.61 | 2.194 |
| 3 | 4 | 2.456 | 102.45 | 4.270 |
| 7 | 2 | 2.972 | 51.77 | 2.883 |
| 7 | 4 | 3.202 | 78.89 | 5.579 |

Read carefully: the single-digit `output tok/s` is **total output over a wall that is
dominated by 250K prefill**, so it is a valid user-facing aggregate and **not** decode
throughput. It is not comparable to the C1 k-sweep numbers.

The one substantive thing this does show: **at 250K, k=7's acceptance advantage widens under
concurrency** — k=7 leads by 17.7% at C2 (2.972 vs 2.526) and 30.4% at C4 (3.202 vs 2.456),
versus only 5.6% at C1 (2.738 vs 2.594). That *reverses* the C1 conclusion's mechanism: k=3
won at C1 by making each pass cheaper, but under concurrency the pass cost difference was
only −1.7% to −1.9% while acceptance still favoured k=7. So **the k=3 advantage at 250K does
not carry to C2/C4**, and the honest recommendation is that no k change is justified.

## Consequence for the Phase 3 decision gate

The review's gate was "if a k clearly wins at 126K/250K, add C2/C4 before changing the
production default". The C1 win was real (+8.7% at 250K) but the C2/C4 evidence now shows the
acceptance term dominating once requests overlap, so:

- **no production k change is recommended**;
- the C1 +8.7% is a genuine single-stream result and stands as measured;
- a context-dependent k policy would need a decode-window metric that works under
  concurrency, which this engine's global counters cannot provide without per-request
  instrumentation.

## Rule reinforced

This is the same family as the retired `full_wall - independent_prefill_wall` method: **a
metric whose definition does not match the interval it divides**. The guard is to check, for
every ratio, that the numerator and denominator are accumulated over the *same* interval —
and when they cannot be, to report `null` rather than a number.
