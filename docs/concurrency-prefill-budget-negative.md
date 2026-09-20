# Prefill-budget diagnostic — NEGATIVE: the budget does not control the interference

The review's confirmation test was: shrink the prefill chunk, and if concurrent decode ITL /
generation throughput recovers while TTFT worsens, the interference is a scheduler property
worth fixing. **It does not recover.** The interference is structural, not a tunable
scheduling parameter.

## Setup

126K C2 on the M7 production path (k=7, 1350 MHz / 180 W), `output = 512`, exact-token unique
prompts, fresh engine per budget. Block size was **832 in all three runs**, so unlike the k
sweep there is no geometry confound — this is a clean single-variable change.

## Result

| `max_num_batched_tokens` | scheduled | block | TTFT₀ | TTFT₁ | first-token skew | steady window | **ITL p50** | ITL p95 | aggregate tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 996 | 832 | 223.0 s | 106.7 s | 116.4 s | 2.14 s | **52.97 ms** | 54.36 ms | 111.4 |
| 2048 (default) | 2020 | 832 | 104.3 s | 212.8 s | 108.4 s | 5.19 s | **53.38 ms** | 53.99 ms | 118.8 |
| 4096 | 4068 | 832 | 103.6 s | 209.0 s | 105.3 s | 6.51 s | **53.27 ms** | 54.15 ms | 121.2 |

`Xid 31 delta = 0` on all three.

**Decode latency under concurrency is 52.97–53.38 ms regardless of a 4x change in the prefill
chunk budget — a spread of 0.4%.** The first-token skew moves from 116 s to 105 s (10%) over
the same range, i.e. it barely responds. Aggregate tok/s is flat within ~8%.

The only thing the budget clearly changes is **TTFT**, and it changes *which request wins the
prefill race*: at 1024 the order flips (req1 first-tokens at 106.7 s while req0 waits until
223.0 s), whereas at 2048/4096 req0 goes first. The skew magnitude is unaffected; only the
identity of the loser changes.

## Interpretation

The interference is **structural**. Two long prefills cannot proceed concurrently because the
total prefill work is fixed and the request that finishes first must then share the GPU with
the other's remaining prefill. No chunk size removes this: making chunks smaller only changes
how often the scheduler switches, not how much total prefill work remains between one
request's first token and the other's. That is why ITL is invariant.

**So there is no scheduling knob to turn here.** The options are workload-level (avoid mixing a
long prefill with an in-flight decode), capacity-level (enough KV to admit both prefills), or
accept it.

## A measurement defect found in this harness, for the third time in this family

My `steady_tok_s` divided the in-window generation-token delta by the **full steady window**.
The counter delta spans first-sample to last-sample, which is shorter than the window, so the
rate was under-reported — at budget 1024 by 35% (59.3 instead of 111.4), which is exactly the
kind of spurious "throughput collapsed" signal that started this whole investigation.

Corrected (dividing by the sample span):

| budget | tok/s, full-window denominator (wrong) | tok/s, sample span (correct) |
| --- | --- | --- |
| 1024 | 59.3 | **111.4** |
| 2048 | 91.5 | **118.8** |
| 4096 | 93.1 | **121.2** |

The harness now records `steady_sample_span_s` and uses it, keeping the wrong value as
`steady_tok_s_window_denom` so the mistake cannot return unnoticed.

This is the **third instance of one defect class in this session**: a ratio whose numerator and
denominator cover different intervals. The first was the original `max_i(end_i − first_i)`
decode window (prefill counted as decode); the second was request-total tokens over the steady
window; this is the counter delta over a longer window. The guard is mechanical: for every
ratio, confirm the numerator and denominator span the same interval, and if they cannot, report
`null` or a bounded range rather than a number.
