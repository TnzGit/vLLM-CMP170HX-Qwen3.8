# Long-context service policy: `MAX_SEQS=1` dominates `MAX_SEQS=2`

A service-policy A/B, not a kernel benchmark. It answers whether admitting a second
ultra-long request while one is decoding is worth it.

## Setup

| dimension | value |
| --- | --- |
| path | M7 mixed-FP8, k=7, FULL graph, NSEG 35 |
| context | **126K**, output 512, two simultaneous exact-token unique prompts |
| clocks / power | 1350 MHz locked, 180 W |
| engines | **fresh per arm** |
| order | **ABBA** (ms=1, ms=2, ms=2, ms=1) so drift cannot masquerade as a policy difference |
| Xid | **delta 0 on all four arms** |
| block geometry | 832 in every arm (no geometry confound) |

## Result

| arm | makespan | TTFT₀ | TTFT₁ | completion₀ | completion₁ | useful agg tok/s | per-request tok/s | ITL p50 | KV max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ms=1 r1 | 215.0 s | 100.9 s | 209.4 s | **107.0 s** | 215.0 s | 4.76 | **84.5 / 91.9** | — | 0.231 |
| ms=1 r2 | 216.7 s | 102.1 s | 211.1 s | **108.2 s** | 216.7 s | 4.73 | **83.9 / 91.3** | — | 0.225 |
| ms=2 r1 | 219.2 s | 102.9 s | 211.7 s | 217.7 s | 219.2 s | 4.67 | **4.5** / 68.0 | 53.3 ms | 0.459 |
| ms=2 r2 | 220.4 s | 212.6 s | 103.5 s | 220.4 s | 219.1 s | 4.65 | 65.5 / **4.4** | 53.5 ms | 0.449 |

Medians:

| arm | makespan | useful agg | first-completion | slowest request |
| --- | --- | --- | --- | --- |
| **ms=1** | **215.8 s** | **4.75 tok/s** | **107.6 s** | **84.2 tok/s** |
| ms=2 | 219.8 s | 4.66 tok/s | 218.4 s | **4.4 tok/s** |

`MAX_SEQS=2` versus 1: **makespan +1.8%**, **useful aggregate −1.8%**, **first-completion
2.0x later**, and the slowest request **19x slower** (4.4 vs 84.2 tok/s).

## Decision

The gate was: prefer `MAX_SEQS=1` if `MAX_SEQS=2` improves makespan/useful throughput by less
than ~5% but causes an order-of-magnitude worse first-request decode stall. Here the throughput
"benefit" is **negative** and the stall is **19x**.

**Long-context service profile uses `MAX_SEQS=1`.**

### The Pareto reading

```
MAX_SEQS=1:  total work completion   215.8 s   (best)
             first-user latency      107.6 s   (best, by 2x)
             decode smoothness        84.2 tok/s per request, no stall

MAX_SEQS=2:  total work completion   219.8 s   (1.8% worse)
             first-user latency      218.4 s   (2.0x worse)
             second-user latency     ~219 s    (no better than ms=1)
             aggregate throughput    4.66 tok/s (1.8% worse)
```

`MAX_SEQS=1` is not a tradeoff — it is **better on every measured axis**, including the one
that `MAX_SEQS=2` was supposed to help (total completion). The reason is that the GPU is
already saturated by a single long-context request's prefill+decode, so concurrency adds
interference without adding throughput; it only redistributes latency unfairly (one request
gets 4.4 tok/s while the other gets 65.5).

### Why the second request's decode collapses

This is the Case B mechanism measured earlier: the request whose prefill finishes first then
decodes while the other is still prefilling, for ~110 s at 126K. Its decode rate during that
window falls to **4.5 tok/s** against 84.5 when it has the GPU to itself — a **19x** stall,
matching the first-token skew measured independently in the concurrency telemetry.

## Step C — no concurrency-k gate is needed

Because long context now uses `MAX_SEQS=1`, a "126K C2/C4 k comparison" is **not a
production-relevant gate**: with one resident sequence there is no concurrent-decode regime to
tune k for. And 250K C4 was already shown to be capacity/admission-limited while 250K C2 cannot
form a usable steady decode window on unique prompts.

**So the concurrency-k gate is dropped**, and k is decided from C1 data alone — which is also
the only regime the long-context profile actually runs in.

## Artifacts

`bench/int8-g64/maxseqs-ab/` holds the four raw cell JSONs (with per-1s timelines).
