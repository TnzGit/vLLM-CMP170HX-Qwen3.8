# Table C — int64 widening: paired 4K regression test

**Result: the widening costs ~0.3–0.6%, not the ~4% that phase 1 suggested.**

## Method

- **Paired and interleaved (ABBA)**: for each round the arm order alternates
  (`old,fixed` then `fixed,old`), and the only difference between arms is one source
  file swapped on disk before each round, with the engine restarted. The arm label
  therefore cannot drift from the code — each log line records
  `int64-markers=0|1` read back from the file actually in place.
- **5 independent rounds per arm per concurrency level** (20 engine starts total).
- **The historical metric**: `ms/speculative iteration = decode_s * 1000 /
  vllm:spec_decode_num_drafts_total`, with `decode_s` the direct first-content-chunk to
  stream-end interval. No prefill subtraction.
- Same checkpoint (`Qwen3.8-27B-WA16-AutoRound-fast`), same prompt token IDs per round,
  same graph mode (eager, matching the phase-1 cells being audited), same corpus.
- **1350 MHz locked + 180 W**; verified at every round from `clocks.sm`.
- `Xid 31 delta = 0` on all 20 arms.

## Result

| concurrency | arm | ms/spec-iteration (median) | mean | sd | accepted/pass | output tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | old | **41.524** | 41.504 | 0.330 | 2.865 | 69.0 |
| C1 | fixed | **41.789** | 41.842 | 0.580 | 2.954 | 70.1 |
| | | **+0.64%** | | | | |
| C2 | old | **11.177** | 11.118 | 0.193 | 3.176 | 143.6 |
| C2 | fixed | **11.207** | 11.159 | 0.287 | 2.802 | 127.6 |
| | | **+0.27%** | | | | |

## Verdict

Both deltas are **far below the 2% "no meaningful cost" threshold** in the review's
scale, and below the run-to-run spread at C1 (sd 0.33–0.58 ms on a 41.5 ms figure, i.e.
0.8–1.4%). The widening is therefore **accepted as having no material performance cost**.

**The ~4% suggested by phase 1 was a measurement artifact.** Those numbers came from
`full_request_wall - independent_prefill_wall`, which for a 4K request means differencing
~2.5 s walls to isolate ~0.7 s of decode; the residual difference between the two phase-1
arms was within that method's error. This is the concrete cost of the retired method, and
it is why the review's point D was worth acting on.

## Two observations worth carrying forward

1. **Acceptance differs between arms at C2** (3.176 old vs 2.802 fixed, ~12%), while
   `ms/spec-iteration` is essentially identical (+0.27%). That is the metric separation
   working as intended: the kernel cost per speculative iteration is unchanged, and the
   output-tok/s difference at C2 (143.6 vs 127.6) is driven by acceptance, not by the
   kernel. Any A/B that reported only output tok/s would have wrongly blamed the
   widening.
2. **C1 acceptance is ~2.87–2.95**, in the same regime as the historical 3.3–3.6 and far
   from the k=7 ceiling of 8.0 that the repeated-filler workload produced. The
   reconstructed corpus therefore exercises speculation realistically at 4K.

Note the C1 figure (41.5 ms/iter) is much larger than C2's (11.2), which is expected:
at C1 each speculative iteration serves one request, while at C2 two requests are
verified together, amortising the per-iteration cost. It is not comparable to M7's
22.315 ms/step, which is a **FULL-graph** 4K pass on a different checkpoint and a
different path — that comparison is Table B's job.
