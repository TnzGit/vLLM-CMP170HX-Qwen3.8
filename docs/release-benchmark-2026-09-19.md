# Release benchmark: long-context speculative decoding on CMP 170HX

Decode throughput (output tokens / second), median of 2 rounds, on the
path repaired by `patches/spec-decode-attn-int64-block-id.patch`.

Configuration: `Qwen3.8-27B-W4A16-AutoRound-fast` + DFlash2 W4A16 k=7,
`TRITON_ATTN` + `int8_per_token_head` KV, `max-model-len=262144`,
`max-num-seqs=4`, eager graph mode, clocks pinned. Prompts are **exact**
token counts and distinct per request and per round.

Decode is separated from prefill by bracketing each batch with a
prefill-only run of the same shape; a raw wall-clock figure would be
prefill-dominated at long context.

> **This document is a capacity / concurrency / correctness qualification, not a
> performance comparison.** See the note at the end for why: the workload is exact-token
> but built from heavily repeated filler, which drives DFlash acceptance to 7.5-8.0
> against a k=7 ceiling, whereas the historical baselines run at ~3.3-3.6 accepted
> tokens per step. Output tok/s from the two workloads are not comparable.

---

## The table

| ctx | conc | old-same-path | **fixed** | historical mixed-FP8 | modeled ceiling |
|---|---|---|---|---|---|
| 4K | C1 | 179.7 | **172.9** | 144.8 / 160.0 (NSEG 16/32) | 121.8 / 131.2 (def/greedy) |
| 4K | C2 | 334.8 | **321.8** | 144.8 / 160.0 (NSEG 16/32) | 195.5 / 214.6 (def/greedy) |
| 4K | C4 | **FAULTED** | **580.5** | 144.8 / 160.0 (NSEG 16/32) | 278.9 / 285.7 (def/greedy) |
| 16K | C1 | faulted (pre-fix IMA) | **167.1** | -- no record | 121.8 / 131.2 (def/greedy) |
| 16K | C2 | faulted (pre-fix IMA) | **260.0** | -- no record | 195.5 / 214.6 (def/greedy) |
| 16K | C4 | faulted (pre-fix IMA) | **320.3** | -- no record | 278.9 / 285.7 (def/greedy) |
| 32K | C1 | faulted (pre-fix IMA) | **149.4** | -- no record | 121.8 / 131.2 (def/greedy) |
| 32K | C2 | faulted (pre-fix IMA) | **218.5** | -- no record | 195.5 / 214.6 (def/greedy) |
| 32K | C4 | faulted (pre-fix IMA) | **231.1** | -- no record | 278.9 / 285.7 (def/greedy) |
| 65K | C1 | faulted (pre-fix IMA) | **151.7** | -- no record | 121.8 / 131.2 (def/greedy) |
| 65K | C2 | faulted (pre-fix IMA) | **131.9** | -- no record | 195.5 / 214.6 (def/greedy) |
| 65K | C4 | faulted (pre-fix IMA) | **183.2** | -- no record | 278.9 / 285.7 (def/greedy) |
| 126K | C1 | faulted (pre-fix IMA) | **90.4** | 54.2 / 68.2 (NSEG 16/32) | -- out of range |
| 126K | C2 | faulted (pre-fix IMA) | **69.5** | 54.2 / 68.2 (NSEG 16/32) | -- out of range |
| 126K | C4 | faulted (pre-fix IMA) | **71.4** | 54.2 / 68.2 (NSEG 16/32) | -- out of range |
| 250K | C1 | faulted (pre-fix IMA) | **94.7** | 32.5 / 42.2 (NSEG 16/32) | -- out of range |
| 250K | C2 | faulted (pre-fix IMA) | not measured -- prefill-time bound | 32.5 / 42.2 (NSEG 16/32) | -- out of range |
| 250K | C4 | faulted (pre-fix IMA) | not measured -- prefill-time bound | 32.5 / 42.2 (NSEG 16/32) | -- out of range |

## Column provenance and validity

| column | source | valid at | why it is not measured here |
|---|---|---|---|
| old-same-path | this repo, pre-fix build (measured for this table) | 4K C1/C2 only | it faults at **4K C4** as well as at every >=16K cell -- 16 requests of 4K (65K cumulative tokens) was enough. So the pre-fix build could not serve even short-context concurrency, and the missing cells are missing **by construction**. That absence is the finding |
| **fixed** | `bench/int8-g64/release_bench.py` | all listed cells | -- |
| historical mixed-FP8 | `docs/cmp170hx-mixed-fp8-engineering.md` | 4K, 126K, 250K | **NOT the M7 baseline.** The two values are NSEG 16/32 of the *same* FP8 route, `not` concurrency, and they are not best-known. The genuine historical M7 (NSEG 35) is `22.315 / 35.230 / 46.941 ms/step` at 4K/126K/250K -- see `docs/m7-historical-contract.md`. No record exists at 16K/32K/65K |
| modeled ceiling | -- | -- | **conceptually wrong as populated.** This column holds *measured* `CTX=fast` (bf16 KV, 64k) throughput from `single-user/README.md`, which is a different configuration, not a modelled engineering-effective ceiling. The engineering reference figures are 4K 170-200 tok/s, 126K ~131.2 tok/s, 250K ~100 tok/s, and they are modelling references, not specifications. The column is renamed `CTX=fast measured` below and must not be read as a ceiling for this path. |

## Gaps, stated rather than hidden

- **250K C2/C4**: not measured, by explicit decision. Each cell needs four prefill passes of ~24-50 minutes at the measured ~320 tok/s prefill rate, and 4x250K = 1.0M tokens approaches the 1.149M-token KV pool. The arithmetic, not the result, is the reason.
- **old-same-path at >=16K**: not measurable, as above.
- **acceptance** came back `None` on several C1 cells: the `SpecDecoding metrics` logger is periodic, so a bracket containing few decode steps can miss its window. Decode throughput is unaffected (it is measured from wall time), but per-cell acceptance should not be quoted where it is blank.

## Correctness during the run

Xid 31 count was 81 before the benchmark and 81 after: **zero new illegal accesses across all 16 measured fixed-path cells** (6 contexts x 3 concurrency levels = 18 possible, minus the 2 skipped 250K concurrency cells), covering 31 round-records, including the 250K cell. The fix holds under benchmark load, not only under the dedicated requalification (handover 57-58).


---

## Status of this document (corrected)

Per review, this benchmark is **a capacity / concurrency / correctness qualification**, and
its throughput numbers must not be used to declare a production performance winner.

What it validly establishes:

- the repaired path serves **250K at C1**, and every measured context/concurrency cell
  runs clean;
- **zero Xid 31 delta** across the whole run (81 before, 81 after);
- 16 fixed-path cells measured (18 possible, 250K C2/C4 deliberately skipped).

What it does **not** establish, and must not be quoted as:

1. **It is not comparable with the historical M7 baseline.** The `historical mixed-FP8`
   column holds NSEG 16/32 values of the FP8 route, not M7. M7 is NSEG 35 with
   `22.315 / 35.230 / 46.941 ms/step` -- see `docs/m7-historical-contract.md`.
2. **The graph contract differs.** These cells ran **eager** (`cudagraph_mode=NONE`),
   whereas M7 is **FULL** CUDA Graph. A graph-contract-matched comparison is required
   before any A/B.
3. **The workload is not a speculative-performance workload.** Prompts are built by
   repeating one filler sentence, which lifts DFlash acceptance to ~7.5-8.0 against a
   k=7 ceiling; the historical baselines run at ~3.3-3.6 accepted tokens/step. Comparing
   output tok/s across those two workloads is meaningless.
4. **250K decode is not a stable number.** Its decode interval is obtained by
   subtracting two ~1424 s walls, leaving ~1-2 s of signal; the two rounds gave 119.4 and
   70.0 tok/s. Retired: see below.

### Retired measurement method

`full_request_wall - independent_prefill_wall` is **abandoned for 126K and 250K**. Two
~1400 s measurements cannot be differenced to estimate a ~1 s decode interval. The
historical protocol already had the right method and this benchmark should have used it:
`bench/context_ab.py` streams the response, takes TTFT from the first content chunk, and
measures decode as `end - first` with `ms/step = decode_s * 1000 /
vllm:spec_decode_num_drafts_total`. That is a direct decode interval and a true
per-speculative-step metric. All A/B work from here uses it.

### Still to be done (phase 2)

`docs/m7-historical-contract.md` records the recovered contract. A replay must name and
hash its corpus (the frozen corpus is **gone** from the host), record acceptance, and
state whether it used 1350 MHz or the historical 180 W operating point.
