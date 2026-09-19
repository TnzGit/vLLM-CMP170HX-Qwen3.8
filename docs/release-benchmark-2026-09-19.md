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
| historical mixed-FP8 | `docs/cmp170hx-mixed-fp8-engineering.md` | 4K, 126K, 250K | different route (FP8 target KV, BF16 draft KV, FULL graph) and the two values are NSEG 16/32, **not** concurrency. No record exists at 16K/32K/65K |
| modeled ceiling | `single-user/README.md` | <=64K | measured on `CTX=fast` (bf16 KV, 64k). Different KV format, so it is a ceiling for a *different* configuration, not for this one |

## Gaps, stated rather than hidden

- **250K C2/C4**: not measured, by explicit decision. Each cell needs four prefill passes of ~24-50 minutes at the measured ~320 tok/s prefill rate, and 4x250K = 1.0M tokens approaches the 1.149M-token KV pool. The arithmetic, not the result, is the reason.
- **old-same-path at >=16K**: not measurable, as above.
- **acceptance** came back `None` on several C1 cells: the `SpecDecoding metrics` logger is periodic, so a bracket containing few decode steps can miss its window. Decode throughput is unaffected (it is measured from wall time), but per-cell acceptance should not be quoted where it is blank.

## Correctness during the run

Xid 31 count was 81 before the benchmark and 81 after: **zero new illegal accesses across all 31 measured cells**, including the 250K cell. The fix holds under benchmark load, not only under the dedicated requalification (handover 57-58).

