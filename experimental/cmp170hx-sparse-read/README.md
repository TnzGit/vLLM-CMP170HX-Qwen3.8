# CMP170HX verifier-only sparse KV read prototype

This experiment asks one narrow question:

> With the full long-context KV cache still resident on the GPU, can the M7
> speculative verifier read only a bounded subset of historical pages and make
> 126K/250K decode cost approach the 32K–65K regime?

It is **not** CPU/NVMe KV offload, not a scheduler redesign, and not a claim of
exact full-history attention.

## Scope

The prototype modifies only the M7 static-FP8 split-KV verifier.

Unchanged:

- canonical/write-side vLLM block tables;
- KV allocation and residency;
- prefill;
- absolute Q/K RoPE positions;
- 48 GDN recurrent layers/state;
- DFlash2 draft KV and proposal path;
- prefix-cache ownership;
- scheduler/admission policy.

Changed, behind an opt-in environment variable:

- the target verifier receives a second, persistent read-only block table;
- above the configured budget, that table contains a sink prefix plus the
  newest pages, in chronological order;
- the verifier receives the corresponding compact causal length.

No K/V bytes move.  Only page IDs are copied into a graph-stable read view.

## Why compact causal positions are valid for this gate

The cached K tensors and query Q tensors retain their original absolute RoPE
coordinates.  The compact coordinate is used only for:

1. indexing the selected page list; and
2. the verifier's causal mask.

All selected historical pages precede the live query suffix and the live tail is
always the final selected page range.  Therefore the verifier can mask by the
compact order without rewriting RoPE positions.

This is deliberately a **semantic approximation** above budget: unselected
full-attention history is invisible to the 16 target attention layers for that
verify pass.  GDN recurrent state still carries the model's recurrent history.

## Runtime flags

All flags are boot-static for a server process.

```bash
export VLLM_SPARSE_KV_READ=1
export VLLM_SPARSE_KV_READ_TOKENS=32000
export VLLM_SPARSE_KV_READ_SINK_TOKENS=1024
export VLLM_SPARSE_KV_READ_MAX_BLOCKS=8192
```

- `TOKENS` is the total verifier attention budget, including the sink.
- The budget is rounded **up** to the verifier's runtime kernel block size.
- `SINK_TOKENS` is also rounded up to that block size.
- At least one selected block is always reserved for the live/recent tail.
- Below budget the sparse read table is an identity copy, which is the main
  correctness control.
- `MAX_BLOCKS` sizes the persistent read table.  8192 is intentionally cheap
  and covers the planned 32K/48K/65K gates even if the verifier kernel block is
  only 16 tokens.

The implementation prints one line on first verifier use:

```text
[cmp-sparse-read] enabled verifier-only sink+recent view: ...
```

Read the **actual** `kernel_block`, `budget_blocks`, and `sink_blocks` from
that line.  Do not infer them from launcher intent.

## Install

The normal vLLM patch stack and M7 mixed-FP8 series must already be present.

```bash
experimental/cmp170hx-sparse-read/install.sh --dry-run
experimental/cmp170hx-sparse-read/install.sh --apply
experimental/cmp170hx-sparse-read/install.sh --check
```

The production runtime and any shadow/test vLLM site used through
`PYTHONPATH` must be patched separately.  Do not assume one install shadows
the other.

To undo only this experiment:

```bash
experimental/cmp170hx-sparse-read/install.sh --reverse
```

The installer restores the exact pre-experiment file from the backup created by
`--apply`.

## Intended first gate

Use the production **long** profile:

- Qwen3.8-27B target W4A16;
- target static FP8 KV / M7 mixed-FP8 verifier;
- DFlash2 W4A16;
- `k=3`;
- `MAX_SEQS=1`;
- `VLLM_SPEC_DECODE_ATTN_SEGMENTS=35`;
- FULL graph;
- 1350 MHz locked / 180 W;
- research port 8002 only.

A/B:

1. dense M7: `VLLM_SPARSE_KV_READ=0`;
2. sparse 32K;
3. sparse 48K;
4. sparse 65K.

Important contexts: 4K, 32K, 65K, 126K, 250K.  Use a fresh engine for every
formal context/config cell.

The 4K arm must be numerically identical to dense M7 because the selected table
is an identity copy there.  Above budget, output equality is **not expected**;
the experiment has changed attention semantics by design.

## GO / NO-GO

This branch is worth further retrieval work only if all of the following hold:

- no Xid increase;
- no illegal memory access / graph-address instability;
- below-budget identity control passes;
- 126K and 250K verifier latency tracks the selected working-set size rather
  than logical history;
- 250K end-to-end decode shows a structural improvement large enough to justify
  quality work (target: clearly above measurement spread; 90+ tok/s is the
  practical high-value region, not a hard correctness gate).

If the verifier still scales with logical 250K despite a verified compact read
table, stop.  Do **not** add CPU offload, NVMe, retrieval scoring, or scheduler
complexity to rescue it.

## Explicit non-goals

Do not add in this phase:

- Mean-K / SubBlockMeanK retrieval;
- host-pinned KV;
- NVMe;
- dynamic per-request budgets;
- DFlash2 sparse draft KV;
- changes to target page geometry;
- multi-sequence long-context qualification.

Those only become rational after the oracle read-view gate wins.
