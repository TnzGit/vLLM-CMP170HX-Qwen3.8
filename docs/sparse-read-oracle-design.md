# Sparse-read oracle design for M7

Status: **unqualified research prototype**.  This document defines the experiment
before GPU execution so the local runner does not choose or reinterpret the
method after seeing results.

## 1. Hypothesis

The qualified M7 long-context path keeps the full target KV cache on the CMP170HX.
At 250K, target verification dominates the step because every one of the 16
full-attention layers scans the complete target history.

The experiment asks whether changing **only the verifier read view** from the
complete canonical page list to a bounded page list makes verifier cost depend on
the bounded list rather than on logical history.

This is a test of the architectural ceiling before implementing retrieval.

## 2. State that remains canonical

The following always remain full-history and authoritative:

- scheduler token count;
- request `num_computed_tokens`;
- target KV allocation;
- target KV writes;
- prefix hashes / ownership;
- DFlash2 draft state;
- GDN recurrent state;
- absolute RoPE positions.

The prototype never frees or copies a KV page.

## 3. Two coordinate systems

For a 250K request with a 32K sparse verifier budget:

```text
logical coordinate:
  0 ------------------------------------------------------------- 250K
  positions / GDN / KV writes remain here

verifier read coordinate:
  [sink pages] [recent pages........................] [live q tail]
  0 --------------------------------------------------------- ~32K
```

K and Q tensors are already RoPE-transformed at their original logical
positions.  The second coordinate is used only to enumerate selected pages and
apply causality.  It must never be written back into request positions.

## 4. Page policy used by the oracle

This phase intentionally uses no learned/content retrieval.

For a budget of `B` verifier pages and `S` sink pages:

- if logical pages <= B: identity table;
- otherwise:
  - keep pages `[0, S)`;
  - keep the newest `B-S` pages;
  - preserve chronological order;
  - always include the physical page(s) holding the live speculative suffix.

The token knobs are rounded up to the runtime verifier block size.  The runtime
block size is measured, never assumed from comments.

This policy is useful because it provides all three controls needed for the
first gate:

1. identity below budget;
2. non-contiguous physical page IDs above budget (sink + tail);
3. bounded attention work with zero KV transfer.

## 5. CUDA Graph contract

The sparse block table and sparse sequence-length buffer are allocated in
`SpecDecodeAttention.__init__`, beside the existing M7 partial buffers.

Every verifier call launches a small Triton table-builder before the attention
partial kernel.  The builder writes into persistent buffers.  No tensor is
allocated or resized after verifier workspace construction.

Therefore FULL graph capture sees stable addresses.

The builder currently runs once per full-attention layer.  That is intentionally
left simple for the oracle.  If the experiment wins and its table-build overhead
is measurable, moving construction to a once-per-forward metadata stage is a
separate optimization.

## 6. Why this does not prove KVMem quality

Above budget, this prototype is sparse attention by construction.

It proves only:

- the vLLM/M7 execution path can consume a compact read table;
- long-context verifier work can or cannot be bounded;
- the resulting end-to-end throughput ceiling.

It does **not** prove that sink+recent preserves task quality.  A middle-history
needle is expected to disappear unless it happens to be selected.

Content-aware retrieval is a second project and should begin only after this
gate wins.

## 7. Measurement matrix

Production long profile only:

| item | value |
| --- | --- |
| target | Qwen3.8-27B W4A16 |
| target KV | static FP8, M7 |
| draft | DFlash2 W4A16 |
| k | 3 |
| MAX_SEQS | 1 |
| NSEG | 35 |
| graph | FULL |
| clock / power | 1350 MHz / 180 W |
| port | 8002 |

Arms:

- dense M7;
- sparse 32K;
- sparse 48K;
- sparse 65K.

Contexts:

- 4K;
- 32K;
- 65K;
- 126K;
- 250K.

For formal e2e cells:

- fresh engine per arm/context;
- identical prompt bytes across arms;
- exact server-reported prompt-token contract;
- 3 rounds if the first pass is stable;
- report Xid delta;
- record parsed runtime config and EngineCore environment;
- abort a poisoned engine after any CUDA fault.

## 8. Primary metrics

Keep categories separate.

### Kernel gate

`bench/sparse-read/spec_attn_sparse_ctx_scan.py`

Report:

- dense us/layer;
- sparse us/layer;
- active sparse tokens;
- table-policy equality;
- below-budget output identity.

This decides whether e2e work is justified.  It is not an e2e throughput claim.

### E2E gate

`bench/sparse-read/e2e_decode_gate.py`

Report:

- streamed decode tok/s on first-content -> completion;
- counter-derived ms/spec-iteration on one counter-snapshot interval;
- counter-derived tok/step on the same snapshot interval;
- TTFT separately;
- prompt/output hashes;
- preemption delta.

Do not divide a request-total counter by the streamed decode window.

## 9. Expected order of magnitude, not a pass criterion

Measured dense k=3 pass costs before this branch:

| context | dense ms/spec-iteration |
| --- | ---: |
| 32K | 23.364 |
| 48K | 24.756 |
| 65K | 26.122 |
| 126K | 30.294 |
| 250K | 38.380 |

If a 250K logical request with a 32K read view behaves structurally like the
32K regime, an end-to-end result around 100 tok/s is plausible.  This is a
modelled expectation, **not measured evidence** and not a required number.

The useful decision boundary is qualitative:

- a large, repeatable structural win -> continue to retrieval;
- no meaningful verifier scaling change -> stop immediately.

## 10. Work explicitly deferred after a GO

Only after the oracle wins:

1. capture query Q at the turn boundary;
2. build a GPU-resident semantic index (likely sub-block summaries aggregated
   to physical verifier pages);
3. select sink + mandatory + recent + content top-K;
4. evaluate quality/needle/multi-constraint workloads;
5. only then decide whether draft KV should use the same read view.

CPU/NVMe KV tiers remain a separate capacity feature.  They are not needed to
prove the present speed hypothesis on a 64 GiB card.
