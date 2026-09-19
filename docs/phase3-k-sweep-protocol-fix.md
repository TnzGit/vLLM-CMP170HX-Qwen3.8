# Phase 3 — a corrupted run, caught, and the protocol corrected

The first k-sweep produced a k=3 4K cell of **2–6 tok/s** against an expected ~115–137. It
was not a k=3 property. This records how it was caught and what changed.

## What the corrupted run showed

| round | ms/output-token | tok/s |
| --- | --- | --- |
| 0 | 404.30 | 2.47 |
| 1 | 152.85 | 6.54 |
| 2 (profiled) | 453.51 | 2.21 |

and then `stop_profile failed: Remote end closed connection without response`, followed by
`ConnectionRefusedError` for every later cell.

## Why it was not a real result

Three independent checks:

1. **a clean k=3 re-measure** (no profiler, fresh engine) gives **111.7 tok/s** and
   20.87 ms/spec-iteration — normal;
2. **`dmesg` shows no Xid at all** in the window, so no GPU fault;
3. the engine **died mid-run** — the profiler-stop call lost the connection and the next
   cell could not connect. The timings were recorded around a dying process.

## Root cause: an environment variable left in from the Phase 2 diagnosis

While chasing Phase 2's draft/target separation I added
`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` (it is what makes
`record_function_or_nullcontext("gpu_model_runner: draft")` a real scope rather than a null
context). I then left it set in the Phase 3 driver, which also ran the profiler across whole
rounds.

Controlled comparisons at 4K, k=7, same engine otherwise:

| condition | ms/spec-iteration | tok/s |
| --- | --- | --- |
| no profiler, no scopes env | 22.48 | 115.3 |
| profiler config, no scopes env | 22.15 | 118.3 |
| profiler config + scopes env | 22.15 | 118.3 |

So neither flag alone reproduces the slowdown, which means the corruption was the
**combination** of scope instrumentation with a profiler left running across a long cell on
a hybrid model — and possibly GPU contention from a leftover engine. The honest statement is
that the mechanism is **not fully isolated**; what is established is that the cell was not a
measurement, because the engine died and the clean re-measure disagrees by 20x.

## What changed in v2

| change | reason |
| --- | --- |
| **no profiler at all** | the six required metrics do not need one; component attribution was Phase 2's job and is complete |
| `VLLM_CUSTOM_SCOPES_FOR_PROFILING` explicitly unset | it was v1's contamination |
| **fresh engine per (k, context)** | isolates a crash to one cell instead of poisoning the sweep |
| **engine-liveness check before each cell** | v1 kept going against a dead engine |
| **spread gate** | `MAX_ROUND_SPREAD = 3.0`: a cell whose rounds disagree by more than 3x exits **rc=4** and aborts the sweep instead of being banked |
| block size recorded per engine | it is not constant (see below) |

The gate is the same discipline that had already been applied to the ABBA and B2 drivers
after those wasted runs; Phase 3 v1 is what happens without it.

## A second correction: the page geometry is not 896

The Phase 4A audit describes the contract as `BLOCK_SIZE = 896`. The **observed** values from
`Setting attention block size to N` are:

| run | block size |
| --- | --- |
| k=7 (Phase 2) | 832 |
| k=3 (v1) | 816 |
| k=3 (clean probe) | 800 |
| k=7 (probe) | 800 |

The value is derived from page-size arithmetic
(`vllm/platforms/interface.py`, "so the quantized primary KV page covers the
higher-precision padded-spec page") and therefore **depends on k and on the mamba page
size**, not on a fixed 896. The kernel's own comment says "TILE divides the 896-token
production page", which describes the *design* geometry rather than what these runs
actually launched with. Every v2 cell records its block size so the contract is taken from
the running engine, not from the comment.

This is the same rule this project keeps relearning: **read the effective config from the
running engine, never from the source comment or the launcher.**
