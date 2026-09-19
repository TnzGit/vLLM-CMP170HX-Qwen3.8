# Phase 2 — M7 replay and the same-checkpoint Path A / Path B comparison

Three tables, three different questions. They must not be mixed.

**Headline:** M7 mixed-FP8 **remains the fastest long-context production path**, by a
margin that grows with context (+7.9% at 4K to +139% at 250K in ms/speculative-iteration).
The repaired `int8_per_token_head` path is competitive only at short context.

---

## Table A — M7 reconstructed-protocol replay

*Question: does the frozen historical baseline still reproduce?*

| ctx | frozen M7 | reconstructed (median) | delta |
| --- | --- | --- | --- |
| 4K | 22.315 | **22.254** | **−0.27%** |
| 65K | — | 29.153 | — |
| 126K | 35.230 | **35.717** | **+1.38%** |
| 250K | 46.941 | **48.329** | **+2.96%** |

Conditions: M7 service as deployed (`SEGMENTS=35`, `VLLM_FP8_SPEC_FULL_CG=1`, i.e. FULL
graph), DFlash2 k=7, `Uncensored` checkpoint, **180 W without a clock lock** (historical
condition; actual clocks 1140/1455/1470 MHz recorded per round), historical harness
`bench/context_ab.py`, fresh engine per context, 3 rounds each. Every context
`Xid 31 delta = 0`.

**Verdict: the frozen baseline reproduces.** 4K is inside the run-to-run spread. The
residual grows monotonically with context (−0.27% → +1.38% → +2.96%), which is structured
rather than noise, and its cause is **not isolated** — candidates are corpus content (a
reconstructed corpus, not token-identical) and clock variation between rounds. Per the
review's guidance, 1–3% at long context is not grounds to call the historical numbers
wrong.

This is a **reconstructed-protocol replay**, not a token-for-token replay: the frozen
corpus is gone from the host.

---

## Table B — same-checkpoint fair A/B @ 1350 MHz / 180 W

*Question: today, on identical inputs, which path is faster?*

Both paths on `Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4`, identical prompt token IDs
(same corpus, same salt), DFlash2 k=7, output 256, FULL graph, fresh engine per cell,
3 rounds per cell, empty `VLLM_API_KEY` on both (see "contract equality" below).

| ctx | Path | ms/spec-iteration | sd | accepted/pass | output tok/s |
| --- | --- | --- | --- | --- | --- |
| 4K | **A** mixed-FP8 | **22.306** | 0.140 | 3.08 | 137.3 |
| 4K | B int8 | 24.065 | 0.241 | 3.08 | 126.1 |
| 65K | **A** | **29.258** | 0.454 | 2.91 | 97.4 |
| 65K | B | 46.438 | 0.438 | 2.86 | 60.9 |
| 126K | **A** | **35.685** | 0.278 | 3.38 | 93.8 |
| 126K | B | 68.827 | 0.515 | 3.08 | 45.2 |
| 250K | **A** | **48.292** | 0.325 | 3.30 | 69.4 |
| 250K | B | 115.408 | 0.829 | 2.60 | 21.9 |

| ctx | A ms/iter | B ms/iter | **delta** | A tok/s | B tok/s | A tok/step | B tok/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4K | 22.306 | 24.065 | **+7.9%** | 137.3 | 126.1 | 3.08 | 3.08 |
| 65K | 29.258 | 46.438 | **+58.7%** | 97.4 | 60.9 | 2.91 | 2.86 |
| 126K | 35.685 | 68.827 | **+92.9%** | 93.8 | 45.2 | 3.38 | 3.08 |
| 250K | 48.292 | 115.408 | **+139.0%** | 69.4 | 21.9 | 3.30 | 2.60 |

Every cell `Xid 31 delta = 0`. Clocks locked at 1350 MHz throughout (recorded per round).

**Path A wins at every context, and the gap scales monotonically with context length**
(+7.9% → +58.7% → +92.9% → +139.0%).

### Why: per-iteration cost, not speculation efficiency

`accepted/pass` is closely matched at 4K, 65K and 126K (3.08/3.08, 2.91/2.86, 3.38/3.08),
and only at 250K does Path B fall behind (2.60 vs 3.30). So the gap is dominated by
**target verifier iteration cost**, which is what M7's own freeze document predicted: M7's
work was almost entirely verifier-side, and its 126K profile assigned 42.2% of the step to
FP8 verifier partials.

The monotone context scaling is diagnostic: a 4K-sized constant overhead would show as a
fixed ms gap, not a growing one. A cost that grows with KV length is verifier scan cost —
i.e. the split-KV FP8 verifier with NSEG=35 (140 CTAs on 140 SMs) against the int8
Triton verifier on this path.

### Contract equality (important for validity)

The arms differ **only** in what the paths necessarily differ in: target KV dtype and
layout, attention backend, and verifier kernel (NSEG 35 FP8 vs the int8 split-KV kernel).
Everything else was equalised deliberately:

- **same checkpoint**, same drafter, same k, same corpus, same prompt token IDs;
- **same graph mode (FULL)** — this is the correction to phase 1, which measured eager;
- **same clocks and power** (1350 MHz / 180 W);
- **same server contract**: the historical harness sends no `Authorization` header, so
  Path B is started with an empty `VLLM_API_KEY` to match Path A exactly. This was a real
  bug on the first attempt — Path B's non-empty key 401'd every request and produced no
  data at all. Matching the contract was chosen over patching the historical harness, so
  the harness stays byte-identical and the arms lose a difference instead of gaining one.

### Cross-check against Table A

B2's Path A figures (22.306 / 29.258 / 35.685 / 48.292) coincide with the B1 replay
(22.254 / 29.153 / 35.717 / 48.329) to within 1%. So locking 1350 MHz barely moves M7, and
Tables A and B rest on a consistent baseline. That agreement is itself evidence the B2
harness swap did not perturb the reference path.

---

## Table C — int64 widening paired regression

*Question: did the correctness fix cost performance?*

| concurrency | old ms/iter | fixed ms/iter | delta |
| --- | --- | --- | --- |
| C1 | 41.524 | 41.789 | **+0.64%** |
| C2 | 11.177 | 11.207 | **+0.27%** |

Paired ABBA, 5 rounds per arm per concurrency, 1350 MHz / 180 W, historical metric, engine
restarted per round with the arm read back from the file on disk. `Xid 31 delta = 0` on all
20 arms. **Both deltas are below the 2% threshold and below C1's own run-to-run spread**, so
the widening is accepted as having no material cost. The ~4% phase 1 implied was an artifact
of differencing two ~2.5 s walls to isolate ~0.7 s of decode.

---

## The seven questions

**1. Does the reconstructed M7 come close to the frozen historical result?**
Yes. 4K −0.27%, 126K +1.38%, 250K +2.96%. The residual is monotone in context and not yet
explained (corpus content or clock variance); it is not large enough to impugn the
historical numbers.

**2. On the same checkpoint, workload, 1350 MHz/180 W — which is faster at 4K/65K/126K/250K?**
**M7 mixed-FP8 (Path A) at all four**, by +7.9% / +58.7% / +92.9% / +139.0% in
ms/spec-iteration.

**3. Where does the difference come from — verifier, acceptance, draft, or graph overhead?**
**Target verifier iteration cost.** Acceptance is matched (3.08/3.08, 2.91/2.86, 3.38/3.08)
and only diverges at 250K (3.30/2.60), while the cost gap scales monotonically with KV
length. Draft and graph configuration are identical across arms. This phase did **not**
capture a new component-level profile, so the split *within* the iteration (verifier vs
Marlin vs GDN) is **not re-measured** — that is question 6's caveat.

**4. Should M7 still be the production-performance path?**
**Yes, for long context.** It is the faster path at every measured context and the gap
widens exactly where production cares most (126K/250K). It also reproduces its own
historical baseline within 3%.

**5. What role should the repaired int8 path play?**
It is now **correct** (zero Xid delta across every requalification and every B2 cell) and
**competitive at short context only** (+7.9% at 4K). Its value is as the *cheaper-capacity*
option if a smaller KV footprint matters, and as a validated alternative — not as the
long-context performance leader. It should not replace Path A on throughput grounds.

**6. Is the 126K Amdahl breakdown still near verifier ~42.2% / Marlin ~43.4% / GDN 3.7% /
other 10.3%?**
**Not re-measured this phase.** No new kernel-level profile was captured, so this must be
answered as unknown rather than restated. The B2 evidence is *consistent* with a
verifier-dominated deficit (the gap grows with KV length), but consistency is not a profile.
Re-profiling the 126K step on both paths is the natural next measurement.

**7. Next optimisation priority?**
Answerable only from the above, so: **the verifier**, then DFlash depth, then Marlin.
Reasoning, in order of expected value:

- **verifier** — it is the measured locus of Path B's deficit and, at 42.2% of M7's own
  126K step, still the single largest addressable block. Path B's gap is a verifier
  problem, so closing it (or porting M7's NSEG-35 geometry to the int8 kernel) is the
  highest-value work;
- **DFlash k scan** — acceptance is 3.0–3.4 and flat, so a k=3/5/7 sweep is a real but
  bounded lever, and it is cheap;
- **Marlin / SwiGLU** — 43.4% of the step, but the phase-1 W4A8 result at 4K was negative
  and no new evidence changes that; keep it behind the verifier until a profile shows the
  Marlin block has moved.

---

## What this phase did not do

- did not re-run the phase-1 release matrix (kept as capacity/correctness qualification);
- did not call the reconstructed corpus an exact replay;
- did not declare a path winner from different checkpoints (Table B is same-checkpoint);
- did not use repeated filler for the performance verdict;
- did not use large-number subtraction to measure long-context decode — the historical
  direct-decode-interval metric was used throughout;
- did not rewrite PR #1 history, and did not restructure PRs #2/#3/#4.

## Artifacts

| file | contents |
| --- | --- |
| `docs/m7-historical-contract.md` | the recovered M7 contract, corpus analysis, replay plan |
| `docs/m7-replay-table-a.md` | Table A with conditions |
| `docs/b1-replay-raw.txt` | raw B1 replay log |
| `docs/phase2-path-ab-comparison.md` | this document |
| `docs/b2-ab-raw.txt` | raw B2 log |
| `bench/int8-g64/b2-summary.json` | Table B machine-readable |
| `bench/int8-g64/int64-abba/` | 20 raw JSON arms for Table C |
| `docs/int64-paired-regression.md` | Table C with method |
