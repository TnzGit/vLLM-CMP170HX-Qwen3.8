# Phase 4A — SageAttention vs the M7 verifier: read-only dataflow audit

**No code was written for this phase.** It is a source audit of the two kernels against the
real contract, to decide which (if any) Sage-inspired idea is worth an experiment. The
default answer is "not worth it" unless a row below shows the current kernel *lacks* the
technique **and** the technique addresses a measured bottleneck.

## The real contract (from the source, not from Sage's assumptions)

`_spec_attn_partial_q8_g6_fp8`, the production path
(`mixed-fp8-test-site/vllm/v1/attention/ops/spec_decode_attn.py:205`):

| dimension | value |
| --- | --- |
| arch | SM80 |
| head dim | **D = 256** |
| query heads / KV heads | Hq = 24, Hkv = 4, **GQA = 6** |
| query tokens | **q ≤ 8** (qmax = 10 derived; kernel splits rows 32 + 16) |
| KV | **paged, static FP8 (E4M3FN)**, decoded through an exact BF16 LUT |
| page geometry | **BLOCK_SIZE = 896** |
| split-KV | **NSEG = 35** (140 CTAs on 140 SMs) |
| tile | `TILE = 32`, `num_warps = 4`, `num_stages = 1` |
| graph | FULL CUDA graph, so all workspace addresses are fixed |

Grid: `(num_reqs, Hkv, NSEG)`. Each CTA owns one `(request, kv_head, segment)` and computes
all 48 useful query rows as **32 + 16**, loading each 32-token K/V tile once and reusing it
for both row groups.

Measured bottleneck (Phase 2): **verifier partial = 12.653 ms/pass at 126K (35.9%) and
24.888 ms/pass at 250K (52.2%)**, growing with context. So the thing to attack is KV-scan
cost per pass.

## Audit table

| idea | current M7 already has? | SM80 / D256 applicable? | bottleneck affected | risk | estimated whole-verifier upside |
| --- | --- | --- | --- | --- | --- |
| K/V load pipeline | yes — `tl.load` of a 32-token tile, K and V both, once per tile, reused by both row groups | yes | KV traffic (the dominant term) | — | none; already single-load-per-tile |
| `cp.async` overlap / software pipelining | **no** — `num_stages = 1`, no explicit async copy | yes in principle | load latency hiding | **high**: the isolated cp.async-K experiment already failed once (E-series negative) | low; do not repeat |
| shared-memory permutation / swizzle | not applicable — Triton manages it; no manual smem staging in this kernel | n/a | — | — | none |
| `ldmatrix` feed pattern | implicit via Triton `tl.dot` | n/a (Triton decides) | MMA feed | — | none without leaving Triton |
| per-warp / per-fragment scale ownership | no — **one scalar** `static_k_scale` and `static_v_scale` for the whole tile | yes | scale application is a multiply, not a reduction | low | **negligible**: scales are scalars, not a per-fragment cost |
| mask only on the final partial tile | **partially** — `k_ok = pos < kv_len` is computed every tile and applied to the load and the score mask | yes | mask arithmetic | low | small; masks are cheap vs the load |
| `sm_scale` folding | **partially** — `scale` is folded into `qs0/qs1` at load time, but `static_k_scale` is applied *after* the dot | yes | one multiply on the score matrix | low | **small but real**: fold `static_k_scale` into the q scale so the post-dot multiply disappears |
| `exp2` softmax | **no** — uses `tl.exp` | yes | softmax cost | low | small; softmax is not the measured bottleneck |
| GQA KV-head mapping | yes — `kvh * 6 + rg` packs the 6 query heads per KV head directly | n/a | — | — | none; already optimal for G=6 |
| reduction topology (denominator) | **yes, and this is the interesting one** — `l0`/`l1` accumulate with a full `tl.sum(p0, 1)` over the 32-wide tile, and the running rescale `acc * a` happens every tile | — | **the softmax reduction is on the critical path of every tile** | medium | **the best candidate: see 4B** |
| register lifetime | FP16 `acc` (32×256 and 16×256) with a `.to(float32)` round-trip every tile | — | register pressure / conversions | high | see 4C |

## What the audit rules out

- **cp.async pipelining**: the current kernel has `num_stages = 1` and no explicit async
  copy, so in principle there is overlap to win. But this exact direction **already failed**
  as an isolated experiment in this project's history, and the review explicitly forbids
  redoing it. Ruled out.
- **per-fragment Q/K scales**: the current kernel uses *scalar* scales, so "per-warp scale
  ownership" from Sage has nothing to improve — there is no per-fragment scale to own.
  This is the accuracy-oriented family the review deprioritised (4D), and the audit confirms
  it cannot connect to verifier latency here.
- **shared-memory swizzle / ldmatrix feed**: not exposed at the Triton level. Changing them
  means leaving Triton, which is a rewrite, not a tuning experiment.
- **`sm_scale` folding** is the one genuinely cheap item: `static_k_scale` is applied after
  the dot, and folding it into `qs0/qs1` removes a 32×32 multiply per tile. It is small
  enough that it should be *bundled into* 4B rather than run as its own experiment.

## Conclusion of 4A, and the ordering it implies

The audit does **not** find a large, un-tapped dataflow idea. It finds one structural item
worth an experiment and one worth a resource check:

1. **4B — tensor-core softmax denominator / rowsum.** The kernel currently reduces the
   denominator with `tl.sum(p, 1)` per tile on the FP32 path, and the running accumulator is
   rescaled every tile. A tensor-core rowsum is the Sage idea that maps onto this and does
   **not** change the KV representation. Proceed to 4B.
2. **4C — two-level PV accumulation**, but only if the register/occupancy gate passes. The
   audit shows the current kernel already converts FP16→FP32→FP16 on `acc` every tile, so a
   two-level accumulator is a plausible fit; but the review records that E33/E36 already hit
   255 regs + spill on a D256 register accumulator. Check ptxas first.
3. **Everything else is ruled out or downgraded**, with the reasons above.

## Explicitly not done

- no SageAttention integration, and none proposed;
- no D64/D128 stand-in for D256;
- no SM89 FP8-PV;
- no dense-contiguous-KV assumption;
- no kernel-only TOPS claim treated as a production win;
- no re-run of the failed isolated cp.async-K experiment.
