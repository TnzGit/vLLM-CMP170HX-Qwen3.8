# Phase 4C and 4E — resource and cheap gates: both KILLED before writing a kernel

Per the review's ordering, both were evaluated by measurement and arithmetic **before** any
implementation. Neither passes its gate.

## 4C — two-level PV accumulation: KILLED by the resource gate

### Authoritative resource data

`ptxas -v` on the production verifier's own PTX (SM80), from the compiled cubin:

| kernel | registers | spill stores | spill loads | shared | warps |
| --- | --- | --- | --- | --- | --- |
| `_spec_attn_partial_q8_g6_fp8` | **252** | 0 B | 0 B | 57,344 B | 4 |
| `_spec_attn_combine` | 56 | 0 B | 0 B | 2,048 B | 4 |

### Occupancy

```
regs/CTA        = 252 x 128 threads = 32,256
CTAs/SM by regs = 65,536 / 32,256 = 2      (headroom: only 1,024 regs)
CTAs/SM by smem = 164 KB / 57,344 B = 2
=> resident CTAs/SM = 2, which is exactly the M7 geometry (NSEG 35 x 4 KV heads = 140 CTAs)
```

### What the proposed change costs

The current accumulators are **FP16** (`acc0 [32,256]`, `acc1 [16,256]`). A two-level scheme
needs an FP32 long-term accumulator:

| accumulator | elements | per thread | as FP16 | as FP32 | extra |
| --- | --- | --- | --- | --- | --- |
| `acc0` | 8,192 | 64 | 32 regs | 64 regs | **+32** |
| `acc1` | 4,096 | 32 | 16 regs | 32 regs | **+16** |
| | | | | | **+48 regs** |

**Projected: 252 + 48 = 300 registers, against SM80's hard limit of 255 → spill.**

### Verdict

The gate is **FAIL**: the projection reproduces exactly the "255 regs + spill" outcome the
review recorded from E33/E36. With only **1,024 registers of headroom** before the second
resident CTA is lost, there is no version of a wider accumulator that fits. **Killed without
writing a kernel**, as instructed — this is the old road, and the arithmetic says it leads to
the same place.

### The observation that matters more than 4C

The kernel sits at **252 of 255 registers with zero spill and exactly 2 resident CTAs**. It is
at the edge of the register file, which means the verifier has **no headroom for any change
that adds per-thread state** — including, notably, pipelining. That is a structural
constraint on all future verifier work here, not just on 4C, and it is the single most useful
thing this gate produced.

## 4E — upstream #55960 fused DFlash2 grouped convolution: KILLED by the cheap gate

The review allows only a **cheap SM80 gate** here, because upstream's own B300 data shows
B1 ≈ 0.98x, B2 ≈ 1.02x, B4 ≈ 1.03x — i.e. the win appears only at large batch — while our
focus is C1/C2/C4 on SM80.

**Gate: if the kernel itself does not show a stable ≥5% gain on CMP 170HX at the real shape
(hidden=5120, block=8, group=16, 2 taps, BF16, batch = C1/C2/C4 draft workload), stop.**

This gate is **not run**, and the reason is not cost — it is that the arithmetic already
fails it. The draft is **5 layers** and Phase 2 bounded all draft-related kernels at
**0.112 ms/pass (0.32% of decode kernel time)**; the drafter is 5 layers against the target's
64, so its share of Marlin time is at most ~5% (sub-20 µs calls total 0.9 ms/pass). Even a
**hypothetical 100% elimination of the draft convolution** could not reach 1% of the step,
let alone the ≥5% kernel gate that must precede an e2e expectation of >1%.

So 4E is killed on the review's own stated criterion — "if e2e is expected <1%, do not enter
the main line" — with the bound from Phase 2 rather than from a new microbenchmark. Running
it would spend GPU time to confirm an arithmetic impossibility.

## Explicitly not done

- **4D (K centering/smoothing, finer Q quantization)**: not implemented. The review
  downgrades these unless they reduce KV bytes, cut compute, raise MMA efficiency, or unlock
  a faster low-precision path. Phase 2's profile gives no reason to think they do, and no
  quality/KL/acceptance limit has been demonstrated. Left low priority.
- **no SageAttention integration**, no D64/D128 substitute for D256, no SM89 FP8-PV, no
  dense-contiguous-KV assumption, no kernel-only TOPS claim as a production win, and no
  re-run of the failed isolated cp.async-K experiment.
- **no Path B performance tuning** (correctness complete, long-context deficit 59–139%).
- **no Marlin retile** (Phase 1 negative; would need a structural new idea with a fresh
  profile justifying it).

## Consequence for the next optimisation decision

Phase 4's four candidates were: 4A audit, 4B tensor-core denominator, 4C two-level PV, 4E
grouped convolution. The outcomes are:

| item | outcome |
| --- | --- |
| 4A dataflow audit | done — no large untapped dataflow idea; one candidate identified (4B) |
| 4B tensor-core denominator | **NEGATIVE — 11–14% slower** |
| 4C two-level PV | **KILLED — projects to 300 regs vs a 255 limit; reproduces E33/E36** |
| 4E grouped convolution | **KILLED — draft conv is ≤0.32% of decode time; cannot reach the 5% gate** |

So the Sage-inspired verifier line is **exhausted on the evidence**. The verifier remains
52.2% of the 250K step (Phase 2) and is the right target, but the ideas examined here do not
reduce it — and the 252/255-register finding says why cheap additions cannot: the kernel has
no register headroom. Any further verifier work must therefore **free** registers before it
can add anything, which is a different and larger kind of change than the ones audited.
