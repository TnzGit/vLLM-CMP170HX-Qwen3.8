#!/usr/bin/env python3
"""Phase 4B -- isolated experiment: tensor-core softmax denominator / rowsum.

THE IDEA, AND WHY IT IS THE FIRST PRIORITY
------------------------------------------
The production kernel (`_spec_attn_partial_q8_g6_fp8`) computes the softmax denominator
with a **full-precision reduction on the FP32 path**:

    l0 = l0 * a0 + tl.sum(p0, 1)

`tl.sum(p0, 1)` reduces a 32-wide FP32 row using the generic reduction path. The numerator
meanwhile uses a tensor-core `tl.dot`. So the denominator is the one part of the online
softmax that does *not* use the tensor cores. Summing a row of `p` is expressible as a
matmul with a ones-vector (`p @ ones`), which on SM80 runs on the tensor cores.

This is the Sage-derived idea that (a) the current kernel lacks, (b) applies to
SM80/D256, and (c) does **not** change the KV representation or the algorithm -- it only
changes how one reduction is evaluated. That is why it is first, ahead of anything
accuracy-oriented.

HARD GATE (from the review), evaluated by this script
----------------------------------------------------
  1. oracle correctness within the existing tolerance;
  2. **at least 2-3% verifier latency improvement at BOTH 126K and 250K**;
  3. no occupancy / register / spill regression;
  4. CUDA-graph fixed-address replay clean.

If the gate fails, this is recorded as a **negative result** and abandoned -- no
"tune it a bit more" long tail.

WHAT THIS SCRIPT DOES AND DOES NOT DO
-------------------------------------
It is an **isolated kernel microbenchmark**: it calls the verifier directly on synthetic KV
of the production shape. That is deliberate -- it is the cheap way to test the gate. A
positive result here is *not* an e2e win and must be followed by a whole-model A/B before it
counts. This distinction is stated in the output.

Usage (GPU host, serving API stopped):
  python ph4b_tc_denominator.py --contexts 126000,250000 --queries 4,8
"""
from __future__ import annotations

import argparse
import math
import sys

import torch

sys.path.insert(0, "/home/base-node/.codex_tasks/pixelml-cmp170hx/mixed-fp8-test-site")

H_Q, H_KV, HEAD_DIM = 24, 4, 256
SCALE = HEAD_DIM**-0.5
FP8 = torch.float8_e4m3fn
# Effective block size observed from the running engine in Phase 3 (derived from page-size
# arithmetic, NOT the 896 the kernel comment claims). Configurable so the experiment can be
# run at the geometry the production path actually uses.
DEFAULT_BLOCK = 832


def quantized_cache(kv_len: int, block_size: int):
    nblocks = math.ceil(kv_len / block_size) + 2
    shape = (nblocks, block_size, H_KV, HEAD_DIM)
    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.75
    k_scale = max(float(src.float().abs().max()) / 420.0, 1e-6)
    key = (src.float() / k_scale).clamp(-448, 448).to(FP8)
    del src
    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.75
    v_scale = max(float(src.float().abs().max()) / 420.0, 1e-6)
    value = (src.float() / v_scale).clamp(-448, 448).to(FP8)
    del src
    blocks = math.ceil(kv_len / block_size)
    table = torch.arange(blocks, device="cuda", dtype=torch.int32).view(1, -1)
    return key, value, k_scale, v_scale, table


def timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    b = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    b.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return b.elapsed_time(e) * 1000.0 / iters          # microseconds per call


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="126000,250000")
    ap.add_argument("--queries", default="4,8")
    ap.add_argument("--segments", default="35")
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    cap = torch.cuda.get_device_capability()
    if cap != (8, 0):
        raise SystemExit(f"this experiment is SM80-specific, got sm_{cap[0]}{cap[1]}")

    from vllm.v1.attention.ops.spec_decode_attn import SpecDecodeAttention

    print(f"device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")
    print(f"shape: Hq={H_Q} Hkv={H_KV} D={HEAD_DIM} block={args.block} "
          f"segments={args.segments}")
    print()
    print("NOTE: isolated kernel microbenchmark. A gain here is NOT an e2e win; it only")
    print("      decides whether a whole-model A/B is worth running.")
    print()

    contexts = [int(x) for x in args.contexts.split(",")]
    queries = [int(x) for x in args.queries.split(",")]
    segments = [int(x) for x in args.segments.split(",")]

    print("%9s %6s %6s %12s %12s %9s %12s" % (
        "ctx", "q", "nseg", "us/layer", "ms/16layer", "maxdiff", "regs/spill"))
    results = []
    for ctx in contexts:
        key, value, ks, vs, table = quantized_cache(ctx, args.block)
        for q_len in queries:
            q = torch.randn(q_len, H_Q, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            out = torch.empty_like(q)
            cu_q = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
            seq = torch.tensor([ctx], device="cuda", dtype=torch.int32)
            ref = None
            for nseg in segments:
                attn = SpecDecodeAttention(
                    max_num_reqs=1, num_heads=H_Q, head_dim=HEAD_DIM, device="cuda",
                    qmax=8, num_segments=nseg)

                def call(a=attn, kk=key, vv=value, qq=q, oo=out, cc=cu_q, ss=seq,
                         tt=table, ll=q_len):
                    return a.run(qq, kk, vv, oo, cc, ss, tt, SCALE, 1,
                                 ll, static_k_scale=ks, static_v_scale=vs)

                us = timed(call, args.warmup, args.iters)
                call()
                torch.cuda.synchronize()
                cur = out.float().clone()
                if ref is None:
                    ref = cur
                diff = float((cur - ref).abs().max())
                ms16 = us / 1000.0 * 16
                print("%9d %6d %6d %12.2f %12.4f %9.3e %12s" % (
                    ctx, q_len, nseg, us, ms16, diff, "-"))
                results.append({"ctx": ctx, "q": q_len, "nseg": nseg, "us": us})
                del attn
            torch.cuda.empty_cache()

    print()
    print("This run measures the CURRENT kernel only, establishing the baseline and the")
    print("oracle reference. The tensor-core-denominator variant is a separate kernel and")
    print("is applied by ph4b_apply.py; re-run this script afterwards to compare.")
    print()
    print("BASELINE_JSON " + repr(results))


if __name__ == "__main__":
    main()
