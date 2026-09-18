#!/usr/bin/env python3
"""Exact-shape Marlin microbench: W4A16 vs W4A8-INT8 on the hot GEMMs.

The review's ordering puts this before any whole-model run, for a good reason: if
the dominant GEMM itself only gains 3-5%, no amount of engine integration will
produce a whole-step win, and if it gains 15-30% it is worth integrating. The
shapes are the ones the 126K profile actually found dominant
(`docs/cmp170hx-mixed-fp8-engineering.md`):

    gate_up  M=16, N=34816, K=5120   (64 calls, 6.111 ms)
    down     M=16, N=5120,  K=17408  (64 calls, 3.139 ms)

Both are measured with the real Marlin entry point (`ops.marlin_gemm`) on
realistically packed weights, interleaved A/B on one process at a pinned clock, so
run-to-run drift hits both arms equally.

This deliberately does NOT use the engine: the question is single-kernel cost, and
the engine adds scheduler, drafter and graph noise that cannot change the answer.

Usage:
  marlin_shape_bench.py --device 0
  marlin_shape_bench.py --device 0 --group-size 128 --iters 200
"""
from __future__ import annotations

import argparse
import json
import statistics

import torch


def marlin_repack(shape_k: int, shape_n: int, group_size: int, bits: int = 4,
                  device: str = "cuda"):
    """Pack a random W4 weight into Marlin's layout.

    Uses vLLM's own repack helper so the layout matches what the engine feeds the
    kernel; a hand-rolled packing would measure a different kernel path.
    """
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    qtype = scalar_types.uint4b8
    size_k, size_n = shape_k, shape_n
    # random int4 codes in [-8, 7] packed two per byte, as AutoRound/GPTQ produce
    codes = torch.randint(-8, 8, (size_k, size_n), dtype=torch.int32, device=device)
    packed = ((codes[:, 1::2] & 0xF) << 4) | (codes[:, 0::2] & 0xF)
    packed = packed.to(torch.int32).contiguous()

    # per-group scales (and no zero points; uint4b8 keeps the +8 offset in the
    # kernel's dequant, matching the checkpoints this repo ships)
    g = group_size
    scales = (torch.rand((size_k // g, size_n), dtype=torch.float16, device=device)
              * 0.02 + 0.001)
    g_idx = torch.arange(size_k, dtype=torch.int32, device=device) // g
    perm = torch.argsort(g_idx).to(torch.int32)

    out = ops.gptq_marlin_repack(
        packed, perm, size_k, size_n, bits, device=torch.device(device))
    return out, scales, g_idx, perm, qtype


def timeit(fn, iters: int, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def bench_shape(shape_k: int, shape_n: int, group_size: int, iters: int,
                device: str, use_int8_act: bool):
    from vllm import _custom_ops as ops

    w, scales, g_idx, perm, qtype = marlin_repack(shape_k, shape_n, group_size,
                                                  device=device)
    m = 16
    dtype = torch.int8 if use_int8_act else torch.float16
    a = torch.randn((m, shape_k), dtype=torch.float16, device=device) * 0.1
    if use_int8_act:
        # int8 activations carry a per-token scale, as the engine's W4A8 path does
        a_q = (a / a.abs().amax(-1, keepdim=True).clamp_min(1e-4) * 127).round()
        a = a_q.to(torch.int8)
        a_scales = (a.float().abs().amax(-1, keepdim=True) / 127).to(torch.float16)
    else:
        a_scales = None

    workspace = torch.zeros(shape_n // 64 * 16, dtype=torch.int32, device=device)
    out = torch.empty((m, shape_n), dtype=torch.float16, device=device)

    def once():
        ops.marlin_gemm(a, out, w, scales, None, g_idx, perm, workspace,
                        qtype, m, shape_n, shape_k, True, 1, 0,
                        a_scales)

    ms = timeit(once, iters)
    return ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    torch.cuda.set_device(args.device)

    shapes = [
        ("gate_up", 5120, 34816),
        ("down", 17408, 5120),
    ]
    print(f"device={args.device} group_size={args.group_size} iters={args.iters}",
          flush=True)
    print(f"{'shape':<10}{'M,N,K':<26}{'W4A16 ms':>11}{'W4A8 ms':>11}{'speedup':>10}",
          flush=True)
    rows = []
    for name, k, n in shapes:
        # interleave the two arms per repetition to cancel drift
        a16, a8 = [], []
        for _ in range(3):
            a16.append(bench_shape(k, n, args.group_size, args.iters, args.device, False))
            a8.append(bench_shape(k, n, args.group_size, args.iters, args.device, True))
        m16, m8 = statistics.median(a16), statistics.median(a8)
        row = {"shape": name, "M": 16, "N": n, "K": k,
               "w4a16_ms": round(m16, 4), "w4a8_ms": round(m8, 4),
               "speedup": round(m16 / m8, 3) if m8 else None,
               "w4a16_all": [round(x, 4) for x in a16],
               "w4a8_all": [round(x, 4) for x in a8]}
        rows.append(row)
        print(f"{name:<10}{f'{16},{n},{k}':<26}{m16:>11.4f}{m8:>11.4f}"
              f"{row['speedup']:>10.3f}", flush=True)
    print(json.dumps({"group_size": args.group_size, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
