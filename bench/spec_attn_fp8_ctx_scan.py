"""Microbenchmark the real SM80 static-FP8 speculative verifier geometry.

Unlike ``spec_attn_ctx_scan.py``, this uses the CMP 170HX production target shape:
E4M3 KV, block size 896, 24 query heads, 4 KV heads, and head size 256.  It
compares several split counts on identical cache bytes and reports both latency and
the maximum output difference from the 16-split baseline.

Run with the API service stopped so another CUDA context does not perturb timings::

    python bench/spec_attn_fp8_ctx_scan.py
    python bench/spec_attn_fp8_ctx_scan.py --contexts 126000,250000 --queries 4,8,12
"""

from __future__ import annotations

import argparse
import math

import torch

from vllm.v1.attention.ops.spec_decode_attn import SpecDecodeAttention


H_Q, H_KV, HEAD_DIM, BLOCK_SIZE = 24, 4, 256, 896
SCALE = HEAD_DIM**-0.5
FP8 = torch.float8_e4m3fn


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def quantized_cache(kv_len: int):
    nblocks = math.ceil(kv_len / BLOCK_SIZE) + 2
    shape = (nblocks, BLOCK_SIZE, H_KV, HEAD_DIM)

    # Quantize K and V separately so the peak temporary footprint stays modest.
    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.75
    k_scale = max(float(src.float().abs().max()) / 420.0, 1e-6)
    key = (src.float() / k_scale).clamp(-448, 448).to(FP8)
    del src
    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.75
    v_scale = max(float(src.float().abs().max()) / 420.0, 1e-6)
    value = (src.float() / v_scale).clamp(-448, 448).to(FP8)
    del src

    blocks = math.ceil(kv_len / BLOCK_SIZE)
    table = torch.arange(blocks, device="cuda", dtype=torch.int32).view(1, -1)
    return key, value, k_scale, v_scale, table


def timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=csv_ints, default=[4096, 126000, 250000])
    parser.add_argument("--queries", type=csv_ints, default=[4, 6, 8, 10, 12])
    parser.add_argument("--segments", type=csv_ints, default=[8, 16, 32, 64])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an exact SM80 CUDA GPU")

    torch.manual_seed(17)
    print(
        f"device={torch.cuda.get_device_name()} block={BLOCK_SIZE} "
        f"Hq/Hkv/D={H_Q}/{H_KV}/{HEAD_DIM}"
    )
    print("context query nseg us/layer ms/16layers maxdiff-vs-nseg16")

    for context in args.contexts:
        key, value, ks, vs, block_table = quantized_cache(context)
        for query_len in args.queries:
            q = torch.randn(query_len, H_Q, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            out = torch.empty_like(q)
            cu_q = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
            seq = torch.tensor([context], device="cuda", dtype=torch.int32)
            outputs: dict[int, torch.Tensor] = {}
            latencies: dict[int, float] = {}

            for nseg in args.segments:
                attn = SpecDecodeAttention(
                    max_num_reqs=1,
                    num_heads=H_Q,
                    head_dim=HEAD_DIM,
                    device="cuda",
                    qmax=max(args.queries),
                    num_segments=nseg,
                )

                def call(a=attn):
                    return a.run(
                        q,
                        key,
                        value,
                        out,
                        cu_q,
                        seq,
                        block_table,
                        SCALE,
                        1,
                        query_len,
                        static_k_scale=ks,
                        static_v_scale=vs,
                    )

                latencies[nseg] = timed(call, args.warmup, args.iters)
                call()
                torch.cuda.synchronize()
                outputs[nseg] = out.float().clone()
                del attn

            reference = outputs[16]
            for nseg in args.segments:
                diff = float((outputs[nseg] - reference).abs().max())
                usec = latencies[nseg]
                print(
                    f"{context:7d} {query_len:5d} {nseg:4d} "
                    f"{usec:8.1f} {usec * 16 / 1000:11.3f} {diff:.6f}"
                )
            del q, out, cu_q, seq, outputs

        del key, value, block_table
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
