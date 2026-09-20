"""Measure the M7 verifier with a dense canonical KV cache and sparse read view.

Requires:
  - exact SM80
  - normal repository patches
  - experimental/cmp170hx-mixed-fp8 M7 series
  - experimental/cmp170hx-sparse-read prototype

The cache remains fully resident.  For each logical context this script compares
the ordinary dense M7 verifier with the sparse sink+recent read table and checks
that the GPU-built table exactly matches a CPU reference policy.

Below budget the read table is an identity copy, so output must be bit-identical.
Above budget output equality is not expected because attention semantics changed.

Example:
  python bench/sparse-read/spec_attn_sparse_ctx_scan.py
  python bench/sparse-read/spec_attn_sparse_ctx_scan.py \
      --contexts 32000,126000,250000 --budgets 32000,48000,65000 --queries 4
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import torch

from vllm.v1.attention.ops.spec_decode_attn import SpecDecodeAttention


H_Q, H_KV, HEAD_DIM = 24, 4, 256
# M7 currently presents a 896-token target verifier page.  The benchmark does
# not assume that: it reads key.shape[1] for all policy math and reports it.
CACHE_BLOCK_SIZE = 896
SCALE = HEAD_DIM**-0.5
FP8 = torch.float8_e4m3fn


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Expected:
    block_ids: list[int]
    seq_len: int
    budget_blocks: int
    sink_blocks: int


def expected_policy(
    kv_len: int,
    block_size: int,
    budget_tokens: int,
    sink_tokens: int,
) -> Expected:
    src_blocks = cdiv(kv_len, block_size)
    budget_blocks = cdiv(budget_tokens, block_size)
    if budget_blocks <= 0:
        raise ValueError("budget must contain at least one block")
    if src_blocks <= budget_blocks:
        return Expected(list(range(src_blocks)), kv_len, budget_blocks, 0)

    sink_blocks = min(
        cdiv(sink_tokens, block_size) if sink_tokens else 0,
        max(0, budget_blocks - 1),
    )
    recent_blocks = budget_blocks - sink_blocks
    recent_start = src_blocks - recent_blocks
    ids = list(range(sink_blocks)) + list(range(recent_start, src_blocks))
    tail_tokens = kv_len - (src_blocks - 1) * block_size
    seq_len = (budget_blocks - 1) * block_size + tail_tokens
    return Expected(ids, seq_len, budget_blocks, sink_blocks)


def quantized_cache(kv_len: int):
    nblocks = cdiv(kv_len, CACHE_BLOCK_SIZE) + 2
    shape = (nblocks, CACHE_BLOCK_SIZE, H_KV, HEAD_DIM)

    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.75
    k_scale = max(float(src.float().abs().max()) / 420.0, 1e-6)
    key = (src.float() / k_scale).clamp(-448, 448).to(FP8)
    del src

    src = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.75
    v_scale = max(float(src.float().abs().max()) / 420.0, 1e-6)
    value = (src.float() / v_scale).clamp(-448, 448).to(FP8)
    del src

    blocks = cdiv(kv_len, CACHE_BLOCK_SIZE)
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


def make_attention(enabled: bool, budget: int, sink: int, qmax: int, nseg: int):
    os.environ["VLLM_SPARSE_KV_READ"] = "1" if enabled else "0"
    os.environ["VLLM_SPARSE_KV_READ_TOKENS"] = str(budget if enabled else 0)
    os.environ["VLLM_SPARSE_KV_READ_SINK_TOKENS"] = str(sink)
    os.environ.setdefault("VLLM_SPARSE_KV_READ_MAX_BLOCKS", "8192")
    attn = SpecDecodeAttention(
        max_num_reqs=1,
        num_heads=H_Q,
        head_dim=HEAD_DIM,
        device="cuda",
        qmax=qmax,
        num_segments=nseg,
    )
    if enabled and not hasattr(attn, "sparse_block_table"):
        raise RuntimeError(
            "SpecDecodeAttention has no sparse-read buffers; install "
            "experimental/cmp170hx-sparse-read first"
        )
    return attn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--contexts",
        type=csv_ints,
        default=[4096, 32000, 65000, 126000, 250000],
    )
    ap.add_argument(
        "--budgets",
        type=csv_ints,
        default=[32000, 48000, 65000],
    )
    ap.add_argument("--queries", type=csv_ints, default=[4])
    ap.add_argument("--sink-tokens", type=int, default=1024)
    ap.add_argument("--segments", type=int, default=35)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an exact SM80 CUDA GPU")

    torch.manual_seed(170)
    print(
        "context query budget logical_block active_tokens dense_us/layer "
        "sparse_us/layer speedup maxdiff below_budget_identity table_ok"
    )

    for context in args.contexts:
        key, value, ks, vs, block_table = quantized_cache(context)
        block_size = int(key.shape[1])

        for query_len in args.queries:
            q = torch.randn(
                query_len, H_Q, HEAD_DIM, device="cuda", dtype=torch.bfloat16
            )
            cu_q = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
            seq = torch.tensor([context], device="cuda", dtype=torch.int32)

            dense_out = torch.empty_like(q)
            dense = make_attention(
                False,
                0,
                args.sink_tokens,
                max(args.queries),
                args.segments,
            )

            def dense_call():
                return dense.run(
                    q,
                    key,
                    value,
                    dense_out,
                    cu_q,
                    seq,
                    block_table,
                    SCALE,
                    1,
                    query_len,
                    static_k_scale=ks,
                    static_v_scale=vs,
                )

            dense_us = timed(dense_call, args.warmup, args.iters)
            dense_call()
            torch.cuda.synchronize()
            dense_ref = dense_out.clone()

            for budget in args.budgets:
                sparse_out = torch.empty_like(q)
                sparse = make_attention(
                    True,
                    budget,
                    args.sink_tokens,
                    max(args.queries),
                    args.segments,
                )

                def sparse_call(a=sparse, out=sparse_out):
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

                sparse_us = timed(sparse_call, args.warmup, args.iters)
                sparse_call()
                torch.cuda.synchronize()

                expected = expected_policy(
                    context, block_size, budget, args.sink_tokens
                )
                got_len = int(sparse.sparse_seq_lens[0].item())
                got_ids = (
                    sparse.sparse_block_table[0, : len(expected.block_ids)]
                    .cpu()
                    .tolist()
                )
                table_ok = got_len == expected.seq_len and got_ids == expected.block_ids
                if not table_ok:
                    raise AssertionError(
                        "sparse table mismatch: "
                        f"context={context} budget={budget} "
                        f"got_len={got_len} exp_len={expected.seq_len} "
                        f"got_head={got_ids[:8]} exp_head={expected.block_ids[:8]} "
                        f"got_tail={got_ids[-8:]} exp_tail={expected.block_ids[-8:]}"
                    )

                maxdiff = float((sparse_out.float() - dense_ref.float()).abs().max())
                below = context <= expected.budget_blocks * block_size
                identity = (not below) or maxdiff == 0.0
                if below and not identity:
                    raise AssertionError(
                        f"below-budget identity failed: ctx={context} budget={budget} "
                        f"maxdiff={maxdiff}"
                    )

                print(
                    f"{context:7d} {query_len:5d} {budget:6d} {block_size:13d} "
                    f"{expected.seq_len:13d} {dense_us:14.1f} {sparse_us:15.1f} "
                    f"{dense_us / sparse_us:7.3f} {maxdiff:7.5f} "
                    f"{str(identity):>21s} {str(table_ok):>8s}",
                    flush=True,
                )
                del sparse, sparse_out

            del dense, dense_out, dense_ref, q, cu_q, seq

        del key, value, block_table
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
