"""Correctness/smoke bench for the standalone V7 SM80 CUDA verifier.

This bench deliberately imports a JIT-built extension from
``experimental/cmp170hx-mixed-fp8/cuda_prototype``.  It never imports vLLM,
FlashInfer, or changes the active experimental patch series.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experimental/cmp170hx-mixed-fp8/cuda_prototype/v7_verifier.cu"

H_Q = 24
H_KV = 4
D = 256
QMAX = 8
BLOCK = 896
NSEG = 35


def build_extension(verbose: bool):
    import torch
    from torch.utils.cpp_extension import load

    return load(
        name="v7_sm80_cuda_verifier",
        sources=[str(SRC)],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        with_cuda=True,
        verbose=verbose,
    )


def e4m3_lut(device):
    import torch

    values = []
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0xF and mantissa == 0x7:
            value = 0.0
        elif exponent == 0:
            value = mantissa * (2.0**-9)
        else:
            value = (1.0 + mantissa * 0.125) * (2.0 ** (exponent - 7))
        values.append(sign * value)
    return torch.tensor(values, device=device, dtype=torch.bfloat16)


def quantize_fp8(x, scale):
    import torch

    return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(
        torch.uint8
    )


def make_case(
    device,
    lengths,
    q_lens,
    int64_indices=False,
    physical_block_base=0,
):
    import torch

    assert len(lengths) == len(q_lens)
    requests = len(lengths)
    max_len = max(lengths)
    pages = math.ceil(max_len / BLOCK)
    physical_blocks = physical_block_base + pages
    k_scale = 0.03125
    v_scale = 0.02734375
    # The raw cache is intentionally kept as uint8: this is the SM80 path's
    # zero-copy view of PyTorch's E4M3 bytes, not a new cache format.
    if physical_block_base:
        # Avoid materializing full FP32 staging tensors for the 4 GiB high-ID
        # case.  Every raw byte is valid input to the fail-closed LUT decoder.
        shape = (physical_blocks, BLOCK, H_KV, D)
        k_cache = torch.randint(0, 256, shape, device=device, dtype=torch.uint8)
        v_cache = torch.randint(0, 256, shape, device=device, dtype=torch.uint8)
    else:
        k_src = torch.randn(
            physical_blocks, BLOCK, H_KV, D, device=device, dtype=torch.float32
        )
        v_src = torch.randn(
            physical_blocks, BLOCK, H_KV, D, device=device, dtype=torch.float32
        )
        k_cache = quantize_fp8(k_src, k_scale).contiguous()
        v_cache = quantize_fp8(v_src, v_scale).contiguous()

    total_q = sum(q_lens)
    q = torch.randn(total_q, H_Q, D, device=device, dtype=torch.bfloat16)
    cu_cpu = [0]
    for q_len in q_lens:
        cu_cpu.append(cu_cpu[-1] + q_len)
    index_dtype = torch.int64 if int64_indices else torch.int32
    cu_q = torch.tensor(cu_cpu, device=device, dtype=index_dtype)
    seq = torch.tensor(lengths, device=device, dtype=index_dtype)
    block_table = torch.arange(
        physical_block_base,
        physical_block_base + pages,
        device=device,
        dtype=index_dtype,
    ).repeat(requests, 1)
    lut = e4m3_lut(device)
    workspace_n = requests * H_Q * QMAX * NSEG
    part_o = torch.empty(workspace_n, D, device=device, dtype=torch.float32)
    part_m = torch.empty(workspace_n, device=device, dtype=torch.float32)
    part_l = torch.empty(workspace_n, device=device, dtype=torch.float32)
    out = torch.empty_like(q)
    return (
        q,
        k_cache,
        v_cache,
        block_table,
        seq,
        cu_q,
        part_o,
        part_m,
        part_l,
        lut,
        out,
        k_scale,
        v_scale,
    )


def reference(case):
    import torch

    (
        q,
        k_cache,
        v_cache,
        block_table,
        seq,
        cu_q,
        _part_o,
        _part_m,
        _part_l,
        lut,
        out,
        k_scale,
        v_scale,
    ) = case
    ref = torch.empty_like(out)
    for req in range(seq.numel()):
        q_start = int(cu_q[req].item())
        q_stop = int(cu_q[req + 1].item())
        q_len = q_stop - q_start
        kv_len = int(seq[req].item())
        positions = torch.arange(kv_len, device=q.device)
        pages = positions // BLOCK
        slots = positions % BLOCK
        physical = block_table[req, pages].long()
        for kvh in range(H_KV):
            k = lut[k_cache[physical, slots, kvh].long()].float() * k_scale
            v = lut[v_cache[physical, slots, kvh].long()].float() * v_scale
            for qi in range(q_len):
                q_pos = kv_len - q_len + qi
                allowed = positions <= q_pos
                for group in range(H_Q // H_KV):
                    h = kvh * (H_Q // H_KV) + group
                    q_scaled = (
                        q[q_start + qi, h].float() * (1.0 / math.sqrt(D))
                    ).to(torch.bfloat16).float()
                    scores = q_scaled @ k.T
                    scores = scores.masked_fill(~allowed, float("-inf"))
                    probs = torch.softmax(scores, dim=0)
                    ref[q_start + qi, h] = (probs[:, None] * v).sum(dim=0).to(
                        ref.dtype
                    )
    return ref


def run_case(
    ext,
    lengths,
    q_lens,
    int64_indices=False,
    physical_block_base=0,
):
    import torch

    case = make_case(
        torch.device("cuda"),
        lengths,
        q_lens,
        int64_indices,
        physical_block_base,
    )
    (
        q,
        k_cache,
        v_cache,
        block_table,
        seq,
        cu_q,
        part_o,
        part_m,
        part_l,
        lut,
        out,
        k_scale,
        v_scale,
    ) = case
    scale = 1.0 / math.sqrt(D)
    ext.partial(
        q,
        k_cache,
        v_cache,
        block_table,
        seq,
        cu_q,
        part_o,
        part_m,
        part_l,
        lut,
        k_scale,
        v_scale,
        scale,
    )
    ext.combine(part_o, part_m, part_l, out, cu_q)
    torch.cuda.synchronize()
    ref = reference(case)
    # The candidate intentionally rounds the running value accumulator to
    # FP16, matching the existing static-FP8 q8 specialization before the
    # final BF16 output conversion.
    err = (out.float() - ref.float()).abs().max().item()
    if not math.isfinite(err) or err > 0.08:
        raise AssertionError(
            f"V7 mismatch lengths={lengths} q_lens={q_lens} "
            f"int64={int64_indices}: max_abs={err:.6f}"
        )
    print(
        f"PASS lengths={lengths} q_lens={q_lens} int64={int64_indices} "
        f"block_base={physical_block_base} max_abs={err:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--high-block-id", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        print(f"SKIP: PyTorch unavailable ({exc})")
        return 0
    if not torch.cuda.is_available():
        print("SKIP: CUDA is unavailable; V7 is fixed to SM80")
        return 0
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 0):
        print(f"SKIP: active GPU is SM{major}{minor}, V7 requires SM80")
        return 0

    ext = build_extension(args.verbose)
    print("resources", dict(ext.resources()))
    if args.build_only:
        print(f"PASS: built {SRC}")
        return 0

    run_case(ext, [895], [5])
    run_case(ext, [896], [8])
    run_case(ext, [897, 4097], [5, 8], int64_indices=True)
    if args.full:
        run_case(ext, [897], [8])
        run_case(ext, [895, 896, 897, 4097], [5, 8, 6, 1])
        run_case(ext, [1500], [6])
        run_case(ext, [1500], [7])
        run_case(ext, [8192], [8])
        run_case(ext, [4097, 1300, 8192, 64], [5, 8, 6, 1])
        run_case(ext, [32768], [8])
        run_case(ext, [65536], [8])
    if args.high_block_id:
        block_stride = BLOCK * H_KV * D
        first_high_block = (2**31) // block_stride + 1
        gib = 2 * (first_high_block + 1) * block_stride / 2**30
        print(
            f"high_block_id={first_high_block} stride={block_stride} "
            f"two_cache_bytes={gib:.2f} GiB"
        )
        run_case(
            ext,
            [895],
            [5],
            int64_indices=True,
            physical_block_base=first_high_block,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
