"""Standalone ``SpecDecodeAttention`` adapter for V7 microbenchmarks.

This module is loaded only through ``bench/spec_attn_fp8_ctx_scan.py --module``.
It does not register a vLLM custom op or modify the qualified dispatch.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


H_Q = 24
H_KV = 4
HEAD_DIM = 256
QMAX = 8
BLOCK_SIZE = 896
NSEG = 35


@lru_cache(maxsize=1)
def _extension():
    source = Path(__file__).with_name("v7_verifier.cu")
    return load(
        name="v7_sm80_cuda_verifier",
        sources=[str(source)],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        with_cuda=True,
        verbose=False,
    )


def _e4m3_lut(device: torch.device) -> torch.Tensor:
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


class SpecDecodeAttention:
    """Fixed-geometry adapter implementing the benchmark's existing ABI."""

    def __init__(
        self,
        max_num_reqs,
        num_heads,
        head_dim,
        device,
        qmax,
        num_segments=NSEG,
    ):
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise RuntimeError("V7 requires exact SM80")
        if (num_heads, head_dim, qmax, num_segments) != (
            H_Q,
            HEAD_DIM,
            QMAX,
            NSEG,
        ):
            raise ValueError(
                "V7 requires Hq=24, D=256, qmax=8 and NSEG=35; got "
                f"{num_heads}, {head_dim}, {qmax}, {num_segments}"
            )
        self.max_num_reqs = max_num_reqs
        count = max_num_reqs * H_Q * QMAX * NSEG
        self.part_o = torch.empty(count, HEAD_DIM, dtype=torch.float32, device=device)
        self.part_m = torch.empty(count, dtype=torch.float32, device=device)
        self.part_l = torch.empty(count, dtype=torch.float32, device=device)
        self.fp8_lut = _e4m3_lut(torch.device(device))
        self.ext = _extension()

    def run(
        self,
        q,
        key_cache,
        value_cache,
        out,
        cu_seqlens_q,
        seqused_k,
        block_table,
        scale,
        num_reqs,
        max_query_len,
        k_scale_cache=None,
        v_scale_cache=None,
        static_k_scale=None,
        static_v_scale=None,
    ):
        if k_scale_cache is not None or v_scale_cache is not None:
            raise ValueError("V7 supports static FP8 scales only")
        if static_k_scale is None or static_v_scale is None:
            raise ValueError("V7 requires static K/V scales")
        if (
            num_reqs > self.max_num_reqs
            or max_query_len > QMAX
            or key_cache.shape[1:] != (BLOCK_SIZE, H_KV, HEAD_DIM)
        ):
            raise ValueError("request/query/cache geometry is outside V7")
        key_raw = (
            key_cache
            if key_cache.dtype == torch.uint8
            else key_cache.view(torch.uint8)
        )
        value_raw = (
            value_cache
            if value_cache.dtype == torch.uint8
            else value_cache.view(torch.uint8)
        )
        self.ext.partial(
            q,
            key_raw,
            value_raw,
            block_table,
            seqused_k,
            cu_seqlens_q,
            self.part_o,
            self.part_m,
            self.part_l,
            self.fp8_lut,
            float(static_k_scale),
            float(static_v_scale),
            float(scale),
        )
        self.ext.combine(
            self.part_o,
            self.part_m,
            self.part_l,
            out,
            cu_seqlens_q,
        )
        return out

