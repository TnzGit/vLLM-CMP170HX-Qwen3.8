"""Correctness test for the experimental mixed-FP8 verifier patch.

Run on the CMP 170HX after applying, in order:
  patches/spec-decode-attn.patch
  patches/spec-decode-int8-kv.patch
  experimental/cmp170hx-mixed-fp8/patches/spec-decode-fp8-kv-sm80.patch

The test intentionally bypasses FlashInfer dispatch. It verifies the kernel math
first: a raw E4M3 paged cache plus static K/V scales must match an explicitly
dequantized reference built from the exact same FP8 bytes.
"""

import argparse
import math
import sys

import torch

from vllm.v1.attention.ops.spec_decode_attn import SpecDecodeAttention


torch.manual_seed(7)
DEV = "cuda"
H_Q = 24
H_KV = 4
D = 256
BLOCK_SIZE = 64
SCALE = D**-0.5
FP8 = torch.float8_e4m3fn


def _quantize_static(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    # Keep a little headroom below E4M3's 448 finite maximum so random maxima
    # do not sit exactly on the clipping edge.
    amax = float(x.float().abs().max().item())
    scale = max(amax / 420.0, 1e-6)
    q = (x.float() / scale).clamp(-448.0, 448.0).to(FP8)
    return q, scale


def make(kv_lens: list[int], q_len: int):
    batch = len(kv_lens)
    max_blocks = max(math.ceil(length / BLOCK_SIZE) for length in kv_lens)
    num_blocks = batch * max_blocks + 16

    k_src = torch.randn(
        num_blocks, BLOCK_SIZE, H_KV, D, device=DEV, dtype=torch.bfloat16
    )
    v_src = torch.randn_like(k_src)
    k_fp8, k_scale = _quantize_static(k_src)
    v_fp8, v_scale = _quantize_static(v_src)

    perm = torch.randperm(num_blocks, device=DEV)
    block_table = torch.zeros(
        batch, max_blocks, dtype=torch.int32, device=DEV
    )
    for req, length in enumerate(kv_lens):
        nblocks = math.ceil(length / BLOCK_SIZE)
        lo = req * max_blocks
        block_table[req, :nblocks] = perm[lo : lo + nblocks].to(torch.int32)

    q = torch.randn(
        batch * q_len, H_Q, D, device=DEV, dtype=torch.bfloat16
    )
    cu_q = torch.arange(
        0, batch * q_len + 1, q_len, dtype=torch.int32, device=DEV
    )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=DEV)
    return q, k_fp8, v_fp8, k_scale, v_scale, block_table, cu_q, seq_lens


def reference(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    k_scale: float,
    v_scale: float,
    block_table: torch.Tensor,
    kv_lens: list[int],
    q_len: int,
) -> torch.Tensor:
    outs = []
    group = H_Q // H_KV
    for req, length in enumerate(kv_lens):
        nblocks = math.ceil(length / BLOCK_SIZE)
        block_ids = block_table[req, :nblocks].long()
        k = (k_fp8[block_ids].float() * k_scale).reshape(-1, H_KV, D)[:length]
        v = (v_fp8[block_ids].float() * v_scale).reshape(-1, H_KV, D)[:length]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        qq = q[req * q_len : (req + 1) * q_len].float()

        scores = torch.einsum("qhd,khd->hqk", qq, k) * SCALE
        q_pos = torch.arange(length - q_len, length, device=DEV)[:, None]
        k_pos = torch.arange(length, device=DEV)[None, :]
        scores = scores.masked_fill((k_pos > q_pos)[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return torch.cat(outs, dim=0)


def run_case(att: SpecDecodeAttention, kv_lens: list[int], q_len: int) -> float:
    (
        q,
        k_fp8,
        v_fp8,
        k_scale,
        v_scale,
        block_table,
        cu_q,
        seq_lens,
    ) = make(kv_lens, q_len)
    out = torch.empty_like(q)
    att.run(
        q,
        k_fp8,
        v_fp8,
        out,
        cu_q,
        seq_lens,
        block_table,
        SCALE,
        len(kv_lens),
        q_len,
        static_k_scale=k_scale,
        static_v_scale=v_scale,
    )
    ref = reference(
        q,
        k_fp8,
        v_fp8,
        k_scale,
        v_scale,
        block_table,
        kv_lens,
        q_len,
    )
    return float((out.float() - ref).abs().max().item())


def run_high_block_case(att: SpecDecodeAttention) -> float:
    """Exercise block-id address arithmetic just above the int32 boundary.

    A block contains 65,536 FP8 elements. Block 32,780 therefore begins above
    2**31 elements. Only that referenced block is initialized; the rest of the
    allocation exists solely to make the high physical block id valid.
    """
    block_id = 32_780
    kv_len = BLOCK_SIZE
    q_len = 5
    shape = (block_id + 1, BLOCK_SIZE, H_KV, D)
    k_fp8 = torch.empty(shape, device=DEV, dtype=FP8)
    v_fp8 = torch.empty(shape, device=DEV, dtype=FP8)
    k_src = torch.randn(1, BLOCK_SIZE, H_KV, D, device=DEV, dtype=torch.bfloat16)
    v_src = torch.randn_like(k_src)
    k_block, k_scale = _quantize_static(k_src)
    v_block, v_scale = _quantize_static(v_src)
    k_fp8[block_id].copy_(k_block[0])
    v_fp8[block_id].copy_(v_block[0])
    block_table = torch.tensor([[block_id]], device=DEV, dtype=torch.int32)
    q = torch.randn(q_len, H_Q, D, device=DEV, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, q_len], device=DEV, dtype=torch.int32)
    seq_lens = torch.tensor([kv_len], device=DEV, dtype=torch.int32)
    out = torch.empty_like(q)
    att.run(
        q,
        k_fp8,
        v_fp8,
        out,
        cu_q,
        seq_lens,
        block_table,
        SCALE,
        1,
        q_len,
        static_k_scale=k_scale,
        static_v_scale=v_scale,
    )
    ref = reference(
        q,
        k_fp8,
        v_fp8,
        k_scale,
        v_scale,
        block_table,
        [kv_len],
        q_len,
    )
    return float((out.float() - ref).abs().max().item())


def main() -> int:
    global BLOCK_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="return success instead of failure when CUDA/sm80 is unavailable",
    )
    parser.add_argument(
        "--high-block-id",
        action="store_true",
        help="also run the ~4 GiB high physical block-id regression",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=BLOCK_SIZE,
        help="paged-KV block size used by the synthetic cache",
    )
    parser.add_argument(
        "--production-page-boundaries",
        action="store_true",
        help="exercise the 895/896/897-token boundaries used by the CMP profile",
    )
    args = parser.parse_args()

    BLOCK_SIZE = args.block_size
    if args.high_block_id and BLOCK_SIZE != 64:
        parser.error("--high-block-id currently requires --block-size 64")

    if not torch.cuda.is_available():
        print("SKIP: CUDA is required")
        return 0 if args.allow_skip else 2

    cap = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()} capability={cap}")
    if cap != (8, 0):
        print("SKIP: this regression requires GA100/sm80")
        return 0 if args.allow_skip else 2

    att = SpecDecodeAttention(
        max_num_reqs=16,
        num_heads=H_Q,
        head_dim=D,
        device=DEV,
        qmax=64,
    )

    cases = [
        ([1500], 8),
        ([8192], 8),
        ([4097, 1300], 5),
        ([32768], 8),
        ([65536], 16),
        ([8192, 1500], 64),
    ]
    if args.production_page_boundaries:
        cases = [
            ([895], 5),
            ([896], 8),
            ([897], 8),
            ([4097], 8),
            ([4097, 1300], 5),
        ] + cases
    failed = False
    for kv_lens, q_len in cases:
        err = run_case(att, kv_lens, q_len)
        # BF16 tensor-core accumulation dominates this tolerance; the reference
        # dequantizes the exact FP8 bytes, so this is not a quantization-quality
        # comparison.
        ok = err < 0.08
        print(
            f"kv={kv_lens} q={q_len}: max|kernel-dequant_ref|={err:.5f} "
            f"{'OK' if ok else 'FAIL'}"
        )
        failed |= not ok

    if args.high_block_id:
        err = run_high_block_case(att)
        ok = err < 0.08
        print(
            "high_block_id=32780 q=5: "
            f"max|kernel-dequant_ref|={err:.5f} {'OK' if ok else 'FAIL'}"
        )
        failed |= not ok

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
