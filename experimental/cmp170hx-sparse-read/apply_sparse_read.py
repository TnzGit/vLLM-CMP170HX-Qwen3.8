#!/usr/bin/env python3
"""Install the CMP170HX sparse verifier read-view prototype into a patched vLLM tree.

This deliberately modifies only v1/attention/ops/spec_decode_attn.py after the
normal repository patch stack and experimental/cmp170hx-mixed-fp8 M7 series are
already installed.  It does not alter the scheduler, KV allocator, write-side
block table, GDN state, prefill, or DFlash2 draft KV.

The prototype keeps the canonical KV cache fully GPU resident.  For speculative
verification only, it builds a compact, graph-stable read block table containing
an optional sink prefix plus the newest pages up to a fixed token budget.

Use --dry-run before --apply.  --reverse restores the exact backup made by the
first successful --apply.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

BEGIN = "# BEGIN cmp170hx sparse-read oracle"
END = "# END cmp170hx sparse-read oracle"

KERNEL_INSERT = r'''
# BEGIN cmp170hx sparse-read oracle
@triton.jit
def _cmp_sparse_recent_block_table(
    src_bt_ptr,
    src_seq_ptr,
    dst_bt_ptr,
    dst_seq_ptr,
    src_stride,
    dst_stride,
    BLOCK_SIZE: tl.constexpr,
    BUDGET_BLOCKS: tl.constexpr,
    SINK_BLOCKS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Build a compact chronological verifier-only block table.

    The canonical table is never modified.  Below budget this is an identity
    copy.  Above budget it keeps SINK_BLOCKS from the prefix and fills the rest
    with the newest pages.  The final source page is always present, so the
    speculative query suffix remains the final q_len tokens in the compact
    causal coordinate system.  K/Q RoPE coordinates are untouched because the
    cached tensors themselves remain at their original absolute positions.
    """
    req = tl.program_id(0)
    chunk = tl.program_id(1)
    kv_len = tl.load(src_seq_ptr + req)
    src_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    keep_blocks = tl.minimum(src_blocks, BUDGET_BLOCKS)
    is_sparse = src_blocks > BUDGET_BLOCKS

    # Always leave at least one slot for the live tail when sparsifying.
    sink_blocks = tl.minimum(SINK_BLOCKS, tl.maximum(keep_blocks - 1, 0))
    recent_blocks = keep_blocks - sink_blocks
    recent_start = src_blocks - recent_blocks

    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    valid = offs < keep_blocks
    sparse_src = tl.where(
        offs < sink_blocks,
        offs,
        recent_start + (offs - sink_blocks),
    )
    src_idx = tl.where(is_sparse, sparse_src, offs)
    block_id = tl.load(
        src_bt_ptr + req * src_stride + src_idx,
        mask=valid,
        other=0,
    )
    tl.store(
        dst_bt_ptr + req * dst_stride + offs,
        block_id,
        mask=valid,
    )

    tail_tokens = tl.where(
        src_blocks > 0,
        kv_len - (src_blocks - 1) * BLOCK_SIZE,
        0,
    )
    compact_len = tl.where(
        is_sparse,
        (keep_blocks - 1) * BLOCK_SIZE + tail_tokens,
        kv_len,
    )
    tl.store(dst_seq_ptr + req, compact_len, mask=chunk == 0)
# END cmp170hx sparse-read oracle
'''.strip("
")

CTOR_INSERT = r'''
        # BEGIN cmp170hx sparse-read oracle
        # Experimental verifier-only read view.  Buffers are allocated with the
        # verifier workspace so FULL CUDA Graph captures stable addresses.
        self.sparse_read_enabled = (
            os.environ.get("VLLM_SPARSE_KV_READ", "0") == "1"
        )
        self.sparse_read_tokens = int(
            os.environ.get("VLLM_SPARSE_KV_READ_TOKENS", "0")
        )
        self.sparse_read_sink_tokens = int(
            os.environ.get("VLLM_SPARSE_KV_READ_SINK_TOKENS", "1024")
        )
        self.sparse_read_max_blocks = int(
            os.environ.get("VLLM_SPARSE_KV_READ_MAX_BLOCKS", "8192")
        )
        self._sparse_read_reported = False
        self.sparse_block_table = None
        self.sparse_seq_lens = None
        if self.sparse_read_enabled:
            if self.sparse_read_tokens <= 0:
                raise ValueError(
                    "VLLM_SPARSE_KV_READ_TOKENS must be > 0 when sparse read is enabled"
                )
            if self.sparse_read_sink_tokens < 0:
                raise ValueError(
                    "VLLM_SPARSE_KV_READ_SINK_TOKENS must be >= 0"
                )
            if self.sparse_read_max_blocks <= 0:
                raise ValueError(
                    "VLLM_SPARSE_KV_READ_MAX_BLOCKS must be > 0"
                )
            self.sparse_block_table = torch.empty(
                (max_num_reqs, self.sparse_read_max_blocks),
                dtype=torch.int32,
                device=device,
            )
            self.sparse_seq_lens = torch.empty(
                max_num_reqs,
                dtype=torch.int32,
                device=device,
            )
        # END cmp170hx sparse-read oracle
'''.rstrip()

RUN_INSERT = r'''
        # BEGIN cmp170hx sparse-read oracle
        attn_block_table = block_table
        attn_seq_lens = seqused_k
        if self.sparse_read_enabled:
            # This research branch is intentionally scoped to the qualified M7
            # static-FP8 verifier.  Fail closed on any other path.
            if not static_quant:
                raise RuntimeError(
                    "CMP sparse read is qualified only for static-FP8 verification"
                )
            block_size = int(key_cache.shape[1])
            budget_blocks = triton.cdiv(self.sparse_read_tokens, block_size)
            if budget_blocks <= 0:
                raise RuntimeError("CMP sparse read resolved to an empty budget")
            if budget_blocks > self.sparse_read_max_blocks:
                raise RuntimeError(
                    "CMP sparse read budget needs "
                    f"{budget_blocks} blocks but VLLM_SPARSE_KV_READ_MAX_BLOCKS="
                    f"{self.sparse_read_max_blocks}"
                )
            sink_blocks = min(
                triton.cdiv(self.sparse_read_sink_tokens, block_size)
                if self.sparse_read_sink_tokens
                else 0,
                max(0, budget_blocks - 1),
            )
            assert self.sparse_block_table is not None
            assert self.sparse_seq_lens is not None
            chunk = 256
            _cmp_sparse_recent_block_table[
                (num_reqs, triton.cdiv(budget_blocks, chunk))
            ](
                block_table,
                seqused_k,
                self.sparse_block_table,
                self.sparse_seq_lens,
                block_table.stride(0),
                self.sparse_block_table.stride(0),
                BLOCK_SIZE=block_size,
                BUDGET_BLOCKS=budget_blocks,
                SINK_BLOCKS=sink_blocks,
                CHUNK=chunk,
                num_warps=4,
            )
            attn_block_table = self.sparse_block_table
            attn_seq_lens = self.sparse_seq_lens
            if not self._sparse_read_reported:
                print(
                    "[cmp-sparse-read] enabled verifier-only sink+recent view: "
                    f"requested_tokens={self.sparse_read_tokens} "
                    f"sink_tokens={self.sparse_read_sink_tokens} "
                    f"kernel_block={block_size} budget_blocks={budget_blocks} "
                    f"sink_blocks={sink_blocks} max_blocks={self.sparse_read_max_blocks}",
                    flush=True,
                )
                self._sparse_read_reported = True
        # END cmp170hx sparse-read oracle
'''.rstrip()

CTOR_ANCHOR = '''        self.fp8_lut = torch.tensor(
            _build_e4m3fn_lut(), dtype=torch.bfloat16, device=device
        )'''

RUN_ANCHOR = '''        else:
            kernel_key_cache = key_cache
            kernel_value_cache = value_cache
        assert max_query_len <= self.qmax, "too many query tokens per request for this kernel"'''

SPECIAL_CALL_OLD = '''                q, kernel_key_cache, kernel_value_cache, block_table,
                seqused_k, cu_seqlens_q,'''
SPECIAL_CALL_NEW = '''                q, kernel_key_cache, kernel_value_cache, attn_block_table,
                attn_seq_lens, cu_seqlens_q,'''

GENERIC_CALL_OLD = '''                q, kernel_key_cache, kernel_value_cache, block_table, seqused_k, cu_seqlens_q,'''
GENERIC_CALL_NEW = '''                q, kernel_key_cache, kernel_value_cache, attn_block_table, attn_seq_lens, cu_seqlens_q,'''


def require_m7(text: str) -> None:
    required = (
        "_spec_attn_partial_q8_g6_fp8",
        "_e4m3fn_to_bf16_ldg_nc",
        "static_k_scale=None, static_v_scale=None",
        "SEG_TILE=triton.next_power_of_2(self.nseg)",
    )
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise RuntimeError(
            "M7 mixed-FP8 verifier prerequisites are missing: " + ", ".join(missing)
        )


def transformed(text: str) -> str:
    if BEGIN in text:
        raise RuntimeError("sparse-read prototype is already applied")
    require_m7(text)

    kernel_anchor = "\n\n@triton.jit\ndef _spec_attn_partial("
    if text.count(kernel_anchor) != 1:
        raise RuntimeError("unexpected _spec_attn_partial anchor count")
    text = text.replace(
        kernel_anchor,
        "\n\n" + KERNEL_INSERT + "\n\n@triton.jit\ndef _spec_attn_partial(",
        1,
    )

    if text.count(CTOR_ANCHOR) != 1:
        raise RuntimeError("unexpected FP8 LUT constructor anchor count")
    text = text.replace(CTOR_ANCHOR, CTOR_ANCHOR + "\n" + CTOR_INSERT, 1)

    if text.count(RUN_ANCHOR) != 1:
        raise RuntimeError("unexpected verifier run anchor count")
    run_replacement = RUN_ANCHOR.replace(
        '        assert max_query_len <= self.qmax, "too many query tokens per request for this kernel"',
        RUN_INSERT
        + '\n        assert max_query_len <= self.qmax, "too many query tokens per request for this kernel"',
    )
    text = text.replace(RUN_ANCHOR, run_replacement, 1)

    if text.count(SPECIAL_CALL_OLD) != 1:
        raise RuntimeError("unexpected q8/G6 specialized call anchor count")
    text = text.replace(SPECIAL_CALL_OLD, SPECIAL_CALL_NEW, 1)

    if text.count(GENERIC_CALL_OLD) != 1:
        raise RuntimeError("unexpected generic verifier call anchor count")
    text = text.replace(GENERIC_CALL_OLD, GENERIC_CALL_NEW, 1)

    stride_old = "                block_table.stride(0),"
    stride_new = "                attn_block_table.stride(0),"
    if text.count(stride_old) != 2:
        raise RuntimeError(
            f"unexpected block-table stride anchor count: {text.count(stride_old)}"
        )
    text = text.replace(stride_old, stride_new)

    return text


def find_site(explicit: str | None) -> Path:
    if explicit:
        site = Path(explicit).resolve()
    else:
        import vllm  # type: ignore

        site = Path(inspect.getfile(vllm)).resolve().parent
    target = site / "v1/attention/ops/spec_decode_attn.py"
    if not target.is_file():
        raise RuntimeError(f"not a patched vLLM package: {site}")
    return site


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--reverse", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    site = find_site(args.site)
    target = site / "v1/attention/ops/spec_decode_attn.py"
    backup = target.with_suffix(target.suffix + ".sparse-read-oracle.bak")
    text = target.read_text()

    if args.reverse:
        if not backup.is_file():
            raise SystemExit(f"no sparse-read backup to restore: {backup}")
        target.write_text(backup.read_text())
        backup.unlink()
        print(f"restored {target}")
        return

    if args.check:
        require_m7(text)
        if BEGIN not in text or "attn_block_table = self.sparse_block_table" not in text:
            raise SystemExit("sparse-read prototype is not fully installed")
        compile(text, str(target), "exec")
        print(f"sparse-read check OK: {target}")
        return

    new_text = transformed(text)
    compile(new_text, str(target), "exec")
    if args.dry_run or not args.apply:
        print(
            f"sparse-read dry-run OK: {target} "
            f"({len(new_text) - len(text):+d} bytes)"
        )
        return

    if backup.exists():
        raise SystemExit(
            f"refusing to overwrite existing backup: {backup}; "
            "use --check or --reverse"
        )
    backup.write_text(text)
    target.write_text(new_text)
    print(f"installed sparse-read prototype: {target}")
    print(f"backup: {backup}")


if __name__ == "__main__":
    main()
