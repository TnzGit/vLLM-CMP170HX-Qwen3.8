"""Opt-in INT8-G64 KV writer and E38 attention bridge.

The file is copied into ``vllm/v1/attention/ops/int8_g64.py`` by the test
launcher.  It deliberately has no import-time CUDA work: the Triton writer and
the C++/CUDA E38 module are initialized lazily on the first real cache update.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl


GROUP = 64
Q_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
BLOCK_TOKENS = 64
SCALE_GROUPS = HEAD_DIM // GROUP  # four G=64 scales per token
_DEBUG_VIEWS_DONE = False
_DEBUG_WRITES_DONE = False
_DEBUG_BRIDGE_DONE = False


def _debug_file(message: str) -> None:
    if os.environ.get("VLLM_INT8_G64_DEBUG") != "1":
        return
    try:
        with open("/tmp/int8g64-debug.log", "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except OSError:
        pass


@triton.jit
def _store_g64_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_scale_ptr,
    value_scale_ptr,
    slot_ptr,
    key_tok_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    value_tok_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    key_blk_stride: tl.constexpr,
    key_head_cache_stride: tl.constexpr,
    key_slot_stride: tl.constexpr,
    value_blk_stride: tl.constexpr,
    value_head_cache_stride: tl.constexpr,
    value_slot_stride: tl.constexpr,
    key_scale_blk_stride: tl.constexpr,
    key_scale_head_stride: tl.constexpr,
    key_scale_slot_stride: tl.constexpr,
    value_scale_blk_stride: tl.constexpr,
    value_scale_head_stride: tl.constexpr,
    value_scale_slot_stride: tl.constexpr,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    groups: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_ptr + token).to(tl.int64)
    valid = slot >= 0
    block = slot // block_size
    offset = slot % block_size
    dims = tl.arange(0, 64)
    for group in range(groups):
        d = group * 64 + dims
        mask = valid & (d < head_size)
        k = tl.load(
            key_ptr + token * key_tok_stride + head * key_head_stride + d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            value_ptr + token * value_tok_stride + head * value_head_stride + d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        k_scale = tl.maximum(tl.max(tl.abs(k), axis=0) / 127.0, 1e-6)
        v_scale = tl.maximum(tl.max(tl.abs(v), axis=0) / 127.0, 1e-6)
        k_base = (
            block * key_scale_blk_stride
            + head * key_scale_head_stride
            + offset * key_scale_slot_stride
            + group
        )
        v_base = (
            block * value_scale_blk_stride
            + head * value_scale_head_stride
            + offset * value_scale_slot_stride
            + group
        )
        tl.store(key_scale_ptr + k_base, k_scale.to(tl.float16), mask=valid)
        tl.store(value_scale_ptr + v_base, v_scale.to(tl.float16), mask=valid)
        kq = tl.extra.cuda.libdevice.round(k / k_scale)
        vq = tl.extra.cuda.libdevice.round(v / v_scale)
        kq = tl.clamp(kq, -127.0, 127.0).to(tl.int8)
        vq = tl.clamp(vq, -127.0, 127.0).to(tl.int8)
        k_base = (
            block * key_blk_stride
            + head * key_head_cache_stride
            + offset * key_slot_stride
            + d
        )
        v_base = (
            block * value_blk_stride
            + head * value_head_cache_stride
            + offset * value_slot_stride
            + d
        )
        tl.store(key_cache_ptr + k_base, kq, mask=mask)
        tl.store(value_cache_ptr + v_base, vq, mask=mask)


def reshape_and_cache_int8_g64(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write arbitrary prefill/decode tokens into the E38 HND cache layout."""
    if key.device.type != "cuda" or value.device.type != "cuda":
        raise ValueError("INT8-G64 cache writer requires CUDA")
    if key_cache.dtype != torch.int8 or value_cache.dtype != torch.int8:
        raise TypeError("INT8-G64 code views must be int8")
    if key_scale.dtype != torch.float16 or value_scale.dtype != torch.float16:
        raise TypeError("INT8-G64 scales must be float16")
    if key.shape[-1] != HEAD_DIM or value.shape[-1] != HEAD_DIM:
        raise ValueError("E38 currently supports D=256 only")
    if key_cache.stride(-1) != 1 or key_cache.stride(-2) != HEAD_DIM:
        raise ValueError("INT8-G64 requires contiguous HND cache views")
    block_size = key_cache.shape[2]
    groups = (HEAD_DIM + GROUP - 1) // GROUP
    grid = (key.shape[0], key.shape[1])
    global _DEBUG_WRITES_DONE
    if os.environ.get("VLLM_INT8_G64_DEBUG") == "1" and not _DEBUG_WRITES_DONE:
        _DEBUG_WRITES_DONE = True
        _debug_file(
            "writer key=%s key_stride=%s value_stride=%s cache=%s cache_stride=%s "
            "ks=%s ks_stride=%s slots=%d min=%d max=%d storage=%d offset=%d"
            % (tuple(key.shape), tuple(key.stride()), tuple(value.stride()),
               tuple(key_cache.shape), tuple(key_cache.stride()),
               tuple(key_scale.shape), tuple(key_scale.stride()), int(slot_mapping.numel()),
               int(slot_mapping.min().item()) if slot_mapping.numel() else -1,
               int(slot_mapping.max().item()) if slot_mapping.numel() else -1,
               int(key_cache.untyped_storage().nbytes()), int(key_cache.storage_offset()))
        )
        print(
            "INT8-G64 writer debug:",
            "key", tuple(key.shape), tuple(key.stride()), str(key.dtype),
            "value", tuple(value.shape), tuple(value.stride()),
            "cache", tuple(key_cache.shape), tuple(key_cache.stride()),
            "k_scale", tuple(key_scale.shape), tuple(key_scale.stride()),
            "slot", int(slot_mapping.numel()),
            "slot_min", int(slot_mapping.min().item()) if slot_mapping.numel() else -1,
            "slot_max", int(slot_mapping.max().item()) if slot_mapping.numel() else -1,
            "storage_bytes", int(key_cache.untyped_storage().nbytes()),
            "storage_offset", int(key_cache.storage_offset()),
            flush=True,
        )
    _store_g64_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        key_scale,
        value_scale,
        slot_mapping,
        key_tok_stride=key.stride(0),
        key_head_stride=key.stride(1),
        value_tok_stride=value.stride(0),
        value_head_stride=value.stride(1),
        key_blk_stride=key_cache.stride(0),
        key_head_cache_stride=key_cache.stride(1),
        key_slot_stride=key_cache.stride(2),
        value_blk_stride=value_cache.stride(0),
        value_head_cache_stride=value_cache.stride(1),
        value_slot_stride=value_cache.stride(2),
        key_scale_blk_stride=key_scale.stride(0),
        key_scale_head_stride=key_scale.stride(1),
        key_scale_slot_stride=key_scale.stride(2),
        value_scale_blk_stride=value_scale.stride(0),
        value_scale_head_stride=value_scale.stride(1),
        value_scale_slot_stride=value_scale.stride(2),
        block_size=block_size,
        head_size=HEAD_DIM,
        groups=groups,
        num_warps=4,
        num_stages=2,
    )


def _storage_view(
    raw: torch.UntypedStorage,
    dtype: torch.dtype,
    offset_bytes: int,
    shape: tuple[int, ...],
) -> torch.Tensor:
    item = torch.tensor([], dtype=dtype).element_size()
    if offset_bytes % item:
        raise RuntimeError(f"unaligned INT8-G64 storage offset {offset_bytes}")
    base = torch.empty(0, dtype=dtype, device="cuda")
    strides = tuple(
        torch.empty(shape, dtype=dtype, device="meta").stride()
    )
    return torch.as_strided(
        base.set_(raw, offset_bytes // item, shape, strides), shape, strides
    )


def g64_views(kv_cache: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Recover four page-local views from the allocator-owned per-layer tensor.

    The allocator pads every physical page to its own stride, so the layout is
    [page][K 65536][V 65536][Kscale 2048][Vscale 2048][padding]; the four views
    share the page stride and differ only by in-page offset.  This replaces the
    original contiguous global-plane ABI, which rejected padded pages outright.
    Overlapping or misaligned pages are still rejected rather than reinterpreted.
    """
    if kv_cache.dtype != torch.int8 or kv_cache.ndim != 4:
        raise ValueError("page-local INT8-G64 requires an int8 rank-four key view")
    blocks, heads, tokens, dim = kv_cache.shape
    if blocks < 1 or (heads, tokens, dim) != (KV_HEADS, BLOCK_TOKENS, HEAD_DIM):
        raise ValueError(
            f"page-local INT8-G64 requires [positive blocks,{KV_HEADS},{BLOCK_TOKENS},{HEAD_DIM}], "
            f"got {tuple(kv_cache.shape)}"
        )
    page_bytes = kv_cache.stride(0)
    if tuple(kv_cache.stride()[1:]) != (BLOCK_TOKENS * HEAD_DIM, HEAD_DIM, 1):
        raise ValueError("page-local INT8-G64 requires HND within each page")
    code_bytes = 2 * KV_HEADS * BLOCK_TOKENS * HEAD_DIM
    scale_bytes = 2 * KV_HEADS * BLOCK_TOKENS * SCALE_GROUPS * 2
    page_payload = code_bytes + scale_bytes
    if page_bytes < page_payload or page_bytes % 2:
        raise ValueError(
            "page-local INT8-G64 pages must not overlap and must align FP16 "
            f"scales: stride(0)={page_bytes} < {page_payload}"
        )
    base = kv_cache.storage_offset() * kv_cache.element_size()
    if base % 2:
        raise ValueError("page-local INT8-G64 base must align FP16 scales")
    raw = kv_cache.untyped_storage()
    end = base + (blocks - 1) * page_bytes + page_payload
    if end > raw.nbytes():
        raise ValueError(
            f"page-local INT8-G64 storage overrun: end={end}, bytes={raw.nbytes()}"
        )

    def view(dtype: torch.dtype, relative_offset: int, shape, stride):
        item = 1 if dtype == torch.int8 else 2
        return torch.empty(0, dtype=dtype, device=kv_cache.device).set_(
            raw, (base + relative_offset) // item, shape, stride
        )

    codes = (blocks, KV_HEADS, BLOCK_TOKENS, HEAD_DIM)
    scales = (blocks, KV_HEADS, BLOCK_TOKENS, SCALE_GROUPS)
    scale_stride = (page_bytes // 2, BLOCK_TOKENS * SCALE_GROUPS, SCALE_GROUPS, 1)
    views = (
        kv_cache,
        view(torch.int8, KV_HEADS * BLOCK_TOKENS * HEAD_DIM, codes,
             (page_bytes, BLOCK_TOKENS * HEAD_DIM, HEAD_DIM, 1)),
        view(torch.float16, code_bytes, scales, scale_stride),
        view(torch.float16, code_bytes + scale_bytes // 2, scales, scale_stride),
    )
    global _DEBUG_VIEWS_DONE
    if os.environ.get("VLLM_INT8_G64_DEBUG") == "1" and not _DEBUG_VIEWS_DONE:
        _DEBUG_VIEWS_DONE = True
        _debug_file(
            "page-local views shape=%s stride=%s storage=%d base=%d page_bytes=%d "
            "ptrs=%s"
            % (tuple(kv_cache.shape), tuple(kv_cache.stride()), int(raw.nbytes()),
               int(base), int(page_bytes), [int(x.data_ptr()) for x in views])
        )
    return views


@lru_cache(maxsize=1)
def _load_e38():
    from torch.utils.cpp_extension import load

    src = os.environ.get("VLLM_INT8_G64_SRC")
    include = os.environ.get("VLLM_INT8_G64_NINFER_ROOT")
    if not src or not include:
        raise RuntimeError(
            "INT8-G64 needs VLLM_INT8_G64_SRC and VLLM_INT8_G64_NINFER_ROOT"
        )
    src = str(Path(src).expanduser())
    include = str(Path(include).expanduser())
    return load(
        name=os.environ.get("VLLM_INT8_G64_EXT_NAME", "vllm_int8_g64_e38_strided"),
        sources=[src],
        extra_include_paths=[include],
        extra_cuda_cflags=["-O3", "-maxrregcount=170", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


def run_e38_attention(
    q: torch.Tensor,
    pos: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    block_tables: torch.Tensor,
    valid_columns: torch.Tensor,
    output: torch.Tensor,
    partial_acc: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    scale: float,
    split_count: int,
    logical_capacity: int,
) -> None:
    """Launch E38 partial+reduce for C=1..4 fixed eight-query verifies."""
    batch = block_tables.shape[0]
    if batch < 1 or batch > 4:
        raise ValueError("E38 whole-model bridge supports C1..C4")
    if q.shape != (batch * 8, Q_HEADS, HEAD_DIM):
        raise ValueError(f"expected q [{batch * 8},24,256], got {tuple(q.shape)}")
    if pos.numel() != batch * 8:
        raise ValueError("pos must contain eight positions per request")
    global _DEBUG_BRIDGE_DONE
    if os.environ.get("VLLM_INT8_G64_DEBUG") == "1" and not _DEBUG_BRIDGE_DONE:
        _DEBUG_BRIDGE_DONE = True
        _debug_file(
            "bridge q=%s qstride=%s k=%s kstride=%s v=%s vstride=%s ks=%s ksstride=%s "
            "tables=%s tablesstride=%s tables0=%s valid=%s pos=%s split=%d cap=%d"
            % (tuple(q.shape), tuple(q.stride()), tuple(key_cache.shape), tuple(key_cache.stride()),
               tuple(value_cache.shape), tuple(value_cache.stride()), tuple(key_scale.shape),
               tuple(key_scale.stride()), tuple(block_tables.shape), tuple(block_tables.stride()),
               block_tables[0, : min(8, block_tables.shape[1])].tolist(), valid_columns.tolist(),
               pos[: min(8, pos.numel())].tolist(), int(split_count), int(logical_capacity))
        )
        print(
            "INT8-G64 bridge debug:",
            "q", tuple(q.shape), tuple(q.stride()),
            "k", tuple(key_cache.shape), tuple(key_cache.stride()),
            "v", tuple(value_cache.shape), tuple(value_cache.stride()),
            "ks", tuple(key_scale.shape), tuple(key_scale.stride()),
            "vs", tuple(value_scale.shape), tuple(value_scale.stride()),
            "tables", tuple(block_tables.shape), tuple(block_tables.stride()),
            "tables0", block_tables[0, : min(8, block_tables.shape[1])].tolist(),
            "valid", valid_columns.tolist(), "pos", pos[: min(8, pos.numel())].tolist(),
            "split", int(split_count), "capacity", int(logical_capacity),
            flush=True,
        )
    ext = _load_e38()
    table_rows = torch.arange(batch, device=q.device, dtype=torch.int32)
    ext.partial_batch(
        q,
        pos,
        key_cache,
        value_cache,
        key_scale,
        value_scale,
        block_tables,
        valid_columns,
        table_rows,
        partial_acc,
        partial_m,
        partial_l,
        float(scale),
        int(split_count),
        int(logical_capacity),
    )
    ext.reduce_batch(partial_acc, partial_m, partial_l, pos, output, int(split_count))


# --- G64 prefill (warmup, chunked prefill, continuation) ---
@triton.jit
def _g64_prefill_kernel(
    q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, out_ptr,
    cu_q_ptr, seqused_ptr, table_ptr,
    tile_batch_ptr, tile_start_ptr, tile_row_ptr,
    scale,
    stride_kb, stride_ks,
    stride_qt, stride_qh,
    stride_ot, stride_oh,
    stride_tb,
    KVH: tl.constexpr, GROUPS: tl.constexpr, PAGE: tl.constexpr,
    DIM: tl.constexpr, BLOCK_N: tl.constexpr, Q_TILE: tl.constexpr,
    QH_PER_KV: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h // QH_PER_KV
    batch = tl.load(tile_batch_ptr + pid_t).to(tl.int64)
    q_start = tl.load(tile_start_ptr + pid_t).to(tl.int64)
    seq_len = tl.load(seqused_ptr + batch)
    # vLLM contract: context_len = seqused_k - current_batch_query_len; a new
    # query at row i sits at absolute position context_len + i, so a
    # continuation prefill sees the already-cached keys plus its own prefix.
    q_len_b = tl.load(cu_q_ptr + batch + 1).to(tl.int64) - tl.load(cu_q_ptr + batch).to(tl.int64)
    context_len = seq_len.to(tl.int64) - q_len_b

    q_off = tl.arange(0, Q_TILE)
    mask_q = (q_start + q_off) < q_len_b
    d = tl.arange(0, DIM)
    grp = tl.arange(0, GROUPS)

    q_row = tl.load(tile_row_ptr + pid_t).to(tl.int64) + q_off  # global rows from host
    q = tl.load(q_ptr + (q_row[:, None] * stride_qt + pid_h * stride_qh) + d[None, :],
                mask=mask_q[:, None], other=0.0).to(tl.float32)

    m_i = tl.full((Q_TILE,), float("-inf"), tl.float32)
    l_i = tl.zeros((Q_TILE,), tl.float32)
    acc = tl.zeros((Q_TILE, DIM), tl.float32)

    for n0 in range(0, seq_len, BLOCK_N):
        pos = n0 + tl.arange(0, BLOCK_N)
        mask_n = pos < seq_len
        page = tl.load(table_ptr + batch.to(tl.int64) * stride_tb + pos // PAGE,
                       mask=mask_n, other=0).to(tl.int64)
        off = pos % PAGE
        page2 = page[:, None]
        # Views are plane-based; only page stride and in-page offsets here.
        k_base = page2 * stride_kb + kv_h * DIM * PAGE + off[:, None] * DIM + d[None, :]
        v_base = page2 * stride_kb + kv_h * DIM * PAGE + off[:, None] * DIM + d[None, :]
        ks_base = page2 * stride_ks + kv_h * (GROUPS * PAGE) + off[:, None] * GROUPS + grp[None, :]
        vs_base = page2 * stride_ks + kv_h * (GROUPS * PAGE) + off[:, None] * GROUPS + grp[None, :]
        kc = tl.load(k_ptr + k_base, mask=mask_n[:, None], other=0).to(tl.float32)
        vc = tl.load(v_ptr + v_base, mask=mask_n[:, None], other=0).to(tl.float32)
        ksc = tl.load(ks_ptr + ks_base, mask=mask_n[:, None], other=0).to(tl.float32)
        vsc = tl.load(vs_ptr + vs_base, mask=mask_n[:, None], other=0).to(tl.float32)
        gsz: tl.constexpr = DIM // GROUPS
        kq = tl.reshape(kc, (BLOCK_N, GROUPS, gsz)) * ksc[:, :, None]
        vv = tl.reshape(vc, (BLOCK_N, GROUPS, gsz)) * vsc[:, :, None]
        kq = tl.reshape(kq, (BLOCK_N, DIM))
        vv = tl.reshape(vv, (BLOCK_N, DIM))
        # Tensor-core QK^T. The previous broadcast-multiply-sum materialised a
        # [Q_TILE, BLOCK_N, DIM] fp32 temporary and ran on the FP32 pipe, which
        # made prefill quadratic in wall time (22x slower than the bf16 baseline
        # at 3756 prompt tokens). tl.dot on bf16 operands uses HMMA instead.
        s = tl.dot(q.to(tl.bfloat16), tl.trans(kq.to(tl.bfloat16))).to(tl.float32) * scale
        # Causal against absolute positions, matching triton_unified_attention's
        # `dist = context_len + qpos - seq_offset >= 0` condition.
        abs_pos = context_len + q_start + q_off
        s = tl.where((pos[None, :] <= abs_pos[:, None]) & mask_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv.to(tl.bfloat16)).to(tl.float32)
        m_i = m_new
    denom = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / denom[:, None]
    tl.store(out_ptr + (q_row[:, None] * stride_ot + pid_h * stride_oh) + d[None, :],
             acc.to(tl.bfloat16), mask=mask_q[:, None])


def unified_attention_g64(q, k_cache, v_cache, k_scale, v_scale, out,
                          cu_seqlens_q, seqused_k, block_table,
                          softmax_scale, max_seqlen_q, causal=True,
                          q_tile=16):
    """Prefill-shaped G64 paged attention.

    Graph-safe: no device sync; tile tables sized by max_seqlen_q only.
    """
    B = cu_seqlens_q.shape[0] - 1
    num_qh = q.shape[1]
    tiles_per_b = triton.cdiv(max_seqlen_q, q_tile)
    n_tiles = B * tiles_per_b
    device = q.device
    # tile i covers batch i//tiles_per_b, rows [ (i%tiles)*q_tile, +q_tile )
    tile_batch = torch.arange(n_tiles, device=device, dtype=torch.int32) // tiles_per_b
    tile_start = (torch.arange(n_tiles, device=device, dtype=torch.int32) % tiles_per_b) * q_tile
    tile_row = cu_seqlens_q.to(torch.int32)[:B].repeat_interleave(tiles_per_b) + tile_start
    grid = (n_tiles, num_qh)
    _g64_prefill_kernel[grid](
        q, k_cache, v_cache, k_scale, v_scale, out,
        cu_seqlens_q.to(torch.int32), seqused_k.to(torch.int32), block_table.to(torch.int32),
        tile_batch, tile_start, tile_row,
        softmax_scale,
        k_cache.stride(0), k_scale.stride(0),
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        block_table.stride(0),
        KVH=k_cache.shape[1], GROUPS=4, PAGE=k_cache.shape[2],
        DIM=k_cache.shape[3], BLOCK_N=64, Q_TILE=q_tile,
        QH_PER_KV=num_qh // k_cache.shape[1],
    )
