"""G64 prefill v2: tiled queries, graph-safe host (no .item() on captured paths).

Query rows are tiled: one program handles Q_TILE query rows for one (batch,
q-head). The host builds per-tile (batch, q_start) tables on device so the
launch shape is a pure function of max_seqlen_q and batch — no device sync —
keeping CUDA-graph capture legal. Views are plane-based (each pointer starts
at its own plane's first element); the kernel adds only page stride and
in-page offsets, per test_g64_addr_probe.py.
"""
import torch
import triton
import triton.language as tl


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
