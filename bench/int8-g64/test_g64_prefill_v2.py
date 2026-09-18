"""v2 prefill regression: longer sequences, causal, multi-batch, both geometries.

Query rows are tiled; this test exercises tiling (>Q_TILE tokens), causal
masking on absolute positions, and page-table indirection with 2+ pages.
"""
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from page_local_candidate import g64_views, reshape_and_cache_int8_g64
from int8_g64_prefill_v2 import unified_attention_g64

H, N, D = 4, 64, 256
for label, page_stride, base in [("padded", 1777664, 16), ("compact", 135168, 0)]:
    torch.manual_seed(7)
    B = 2
    pages_per_b = 3
    raw = torch.full((base + B * pages_per_b * 1777664,), 85, device="cuda", dtype=torch.int8)
    kv = torch.as_strided(raw, (B * pages_per_b, H, N, D), (page_stride, 16384, 256, 1), base)
    k, v, ks, vs = g64_views(kv)
    T = pages_per_b * N - 5  # 187 tokens: crosses page boundaries, not multiple of 64
    # Distinct data per batch: if the kernel read the wrong pages, the oracle
    # for batch 1 would diverge (identical data would hide the bug).
    keys = [torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(B)]
    vals = [torch.randn_like(kk) for kk in keys]
    slots = torch.cat([torch.arange(T) + b * pages_per_b * N for b in range(B)]).cuda().to(torch.int64)
    reshape_and_cache_int8_g64(torch.cat(keys), torch.cat(vals), k, v, ks, vs, slots)
    torch.cuda.synchronize()
    # prefill queries: 100 tokens for batch 0, 130 for batch 1
    q_lens = [187, 130]
    q = torch.randn(sum(q_lens), 24, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), device="cuda", dtype=torch.int32)
    seqused = torch.tensor([T, T], device="cuda", dtype=torch.int32)

    table = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)
    unified_attention_g64(q=q, k_cache=k, v_cache=v, k_scale=ks, v_scale=vs,
                          out=out, cu_seqlens_q=cu_q, seqused_k=seqused,
                          block_table=table, softmax_scale=0.0625,
                          max_seqlen_q=max(q_lens), causal=True)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all(), label
    # CPU oracle over written tokens only, per batch
    deq_k = (k.float() * ks.float().repeat_interleave(64, -1))   # [P,H,N,D]
    deq_v = (v.float() * vs.float().repeat_interleave(64, -1))
    max_err = 0.0
    kvh_all = torch.arange(24, device="cuda") // 6
    for b, qlen in enumerate(q_lens):
        rows = table[b].tolist()
        # Concatenate per-page [N,H,D] slices along the token axis. The earlier
        # stack->permute(1,0,2,3)->reshape flattened h-major and scrambled the
        # token order, which is why this test failed while the kernel was right.
        dkb = torch.cat([deq_k[pq].permute(1, 0, 2) for pq in rows], dim=0)[:T]
        dvb = torch.cat([deq_v[pq].permute(1, 0, 2) for pq in rows], dim=0)[:T]
        qh = q[cu_q[b].item():cu_q[b].item() + qlen].float()
        kvh = torch.arange(24, device="cuda") // 6
        sc = torch.einsum("thd,nhd->thn", qh, dkb[:, kvh]) * 0.0625
        # causal mask
        ctx = T - qlen  # continuing prefill: query i is at absolute position ctx+i
        pos_q = (ctx + torch.arange(qlen, device="cuda"))[:, None, None]
        pos_k = torch.arange(T, device="cuda")[None, None, :]
        sc = sc.masked_fill(pos_k > pos_q, float("-inf"))
        w = torch.softmax(sc, dim=-1)
        ref = torch.einsum("thn,nhd->thd", w, dvb[:, kvh])
        got = out[cu_q[b].item():cu_q[b].item() + qlen].float()
        err = (got - ref).abs().max().item()
        max_err = max(max_err, err)
    print(f"{label}: tiles_test max_abs={max_err:.5f} {'OK' if max_err < 0.05 else 'FAIL'}", flush=True)
    if max_err >= 0.05:
        sys.exit(3)
print("V2_TILING PASS", flush=True)
