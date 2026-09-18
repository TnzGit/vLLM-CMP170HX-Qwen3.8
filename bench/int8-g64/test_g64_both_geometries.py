"""Both allocator geometries must pass the standalone prefill oracle."""
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from page_local_candidate import g64_views, reshape_and_cache_int8_g64
from int8_g64_prefill import unified_attention_g64

T, H, N, D = 16, 4, 64, 256
B = 3
kv_head_of_q = torch.arange(24, device="cuda") // 6
for label, page_stride, base in [("padded", 1777664, 16), ("compact", 135168, 0)]:
    torch.manual_seed(11)
    raw = torch.full((base + B * 1777664,), 85, device="cuda", dtype=torch.int8)
    kv = torch.as_strided(raw, (B, H, N, D), (page_stride, 16384, 256, 1), base)
    k, v, ks, vs = g64_views(kv)
    key = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    reshape_and_cache_int8_g64(key, value, k, v, ks, vs,
                               torch.arange(T, device="cuda", dtype=torch.int64))
    torch.cuda.synchronize()
    q = torch.randn(1, 24, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, 24, D, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seqused = torch.tensor([T], device="cuda", dtype=torch.int32)
    table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    unified_attention_g64(q=q, k_cache=k, v_cache=v, k_scale=ks, v_scale=vs,
                          out=out, cu_seqlens_q=cu_q, seqused_k=seqused,
                          block_table=table, softmax_scale=0.0625, causal=True)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    dk = (k[0].float() * ks[0].float().repeat_interleave(64, -1)).permute(1, 0, 2)[:T]
    dv = (v[0].float() * vs[0].float().repeat_interleave(64, -1)).permute(1, 0, 2)[:T]
    scores = torch.einsum("hd,nhd->hn", q[0].float(), dk[:, kv_head_of_q]) * 0.0625
    ref = torch.softmax(scores, dim=-1)
    ref = torch.einsum("hn,nhd->hd", ref, dv[:, kv_head_of_q])
    err = (out[0].float() - ref).abs().max().item()
    print(f"{label}: max_abs={err:.5f} {'OK' if err < 0.05 else 'FAIL'}", flush=True)
    if err >= 0.05:
        sys.exit(3)
print("BOTH_GEOMETRIES PASS", flush=True)
