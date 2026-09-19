"""Verify the exact module that will be installed into the isolated runtime.

Imports runtime-candidate/int8_g64.py by path (not the audit scratch modules) so
the artifact under test is byte-identical to the deployed one:
  1. page-local views on the allocator's padded page stride
  2. writer round-trip with sentinel bytes proving no page bleed
  3. G64 prefill numerics against a dequantized-cache oracle
  4. invalid (overlapping / misaligned) pages must still be rejected
"""
import importlib.util
import pathlib
import sys

import torch

MODULE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "int8_g64.py").resolve()
spec = importlib.util.spec_from_file_location("candidate_int8_g64", MODULE)
g64 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g64)

H, N, D, GROUP = 4, 64, 256, 64
PAYLOAD = 2 * N * H * D + 2 * N * H * GROUP * 2  # 135168
PADDED = 1777664
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


print("== 1. page-local views ==")
torch.manual_seed(5)
B, BASE = 3, 16
raw = torch.full((BASE + B * PADDED,), 85, dtype=torch.int8, device="cuda")
kv = torch.as_strided(raw, (B, H, N, D), (PADDED, N * D, D, 1), BASE)
k, v, ks, vs = g64.g64_views(kv)
check("key view is the allocator view", k.data_ptr() == kv.data_ptr())
check("value plane at +65536", v.data_ptr() - k.data_ptr() == 65536)
check("kscale plane at +131072", ks.data_ptr() - kv.untyped_storage().data_ptr() - BASE == 131072)
check("vscale plane at +133120", vs.data_ptr() - kv.untyped_storage().data_ptr() - BASE == 133120)
check("all four share the page stride", {k.stride(0), v.stride(0), ks.stride(0), vs.stride(0)}
      == {PADDED, PADDED // 2}, f"strides={k.stride(0)},{ks.stride(0)}")

print("== 2. writer round-trip and page isolation ==")
TORCH_T = 20
key = torch.randn(TORCH_T, H, D, dtype=torch.bfloat16, device="cuda")
val = torch.randn_like(key)
g64.reshape_and_cache_int8_g64(key, val, k, v, ks, vs,
                               torch.arange(TORCH_T, device="cuda", dtype=torch.int64))
torch.cuda.synchronize()
rec_k = (k[0].float() * ks[0].float().repeat_interleave(GROUP, -1)).permute(1, 0, 2)[:TORCH_T]
rec_v = (v[0].float() * vs[0].float().repeat_interleave(GROUP, -1)).permute(1, 0, 2)[:TORCH_T]
check("K codes/scales decode to the input",
      (rec_k - key.float()).abs().max().item() < 0.02, f"{(rec_k - key.float()).abs().max().item():.5f}")
check("V codes/scales decode to the input",
      (rec_v - val.float()).abs().max().item() < 0.02, f"{(rec_v - val.float()).abs().max().item():.5f}")
flat = raw.flatten()
page0 = flat[BASE:BASE + PADDED]
check("page 1 untouched (no page bleed)", bool((page0[PAYLOAD:] == 85).all()),
      f"non-sentinel={int((page0[PAYLOAD:] != 85).sum())}")
check("page 1 K plane still sentinel",
      bool((flat[BASE + PADDED:BASE + PADDED + 1024] == 85).all()))

print("== 3. G64 prefill numerics ==")
pages_per_b = 2
raw2 = torch.full((BASE + 2 * pages_per_b * PADDED,), 85, dtype=torch.int8, device="cuda")
kv2 = torch.as_strided(raw2, (2 * pages_per_b, H, N, D), (PADDED, N * D, D, 1), BASE)
k2, v2, ks2, vs2 = g64.g64_views(kv2)
T = pages_per_b * N - 3  # 125 tokens: crosses a page boundary
torch.manual_seed(9)
keys = [torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
vals = [torch.randn_like(x) for x in keys]
slots = torch.cat([torch.arange(T) + b * pages_per_b * N for b in range(2)]).cuda().to(torch.int64)
g64.reshape_and_cache_int8_g64(torch.cat(keys), torch.cat(vals), k2, v2, ks2, vs2, slots)
torch.cuda.synchronize()
q_lens = [70, 55]
q = torch.randn(sum(q_lens), 24, D, dtype=torch.bfloat16, device="cuda")
out = torch.empty_like(q)
cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], device="cuda", dtype=torch.int32)
g64.unified_attention_g64(
    q=q, k_cache=k2, v_cache=v2, k_scale=ks2, v_scale=vs2, out=out,
    cu_seqlens_q=cu_q,
    seqused_k=torch.tensor([T, T], device="cuda", dtype=torch.int32),
    block_table=torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32),
    softmax_scale=0.0625, max_seqlen_q=max(q_lens), causal=True)
torch.cuda.synchronize()
check("prefill output is finite", bool(torch.isfinite(out).all()))
kvh = torch.arange(24, device="cuda") // 6
worst = 0.0
for b, qlen in enumerate(q_lens):
    rows = [b * pages_per_b, b * pages_per_b + 1]
    dk = torch.cat([(k2[r].float() * ks2[r].float().repeat_interleave(GROUP, -1)).permute(1, 0, 2)
                    for r in rows], dim=0)[:T]
    dv = torch.cat([(v2[r].float() * vs2[r].float().repeat_interleave(GROUP, -1)).permute(1, 0, 2)
                    for r in rows], dim=0)[:T]
    qb = q[cu_q[b].item():cu_q[b].item() + qlen].float()
    sc = torch.einsum("thd,nhd->thn", qb, dk[:, kvh]) * 0.0625
    # vLLM contract: new query i sits at absolute position (seqused_k - q_len) + i.
    ctx = T - qlen
    sc = sc.masked_fill(torch.arange(T, device="cuda")[None, None, :]
                        > (ctx + torch.arange(qlen, device="cuda"))[:, None, None],
                        float("-inf"))
    ref = torch.einsum("thn,nhd->thd", torch.softmax(sc, -1), dv[:, kvh])
    err = (out[cu_q[b].item():cu_q[b].item() + qlen].float() - ref).abs().max().item()
    worst = max(worst, err)
    check(f"batch {b} prefill matches oracle", err < 0.05, f"max_abs={err:.5f}")

print("== 4. invalid pages still rejected ==")
for name, stride0, base in [("overlapping_pages", 100000, 16), ("unaligned_scale_base", PADDED, 1)]:
    try:
        bad_raw = torch.zeros(stride0 * 2 + 64 + base, dtype=torch.int8, device="cuda")
        bad = torch.as_strided(bad_raw, (2, H, N, D), (stride0, N * D, D, 1), base)
        g64.g64_views(bad)
        check(f"rejected {name}", False, "no error raised")
    except ValueError:
        check(f"rejected {name}", True)

print()
if failures:
    print("MERGED_BRIDGE FAIL:", failures, flush=True)
    sys.exit(3)
print("MERGED_BRIDGE PASS", flush=True)
