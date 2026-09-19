"""Prefill at batch 3 and 4 — the batch sizes the engine degenerates on.

The engine's G64 runs produce identical garbage for the tail requests at batch
>= 3 while a bf16 baseline is clean, so this closes the component-level gap:
every earlier prefill test used B=2 only. Equal-length prompts match the engine
pattern; a continuation shape (seqused_k > q_len) covers chunked prefill.
"""
import importlib.util
import pathlib
import sys

import torch

MODULE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "int8_g64.py").resolve()
spec = importlib.util.spec_from_file_location("cand", MODULE)
g64 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g64)

H, N, D, GROUP = 4, 64, 256, 64
PADDED = 1777664
kvh = torch.arange(24, device="cuda") // 6
failures = []


def run_case(label, q_lens, seqused_extra=0):
    """seqused_extra>0 makes it a continuation prefill (cached prefix)."""
    B = len(q_lens)
    pages_per_b = max(1, (max(q_lens) + seqused_extra + N - 1) // N)
    torch.manual_seed(100 + B)
    raw = torch.full((16 + B * pages_per_b * PADDED,), 85, dtype=torch.int8, device="cuda")
    kv = torch.as_strided(raw, (B * pages_per_b, H, N, D), (PADDED, N * D, D, 1), 16)
    k, v, ks, vs = g64.g64_views(kv)
    seqused = [q + seqused_extra for q in q_lens]
    keys, vals, slots = [], [], []
    for b in range(B):
        Tb = seqused[b]
        keys.append(torch.randn(Tb, H, D, dtype=torch.bfloat16, device="cuda"))
        vals.append(torch.randn_like(keys[-1]))
        slots.append(torch.arange(Tb) + b * pages_per_b * N)
    g64.reshape_and_cache_int8_g64(torch.cat(keys), torch.cat(vals), k, v, ks, vs,
                                   torch.cat(slots).cuda().to(torch.int64))
    torch.cuda.synchronize()

    q = torch.cat([torch.randn(L, 24, D, dtype=torch.bfloat16, device="cuda") for L in q_lens])
    out = torch.empty_like(q)
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()),
                      device="cuda", dtype=torch.int32)
    table = torch.stack([torch.arange(b * pages_per_b, (b + 1) * pages_per_b)
                         for b in range(B)]).to(torch.int32).cuda()
    g64.unified_attention_g64(
        q=q, k_cache=k, v_cache=v, k_scale=ks, v_scale=vs, out=out,
        cu_seqlens_q=cu,
        seqused_k=torch.tensor(seqused, device="cuda", dtype=torch.int32),
        block_table=table, softmax_scale=0.0625, max_seqlen_q=max(q_lens), causal=True)
    torch.cuda.synchronize()

    worst, per_batch = 0.0, []
    for b, L in enumerate(q_lens):
        rows = list(range(b * pages_per_b, (b + 1) * pages_per_b))
        Tb = seqused[b]
        dk = torch.cat([(k[r].float() * ks[r].float().repeat_interleave(GROUP, -1)).permute(1, 0, 2)
                        for r in rows], dim=0)[:Tb]
        dv = torch.cat([(v[r].float() * vs[r].float().repeat_interleave(GROUP, -1)).permute(1, 0, 2)
                        for r in rows], dim=0)[:Tb]
        qb = q[cu[b].item():cu[b].item() + L].float()
        sc = torch.einsum("thd,nhd->thn", qb, dk[:, kvh]) * 0.0625
        ctx = Tb - L
        sc = sc.masked_fill(torch.arange(Tb, device="cuda")[None, None, :]
                            > (ctx + torch.arange(L, device="cuda"))[:, None, None], float("-inf"))
        ref = torch.einsum("thn,nhd->thd", torch.softmax(sc, -1), dv[:, kvh])
        err = (out[cu[b].item():cu[b].item() + L].float() - ref).abs().max().item()
        per_batch.append(round(err, 5))
        worst = max(worst, err)
    ok = worst < 0.05
    print(f"  {'PASS' if ok else 'FAIL'} {label} per_batch={per_batch}", flush=True)
    if not ok:
        failures.append(label)


print("== equal-length prompts (engine pattern) ==")
for B in (3, 4):
    run_case(f"batch{B}_equal12", [12] * B)
    run_case(f"batch{B}_equal1 (decode-sized)", [1] * B)
print("== continuation prefill (cached prefix, seqused_k > q_len) ==")
run_case("batch4_cont12_of_40", [12] * 4, seqused_extra=28)
run_case("batch3_cont8_of_40", [8] * 3, seqused_extra=32)
print("== uneven lengths ==")
run_case("batch3_uneven", [5, 12, 20])
run_case("batch4_uneven", [5, 12, 20, 7])

print()
if failures:
    print("BATCH34_PREFILL FAIL:", failures, flush=True)
    sys.exit(3)
print("BATCH34_PREFILL PASS", flush=True)
