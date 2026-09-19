"""Standalone E38 decode sweep: batch 1..4, mixed per-request lengths.

Separates the three open suspects for the batch>=3 G64 degeneracy without
starting the engine:

  * suspect 2 (positions / valid_columns pairing): a per-request oracle plus a
    cross-match test that flags request i matching request j's expectation, the
    aliasing signature seen at batch 4;
  * suspect 3 (split_count derived from a batch-wide max_seqlen_k): the sweep
    varies lengths widely and re-runs the same batch at split_count 4/8/16;
  * suspect 1 (workspace reallocation): this harness reuses caller-owned buffers
    of one fixed size, so a clean sweep here points at allocation churn instead.

Oracle: softmax attention with Q emulated as int8-per-64-group (the E38 kernel
quantizes Q the same way), over dequantized K/V read back from the cache views.
"""
import importlib.util
import os
import pathlib
import sys

import torch

MODULE = pathlib.Path(sys.argv[1]).resolve()
SRC = sys.argv[2]
INCLUDE = sys.argv[3]
os.environ["VLLM_INT8_G64_SRC"] = SRC
os.environ["VLLM_INT8_G64_NINFER_ROOT"] = INCLUDE
os.environ["VLLM_INT8_G64_EXT_NAME"] = "vllm_int8_g64_e38_strided"

spec = importlib.util.spec_from_file_location("cand", MODULE)
g64 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g64)

H, KH, D, N, GROUP = 4, 24, 256, 64, 64
PADDED = 1777664
SCALE = 0.0625
pages = 32
torch.manual_seed(4242)
raw = torch.full((16 + pages * PADDED,), 85, dtype=torch.int8, device="cuda")
kv = torch.as_strided(raw, (pages, H, N, D), (PADDED, N * D, D, 1), 16)
caches = g64.g64_views(kv)
key = torch.randn(pages * N, H, D, dtype=torch.bfloat16, device="cuda")
val = torch.randn_like(key)
g64.reshape_and_cache_int8_g64(key, val, *caches,
                               torch.arange(pages * N, device="cuda", dtype=torch.int64))
torch.cuda.synchronize()
ext = g64._load_e38()


def decode(query, tables, lengths, split_count, sentinel=False):
    batch = tables.shape[0]
    positions = (lengths[:, None] - 8
                 + torch.arange(8, device="cuda", dtype=torch.int32)).flatten()
    acc = torch.empty((batch, split_count, 8, KH, D), device="cuda", dtype=torch.bfloat16)
    m = torch.empty((batch, split_count, 8, KH), device="cuda", dtype=torch.float32)
    l = torch.empty_like(m)
    out = torch.empty((batch, 8, KH, D), device="cuda", dtype=torch.bfloat16)
    if sentinel:
        # Online-softmax identity: an unwritten split then contributes nothing.
        # If this alone fixes a case, the kernel failed to cover every
        # (batch, split) pair and the reduce read uninitialised memory.
        acc.zero_(); m.fill_(float("-inf")); l.zero_()
    ext.partial_batch(query, positions, *caches, tables, lengths,
                      torch.arange(batch, device="cuda", dtype=torch.int32),
                      acc, m, l, SCALE, split_count, int(lengths.max()))
    ext.reduce_batch(acc, m, l, positions, out, split_count)
    torch.cuda.synchronize()
    return out


def oracle(query, tables, lengths):
    """Per-request causal oracle over the dequantized cache, Q int8-emulated."""
    _, v, _, vs = caches
    dk_all = (caches[0].float() * caches[2].float().repeat_interleave(GROUP, -1))
    dv_all = (v.float() * vs.float().repeat_interleave(GROUP, -1))
    qf = query.float()
    qs = qf.abs().view(-1, KH, 4, D // 4).amax(-1, keepdim=True).clamp_min(1e-6) / 127.0
    qcode = (qf.view(-1, KH, 4, D // 4) / qs).round().clamp(-127, 127)
    qdeq = (qcode * qs).view(-1, KH, D)
    kvh = torch.arange(KH, device="cuda") // (KH // H)
    refs = []
    for b, L in enumerate(lengths):
        rows = tables[b].tolist()
        dk = torch.cat([dk_all[r].permute(1, 0, 2) for r in rows], 0)[:L]
        dv = torch.cat([dv_all[r].permute(1, 0, 2) for r in rows], 0)[:L]
        qb = qdeq[b * 8:(b + 1) * 8]
        sc = torch.einsum("jhd,nhd->jhn", qb, dk[:, kvh]) * SCALE
        pos_q = (L - 8 + torch.arange(8, device="cuda"))[:, None]
        pos_k = torch.arange(L, device="cuda")[None, :]
        sc = sc.masked_fill(pos_k[None] > pos_q[:, None], float("-inf"))
        refs.append(torch.einsum("jhn,nhd->jhd", torch.softmax(sc, -1), dv[:, kvh]))
    return torch.stack(refs)


def case(label, lengths, split_count=4, sentinel=False):
    B = len(lengths)
    cols = max(1, (max(lengths) + N - 1) // N)
    tables = torch.stack([torch.arange(b * cols, (b + 1) * cols)
                          for b in range(B)]).to(torch.int32).cuda()
    query = torch.randn(B * 8, KH, D, dtype=torch.bfloat16, device="cuda")
    got = decode(query, tables, torch.tensor(lengths, device="cuda", dtype=torch.int32),
                 split_count, sentinel=sentinel)
    ref = oracle(query, tables, lengths)
    errs = [(got[b].float() - ref[b]).abs().max().item() for b in range(B)]
    # Aliasing: does request i match request j's expectation better than its own?
    alias = []
    for i in range(B):
        own = (got[i].float() - ref[i]).abs().max().item()
        for j in range(B):
            if i != j and (got[i].float() - ref[j]).abs().max().item() < own / 4:
                alias.append(f"{i}->{j}")
    ok = max(errs) < 0.05 and not alias and bool(torch.isfinite(got).all())
    print(f"  {'PASS' if ok else 'FAIL'} {label} split={split_count} "
          f"errs={[round(e, 4) for e in errs]}"
          + (f" ALIAS={alias}" if alias else ""), flush=True)
    if not ok and not sentinel:
        got2 = decode(query, tables,
                      torch.tensor(lengths, device="cuda", dtype=torch.int32),
                      split_count, sentinel=True)
        ref2 = oracle(query, tables, lengths)
        errs2 = [(got2[b].float() - ref2[b]).abs().max().item() for b in range(B)]
        fixed = max(errs2) < 0.05
        print(f"       sentinel-init retry: errs={[round(e, 4) for e in errs2]} "
              f"-> {'COVERAGE BUG (uninitialised split read)' if fixed else 'not explained by init'}",
              flush=True)
    return ok


print("== equal lengths ==")
results = []
for B in (1, 2, 3, 4):
    results.append(case(f"batch{B}_len200", [200] * B))
print("== mixed per-request lengths ==")
results.append(case("batch2_mixed", [200, 137]))
results.append(case("batch3_mixed", [200, 137, 64]))
results.append(case("batch4_mixed", [200, 137, 64, 129]))
results.append(case("batch4_tail_equal", [200, 137, 131, 130]))
print("== split_count sensitivity (suspect 3) ==")
for sc in (1, 4, 8, 16):
    results.append(case("batch4_mixed", [200, 137, 64, 129], split_count=sc))
print()
if not all(results):
    print("E38_DECODE_SWEEP FAIL", flush=True)
    sys.exit(3)
print("E38_DECODE_SWEEP PASS", flush=True)
