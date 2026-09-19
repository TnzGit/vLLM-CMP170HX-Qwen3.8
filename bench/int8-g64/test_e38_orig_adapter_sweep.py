"""Same batch sweep against the ORIGINAL contiguous E38 adapter.

Decides whether the batch>=3 decode defect is pre-existing in the E38 kernel or
was introduced by the page-stride port: this uses the untouched
`v7_verifier_e38_int8.cu` with the packed contiguous cache ABI
(`[all K][all V][all Kscales][all Vscales]`, page stride 65536), via the original
remote_int8_g64 module, so nothing from this session's port is in the path.
"""
import os
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
SRC = str(HERE / "v7_verifier_e38_int8.cu")
INCLUDE = os.environ.get("VLLM_INT8_G64_NINFER_ROOT", "")
if not INCLUDE or not pathlib.Path(INCLUDE).is_dir():
    raise SystemExit("set VLLM_INT8_G64_NINFER_ROOT to the NInfer source tree")
os.environ["VLLM_INT8_G64_SRC"] = SRC
os.environ["VLLM_INT8_G64_NINFER_ROOT"] = INCLUDE
os.environ.pop("VLLM_INT8_G64_STRIDED", None)
os.environ["VLLM_INT8_G64_EXT_NAME"] = "vllm_int8_g64_e38"

import remote_int8_g64 as g64  # noqa: E402

H, KH, D, N, GROUP = 4, 24, 256, 64, 64
SCALE = 0.0625
pages = 32
torch.manual_seed(4242)
code = pages * H * N * D
scale_bytes = pages * H * N * (D // GROUP) * 2
raw = torch.empty(2 * code + 2 * scale_bytes, dtype=torch.int8, device="cuda")
kv = raw[:code].view(pages, H, N, D)
caches = g64.g64_views(kv)
key = torch.randn(pages * N, H, D, dtype=torch.bfloat16, device="cuda")
val = torch.randn_like(key)
g64.reshape_and_cache_int8_g64(key, val, *caches,
                               torch.arange(pages * N, device="cuda", dtype=torch.int64))
torch.cuda.synchronize()
# Identity page tables: with the packed ABI, page p is simply the p-th 64-token page.
print("packed ABI: page_stride =", kv.stride(0), "(expect 65536)", flush=True)
ext = g64._load_e38()


def decode(query, tables, lengths, split_count):
    batch = tables.shape[0]
    positions = (lengths[:, None] - 8
                 + torch.arange(8, device="cuda", dtype=torch.int32)).flatten()
    acc = torch.empty((batch, split_count, 8, KH, D), device="cuda", dtype=torch.bfloat16)
    m = torch.empty((batch, split_count, 8, KH), device="cuda", dtype=torch.float32)
    l = torch.empty_like(m)
    out = torch.empty((batch, 8, KH, D), device="cuda", dtype=torch.bfloat16)
    ext.partial_batch(query, positions, *caches, tables, lengths,
                      torch.arange(batch, device="cuda", dtype=torch.int32),
                      acc, m, l, SCALE, split_count, int(lengths.max()))
    ext.reduce_batch(acc, m, l, positions, out, split_count)
    torch.cuda.synchronize()
    return out


def oracle(query, tables, lengths):
    dk_all = caches[0].float() * caches[2].float().repeat_interleave(GROUP, -1)
    dv_all = caches[1].float() * caches[3].float().repeat_interleave(GROUP, -1)
    qf = query.float()
    qs = qf.abs().view(-1, KH, 4, D // 4).amax(-1, keepdim=True).clamp_min(1e-6) / 127.0
    qdeq = ((qf.view(-1, KH, 4, D // 4) / qs).round().clamp(-127, 127) * qs).view(-1, KH, D)
    kvh = torch.arange(KH, device="cuda") // (KH // H)
    refs = []
    for b, L in enumerate(lengths):
        rows = tables[b].tolist()
        dk = torch.cat([dk_all[r].permute(1, 0, 2) for r in rows], 0)[:L]
        dv = torch.cat([dv_all[r].permute(1, 0, 2) for r in rows], 0)[:L]
        qb = qdeq[b * 8:(b + 1) * 8]
        sc = torch.einsum("jhd,nhd->jhn", qb, dk[:, kvh]) * SCALE
        pos_q = (L - 8 + torch.arange(8, device="cuda"))[:, None]
        sc = sc.masked_fill(torch.arange(L, device="cuda")[None, :][None] > pos_q[:, None],
                            float("-inf"))
        refs.append(torch.einsum("jhn,nhd->jhd", torch.softmax(sc, -1), dv[:, kvh]))
    return torch.stack(refs)


def case(label, lengths, split_count=4):
    B = len(lengths)
    cols = max(1, (max(lengths) + N - 1) // N)
    tables = torch.stack([torch.arange(b * cols, (b + 1) * cols)
                          for b in range(B)]).to(torch.int32).cuda()
    query = torch.randn(B * 8, KH, D, dtype=torch.bfloat16, device="cuda")
    got = decode(query, tables, torch.tensor(lengths, device="cuda", dtype=torch.int32),
                 split_count)
    ref = oracle(query, tables, lengths)
    errs = [(got[b].float() - ref[b]).abs().max().item() for b in range(B)]
    ok = max(errs) < 0.05 and bool(torch.isfinite(got).all())
    print(f"  {'PASS' if ok else 'FAIL'} ORIG-ADAPTER {label} split={split_count} "
          f"errs={[round(e, 4) for e in errs]}", flush=True)
    return ok


print("== original contiguous adapter, same sweep ==")
res = []
for B in (1, 2, 3, 4):
    res.append(case(f"batch{B}_len200", [200] * B))
res.append(case("batch4_mixed", [200, 137, 64, 129]))
print()
print("ORIG_ADAPTER_BATCH_SWEEP", "PASS" if all(res) else "FAIL", flush=True)
sys.exit(0 if all(res) else 3)
