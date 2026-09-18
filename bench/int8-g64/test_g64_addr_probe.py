"""Address probe: load the exact bytes the prefill kernel formulas produce.

Uses the same views and the same address arithmetic as int8_g64_prefill, for
page 0 / head 0 / token 0, and compares against the torch views directly.
This isolates addressing from attention math.
"""
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
import triton
import triton.language as tl
from page_local_candidate import g64_views, reshape_and_cache_int8_g64


@triton.jit
def _addr_probe(k_ptr, v_ptr, ks_ptr, vs_ptr, out_ptr,
                stride_kb, stride_ks,
                KVH: tl.constexpr, GROUPS: tl.constexpr, PAGE: tl.constexpr,
                DIM: tl.constexpr):
    # Element (page 0, head 0, token 0, d/g 0) via kernel arithmetic.
    k0 = tl.load(k_ptr + tl.program_id(0) * 0 + 0 * stride_kb + 0 * (DIM * PAGE) + 0 * DIM + 0)
    v0 = tl.load(v_ptr + 0 * stride_kb + 0 * (DIM * PAGE) + 0 * DIM + 0)
    ks0 = tl.load(ks_ptr + 0 * stride_ks + 0 * (GROUPS * PAGE) + 0 * GROUPS + 0)
    vs0 = tl.load(vs_ptr + 0 * stride_ks + 0 * (GROUPS * PAGE) + 0 * GROUPS + 0)
    tl.store(tl.static_range(0) * 0 + out_ptr_dummy, 0) if False else None
    # store via a 4-wide result vector
    res = tl.zeros((4,), dtype=tl.float32)
    res = tl.where(tl.arange(0, 4) == 0, k0.to(tl.float32), res)
    res = tl.where(tl.arange(0, 4) == 1, ks0.to(tl.float32), res)
    res = tl.where(tl.arange(0, 4) == 2, v0.to(tl.float32), res)
    res = tl.where(tl.arange(0, 4) == 3, vs0.to(tl.float32), res)
    tl.store(out_ptr + tl.arange(0, 4), res)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 1, 4, 64, 256
    raw = torch.full((B * 1777664 + 32,), 85, dtype=torch.int8, device="cuda")
    kv = torch.as_strided(raw, (B, H, N, D), (1777664, 16384, 256, 1), 16)
    k, v, ks, vs = g64_views(kv)
    val = torch.zeros(1, H, D, device="cuda", dtype=torch.bfloat16)
    val[0, 0, 0] = 1.0
    val[0, 0, 1] = 0.5
    key = torch.zeros_like(val)
    key[0, 0, 0] = 2.0
    reshape_and_cache_int8_g64(key, val, k, v, ks, vs,
                               torch.zeros(1, dtype=torch.int64, device="cuda"))
    torch.cuda.synchronize()
    out = torch.empty(4, device="cuda", dtype=torch.float32)
    _addr_probe[(1,)](k, v, ks, vs, out,
                      k.stride(0), ks.stride(0),
                      KVH=4, GROUPS=4, PAGE=64, DIM=256)
    torch.cuda.synchronize()
    k_view = k[0, 0, 0, 0].item()
    ks_view = ks[0, 0, 0, 0].item()
    v_view = v[0, 0, 0, 0].item()
    vs_view = vs[0, 0, 0, 0].item()
    print(f"probe  k={out[0].item()} ks={out[1].item()} v={out[2].item()} vs={out[3].item()}")
    print(f"view   k={k_view} ks={ks_view} v={v_view} vs={vs_view}")
    ok = (out[0].item() == k_view and out[1].item() == ks_view
          and out[2].item() == v_view and out[3].item() == vs_view)
    print("ADDR_PROBE", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 3)
