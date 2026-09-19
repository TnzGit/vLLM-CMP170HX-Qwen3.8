"""E38 consumer regression: padded pages must match contiguous cache and oracle."""
import argparse
import importlib.util
import os
import torch
from page_local_candidate import g64_views, reshape_and_cache_int8_g64


def run(ext, caches, query, tables, lengths):
    batch = tables.shape[0]
    positions = (lengths[:, None] - 8 + torch.arange(8, device='cuda', dtype=torch.int32)).flatten()
    acc = torch.empty((batch, 4, 8, 24, 256), device='cuda', dtype=torch.bfloat16)
    m = torch.empty((batch, 4, 8, 24), device='cuda', dtype=torch.float32)
    l = torch.empty_like(m)
    out = torch.empty((batch, 8, 24, 256), device='cuda', dtype=torch.bfloat16)
    ext.partial_batch(query, positions, *caches, tables, lengths,
                      torch.arange(batch, device='cuda', dtype=torch.int32),
                      acc, m, l, 0.0625, 4, 128)
    ext.reduce_batch(acc, m, l, positions, out, 4)
    torch.cuda.synchronize()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('include')
    args = ap.parse_args()
    os.environ['VLLM_INT8_G64_SRC'] = args.source
    os.environ['VLLM_INT8_G64_STRIDED'] = '1'
    os.environ['VLLM_INT8_G64_NINFER_ROOT'] = args.include
    from remote_int8_g64 import _load_e38
    torch.manual_seed(1729)
    ext = _load_e38()
    raw = torch.full((16 + 4 * 1777664,), 85, device='cuda', dtype=torch.int8)
    kv = torch.as_strided(raw, (4, 4, 64, 256), (1777664, 16384, 256, 1), 16)
    caches = g64_views(kv)
    key = torch.randn(256, 4, 256, device='cuda', dtype=torch.bfloat16)
    value = torch.randn_like(key)
    reshape_and_cache_int8_g64(key, value, *caches, torch.arange(256, device='cuda', dtype=torch.int64))
    packed = tuple(t.contiguous() for t in caches)
    # Nonidentity page tables force physical-page addressing beyond block zero.
    tables = torch.tensor([[2, 0], [3, 1]], device='cuda', dtype=torch.int32)
    lengths = torch.tensor([128, 97], device='cuda', dtype=torch.int32)
    query = torch.zeros((16, 24, 256), device='cuda', dtype=torch.bfloat16)
    baseline = run(ext, packed, query, tables, lengths)
    actual = run(ext, caches, query, tables, lengths)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    _, v, _, vs = packed
    dequant_v = v.float() * vs.float().repeat_interleave(64, -1)
    refs = []
    for row, length in zip(tables.tolist(), lengths.tolist()):
        history = dequant_v[row].permute(0, 2, 1, 3).reshape(128, 4, 256)
        refs.append(torch.stack([history[:pos + 1].mean(0).repeat_interleave(6, 0)
                                for pos in range(length - 8, length)]))
    reference = torch.stack(refs)
    err = (actual.float() - reference).abs().max().item()
    assert err < 0.005, f'zero-query mean-V oracle max_abs={err}'
    print(f'PASS zero-Q oracle: max_abs={err}; padded/contiguous bitwise equal', flush=True)
    query.normal_()
    baseline = run(ext, packed, query, tables, lengths)
    actual = run(ext, caches, query, tables, lengths)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    assert torch.isfinite(actual).all()
    print('PASS nonzero-Q two-request mixed-length shuffled-page ABI parity', flush=True)
