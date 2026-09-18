"""GPU regression: allocator page strides must survive view recovery and writes.

Catches global-plane offsets used with page-local allocations, aliasing padding,
missing storage-offset bounds, and writes to invalid slots. No model is loaded.
"""
import argparse
import importlib.util
import json
import torch


def load_module(path):
    spec = importlib.util.spec_from_file_location("g64_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_case(module, page_stride, base_offset):
    # Each page is [K:65536][V:65536][Ks:2048][Vs:2048][padding].
    # Prefix/suffix guards deliberately exercise nonzero storage offsets.
    used = 135168
    raw = torch.full((base_offset + 3 * page_stride + 32,), 85,
                     device="cuda", dtype=torch.int8)
    kv = torch.as_strided(raw, (3, 4, 64, 256),
                          (page_stride, 16384, 256, 1), base_offset)
    k, v, ks, vs = module.g64_views(kv)
    for view, offset in zip((k, v, ks, vs), (0, 65536, 131072, 133120)):
        assert view.data_ptr() - raw.data_ptr() == base_offset + offset
        assert view.stride(0) * view.element_size() == page_stride
    # Distinct pages, heads, tokens and groups; all lanes in each group equal
    # ±127*scale, so the exact quantized codes are independently predictable.
    slots = torch.tensor([0, 63, 64, 127, 128, 191, -1], device="cuda", dtype=torch.int64)
    index = torch.arange(7 * 4 * 4, device="cuda").view(7, 4, 4)
    scales = (2.0 ** (index % 4).float() / 128).half()
    sign = torch.where(index % 2 == 0, 1, -1)
    key = (127 * scales.float() * sign).repeat_interleave(64, -1).bfloat16()
    value = (-key * 2).bfloat16()
    module.reshape_and_cache_int8_g64(key, value, k, v, ks, vs, slots)
    torch.cuda.synchronize()
    for t, slot in enumerate(slots.tolist()[:-1]):
        block, token = divmod(slot, 64)
        torch.testing.assert_close(k[block, :, token], (127 * sign[t]).repeat_interleave(64, -1).to(torch.int8), rtol=0, atol=0)
        torch.testing.assert_close(v[block, :, token], (-127 * sign[t]).repeat_interleave(64, -1).to(torch.int8), rtol=0, atol=0)
        torch.testing.assert_close(ks[block, :, token], scales[t], rtol=0, atol=0)
        torch.testing.assert_close(vs[block, :, token], scales[t] * 2, rtol=0, atol=0)
    # Only the first and final token in every head/page may be modified.
    touched = torch.zeros_like(raw, dtype=torch.bool)
    for slot in slots.tolist()[:-1]:
        block, token = divmod(slot, 64)
        for head in range(4):
            for start, width in ((0, 256), (65536, 256), (131072, 8), (133120, 8)):
                offset = base_offset + block * page_stride + start + (head * 64 + token) * width
                touched[offset:offset + width] = True
    assert torch.all(raw[~touched] == 85), "writer changed unused token, padding, or guard bytes"
    print(json.dumps({"pass": "views_and_writer", "stride_bytes": page_stride,
                      "offset_bytes": base_offset, "used_page_bytes": used}), flush=True)


def check_invalid(module):
    for label, size, stride, offset in [
        ("truncated", 135168 * 2 - 2, 135168, 0),
        ("nonzero_offset_overrun", 135168 * 2, 135168, 16),
        ("overlapping_pages", 135168 * 2, 65536, 0),
        ("unaligned_scales", 135168 * 2 + 64, 135169, 0),
    ]:
        raw = torch.empty(size, device="cuda", dtype=torch.int8)
        kv = torch.as_strided(raw, (2, 4, 64, 256), (stride, 16384, 256, 1), offset)
        try:
            module.g64_views(kv)
        except (ValueError, RuntimeError):
            print(json.dumps({"pass": "rejected_invalid", "case": label}), flush=True)
        else:
            raise AssertionError(f"accepted invalid storage contract: {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module")
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)
    module = load_module(args.module)
    for stride in (1777664, 135168):
        for offset in (0, 16):
            check_case(module, stride, offset)
    check_invalid(module)
    print("PASS: GPU page-layout regression", flush=True)
