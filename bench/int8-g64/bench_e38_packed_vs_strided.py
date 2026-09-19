"""Is the strided page addressing the 17.6x? Time the same E38 work both ways.

Arm A: packed contiguous cache + the untouched v7_verifier_e38_int8.cu, whose
       page arithmetic folds kPagedKVPageSize=64 as a compile-time constant.
Arm B: page-local padded cache + page_stride_adapter.cu, whose page arithmetic is
       a runtime int64 multiply (this session's change).

Identical shapes, identical token data, warmup + many iterations, per-call ms.
If B is orders of magnitude slower than A, the addressing change is the cause;
if they are close, the slowdown lives elsewhere (parallelism, split_count, or
outside attention) and section 22's ranking is wrong.
"""
import importlib.util
import os
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
NINFER = os.environ.get("VLLM_INT8_G64_NINFER_ROOT", "")
if not NINFER or not pathlib.Path(NINFER).is_dir():
    raise SystemExit(
        "set VLLM_INT8_G64_NINFER_ROOT to the NInfer source tree that contains "
        "ops/kernel/ (the E38 kernel includes resolve against it)")
H, KH, D, N, GROUP = 4, 24, 256, 64, 64
PAYLOAD = 135168
PADDED = 1777664
SCALE = 0.0625

os.environ["VLLM_INT8_G64_NINFER_ROOT"] = NINFER


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_cache(module, pages, page_stride, base, seed):
    """Packed ABI: the key view spans only the K plane of a global-plane layout.
    Page-local ABI: one padded page per block, four planes inside each page."""
    torch.manual_seed(seed)
    if page_stride == 65536:
        code = pages * H * N * D
        scale_bytes = pages * H * N * (D // GROUP) * 2
        raw = torch.empty(2 * code + 2 * scale_bytes, dtype=torch.int8, device="cuda")
        kv = raw[:code].view(pages, H, N, D)
    else:
        raw = torch.full((base + pages * page_stride,), 85, dtype=torch.int8, device="cuda")
        kv = torch.as_strided(raw, (pages, H, N, D),
                              (page_stride, N * D, D, 1), base)
    caches = module.g64_views(kv)
    key = torch.randn(pages * N, H, D, dtype=torch.bfloat16, device="cuda")
    val = torch.randn_like(key)
    module.reshape_and_cache_int8_g64(
        key, val, *caches, torch.arange(pages * N, device="cuda", dtype=torch.int64))
    torch.cuda.synchronize()
    return caches


def timeit(ext, caches, batch, lengths, split_count, iters=200):
    cols = max(1, (max(lengths) + N - 1) // N)
    tables = torch.stack([torch.arange(b * cols, (b + 1) * cols)
                          for b in range(batch)]).to(torch.int32).cuda()
    lengths_t = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    q = torch.randn(batch * 8, KH, D, dtype=torch.bfloat16, device="cuda")
    positions = (lengths_t[:, None] - 8
                 + torch.arange(8, device="cuda", dtype=torch.int32)).flatten()
    acc = torch.empty((batch, split_count, 8, KH, D), dtype=torch.bfloat16, device="cuda")
    m = torch.empty((batch, split_count, 8, KH), dtype=torch.float32, device="cuda")
    l = torch.empty_like(m)
    out = torch.empty((batch, 8, KH, D), dtype=torch.bfloat16, device="cuda")
    rows = torch.arange(batch, device="cuda", dtype=torch.int32)
    cap = int(lengths_t.max())

    def once():
        ext.partial_batch(q, positions, *caches, tables, lengths_t, rows,
                          acc, m, l, SCALE, split_count, cap)
        ext.reduce_batch(acc, m, l, positions, out, split_count)

    for _ in range(20):
        once()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        once()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


print("==", "loading extensions", flush=True)
os.environ["VLLM_INT8_G64_SRC"] = str(HERE / "v7_verifier_e38_int8.cu")
os.environ["VLLM_INT8_G64_EXT_NAME"] = "vllm_int8_g64_e38"
os.environ.pop("VLLM_INT8_G64_STRIDED", None)
sys.path.insert(0, str(HERE))
import remote_int8_g64 as packed_mod  # noqa: E402

packed_ext = packed_mod._load_e38()

os.environ["VLLM_INT8_G64_SRC"] = str(HERE / "page_stride_adapter.cu")
os.environ["VLLM_INT8_G64_EXT_NAME"] = "vllm_int8_g64_e38_strided"
strided_mod = load_module("cand_int8_g64", HERE / "int8_g64.py")
strided_ext = strided_mod._load_e38()

PAGES = 256  # keeps batch4 x 3800 tokens inside the page table
packed = make_cache(packed_mod, PAGES, 65536, 0, 7)
strided = make_cache(strided_mod, PAGES, PADDED, 16, 7)

print(f"{'case':<28}{'packed ms':>12}{'strided ms':>12}{'ratio':>9}", flush=True)
for batch, length, split_count in [(1, 3800, 4), (4, 3800, 4), (1, 128, 4),
                                   (1, 3800, 16), (1, 3800, 32), (4, 3800, 32)]:
    lengths = [length] * batch
    a = timeit(packed_ext, packed, batch, lengths, split_count)
    b = timeit(strided_ext, strided, batch, lengths, split_count)
    print(f"b{batch} len{length} split{split_count:<6}{a:>12.4f}{b:>12.4f}{b / a:>9.2f}x",
          flush=True)
