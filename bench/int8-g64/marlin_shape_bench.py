#!/usr/bin/env python3
"""Exact-shape Marlin microbench: W4A16 vs W4A8-INT8 on the hot GEMMs.

WHY THIS EXISTS
---------------
If the dominant GEMM gains only 3-5%, no engine integration produces a whole-step
win; if it gains 15-30% it is worth integrating. The shapes are the ones the 126K
profile found dominant (docs/cmp170hx-mixed-fp8-engineering.md):

    gate_up  M=16, N=34816, K=5120   (64 calls, 6.111 ms)
    down     M=16, N=5120,  K=17408  (64 calls, 3.139 ms)

WHAT WAS WRONG BEFORE
---------------------
An earlier revision hand-rolled the INT4 packing and was wrong three ways
(handover 35.2): it packed two nibbles along N where vLLM packs eight 4-bit
values along K; it passed a `device=` argument that 0.27.1's
`gptq_marlin_repack` does not accept; and it derived the INT8 activation scale
*after* overwriting the tensor with the quantized values, collapsing the scale
to ~1. It is replaced by this file, which calls only verified entry points:

  * `apply_gptq_marlin_linear` -- the function the engine itself calls, so the
    measured path is the production path (it handles padding, repacking and the
    W4A8 activation quantisation internally);
  * that function's own `input_dtype` handling, which routes to
    `marlin_quant_input` for the int8 activation case -- so the activation ABI is
    vLLM's, not ours.

The kernel-only delta and the activation-quant overhead are reported separately,
because the review's gate is ">=10% on the dominant GEMM and >=8% after the
activation-quant overhead is counted".

USAGE
-----
  marlin_shape_bench.py --model <AutoRound W4A16 checkpoint> --layer 3
  marlin_shape_bench.py --model <...> --layer 3 --input-dtype int8

`--input-dtype int8` is the W4A8 arm; the default is the W4A16 baseline. The
same weights are used for both, so the only difference is the activation path --
which is exactly the comparison the review asked for, and it avoids the
"is a synthetic weight representative of production packing" question entirely.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch


def timeit(fn, iters: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def load_layer_weights(model_path: str, layer: int, device: str):
    """Pull the packed GPTQ/Marlin tensors for one layer's gate_up and down."""
    from safetensors import safe_open

    prefixes = {
        "gate_up": [f"model.layers.{layer}.mlp.gate_proj",
                    f"model.layers.{layer}.mlp.up_proj"],
        "down": [f"model.layers.{layer}.mlp.down_proj"],
    }
    suffixes = ("qweight", "scales", "qzeros", "g_idx")
    got: dict[str, dict[str, torch.Tensor]] = {k: {} for k in prefixes}
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise SystemExit(f"no safetensors under {model_path}")
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            keys = set(f.keys())
            for group, plist in prefixes.items():
                for p in plist:
                    for suf in suffixes:
                        name = f"{p}.{suf}"
                        if name in keys and f"{p}.{suf}" not in got[group]:
                            got[group][f"{p}.{suf}"] = f.get_tensor(name).to(device)
    return got


def bench_shape(entry: dict, m: int, iters: int, input_dtype, device: str):
    """Time apply_gptq_marlin_linear on a real gate_up pair (or the down proj)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
    )
    from vllm.scalar_type import scalar_types

    qw = entry["qweight"]
    size_k, size_n = None, None
    # Marlin packs 8 x 4-bit along K, so logical K = qweight.shape[0] * 8
    size_k = qw.shape[0] * 8
    size_n = qw.shape[1]
    sc = entry["scales"]
    zp = entry.get("qzeros")
    g_idx = entry.get("g_idx")
    if g_idx is None:
        g_idx = torch.arange(size_k, dtype=torch.int32, device=device)
    workspace = torch.zeros(max(1, size_n // 64 * 16), dtype=torch.int32, device=device)
    x = (torch.randn((m, size_k), dtype=torch.float16, device=device) * 0.1)

    def once():
        return apply_gptq_marlin_linear(
            input=x, weight=qw, weight_scale=sc, weight_zp=zp,
            g_idx=g_idx, g_idx_sort_indices=None, workspace=workspace,
            wtype=scalar_types.uint4b8, output_size_per_partition=size_n,
            input_size_per_partition=size_k, is_k_full=True,
            input_dtype=input_dtype)

    out = once()
    torch.cuda.synchronize()
    ms = timeit(once, iters)
    return ms, (m, size_n, size_k), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--m", type=int, default=16)
    args = ap.parse_args()
    torch.cuda.set_device(args.device)

    weights = load_layer_weights(args.model, args.layer, args.device)
    for group, entry in weights.items():
        if not entry:
            print(f"WARNING: no packed tensors found for {group}", file=sys.stderr)

    print(f"layer={args.layer} M={args.m} iters={args.iters} "
          f"device={args.device}", flush=True)
    print(f"{'group':<9}{'M,N,K':<24}{'W4A16 ms':>11}{'W4A8 ms':>11}"
          f"{'kernel speedup':>16}", flush=True)
    rows = []
    for group, entry in weights.items():
        if not entry:
            continue
        # Rename the checkpoint's inner keys to the names bench_shape expects
        for pfx in (f"model.layers.{args.layer}.mlp.gate_proj",
                    f"model.layers.{args.layer}.mlp.up_proj",
                    f"model.layers.{args.layer}.mlp.down_proj"):
            if f"{pfx}.qweight" in entry:
                e = {k.split(".")[-1]: v for k, v in entry.items() if k.startswith(pfx)}
                break
        else:
            continue
        try:
            t16, shape, _ = bench_shape(e, args.m, args.iters, None, args.device)
            t8, _, _ = bench_shape(e, args.m, args.iters, torch.int8, args.device)
        except Exception as exc:  # noqa: BLE001
            print(f"{group:<9} FAILED: {type(exc).__name__}: {str(exc)[:90]}",
                  file=sys.stderr)
            continue
        m, n, k = shape
        rows.append({"group": group, "M": m, "N": n, "K": k,
                     "w4a16_ms": round(t16, 4), "w4a8_ms": round(t8, 4),
                     "speedup": round(t16 / t8, 3) if t8 else None})
        print(f"{group:<9}{f'{m},{n},{k}':<24}{t16:>11.4f}{t8:>11.4f}"
              f"{rows[-1]['speedup']:>16.3f}", flush=True)
    print(json.dumps({"layer": args.layer, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
