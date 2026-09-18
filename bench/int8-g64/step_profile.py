#!/usr/bin/env python3
"""Whole-step kernel attribution for the production path.

The repo already has one authoritative profile
(`docs/cmp170hx-mixed-fp8-engineering.md`, 126K: verifier partials 42.2% / target
Marlin GEMMs 43.4% / GatedDeltaNet 3.7% / combine 0.4% / other 10.3%). This
harness reproduces that breakdown so any change can be measured against it, and
extends it with the numbers that decide the next target:

  * per-kernel total time and share over a decode window;
  * the top Marlin shapes with call counts, since the profile above found
    `M=16,N=34816,K=5120` (64 calls) and `M=16,N=5120,K=17408` (64 calls)
    dominating;
  * how much of the step is *not* attention and *not* Marlin, i.e. the fusion
    headroom the review points at.

Method: drive the server with one long generation at a chosen context while a
torch profiler window covers the steady-state decode, then attribute CUDA kernel
time by name. torch.profiler with CUDA activity is enough for per-kernel totals
and does not need nsys.

Usage: step_profile.py --port 8002 --ctx 126000 --tag prod-126k

Runs on the serving host: it imports torch only to read the profiler's
device-time fields, and needs no GPU of its own.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import defaultdict

import torch


import os as _os


def _api_key() -> str:
    """Follow the repo convention: VLLM_API_KEY, else api_key.txt, else empty."""
    env = _os.environ.get("VLLM_API_KEY")
    if env:
        return env
    for cand in ("api_key.txt", _os.path.join(_os.path.dirname(__file__), "api_key.txt")):
        try:
            with open(cand, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            continue
    return ""



FILLER = ("The quick brown fox jumps over the lazy dog. "
          "Pack my box with five dozen liquor jugs. ")

BUCKETS = [
    ("verifier_attention", re.compile(r"spec_attn|partial|combine|reduce|gqa|attention|flash|fmha|decode_i8", re.I)),
    ("marlin_gemm", re.compile(r"marlin|gptq", re.I)),
    ("gdn_mamba", re.compile(r"gdn|mamba|conv1d|ssm|chunk", re.I)),
    ("draft_dflash", re.compile(r"dflash|draft|resample|selector", re.I)),
    ("norm_act", re.compile(r"rms|norm|silu|gelu|act_quant|quant", re.I)),
    ("sampler", re.compile(r"sample|softmax|topk|argmax|rejection|logprob", re.I)),
]


def prompt_for(ctx: int) -> str:
    body = FILLER * max(1, (ctx - 64) * 4 // len(FILLER))
    return f"Read the text below and then answer.\n{body}\nQuestion: reply with OK."


def call(port: int, key: str, prompt: str, max_tokens: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps({"model": "qwen3.8-27b", "prompt": prompt,
                         "max_tokens": max_tokens, "temperature": 0.0,
                         "seed": 0, "ignore_eos": True}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.loads(r.read())
    return {"s": time.perf_counter() - t0, "usage": out.get("usage", {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warm", type=int, default=64)
    a = ap.parse_args()

    call(a.port, a.key, prompt_for(256), 4)  # discard

    from torch.profiler import profile, ProfilerActivity

    # Warm the exact context first so the profile window contains steady-state
    # decode rather than allocator/graph/kernel-selection work.
    call(a.port, a.key, prompt_for(a.ctx), a.warm)

    prof = profile(activities=[ProfilerActivity.CUDA], record_shapes=False)
    prof.start()
    res = call(a.port, a.key, prompt_for(a.ctx), a.max_tokens)
    prof.stop()

    events = prof.key_averages()
    per_kernel = []
    for e in events:
        if e.device_type != torch.autograd.DeviceType.CUDA and not e.self_device_time_total:
            continue
        t = e.self_device_time_total  # microseconds
        if t <= 0:
            continue
        per_kernel.append((e.key, t, e.count, getattr(e, "input_shapes", "")))
    total = sum(t for _, t, _, _ in per_kernel)

    buckets = defaultdict(float)
    for name, t, _, _ in per_kernel:
        for label, rx in BUCKETS:
            if rx.search(name):
                buckets[label] += t
                break
        else:
            buckets["other"] += t

    marlin = sorted([(n, t, c, s) for n, t, c, s in per_kernel
                     if BUCKETS[1][1].search(n)], key=lambda x: -x[1])

    row = {
        "tag": a.tag,
        "ctx": a.ctx,
        "completion_tokens": res["usage"].get("completion_tokens"),
        "wall_s": round(res["s"], 3),
        "total_cuda_ms": round(total / 1000.0, 3),
        "buckets_ms": {k: round(v / 1000.0, 3) for k, v in sorted(buckets.items(), key=lambda x: -x[1])},
        "buckets_pct": {k: round(100.0 * v / total, 1) for k, v in sorted(buckets.items(), key=lambda x: -x[1])},
        "top_kernels": [{"name": n[:70], "ms": round(t / 1000.0, 3), "calls": c}
                        for n, t, c, _ in sorted(per_kernel, key=lambda x: -x[1])[:12]],
        "top_marlin_shapes": [{"name": n[:60], "ms": round(t / 1000.0, 3), "calls": c}
                              for n, t, c, _ in marlin[:6]],
    }
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
