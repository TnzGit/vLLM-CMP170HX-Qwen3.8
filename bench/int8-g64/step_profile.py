#!/usr/bin/env python3
"""Whole-step kernel attribution for the production path.

STATUS: this drives the server; it does NOT itself capture GPU kernels.

An earlier revision of this file wrapped the HTTP request in a client-side
``torch.profiler`` and claimed to reproduce the reference breakdown. That cannot
work: ``torch.profiler`` only sees CUDA work submitted by *its own* process, and
the vLLM engine runs the kernels in a separate worker process. The client
profiler therefore records nothing useful, and the numbers it appeared to
produce were not the server's kernels.

Use one of the two mechanisms below instead. Both run inside the engine worker,
which is the only place the kernels are visible.

Option A -- vLLM's built-in torch profiler (preferred; no extra tooling)
-----------------------------------------------------------------------
Start the engine with profiling enabled, drive it, then stop profiling and read
the trace from ``<dir>/capture_traces``:

    --profiler-config.profiler=torch \\
    --profiler-config.torch_profiler_dir=/path/to/prof_dir

The API exposes start/stop endpoints when the server is built with them; the
simplest reliable driver is to start the server with the config above, send the
requests with ``--drive-only`` below, then stop the server, which flushes the
trace. Attribute kernels from the trace with ``--summarize <trace.json>``.

Option B -- Nsight Systems around the server process
----------------------------------------------------
    nsys profile -o prof --trace=cuda --duration=30 \\
        <the server's ExecStart command>

then ``nsys stats --report cuda_gpu_kern_sum prof.nsys-rep``. This is the
lowest-effort route when the server is already running under systemd, because
nsys can attach to the PID.

This script keeps the two jobs that are safe from the client side:
  * ``--drive-only``: send a warmed long generation at a chosen context so a
    profiler started elsewhere has something to measure;
  * ``--summarize``: bucket kernels from a torch-profiler trace into the same
    categories as the reference profile in
    ``docs/cmp170hx-mixed-fp8-engineering.md`` (126K: verifier partials 42.2% /
    target Marlin GEMMs 43.4% / GatedDeltaNet 3.7% / combine 0.4% / other 10.3%).

Usage:
  step_profile.py --port 8002 --ctx 126000 --tag prod-126k --drive-only
  step_profile.py --summarize trace.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict

FILLER = ("The quick brown fox jumps over the lazy dog. "
          "Pack my box with five dozen liquor jugs. ")

BUCKETS = [
    ("verifier_attention",
     re.compile(r"spec_attn|partial|combine|reduce|gqa|attention|flash|fmha|decode_i8", re.I)),
    ("marlin_gemm", re.compile(r"marlin|gptq", re.I)),
    ("gdn_mamba", re.compile(r"gdn|mamba|conv1d|ssm|chunk", re.I)),
    ("draft_dflash", re.compile(r"dflash|draft|resample|selector", re.I)),
    ("norm_act", re.compile(r"rms|norm|silu|gelu|act_quant|quant", re.I)),
    ("sampler", re.compile(r"sample|softmax|topk|argmax|rejection|logprob", re.I)),
]


def _api_key() -> str:
    """Resolve the serving key.

    Order: VLLM_API_KEY, then api_key.txt (the repo's convention), then the
    local test key the isolated 8002 units themselves set. That last fallback is
    not a secret -- it is written in plain text into the unit files -- but
    without it these harnesses 401 against the lab units, which is a regression
    the first version of this helper introduced.
    """
    env = os.environ.get("VLLM_API_KEY")
    if env:
        return env
    for cand in ("api_key.txt", os.path.join(os.path.dirname(__file__), "api_key.txt")):
        try:
            with open(cand, encoding="utf-8") as f:
                val = f.read().strip()
            if val:
                return val
        except OSError:
            continue
    return "pixelml-bench"


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


def drive(args) -> None:
    call(args.port, args.key, prompt_for(256), 4)      # discard
    call(args.port, args.key, prompt_for(args.ctx), 16)  # warm the context
    print(f"driving {args.max_tokens} tokens at ctx={args.ctx}; start your profiler now",
          flush=True)
    res = call(args.port, args.key, prompt_for(args.ctx), args.max_tokens)
    print(json.dumps({"ctx": args.ctx, "wall_s": round(res["s"], 3),
                      "usage": res["usage"]}), flush=True)


def summarize(path: str) -> None:
    """Bucket kernels from a torch-profiler chrome trace."""
    with open(path, encoding="utf-8") as f:
        tr = json.load(f)
    events = tr.get("traceEvents", tr if isinstance(tr, list) else [])
    per_kernel: dict[str, list] = defaultdict(lambda: [0.0, 0])
    runtime_calls: dict[str, list] = defaultdict(lambda: [0.0, 0])
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        dur = float(e.get("dur", 0.0))  # microseconds
        if dur <= 0 or not name:
            continue
        # Only true device-kernel events count as GPU kernel time. `cuda_runtime`
        # events are CPU-side CUDA API calls (cudaLaunchKernel, cudaMemcpyAsync,
        # ...); adding them to the total would double-count launch overhead as
        # execution time, which is exactly the pollution this summariser exists
        # to avoid. They are reported separately as host overhead.
        if cat in ("kernel", "Kernel"):
            per_kernel[name][0] += dur
            per_kernel[name][1] += 1
        elif cat in ("cuda_runtime", "CudaRuntime"):
            runtime_calls[name][0] += dur
            runtime_calls[name][1] += 1
    if not per_kernel:
        print("no kernel events found -- was the trace captured in the ENGINE process?",
              file=sys.stderr)
        raise SystemExit(2)
    total = sum(v[0] for v in per_kernel.values())
    buckets: dict[str, float] = defaultdict(float)
    for name, (dur, _n) in per_kernel.items():
        for label, rx in BUCKETS:
            if rx.search(name):
                buckets[label] += dur
                break
        else:
            buckets["other"] += dur
    print(f"total CUDA kernel time: {total / 1000.0:.3f} ms")
    print(f"{'bucket':<22}{'ms':>10}{'share':>9}")
    for label, dur in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"{label:<22}{dur / 1000.0:>10.3f}{100.0 * dur / total:>8.1f}%")
    if runtime_calls:
        rt = sum(v[0] for v in runtime_calls.values())
        print(f"\nhost CUDA API time (cuda_runtime, NOT kernel execution): {rt / 1000.0:.3f} ms")
        for name, (dur, n) in sorted(runtime_calls.items(), key=lambda x: -x[1][0])[:5]:
            print(f"  {dur / 1000.0:9.3f} ms  x{n:<6} {name[:60]}")

    print("\ntop kernels:")
    for name, (dur, n) in sorted(per_kernel.items(), key=lambda x: -x[1][0])[:12]:
        print(f"  {dur / 1000.0:9.3f} ms  x{n:<6} {name[:64]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--tag", default="")
    ap.add_argument("--ctx", type=int, default=126000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--drive-only", action="store_true",
                    help="just drive the server so an external profiler has work to measure")
    ap.add_argument("--summarize", default="",
                    help="bucket kernels from a torch-profiler chrome trace")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if args.drive_only:
        drive(args)
        return
    ap.error("choose --drive-only (client side) or --summarize <trace.json>; "
             "this script cannot profile the engine from the client process")


if __name__ == "__main__":
    main()
