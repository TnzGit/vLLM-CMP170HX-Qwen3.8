#!/usr/bin/env python3
"""Isolate prefill cost from decode cost at a chosen context length.

Why this exists: whole-request throughput mixes prefill and decode, and an
earlier version of the A/B compared them that way — which is how a 39 s prefill
was briefly mistaken for a decode problem. This harness measures one context at
a time and reports decode as ms/step, which is independent of DFlash acceptance
and therefore comparable across arms whose acceptance differs.

Method, per repetition:
  * warm the target context (a first request at a new context pays one-off
    allocator / graph / kernel-selection costs worth 20-40% of ms/step);
  * one output token  = that prompt's prefill;
  * `ndec` output tokens = the same prefill plus (ndec-1) decode steps, so the
    difference is decode-only time at that context.
The median over --reps is reported, because a single subtraction carries the
scheduler's state from whatever ran before it.

Usage: ab_split_bench.py --port 8002 --tag g64 --ctx 32000 --reps 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


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


def prompt_for(ctx_tokens: int, idx: int = 0) -> str:
    """Build a prompt of roughly ctx_tokens whose TEXT differs per idx.

    The idx is woven through the body, not just appended, because this engine
    trips an illegal access when the same long prefix is submitted repeatedly in
    one process (recorded in handover 25.3). Same token count, different bytes.
    """
    unit = FILLER + f"[case {idx:04d}] "
    body = unit * max(1, (ctx_tokens - 64) * 4 // len(unit))
    return (f"Read the text below and then answer.\n{body}\n"
            f"Question: reply with OK then the number {idx}.")


def call(port: int, key: str, prompt: str, max_tokens: int) -> dict:
    payload = {"model": "qwen3.8-27b", "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "seed": 0, "ignore_eos": True}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    return {"s": time.perf_counter() - t0, "usage": out.get("usage", {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--decode-tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    p, k = args.port, args.key
    ctx, ndec = args.ctx, args.decode_tokens

    call(p, k, prompt_for(256), 2)  # discard: first request pays startup costs

    pre_s, dec_ms, pf_tok_s, prompt_tokens = [], [], [], 0
    for _ in range(args.reps):
        call(p, k, prompt_for(ctx, 900 + _), 4)          # warm this context
        pre = call(p, k, prompt_for(ctx, 100 + _), 1)    # prefill only
        full = call(p, k, prompt_for(ctx, 200 + _), ndec)  # prefill + decode
        dec_t = max(full["s"] - pre["s"], 1e-6)
        dec_n = full["usage"]["completion_tokens"] - 1
        prompt_tokens = pre["usage"]["prompt_tokens"]
        pre_s.append(pre["s"])
        pf_tok_s.append(prompt_tokens / max(pre["s"], 1e-6))
        dec_ms.append(1000.0 * dec_t / max(dec_n, 1))

    ref = call(p, k, prompt_for(128), 1)
    row = {
        "tag": args.tag,
        "ctx": ctx,
        "prompt_tokens": prompt_tokens,
        "reps": args.reps,
        "prefill_s_med": round(statistics.median(pre_s), 4),
        "prefill_tok_s_med": round(statistics.median(pf_tok_s), 1),
        "decode_steps": ndec - 1,
        "decode_ms_per_step_med": round(statistics.median(dec_ms), 3),
        "decode_ms_per_step_all": [round(x, 3) for x in dec_ms],
        "decode_out_tok_s_med": round(1000.0 / statistics.median(dec_ms), 2),
        "prefill_128_s": round(ref["s"], 4),
    }
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
