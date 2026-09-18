#!/usr/bin/env python3
"""Locked A/B benchmark arm for the INT8-G64 long-context comparison.

One arm = one engine on one port. Locked settings: greedy decoding, fixed seed,
identical prompt tokens per row, warmup requests discarded, per-request latency
percentiles plus aggregate output throughput. Prints one JSON line per row so
arms can be diffed mechanically.

Usage: ab_bench.py --port 8002 --tag g64-4k --ctx 4096 --concurrency 1
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
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
# ~13 tokens per repetition of the filler pair; 4 chars/token on average.
CHARS_PER_TOKEN = 4.0


def build_prompt(ctx_tokens: int, idx: int) -> str:
    body = FILLER * max(1, (ctx_tokens - 64) * 4 // len(FILLER))
    return (f"Read the text below and then answer.\n{body}\n"
            f"Question: reply with the single word OK followed by the number {idx}.")


def call(port: int, key: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": "qwen3.8-27b",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.loads(resp.read())
    dt = time.perf_counter() - t0
    usage = out.get("usage", {})
    return {
        "seconds": dt,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "text_head": out["choices"][0]["text"][:40],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--only-concurrency", type=int, default=None)
    args = ap.parse_args()

    # Warmup is discarded: first request pays prefill/cudagraph/allocator costs.
    call(args.port, args.key, build_prompt(256, 0), 8)

    concs = ((args.only_concurrency,) if args.only_concurrency
             else (1, 2, 4))
    for conc in concs:
        prompts = [build_prompt(args.ctx, i) for i in range(conc)]
        samples = []
        for rep in range(max(1, args.requests // conc)):
            with cf.ThreadPoolExecutor(max_workers=conc) as pool:
                res = list(pool.map(lambda p: call(args.port, args.key, p, args.max_tokens),
                                    prompts))
            samples.extend(r for r in res if r["completion_tokens"] > 0)
            if rep == 0:
                samples.clear()  # drop the first round at this concurrency as warmup
        if not samples:
            print(json.dumps({"tag": args.tag, "concurrency": conc, "error": "no samples"}),
                  flush=True)
            continue
        lat = sorted(s["seconds"] for s in samples)
        toks = sum(s["completion_tokens"] for s in samples)
        wall = sum(lat) / conc  # approximate aggregate wall time
        row = {
            "tag": args.tag,
            "ctx": args.ctx,
            "concurrency": conc,
            "n": len(samples),
            "prompt_tokens": samples[0]["prompt_tokens"],
            "completion_tokens": toks // len(samples),
            "ttft_plus_decode_s_p50": round(statistics.median(lat), 4),
            "lat_p10": round(lat[max(0, len(lat) // 10)], 4),
            "lat_p90": round(lat[min(len(lat) - 1, 9 * len(lat) // 10)], 4),
            "aggregate_out_tok_s": round(toks / wall, 2) if wall > 0 else None,
            "sample_head": samples[0]["text_head"],
        }
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
