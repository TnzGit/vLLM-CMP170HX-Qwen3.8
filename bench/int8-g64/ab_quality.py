#!/usr/bin/env python3
"""End-to-end quality arm: fixed prompts, greedy, per-token logprobs recorded.

Two arms produce JSON files that ab_quality_compare.py diffs. Greedy decoding
makes each arm deterministic, so token-exact agreement between arms is a direct
measure of whether INT8-G64 perturbs the model's chosen tokens, and the mean
logprob delta bounds the numerical drift where tokens do agree.

Usage: ab_quality.py --port 8002 --tag g64 --out /path/quality-g64.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


import os as _os


def _api_key() -> str:
    """Resolve the serving key.

    Order: VLLM_API_KEY, then api_key.txt (the repo's convention), then the
    local test key the isolated 8002 units themselves set. That last fallback is
    not a secret -- it is written in plain text into the unit files -- but
    without it these harnesses 401 against the lab units, which is a regression
    the first version of this helper introduced.
    """
    env = _os.environ.get("VLLM_API_KEY")
    if env:
        return env
    for cand in ("api_key.txt", _os.path.join(_os.path.dirname(__file__), "api_key.txt")):
        try:
            with open(cand, encoding="utf-8") as f:
                val = f.read().strip()
            if val:
                return val
        except OSError:
            continue
    return "pixelml-bench"



# Fixed suite: arithmetic, instruction following, retrieval from context, and
# open-ended prose. Kept short so both arms are cheap to compare.
PROMPTS = [
    "Count from 1 to 20 in words, one number per line.",
    "What is 17 * 23? Show the arithmetic, then give the answer on its own line.",
    "List the first 12 prime numbers, comma separated, then their sum.",
    "Explain in three sentences why the sky appears blue.",
    "Write a Python function that reverses a linked list iteratively.",
    "Summarise the following in one sentence: The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. Sphinx of black quartz, judge my vow.",
    "A train leaves at 09:15 and travels 240 km at 80 km/h. What time does it arrive?",
    "Name the capitals of France, Japan, Peru and Kenya, one per line.",
    "Explain the difference between a process and a thread to a new programmer.",
    "Given the list [3, 1, 4, 1, 5, 9, 2, 6], sort it ascending and state the median.",
    "Translate 'the weather is nice today' into Spanish, German and Italian.",
    "Write a haiku about autumn, then count the syllables in each line.",
]


def one(port: int, key: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": "qwen3.8-27b",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "logprobs": 1,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.loads(resp.read())
    ch = out["choices"][0]
    lp = ch.get("logprobs") or {}
    toks = lp.get("tokens") or []
    vals = lp.get("token_logprobs") or []
    return {
        "text": ch["text"],
        "tokens": toks,
        "token_logprobs": vals,
        "finish_reason": ch.get("finish_reason"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=192)
    args = ap.parse_args()

    one(args.port, args.key, "warmup", 4)  # discard
    results = []
    for i, p in enumerate(PROMPTS):
        t0 = time.perf_counter()
        r = one(args.port, args.key, p, args.max_tokens)
        r["prompt_index"] = i
        r["prompt"] = p
        r["seconds"] = round(time.perf_counter() - t0, 3)
        results.append(r)
        print(f"  [{i:2d}] {len(r['tokens'])} tokens, "
              f"mean_lp={sum(r['token_logprobs']) / max(1, len(r['token_logprobs'])):.4f}", flush=True)

    payload = {"tag": args.tag, "max_tokens": args.max_tokens, "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    n_tok = sum(len(r["tokens"]) for r in results)
    all_lp = [v for r in results for v in r["token_logprobs"]]
    print(f"WROTE {args.out}: {len(results)} prompts, {n_tok} tokens, "
          f"overall mean_lp={sum(all_lp) / len(all_lp):.4f}", flush=True)


if __name__ == "__main__":
    main()
