#!/usr/bin/env python3
"""Bounded INT8-G64 correctness probe against an already-running engine.

Deliberately small and deterministic: it never touches the production port.  It
runs the handover 19.4 step-4 sequence against the isolated unit and prints raw
evidence rather than a verdict:

  1. one greedy request, repeated -> must be byte-identical (determinism)
  2. a long single prompt      -> exercises chunked / continuation prefill
  3. batch 2 and batch 4       -> exercises the C2/C4 verify paths

Usage: bounded_correctness_probe.py [--port 8002] [--key ...] [--prompt-tokens N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
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



# A fixed, low-entropy prompt: any cache-address or scale error shows up as
# incoherent continuation rather than as a plausible-looking sample.
PROMPT = (
    "You are a careful arithmetic assistant.\n"
    "Count from 1 to 40 in words, one number per line, then state the sum of "
    "the integers from 1 to 40.\n"
)
FILLER = "The quick brown fox jumps over the lazy dog. "


def post(port: int, key: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_ready(port: int, key: str, deadline_s: int = 900) -> None:
    start = time.time()
    while time.time() - start < deadline_s:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/health",
                headers={"Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"ready after {time.time() - start:.0f}s", flush=True)
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(5)
    raise SystemExit(f"engine on port {port} never became ready")


def one(port: int, key: str, prompt: str, n: int, seed: int) -> str:
    out = post(port, key, {
        "model": "qwen3.8-27b",
        "prompt": prompt,
        "max_tokens": n,
        "temperature": 0.0,
        "seed": seed,
    })
    return out["choices"][0]["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--long-tokens", type=int, default=6000)
    ap.add_argument("--skip-batches", action="store_true")
    args = ap.parse_args()

    wait_ready(args.port, args.key)

    print("\n== 1. determinism: one greedy request twice ==", flush=True)
    a = one(args.port, args.key, PROMPT, args.tokens, 0)
    b = one(args.port, args.key, PROMPT, args.tokens, 0)
    print(f"identical={a == b}")
    print(f"len={len(a)} chars")
    print("first 200 chars:", repr(a[:200]), flush=True)
    if not a.strip():
        print("EMPTY OUTPUT -- engine produced no text", flush=True)
        sys.exit(3)

    print("\n== 2. long prompt: chunked / continuation prefill ==", flush=True)
    filler = FILLER * (args.long_tokens // 9)
    long_prompt = (
        "Read the following long text, then answer the final question.\n"
        + filler
        + "\nQuestion: how many times does the word 'fox' appear above? "
          "Answer with the number, then write one sentence about foxes.\n"
    )
    c = one(args.port, args.key, long_prompt, args.tokens, 0)
    print(f"len={len(c)} chars")
    print("first 300 chars:", repr(c[:300]), flush=True)
    if not c.strip():
        print("EMPTY OUTPUT on long prompt", flush=True)
        sys.exit(3)

    if args.skip_batches:
        print("\nBOUNDED_PROBE PARTIAL (batches skipped)", flush=True)
        return

    for n_req in (2, 4):
        print(f"\n== 3. batch {n_req}: concurrent greedy requests ==", flush=True)
        import concurrent.futures as cf
        prompts = [PROMPT + f"\n(respond with the count only, request {i})" for i in range(n_req)]
        with cf.ThreadPoolExecutor(max_workers=n_req) as pool:
            results = list(pool.map(
                lambda p: one(args.port, args.key, p, args.tokens, 0), prompts))
        for i, text in enumerate(results):
            print(f"  req {i}: len={len(text)} empty={not text.strip()} "
                  f"head={text[:60]!r}", flush=True)
        if any(not t.strip() for t in results):
            print(f"EMPTY OUTPUT in batch {n_req}", flush=True)
            sys.exit(3)

    print("\nBOUNDED_PROBE OK", flush=True)


if __name__ == "__main__":
    main()
