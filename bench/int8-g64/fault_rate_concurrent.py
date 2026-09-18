#!/usr/bin/env python3
"""Concurrent fault-rate measurement: N requests in flight, not one at a time.

The serial protocol in fault_rate.py pays a full prefill per sample because each
request must finish before the next starts. With `MAX_SEQS=4` and a KV pool that
holds 4 x 56K tokens, four requests can be in flight in the same wall-clock
window, so the same elapsed time yields ~4x the samples.

This matters because the trigger was measured to be **cumulative KV volume per
engine instance** (handover 32.9: ~200-290K tokens, at 16K every 12th request),
not per-request length. Concurrency therefore changes how fast the volume
accumulates but not the underlying condition -- which is exactly what a
throughput measurement wants, and it must be verified rather than assumed: the
first run of this script at a given (length, concurrency) should be checked
against the serial result.

Usage:
  fault_rate_concurrent.py --port 8002 --log <log> --tokenizer <path> \
      --length 16384 --rounds 6 --concurrency 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time
import urllib.request

FILLER = ("The quick brown fox jumps over the lazy dog. "
          "Pack my box with five dozen liquor jugs. ")


def _api_key() -> str:
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


def build_ids(tok, length: int, salt: int) -> list[int]:
    unit = tok.encode(FILLER + f"[c{salt:05d}] ", add_special_tokens=False)
    ids = (unit * (length // len(unit) + 1))[:length]
    if salt and len(ids) > 8:
        k = salt % 8
        ids = ids[k:] + ids[:k]
    assert len(ids) == length
    return ids


def call(port, key, ids, mt):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps({"model": "qwen3.8-27b", "prompt": ids, "max_tokens": mt,
                         "temperature": 0.0, "seed": 0, "ignore_eos": True}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        o = json.loads(r.read())
    return o.get("usage", {})


def log_lines(path):
    return int(subprocess.run(["wc", "-l", path], capture_output=True,
                              text=True).stdout.split()[0])


def restart(unit, log, wait_s=150):
    subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
    time.sleep(3)
    n = log_lines(log)
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    subprocess.run(["systemctl", "--user", "start", unit], capture_output=True)
    for _ in range(wait_s // 5):
        time.sleep(5)
        tail = subprocess.run(["tail", "-n", f"+{n+1}", log],
                              capture_output=True, text=True).stdout
        if "Application startup complete" in tail:
            return True
        if subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState",
                           "--value"], capture_output=True,
                          text=True).stdout.strip() == "failed":
            return False
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--log", required=True)
    ap.add_argument("--unit", default="triton-int8-control-8002.service")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--tag", default="conc")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    ok = fault = 0
    total_tokens = 0
    issued = 0
    t_start = time.perf_counter()
    for rnd in range(args.rounds):
        n = log_lines(args.log)
        batch = [build_ids(tok, args.length, rnd * 100 + i)
                 for i in range(args.concurrency)]
        try:
            with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                usages = list(pool.map(
                    lambda ids: call(args.port, args.key, ids, args.decode_tokens), batch))
            for u in usages:
                if u.get("prompt_tokens") != args.length:
                    raise RuntimeError(f"length mismatch {u.get('prompt_tokens')}")
            tail = subprocess.run(["tail", "-n", f"+{n+1}", args.log],
                                  capture_output=True, text=True).stdout
            if "illegal memory access" in tail:
                raise RuntimeError("fault records in log")
            ok += args.concurrency
            issued += args.concurrency
            total_tokens += args.length * args.concurrency
        except Exception as e:
            fault += args.concurrency
            issued += args.concurrency
            print(f"  round {rnd}: FAULT {type(e).__name__} "
                  f"(after ~{total_tokens} tokens)", flush=True)
            if not restart(args.unit, args.log):
                print("  engine restart FAILED -- aborting", flush=True)
                break
        elapsed = time.perf_counter() - t_start
        print(f"  round {rnd}: ok={ok} fault={fault} issued={issued} "
              f"~{total_tokens} tokens in {elapsed:.0f}s", flush=True)

    elapsed = time.perf_counter() - t_start
    print(json.dumps({
        "tag": args.tag, "length": args.length, "concurrency": args.concurrency,
        "rounds": args.rounds, "ok": ok, "fault": fault,
        "rate": round(fault / max(1, issued), 3),
        "tokens_processed": total_tokens, "elapsed_s": round(elapsed, 1),
        "seconds_per_sample": round(elapsed / max(1, issued), 1),
    }), flush=True)


if __name__ == "__main__":
    main()
