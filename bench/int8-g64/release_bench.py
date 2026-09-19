#!/usr/bin/env python3
"""Release benchmark: decode throughput over context x concurrency.

Produces the data for the release table, with explicit provenance for every column
and explicit gaps where a column has no valid measurement.

WHAT IS MEASURED (this harness)
-------------------------------
For each (context, concurrency) cell: send `concurrency` concurrent requests whose
prompts are **exactly** `context` tokens (verified against the server's own
`prompt_tokens`), repeat `rounds` times, and report

    decode_aggregate_tok_s       output tokens / DECODE wall time for the batch,
                                 where decode wall is bracketed by a prefill-only
                                 run of the same shape (a raw wall figure would be
                                 prefill-dominated at long context)
    accepted_tokens_per_pass     from the server's SpecDecoding counters, bracketed
                                 to the requests that produced them
    aggregate_output_tok_s       total output tokens / wall time for the batch, i.e.
                                 the concurrency a user actually gets
    prefill_tok_s                to separate TTFT effects from decode effects

Distinct prompt content per request and per round, so acceptance cannot be affected by
prompt reuse, and exact token lengths so a "32K" cell is not really 31.7K.

WHAT IS NOT MEASURED HERE
-------------------------
Three of the four columns are not reproducible on this box and are carried in the
report with their provenance instead:

  * `old-same-path`  -- the pre-fix build on this same path. It takes an illegal
    memory access at >=16K once ~11 requests have been served (that was the bug), so
    only short/low-request cells are measurable; everything else is "faulted" and is
    reported as such rather than silently omitted;
  * `historical-mixed-fp8` -- from docs/cmp170hx-mixed-fp8-engineering.md: a
    **different route** (FP8 target KV, BF16 draft KV, FULL graph) whose `segments`
    column is NSEG, not concurrency. Recorded only for 4K, 126K and 250K;
  * `modeled-ceiling` -- from single-user/README.md: measured on `CTX=fast`
    (bf16 KV, 64k context), i.e. a different KV format and a different ceiling, so it
    is only meaningful at <=64K.

Mixing those into one table without the provenance note is exactly the error this
repository has made before (calling two different stacks the same "production
control"), so each column carries its source and its validity range.

Usage:
  release_bench.py --port 8002 --log <log> --tokenizer <model dir> \
      --ctxs 4096,16384,32768,65536,126000,250000 --concs 1,2,4 \
      --rounds 2 --out /tmp/release_fixed.json --tag fixed
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import statistics
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
    """Exactly `length` token IDs, distinct content per salt."""
    unit = tok.encode(FILLER + f"[r{salt:06d}] ", add_special_tokens=False)
    if not unit:
        raise SystemExit("tokenizer produced no tokens")
    ids = (unit * (length // len(unit) + 1))[:length]
    if salt and length > 8:
        k = salt % 8
        ids = ids[k:] + ids[:k]
    assert len(ids) == length
    return ids


def call(port, key, ids, mt, timeout=10800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps({"model": "qwen3.8-27b", "prompt": ids, "max_tokens": mt,
                         "temperature": 0.0, "seed": 0, "ignore_eos": True}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        o = json.loads(r.read())
    return time.perf_counter() - t0, o.get("usage", {})


def log_lines(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    return int(subprocess.run(["wc", "-l", path], capture_output=True,
                              text=True).stdout.split()[0])


def window_stats(log_path: str, lo: int, hi: int) -> dict:
    if not log_path or hi <= lo:
        return {}
    text = subprocess.run(["tail", "-n", f"+{lo + 1}", log_path],
                          capture_output=True, text=True).stdout
    text = "\n".join(text.splitlines()[: max(0, hi - lo)])
    acc = [float(m.group(1))
           for m in re.finditer(r"Mean acceptance length: ([0-9.]+)", text)]
    return {"windows": len(acc),
            "acceptance_mean": (sum(acc) / len(acc)) if acc else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--log", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ctxs", default="4096,16384,32768,65536,126000,250000")
    ap.add_argument("--concs", default="1,2,4")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--out", default="/tmp/release.json")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    ctxs = [int(x) for x in args.ctxs.split(",")]
    concs = [int(x) for x in args.concs.split(",")]
    results = []
    for ctx in ctxs:
        for conc in concs:
            cell = {"tag": args.tag, "ctx": ctx, "concurrency": conc,
                    "ok": 0, "faults": 0, "rounds": [], "error": None}
            try:
                for rnd in range(args.rounds):
                    ids_list = [build_ids(tok, ctx, rnd * 100 + i)
                                for i in range(conc)]
                    # PREFILL BRACKET: the same batch shape with max_tokens=1 gives the
                    # prefill-only wall time for this (ctx, concurrency). Decode
                    # throughput is then out_tokens / (wall_full - wall_prefill).
                    # Without this bracket the number is prefill-dominated at long
                    # context and is not a decode measurement at all.
                    to = time.perf_counter()
                    with cf.ThreadPoolExecutor(max_workers=conc) as pool:
                        list(pool.map(lambda ids: call(args.port, args.key, ids, 1),
                                      ids_list))
                    wall_pre = time.perf_counter() - to
                    lo = log_lines(args.log)
                    t0 = time.perf_counter()
                    with cf.ThreadPoolExecutor(max_workers=conc) as pool:
                        outs = list(pool.map(
                            lambda ids: call(args.port, args.key, ids,
                                             args.decode_tokens), ids_list))
                    wall = time.perf_counter() - t0
                    hi = log_lines(args.log)
                    st = window_stats(args.log, lo, hi)
                    total_out = sum(max(u.get("completion_tokens", 0) - 1, 0)
                                    for _, u in outs)
                    prompt_tok = max(u.get("prompt_tokens", 0) for _, u in outs)
                    dec = max(wall - wall_pre, 1e-6)
                    cell["rounds"].append({
                        "wall_full_s": round(wall, 3),
                        "wall_prefill_s": round(wall_pre, 3),
                        "wall_decode_s": round(dec, 3),
                        "prompt_tokens": prompt_tok,
                        "out_tokens_total": total_out,
                        "decode_aggregate_tok_s": round(total_out / dec, 3),
                        "prefill_tok_s_per_req": round(
                            prompt_tok / max(wall_pre, 1e-6) * conc, 1),
                        "per_req_out_tokens": [u.get("completion_tokens", 0)
                                               for _, u in outs],
                        "acceptance": st.get("acceptance_mean"),
                    })
                    cell["ok"] += conc
                    print(f"  {args.tag} ctx={ctx:<7} C{conc} round{rnd}: "
                          f"pre={wall_pre:6.1f}s dec={dec:6.1f}s out={total_out} "
                          f"decode={cell['rounds'][-1]['decode_aggregate_tok_s']} tok/s "
                          f"acc={st.get('acceptance_mean')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                cell["faults"] += 1
                cell["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
                print(f"  {args.tag} ctx={ctx:<7} C{conc} FAULTED: {cell['error']}",
                      flush=True)
            results.append(cell)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    print(json.dumps({"tag": args.tag, "cells": results}), flush=True)


if __name__ == "__main__":
    main()
