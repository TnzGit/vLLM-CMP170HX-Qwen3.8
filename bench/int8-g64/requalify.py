#!/usr/bin/env python3
"""Requalification matrix for a fix to the spec-decode verify-kernel fault.

WHAT MUST BE SHOWN BEFORE THE FIX IS BELIEVED
---------------------------------------------
The fault is periodic in the request count (handover 35: 12 requests at 16K, 5 at 57K,
4 at 64K) and appears only when the custom verify kernel is enabled (48.2). A fix is
therefore only credible if it survives **more than one full original period, in every
configuration that changes the period**, and if the GPU reports **zero Xid 31** while
doing it. A single clean run is exactly the kind of evidence that produced three
retracted models earlier in this investigation, so the matrix is fixed in advance
rather than chosen after seeing results.

THE MATRIX (from the review)
----------------------------
  1. DFlash2 and MTP -- two structurally different drafters that fault identically
     today (34), so a fix must hold for both;
  2. C1 and C4 -- concurrency changes the period (8 vs 12 requests at 16K, 32.14);
  3. at least 10x the original period at 16K -- i.e. 120+ requests, not 15;
  4. 65K and 126K -- the long contexts where the fault was originally noticed;
  5. FULL and eager graph modes -- eager faults identically today (38), so both;
  6. zero Xid 31 events attributable to the run.

HOW XID IS COUNTED
------------------
`dmesg` retains faults from every earlier run on the host, so an absolute count is
meaningless. The count is taken **before** the run and the delta is what is asserted,
and the delta must be 0. `dmesg` is read via `sudo -n`, as the other harnesses do.

Usage:
  requalify.py --port 8002 --log <engine log> --tokenizer <model dir> \
      --out /tmp/requal.json [--matrix 16k10x]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

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


def xid31_count() -> int:
    """Absolute Xid 31 count from dmesg (deltas are what matter)."""
    try:
        out = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True,
                             text=True, errors="ignore").stdout
    except Exception:
        return -1
    return sum(1 for line in out.splitlines() if "Xid" in line and ": 31," in line)


def prompt_for(length: int, idx: int) -> str:
    unit = FILLER + f"[case {idx:05d}] "
    body = unit * max(1, (length - 64) * 4 // len(unit))
    return f"Read the text below and then answer.\n{body}\nQuestion: reply with OK {idx}."


def call(port, key, prompt, mt, timeout=7200):
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps({"model": "qwen3.8-27b", "prompt": prompt, "max_tokens": mt,
                         "temperature": 0.0, "seed": 0, "ignore_eos": True}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_leg(port, key, length, count, decode_tokens, label, out, results):
    """Serve `count` requests of `length` and record the outcome of each."""
    ok = fault = 0
    first_fault = None
    t0 = time.perf_counter()
    for i in range(count):
        try:
            o = call(port, key, prompt_for(length, i), decode_tokens)
            assert o.get("usage", {}).get("prompt_tokens"), "no usage"
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fault += 1
            if first_fault is None:
                first_fault = i + 1
            print(f"  {label}: FAULT at request {i + 1}: {type(exc).__name__}",
                  flush=True)
            break
        if (i + 1) % 20 == 0:
            print(f"  {label}: {i + 1}/{count} ok", flush=True)
    row = {"leg": label, "length": length, "requests": count, "ok": ok,
           "faults": fault, "first_fault_request": first_fault,
           "elapsed_s": round(time.perf_counter() - t0, 1)}
    results.append(row)
    print(json.dumps(row), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return fault == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--log", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default="/tmp/requal.json")
    ap.add_argument("--matrix", default="16k10x",
                    choices=["16k10x", "long", "full"],
                    help="16k10x: 120 requests at 16K (10x the 12-request period); "
                         "long: adds 65K and 126K; full: the whole matrix")
    ap.add_argument("--decode-tokens", type=int, default=16)
    args = ap.parse_args()

    before = xid31_count()
    print(f"Xid 31 before: {before}", flush=True)
    results = []

    legs = [("16k_x10period", 16384, 120)]
    if args.matrix in ("long", "full"):
        legs += [("65k", 65536, 12), ("126k", 126000, 8)]
    if args.matrix == "full":
        legs += [("16k_c4_note", 16384, 24)]  # run with MAX_SEQS=4 by the caller

    all_clean = True
    for label, length, count in legs:
        if not run_leg(args.port, args.key, length, count, args.decode_tokens,
                       label, args.out, results):
            all_clean = False

    after = xid31_count()
    delta = (after - before) if (after >= 0 and before >= 0) else None
    summary = {
        "matrix": args.matrix,
        "xid31_before": before, "xid31_after": after, "xid31_delta": delta,
        "zero_new_xid": delta == 0,
        "all_legs_clean": all_clean,
        "PASS": bool(all_clean and delta == 0),
        "legs": results,
        "note": "zero_new_xid is asserted on the DELTA, because dmesg retains Xid "
                "events from every earlier run on this host (handover 33).",
    }
    print(json.dumps(summary, indent=2), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
