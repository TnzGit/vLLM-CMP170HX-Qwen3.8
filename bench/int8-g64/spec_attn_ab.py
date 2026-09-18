#!/usr/bin/env python3
"""A/B the cost of the interim workaround: custom spec-decode verify kernel on/off.

WHY
---
The fault is in `SpecDecodeAttention` (`_spec_attn_partial`, handover 47/48/51), and
leaving `VLLM_SPEC_DECODE_ATTN` unset is a working workaround (48.2). The kernel
exists for a reason -- its own source comment says:

    "split-KV Triton attention for speculative-decode batches (1 < queries/request
     <= 10). FA2 does not split the KV sequence when max_seqlen_q > 1, leaving most
     SMs idle."

so the workaround is expected to cost something that **grows with context length**
(the un-split fallback leaves SMs idle in proportion to how much KV there is to scan).
That prediction is what this measures rather than assumes.

METHOD
------
Same engine, checkpoint, DFlash2 k=7, `TRITON_ATTN` + `int8_per_token_head`, clocks
pinned. The only difference is whether `VLLM_SPEC_DECODE_ATTN` is `1` (custom kernel)
or **absent** (fallback). It must be *absent*, not `0`: `_spec_attn_enabled()` tests
`== "1"`, and an earlier round of this project's results was invalidated by a launcher
that exported `1` unconditionally (handover 48.1).

REPORTED QUANTITIES -- and why the acceptance accounting is not optional
------------------------------------------------------------------------
An earlier revision of this script claimed in its docstring to report
accepted-tokens/pass and ms/target-pass while computing only ms/output-token. That is
the same class of error as the `ms/step` mislabelling corrected in handover 29.1 --
a documented metric that the code never produced -- so the acceptance counters are now
actually read. Per measured request it reports:

    ms per output token        wall time / generated tokens
    accepted tokens per pass   from the server's SpecDecoding counters, bracketed to
                               the request that produced them
    estimated ms per target pass
                               ms/output-token x accepted-per-pass, LABELLED as
                               derived (a product of two windowed measurements, not a
                               request-local counter)

plus the raw `drafted` / `accepted` totals so the derivation can be checked.

The counters come from the engine's own `SpecDecoding metrics` lines, which are
**periodic windowed-and-reset**, not per-request cumulative. Each measured request is
therefore bracketed by log line count and only windows inside the bracket are
attributed to it (the alignment fix from handover 29.1).

MODE VERIFICATION
-----------------
The arm's mode is read from `/proc/<engine-pid>/environ`, not from the launcher's
intent and not from a log string, and detection requires a python `argv[0]` so the
compute-sanitizer wrapper cannot be mistaken for the engine. If no engine is found the
script reports UNVERIFIED rather than guessing.

Usage:
  spec_attn_ab.py --port 8002 --log <log> --ctxs 4096,16384,65536 --tag custom_on
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
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


def detect_custom_kernel():
    """True if the running engine has VLLM_SPEC_DECODE_ATTN=1.

    Reads /proc/<pid>/environ of the engine process, which is authoritative: the
    launcher's intent can differ from what the process actually received (48.1).
    Returns None if no engine process can be identified.
    """
    try:
        pids = subprocess.run(["pgrep", "-f", "venv/bin/vllm serve"],
                              capture_output=True, text=True).stdout.split()
    except Exception:
        return None
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        # Engine argv is [<python>, .../venv/bin/vllm, serve, ...]; the
        # compute-sanitizer wrapper also matches pgrep but has the sanitizer binary
        # as argv[0].
        if not cmd or "python" not in cmd[0]:
            continue
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                env = f.read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        for kv in env:
            if kv.startswith("VLLM_SPEC_DECODE_ATTN="):
                val = kv.split("=", 1)[1]
                os.environ["_DETECTED"] = val
                return val == "1"
        os.environ["_DETECTED"] = "(absent)"
        return False
    return None


def prompt_for(ctx: int, idx: int) -> str:
    unit = FILLER + f"[case {idx:05d}] "
    body = unit * max(1, (ctx - 64) * 4 // len(unit))
    return f"Read the text below and then answer.\n{body}\nQuestion: reply with OK {idx}."


def call(port, key, prompt, mt):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps({"model": "qwen3.8-27b", "prompt": prompt,
                         "max_tokens": mt, "temperature": 0.0, "seed": 0,
                         "ignore_eos": True}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=7200) as r:
        o = json.loads(r.read())
    return time.perf_counter() - t0, o.get("usage", {})


def log_lines(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    return int(subprocess.run(["wc", "-l", path], capture_output=True,
                              text=True).stdout.split()[0])


def window_stats(log_path: str, lo: int, hi: int) -> dict:
    """SpecDecoding counters from log lines (lo, hi] -- one request's bracket."""
    if not log_path or hi <= lo:
        return {}
    text = subprocess.run(["tail", "-n", f"+{lo + 1}", log_path],
                          capture_output=True, text=True).stdout
    text = "\n".join(text.splitlines()[: max(0, hi - lo)])
    acc = [float(m.group(1))
           for m in re.finditer(r"Mean acceptance length: ([0-9.]+)", text)]
    drafted = [int(m.group(1)) for m in re.finditer(r"Drafted: ([0-9]+) tokens", text)]
    accepted = [int(m.group(1)) for m in re.finditer(r"Accepted: ([0-9]+) tokens", text)]
    return {
        "windows": len(acc),
        "acceptance_mean": (sum(acc) / len(acc)) if acc else None,
        "drafted": max(drafted) if drafted else None,
        "accepted": max(accepted) if accepted else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--log", required=True)
    ap.add_argument("--ctxs", default="4096,16384,65536")
    ap.add_argument("--decode-tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    custom = detect_custom_kernel()
    print(f"tag={args.tag}: engine env VLLM_SPEC_DECODE_ATTN="
          f"{os.environ.get('_DETECTED', '?')} -> custom kernel {custom}", flush=True)
    if custom is None:
        print("  WARNING: could not identify the engine process; the arm's mode is "
              "UNVERIFIED and must not be reported as a fallback measurement",
              file=sys.stderr)

    rows = []
    for ctx in (int(x) for x in args.ctxs.split(",")):
        out_ms, reps = [], []
        for rep in range(args.reps):
            call(args.port, args.key, prompt_for(ctx, 900 + rep), 4)        # warm
            pre_s, _ = call(args.port, args.key, prompt_for(ctx, 100 + rep), 1)
            lo = log_lines(args.log)
            full_s, u = call(args.port, args.key, prompt_for(ctx, 200 + rep),
                             args.decode_tokens)
            hi = log_lines(args.log)
            st = window_stats(args.log, lo, hi) if args.log else {}
            ct = u.get("completion_tokens", 0) - 1
            ms = (1000.0 * max(full_s - pre_s, 1e-6) / ct) if ct > 0 else None
            if ms:
                out_ms.append(ms)
            reps.append({"ms_per_out_tok": round(ms, 3) if ms else None,
                         "out_tokens": ct,
                         "acceptance": st.get("acceptance_mean"),
                         "windows": st.get("windows"),
                         "drafted": st.get("drafted"),
                         "accepted": st.get("accepted")})
        med = statistics.median(out_ms) if out_ms else None
        accs = [r["acceptance"] for r in reps if r["acceptance"]]
        acc = statistics.mean(accs) if accs else None
        row = {
            "tag": args.tag, "ctx": ctx, "custom_kernel": custom,
            "ms_per_output_token_med": round(med, 3) if med else None,
            "accepted_tokens_per_pass": round(acc, 3) if acc else None,
            "estimated_ms_per_target_pass": round(med * acc, 3) if (med and acc) else None,
            "output_tok_s": round(1000.0 / med, 2) if med else None,
            "per_rep": reps,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"tag": args.tag, "custom_kernel": custom, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
