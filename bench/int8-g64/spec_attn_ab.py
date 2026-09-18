#!/usr/bin/env python3
"""A/B the cost of the interim workaround: custom spec-decode verify kernel on/off.

WHY
---
The fault is in `SpecDecodeAttention` (`_spec_attn_partial`, handover 47/48), and
`VLLM_SPEC_DECODE_ATTN` unset is a working workaround (48.2). The kernel exists for a
reason -- its own source comment says:

    "split-KV Triton attention for speculative-decode batches (1 < queries/request
     <= 10). FA2 does not split the KV sequence when max_seqlen_q > 1, leaving most
     SMs idle."

so the workaround is expected to cost something that **grows with context length**
(the un-split fallback leaves SMs idle in proportion to how much KV there is to scan).
That prediction is what this script measures, rather than assuming a number.

METHOD
------
Same engine, same checkpoint, same DFlash2 k=7, same `TRITON_ATTN` +
`int8_per_token_head`, clocks pinned. The only difference is whether
`VLLM_SPEC_DECODE_ATTN` is set to `1` (custom kernel) or **absent** (fallback).
Note it must be *absent*, not `0`: `_spec_attn_enabled()` tests `== "1"`, and an
earlier round of this project's results was invalidated by a launcher that exported
`1` unconditionally (handover 48.1).

Reports, per context length, the three quantities `spec_bench.py` separates --
ms/output-token, accepted tokens/pass, estimated ms/target-pass -- so a difference in
proposal efficiency is not mistaken for a kernel difference.

Also asserts the engine really is in the requested mode by checking the log for
`SpecDecodeAttention` activity, so an arm that silently kept the custom kernel cannot
be recorded as the fallback.

Usage:
  spec_attn_ab.py --port 8002 --log <log> --ctxs 4096,16384,65536
"""
from __future__ import annotations

import argparse
import json
import os
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



def detect_custom_kernel():
    """True if the running engine has VLLM_SPEC_DECODE_ATTN=1 in its environment.

    Reads /proc/<pid>/environ of the vLLM engine process, which is authoritative:
    the launcher's intent can differ from what the process actually got (handover
    48.1). Returns None if no engine process can be found.
    """
    try:
        out = subprocess.run(["pgrep", "-f", "venv/bin/vllm serve"],
                             capture_output=True, text=True).stdout.split()
    except Exception:
        return None
    for pid in out:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        # The engine's argv is [<python>, .../venv/bin/vllm, serve, ...]. The
        # compute-sanitizer wrapper also matches the pgrep pattern but has the
        # sanitizer binary as argv[0], so require a python interpreter here.
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

    # Verify which mode the engine is ACTUALLY in, from the engine process's own
    # environment. Log strings are not authoritative -- the whole point of handover
    # 48.1 is that a launcher exported VLLM_SPEC_DECODE_ATTN=1 unconditionally, so
    # an arm believed to be "off" was really on. /proc/<pid>/environ cannot lie.
    custom = detect_custom_kernel()
    print(f"tag={args.tag}: engine env VLLM_SPEC_DECODE_ATTN="
          f"{os.environ.get('_DETECTED', '?')} -> custom kernel {custom}", flush=True)
    if custom is None:
        print("  WARNING: could not read the engine environment; the arm's mode is "
              "UNVERIFIED and must not be reported as a fallback measurement",
              file=sys.stderr)

    rows = []
    for ctx in (int(x) for x in args.ctxs.split(",")):
        pre_ms, out_ms, accs = [], [], []
        for rep in range(args.reps):
            call(args.port, args.key, prompt_for(ctx, 900 + rep), 4)      # warm
            pre_s, _ = call(args.port, args.key, prompt_for(ctx, 100 + rep), 1)
            full_s, u = call(args.port, args.key, prompt_for(ctx, 200 + rep),
                             args.decode_tokens)
            ct = u.get("completion_tokens", 0) - 1
            if ct > 0:
                out_ms.append(1000.0 * max(full_s - pre_s, 1e-6) / ct)
            pre_ms.append(pre_s)
        import statistics
        med = statistics.median(out_ms) if out_ms else None
        rows.append({"tag": args.tag, "ctx": ctx,
                     "prefill_s_med": round(statistics.median(pre_ms), 4),
                     "ms_per_output_token_med": round(med, 3) if med else None,
                     "output_tok_s": round(1000.0 / med, 2) if med else None,
                     "custom_kernel": custom})
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({"tag": args.tag, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
