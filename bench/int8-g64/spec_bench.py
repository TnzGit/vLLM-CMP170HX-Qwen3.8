#!/usr/bin/env python3
"""Speculative-decode aware split benchmark: kernel cost vs speculation cost.

Why this exists, and what the earlier version got wrong
------------------------------------------------------
An earlier revision of this harness reported "decode ms/step" computed as

    (full_wall - prefill_wall) / (completion_tokens - 1)

which is **milliseconds per generated output token**, not per DFlash verify
step. DFlash2 accepts ~3.3-3.7 tokens per target pass, so the two differ by that
factor and the old label made a false claim ("ms/step is independent of
acceptance"). This version reports both, and separates the two efficiencies:

    target pass cost        -> ms per verify pass (kernel efficiency)
    accepted tokens / pass  -> proposal efficiency (speculation quality)
    output throughput       -> ms per output token (what a user feels)

Those three move independently: a faster attention kernel with worse acceptance
can lose end-to-end, and a better drafter with an unchanged kernel can win.
Reporting only output tok/s cannot tell those apart, and reporting only ms/pass
cannot tell whether the win reaches the user.

Measured quantities come from the server's own counters (`usage` plus the
`SpecDecoding metrics` log lines), so no client-side timing of individual steps
is attempted.

Usage:
  spec_bench.py --port 8002 --tag g64 --ctx 32768 --reps 3
  spec_bench.py --port 8002 --tag g64 --ctx 32768 --exact-tokens 32768
"""
from __future__ import annotations

import argparse
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


def build_prompt(ctx_tokens: int, idx: int, tokenizer=None) -> str:
    """Distinct text per idx.

    Distinctness matters independently of caching: a byte-identical long prompt
    submitted repeatedly is the shape that the lab's earlier fault investigation
    wrongly blamed, and keeping requests unique removes it as a variable.
    """
    unit = FILLER + f"[case {idx:04d}] "
    body = unit * max(1, (ctx_tokens - 64) * 4 // len(unit))
    return (f"Read the text below and then answer.\n{body}\n"
            f"Question: reply with OK then the number {idx}.")


def call(port: int, key: str, prompt: str, max_tokens: int,
         prompt_token_ids=None) -> dict:
    payload = {"model": "qwen3.8-27b", "max_tokens": max_tokens,
               "temperature": 0.0, "seed": 0, "ignore_eos": True}
    if prompt_token_ids is not None:
        payload["prompt"] = prompt_token_ids
    else:
        payload["prompt"] = prompt
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.loads(r.read())
    return {"s": time.perf_counter() - t0, "usage": out.get("usage", {})}


def spec_stats(log_path: str, since_line: int) -> dict:
    """Read the server's own SpecDecoding counters written after `since_line`."""
    try:
        text = subprocess.run(["tail", "-n", f"+{since_line + 1}", log_path],
                              capture_output=True, text=True).stdout
    except Exception:
        return {}
    acc = [float(m.group(1)) for m in re.finditer(r"Mean acceptance length: ([0-9.]+)", text)]
    drafted = [int(m.group(1)) for m in re.finditer(r"Drafted: ([0-9]+) tokens", text)]
    accepted = [int(m.group(1)) for m in re.finditer(r"Accepted: ([0-9]+) tokens", text)]
    return {
        "acceptance_len_last": acc[-1] if acc else None,
        "acceptance_len_median": statistics.median(acc) if acc else None,
        "drafted_tokens": max(drafted) if drafted else None,
        "accepted_tokens": max(accepted) if accepted else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--decode-tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--log", default="", help="server log, for acceptance counters")
    ap.add_argument("--exact-tokens", type=int, default=0,
                    help="build the prompt as exactly this many token IDs")
    ap.add_argument("--tokenizer", default="", help="tokenizer path for --exact-tokens")
    ap.add_argument("--k", type=int, default=0,
                    help="draft depth to report alongside (informational)")
    args = ap.parse_args()
    p, k = args.port, args.key
    ctx, ndec = args.ctx, args.decode_tokens

    ids = None
    if args.exact_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        # Distinct token content per call, exact length, so length is controlled
        # while the repeated-prefix variable stays removed.
        base = tok.encode(FILLER, add_special_tokens=False)
        need = args.exact_tokens - 8
        seq = (base * (need // len(base) + 1))[:need]
        seq[0:4] = tok.encode("[case]", add_special_tokens=False)[:4] or seq[0:4]
        ids = seq + tok.encode("\nQuestion: reply with OK.", add_special_tokens=False)

    def one(idx: int, mt: int) -> dict:
        if ids is not None:
            # rotate the tail so consecutive calls differ without changing length
            rot = ids[:] 
            rot[-4:] = ids[-(4 + idx % 4):] + ids[-4:-(4 + idx % 4)] if idx else ids[-4:]
            return call(p, k, "", mt, prompt_token_ids=rot)
        return call(p, k, build_prompt(ctx, idx), mt)

    log = args.log
    n0 = 0
    if log and os.path.exists(log):
        n0 = int(subprocess.run(["wc", "-l", log], capture_output=True,
                                text=True).stdout.split()[0])

    one(0, 2)  # discard: startup costs

    pre_s, pre_tok, out_ms, per_rep = [], [], [], []
    for rep in range(args.reps):
        one(900 + rep, 4)                     # warm this context
        pre = one(100 + rep, 1)               # prefill only
        full = one(200 + rep, ndec)           # prefill + decode
        pre_s.append(pre["s"])
        pre_tok.append(pre["usage"]["prompt_tokens"] / max(pre["s"], 1e-6))
        ct = full["usage"]["completion_tokens"] - 1
        dec_t = max(full["s"] - pre["s"], 1e-6)
        out_ms.append(1000.0 * dec_t / max(ct, 1))
        per_rep.append((round(dec_t, 3), ct))

    st = spec_stats(log, n0) if log else {}
    acc_len = st.get("acceptance_len_median")
    med_out_ms = statistics.median(out_ms)

    # ms per target verify pass = ms per output token x accepted tokens per pass.
    # acceptance_len counts accepted tokens per pass including the always-present
    # first token, so passes = output_tokens / acceptance_len.
    ms_per_pass = (med_out_ms * acc_len) if acc_len else None
    passes_per_100 = (100.0 / acc_len) if acc_len else None

    row = {
        "tag": args.tag,
        "ctx": ctx,
        "exact_tokens": args.exact_tokens or None,
        "k": args.k or None,
        "prompt_tokens": None,
        "reps": args.reps,
        "prefill_s_med": round(statistics.median(pre_s), 4),
        "prefill_tok_s_med": round(statistics.median(pre_tok), 1),
        "decode_out_tokens": per_rep[0][1] if per_rep else None,
        # --- the three separated metrics ---
        "ms_per_output_token_med": round(med_out_ms, 3),
        "ms_per_output_token_all": [round(x, 3) for x in out_ms],
        "accepted_tokens_per_pass": acc_len,
        "ms_per_target_pass": round(ms_per_pass, 3) if ms_per_pass else None,
        "target_passes_per_100_out": round(passes_per_100, 1) if passes_per_100 else None,
        "output_tok_s": round(1000.0 / med_out_ms, 2) if med_out_ms else None,
        "drafted_tokens": st.get("drafted_tokens"),
        "accepted_tokens": st.get("accepted_tokens"),
    }
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
