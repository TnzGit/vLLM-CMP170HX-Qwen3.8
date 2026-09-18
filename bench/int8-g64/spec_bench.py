#!/usr/bin/env python3
"""Speculative-decode aware split benchmark: kernel cost vs speculation cost.

Why this exists
---------------
An earlier revision reported "decode ms/step" computed as

    (full_wall - prefill_wall) / (completion_tokens - 1)

`completion_tokens - 1` is the number of **generated output tokens**, not the
number of DFlash verify passes, and DFlash2 accepts ~3.3-3.4 tokens per pass. The
figure was therefore **ms per output token** and the claim that it was
"independent of DFlash acceptance" was wrong. This version reports the three
quantities that move independently:

    ms per output token        what a user feels
    ms per target pass         kernel efficiency  (derived, see below)
    accepted tokens per pass   proposal efficiency

and it labels the middle one honestly as **derived**: vLLM's own
`Mean acceptance length` is `1 + accepted_draft_tokens / num_spec_steps`, so
`ms/output-token x acceptance` estimates pass cost, but it is a product of two
independently windowed measurements rather than a request-local counter. It is
named `estimated_ms_per_target_pass` until the engine exposes per-request
`num_spec_steps` / `num_accepted_draft_tokens`.

Window alignment
----------------
The `SpecDecoding metrics` logger emits **periodic windowed and reset** counters,
not per-request cumulative ones. The previous version read every window after a
start marker and took a median, which need not correspond to the measured reps at
all. This version brackets each measured request: it records the log length
before and after the request and reads only the windows that appeared inside that
bracket, so acceptance is attributed to the request that produced it.

Exact-token mode
----------------
The previous implementation was silently wrong: it assumed a fixed 8-token tail,
and its list-slice rotation could change the list length
(`rot[-4:] = <5 items>` grows the list), so it was unusable for residue work.
This version builds the prompt from real token IDs, pads or trims to the
requested length, and asserts `usage.prompt_tokens == requested` on every call.

`--permute-salts` exists because a fixed salt order cannot distinguish a
per-content acceptance effect from a position/warmup effect: run the same three
salts in three orders and see which one the timing follows.

Usage:
  spec_bench.py --port 8002 --tag g64 --ctx 32768 --reps 3 --log <server log>
  spec_bench.py --port 8002 --tag g64 --exact-tokens 65531 --reps 3 --log <log>
  spec_bench.py --port 8002 --tag probe --permute-salts --ctx 32768 --log <log>
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
    """Resolve the serving key: env, then api_key.txt, then the lab unit default.

    That last fallback is not a secret -- it is written in plain text into the
    isolated units -- but without it these harnesses 401 against them, which is a
    regression an earlier version of this helper introduced.
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


def build_text_prompt(ctx_tokens: int, idx: int) -> str:
    """Distinct text per idx; length is approximate and asserted downstream."""
    unit = FILLER + f"[case {idx:04d}] "
    body = unit * max(1, (ctx_tokens - 64) * 4 // len(unit))
    return (f"Read the text below and then answer.\n{body}\n"
            f"Question: reply with OK then the number {idx}.")


def build_exact_ids(tok, length: int, salt: int) -> list[int]:
    """Exactly `length` token IDs, with content that varies per salt.

    Rotation uses a full-length reassignment, never a slice assignment whose
    right-hand side has a different length, because that silently changes the
    sequence length -- the one thing this mode must guarantee.
    """
    unit = tok.encode(FILLER + f"[s{salt:05d}] ", add_special_tokens=False)
    if not unit:
        raise SystemExit("tokenizer produced no tokens for the filler")
    ids = (unit * (length // len(unit) + 1))[:length]
    if salt and length > 8:
        k = salt % 8
        ids = ids[k:] + ids[:k]          # same length by construction
    assert len(ids) == length, (len(ids), length)
    return ids


def call(port: int, key: str, prompt, max_tokens: int) -> dict:
    payload = {"model": "qwen3.8-27b", "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "seed": 0, "ignore_eos": True}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.loads(r.read())
    return {"s": time.perf_counter() - t0, "usage": out.get("usage", {})}


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
    acc = [float(m.group(1)) for m in re.finditer(r"Mean acceptance length: ([0-9.]+)", text)]
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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--decode-tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--log", default="")
    ap.add_argument("--exact-tokens", type=int, default=0)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--permute-salts", action="store_true",
                    help="run the same three salts in three orders to separate a "
                         "content effect from a position/warmup effect")
    args = ap.parse_args()
    p, k = args.port, args.key
    ndec = args.decode_tokens

    tok = None
    if args.exact_tokens:
        if not args.tokenizer:
            ap.error("--exact-tokens needs --tokenizer")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def one(idx: int, mt: int) -> dict:
        if tok is not None:
            ids = build_exact_ids(tok, args.exact_tokens, idx)
            r = call(p, k, ids, mt)
            got = r["usage"].get("prompt_tokens")
            assert got == args.exact_tokens, (
                f"exact-token mode violated: requested {args.exact_tokens}, "
                f"server saw {got}")
            return r
        return call(p, k, build_text_prompt(args.ctx, idx), mt)

    one(0, 2)  # discard startup

    if args.permute_salts:
        orders = [(200, 201, 202), (202, 200, 201), (201, 202, 200)]
        print("salt-order probe (same three salts, three orders):", flush=True)
        for order in orders:
            row = []
            for salt in order:
                one(salt + 500, 4)
                lo = log_lines(args.log)
                r = one(salt, ndec)
                hi = log_lines(args.log)
                st = window_stats(args.log, lo, hi) if args.log else {}
                ct = r["usage"]["completion_tokens"] - 1
                ms = 1000.0 * r["s"] / max(ct, 1)
                row.append({"salt": salt, "ms_per_out_tok": round(ms, 3),
                            "acceptance": st.get("acceptance_mean")})
            print("  " + json.dumps({"order": list(order), "results": row}), flush=True)
        return

    pre_s, pre_tok, out_ms, reps = [], [], [], []
    for rep in range(args.reps):
        one(900 + rep, 4)                       # warm this context
        pre = one(100 + rep, 1)                 # prefill only
        lo = log_lines(args.log)
        full = one(200 + rep, ndec)             # prefill + decode
        hi = log_lines(args.log)
        st = window_stats(args.log, lo, hi) if args.log else {}
        ct = full["usage"]["completion_tokens"] - 1
        dec_t = max(full["s"] - pre["s"], 1e-6)
        pre_s.append(pre["s"])
        pre_tok.append(pre["usage"]["prompt_tokens"] / max(pre["s"], 1e-6))
        out_ms.append(1000.0 * dec_t / max(ct, 1))
        reps.append({"ms_per_out_tok": round(out_ms[-1], 3),
                     "out_tokens": ct,
                     "acceptance": st.get("acceptance_mean"),
                     "windows": st.get("windows"),
                     "drafted": st.get("drafted"), "accepted": st.get("accepted")})

    med = statistics.median(out_ms)
    accs = [r["acceptance"] for r in reps if r["acceptance"]]
    acc = statistics.mean(accs) if accs else None
    row = {
        "tag": args.tag,
        "ctx": args.ctx,
        "exact_tokens": args.exact_tokens or None,
        "k": args.k or None,
        "reps": args.reps,
        "prefill_s_med": round(statistics.median(pre_s), 4),
        "prefill_tok_s_med": round(statistics.median(pre_tok), 1),
        "ms_per_output_token_med": round(med, 3),
        "ms_per_output_token_all": [round(x, 3) for x in out_ms],
        "accepted_tokens_per_pass": round(acc, 3) if acc else None,
        "estimated_ms_per_target_pass": round(med * acc, 3) if acc else None,
        "target_passes_per_100_out": round(100.0 / acc, 1) if acc else None,
        "output_tok_s": round(1000.0 / med, 2) if med else None,
        "per_rep": reps,
    }
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
