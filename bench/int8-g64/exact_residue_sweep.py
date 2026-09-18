#!/usr/bin/env python3
"""Exact-token residue sweep: find where the long-context fault actually starts.

Why exact tokens: this repository has already had a bug that broke at exactly one
prompt length in 128 (see the launcher notes on the KVarN residue series), so a
sweep labelled "63K / 66K / 70K" from character-counted prompts is not enough to
locate a fault or to declare a clean range. The prompt is built as an exact
number of token IDs and the sweep steps by 1 token across a full residue period
at each modulus that matters for this stack.

For every length the script reports OK or the exact fault, and it restarts the
engine after any fault (a crashed engine keeps returning stale failures and
invalidates everything measured after it -- handover 28.2).

Usage:
  exact_residue_sweep.py --port 8002 --log <log> --tokenizer <path> \
      --start 65500 --end 65620 --decode-tokens 8
  exact_residue_sweep.py --port 8002 --log <log> --tokenizer <path> \
      --centre 66000 --period 128 --decode-tokens 8
"""
from __future__ import annotations

import argparse
import json
import os
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


def build_ids(tok, length: int, salt: int) -> list[int]:
    """Exactly `length` token IDs, distinct content per salt."""
    unit = tok.encode(FILLER + f"[s{salt:05d}] ", add_special_tokens=False)
    if not unit:
        raise SystemExit("tokenizer produced no tokens for the filler")
    body = (unit * (length // len(unit) + 1))[:length]
    # rotate the head so different salts differ early; a full-length reassignment
    # cannot change the length (a slice assignment with a longer RHS would).
    if salt and len(body) > 8:
        k = salt % 8
        body = body[k:] + body[:k]
    assert len(body) == length, f"built {len(body)} tokens, wanted {length}"
    return body


def call(port, key, ids, max_tokens):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps({"model": "qwen3.8-27b", "prompt": ids, "max_tokens": max_tokens,
                         "temperature": 0.0, "seed": 0, "ignore_eos": True}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        o = json.loads(r.read())
    return time.perf_counter() - t0, o.get("usage", {})


def restart(unit, log, wait_s=140):
    subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
    time.sleep(3)
    n = int(subprocess.run(["wc", "-l", log], capture_output=True, text=True).stdout.split()[0])
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    subprocess.run(["systemctl", "--user", "start", unit], capture_output=True)
    for _ in range(wait_s // 5):
        time.sleep(5)
        tail = subprocess.run(["tail", "-n", f"+{n+1}", log],
                              capture_output=True, text=True).stdout
        if "Application startup complete" in tail:
            return True
        st = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState",
                             "--value"], capture_output=True, text=True).stdout.strip()
        if st == "failed":
            return False
    return False


def faults_since(log, n):
    tail = subprocess.run(["tail", "-n", f"+{n+1}", log],
                          capture_output=True, text=True).stdout
    return tail.count("illegal memory access")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--log", required=True)
    ap.add_argument("--unit", default="triton-int8-control-8002.service")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--centre", type=int, default=0)
    ap.add_argument("--period", type=int, default=0,
                    help="with --centre, sweep centre..centre+period-1")
    ap.add_argument("--decode-tokens", type=int, default=8)
    ap.add_argument("--k-residues", default="",
                    help="comma-separated draft depths; for each, also probe the "
                         "lengths implied by 'bad residue = 117 + k' so a "
                         "spec-geometry fault appears as a shift with k instead of "
                         "requiring a full 128-length scan")
    ap.add_argument("--no-restart", action="store_true",
                    help="do not restart the engine after a fault (diagnosis only)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    if args.centre and args.period:
        lengths = list(range(args.centre, args.centre + args.period))
    else:
        lengths = list(range(args.start, args.end + 1))

    if args.k_residues:
        # bad prompt residue is 123 for k=7 in the historical series; each step
        # down in k moves it by 2 (117 + k). Probe each implied length +-1.
        base = args.centre or args.start or 65531
        ks = [int(x) for x in args.k_residues.split(",")]
        extra = []
        for kk in ks:
            implied = base + (kk - 7) * -2
            extra += [implied - 1, implied, implied + 1]
        lengths = sorted(set(extra))
        print(f"k-residue probe (base {base}, bad residue 117+k): lengths {lengths}",
              flush=True)

    first_fault, ok_count, fault_count = None, 0, 0
    for i, L in enumerate(lengths):
        n = int(subprocess.run(["wc", "-l", args.log], capture_output=True,
                               text=True).stdout.split()[0])
        ids = build_ids(tok, L, i)
        try:
            dt, u = call(args.port, args.key, ids, args.decode_tokens)
            got = u.get("prompt_tokens")
            if got != L:
                raise RuntimeError(
                    f"length mismatch: requested {L} tokens, server saw {got}; "
                    "an OK here would not be evidence about length L")
            f = faults_since(args.log, n)
            if f:
                raise RuntimeError(f"{f} fault records in log")
            ok_count += 1
            print(f"  L={L:<7} OK   {dt:6.2f}s  prompt_tokens={u.get('prompt_tokens')}",
                  flush=True)
        except Exception as e:
            fault_count += 1
            if first_fault is None:
                first_fault = L
            print(f"  L={L:<7} FAULT  {type(e).__name__}: {str(e)[:90]}", flush=True)
            if not args.no_restart:
                if restart(args.unit, args.log):
                    print("        engine restarted", flush=True)
                else:
                    print("        ENGINE RESTART FAILED -- stopping", flush=True)
                    break

    print(json.dumps({
        "lengths_tested": len(lengths), "ok": ok_count, "faulted": fault_count,
        "first_fault_length": first_fault,
        "residues": ({m: (first_fault % m if first_fault else None)
                      for m in (16, 32, 64, 128, 448, 896)} if first_fault else None),
    }), flush=True)


if __name__ == "__main__":
    main()
