#!/usr/bin/env python3
"""Fault-rate measurement for the ~64K illegal-access fault.

Section 32.6 of the handover retracted the "discrete at 2^16" model: prompt length
65536 faulted in one run and passed in another, so a single observation at a
single length cannot distinguish a deterministic trigger from a probabilistic
one. For a fault with no confirmed mechanism the useful measurement is a **rate**,
not a boundary.

For each requested length this sends N independent requests (each with distinct
content, exact token count verified against the server's own `prompt_tokens`) and
reports how many faulted. The engine is restarted after every fault, because a
crashed engine keeps returning stale failures and would otherwise inflate the
rate of every subsequent request (handover 28.2).

Usage:
  fault_rate.py --port 8002 --log <log> --tokenizer <path> \
      --lengths 65521,65531,65536,65552 --repeats 8 --decode-tokens 8
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
    unit = tok.encode(FILLER + f"[r{salt:05d}] ", add_special_tokens=False)
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
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        o = json.loads(r.read())
    return time.perf_counter() - t0, o.get("usage", {})


def log_lines(path):
    return int(subprocess.run(["wc", "-l", path], capture_output=True,
                              text=True).stdout.split()[0])


def restart_cmd(cmd, log, wait_s=180):
    """Restart a directly-launched engine and wait for readiness."""
    subprocess.run(["bash", "-c", cmd], capture_output=True)
    for _ in range(wait_s // 5):
        time.sleep(5)
        tail = subprocess.run(["tail", "-n", "80", log],
                              capture_output=True, text=True).stdout
        if "Application startup complete" in tail:
            return True
    return False


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
        st = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState",
                             "--value"], capture_output=True, text=True).stdout.strip()
        if st == "failed":
            return False
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--key", default=_api_key())
    ap.add_argument("--log", required=True)
    ap.add_argument("--unit", default="triton-int8-control-8002.service")
    ap.add_argument("--restart-cmd", default="",
                    help="shell command that stops and restarts the SAME engine "
                         "configuration being measured; required when the arm was "
                         "launched directly rather than via a systemd unit, because "
                         "restarting the unit would silently change the config "
                         "(e.g. --max-num-batched-tokens) and invalidate the arm")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--decode-tokens", type=int, default=8)
    ap.add_argument("--tag", default="rate")
    ap.add_argument("--ready-wait", type=int, default=300)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    lengths = [int(x) for x in args.lengths.split(",")]

    table = {}
    for L in lengths:
        ok = fault = 0
        detail = []
        for i in range(args.repeats):
            n = log_lines(args.log)
            try:
                dt, u = call(args.port, args.key, build_ids(tok, L, i), args.decode_tokens)
                if u.get("prompt_tokens") != L:
                    raise RuntimeError(f"length mismatch {u.get('prompt_tokens')} != {L}")
                tail = subprocess.run(["tail", "-n", f"+{n+1}", args.log],
                                      capture_output=True, text=True).stdout
                if "illegal memory access" in tail:
                    raise RuntimeError("fault records in log")
                ok += 1
                detail.append("ok")
            except Exception as e:
                fault += 1
                detail.append(f"fault:{type(e).__name__}")
                # A crashed engine keeps returning stale failures, so a fault must
                # be followed by a restart of the SAME configuration before the
                # next sample is meaningful (handover 28.2). Only the fault path
                # restarts; on success there is nothing to do.
                restarted = (restart_cmd(args.restart_cmd, args.log, args.ready_wait)
                             if args.restart_cmd else restart(args.unit, args.log))
                if not restarted:
                    print("  engine restart FAILED -- aborting", flush=True)
                    table[L] = {"ok": ok, "fault": fault, "rate": None,
                                "detail": detail, "aborted": True}
                    print(json.dumps({"tag": args.tag, "table": table}), flush=True)
                    return
            print(f"  L={L} rep{i+1}/{args.repeats}: {detail[-1]} "
                  f"(ok={ok} fault={fault})", flush=True)
        table[L] = {"ok": ok, "fault": fault,
                    "rate": round(fault / max(1, ok + fault), 3), "detail": detail}

    print(json.dumps({"tag": args.tag, "repeats": args.repeats,
                      "decode_tokens": args.decode_tokens, "table": table}), flush=True)


if __name__ == "__main__":
    main()
