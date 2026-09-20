#!/usr/bin/env python3
"""Concurrency root-cause telemetry for the 250K C2/C4 anomaly.

WHY THIS EXISTS
---------------
The 250K C2/C4 extension reported ~757 ms per request-iteration against ~38 ms at C1, and
~2-30 tok/s against ~65 tok/s. I first wrote that off as a measurement-definition problem
because the server's own log also reported single-digit generation throughput. That
inference was too quick, and it was also based on a bad argument: I compared CLI flags
between runs, and "the flag is absent from my launcher" says nothing about the value the
engine actually parsed. What matters is the *parsed* `SchedulerConfig` plus what the
scheduler was doing at the time.

FOUR CANDIDATE MECHANISMS (the review's own taxonomy)
----------------------------------------------------
  A. capacity / admission serialization -- the C requests never ran concurrently because
     the KV budget admitted them one at a time. Then preemption=0, Xid=0, rc=0 are all
     expected, and calling it a "C2 verifier benchmark" is simply wrong.
  B. prefill/decode scheduler interference -- running=C and waiting=0, but a long chunked
     prefill of one request is still occupying the GPU while another decodes. The decode
     window then measures *service latency under interference*, not verifier cost.
  C. genuine decode-concurrency collapse -- all requests finished prefill, prompt
     throughput is 0, and generation is still single-digit. Only this case justifies a
     profiler.
  D. measurement-only artifact -- aggregate throughput is fine and only my derived
     ms/spec-iteration is wrong.

This harness produces the evidence needed to tell them apart, and refuses to guess:
it samples server telemetry at ~1 Hz and records per-request first-token and finish
timestamps so the *steady-state* window (where every request is genuinely decoding) can be
isolated.

THE STEADY-STATE WINDOW
-----------------------
    steady_start = max_i(first_output_i)
    steady_end   = min_i(finish_i)
    analyse only [steady_start, steady_end]

Inside that window all C requests are past prefill and none has finished, so the global
counter deltas are attributable to genuine concurrent decode. Outside it they are not.

Usage:
  conc_telemetry.py --port 8002 --tokenizer <dir> --corpus-root <tree> \
      --cells 4096:2,32768:2,126000:2,250000:2,250000:4 --output 256 \
      --out /tmp/conc_telemetry.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import statistics
import threading
import time
import urllib.request

GAUGE_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    # "_by_reason" separates "waiting for KV capacity" from ordinary queueing, which is
    # exactly the Case A (admission) vs Case B (interference) discriminator.
    "vllm:num_requests_waiting_by_reason",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def _key() -> str:
    return os.environ.get("VLLM_API_KEY") or "pixelml-bench"


def scrape(port: int) -> dict:
    """Return every metric of interest plus the raw metric names for discovery."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics",
                                headers={"Authorization": f"Bearer {_key()}"})
    with urllib.request.urlopen(req, timeout=10) as r:
        lines = r.read().decode().splitlines()
    out: dict[str, float] = {}
    for line in lines:
        if line.startswith("#"):
            continue
        head = line.rsplit(" ", 1)
        if len(head) != 2:
            continue
        metric_id = head[0]
        name = metric_id.split("{")[0]
        try:
            val = float(head[1])
        except ValueError:
            continue
        if name not in GAUGE_METRICS:
            continue
        if name == "vllm:num_requests_waiting_by_reason":
            # Keep the reason split: 'capacity' means the KV/scheduling budget could not
            # admit the request at all (Case A), while 'deferred' is a transient
            # constraint. Summing these together would destroy the discriminator.
            for reason in ("capacity", "deferred"):
                if f'reason="{reason}"' in metric_id:
                    out[f"{name}[{reason}]"] = out.get(f"{name}[{reason}]", 0.0) + val
            continue
        # sum across remaining label sets (e.g. per-engine)
        out[name] = out.get(name, 0.0) + val
    return out


class Sampler:
    """1 Hz telemetry sampler running in a background thread."""

    def __init__(self, port: int, interval: float = 1.0):
        self.port = port
        self.interval = interval
        self.samples: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                self.samples.append((t, scrape(self.port)))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)

    def window(self, t0: float, t1: float) -> dict:
        """Aggregate samples inside [t0, t1]."""
        sel = [(t, s) for t, s in self.samples if t0 <= t <= t1]
        if not sel:
            return {"samples": 0}
        keys = set()
        for _, s in sel:
            keys |= set(s)
        agg: dict[str, float] = {}
        for k in keys:
            vals = [s.get(k, 0.0) for _, s in sel]
            if k.endswith("_total"):
                agg[k + "__delta"] = round(vals[-1] - vals[0], 3)
            else:
                agg[k + "__max"] = round(max(vals), 3)
                agg[k + "__mean"] = round(statistics.mean(vals), 3)
        agg["samples"] = len(sel)
        agg["span_s"] = round(sel[-1][0] - sel[0][0], 2)
        return agg


def build_corpus(root, limit_chars: int = 8_000_000) -> str:
    from pathlib import Path
    parts, size = [], 0
    for path in sorted(Path(root).rglob("*")):
        if path.suffix not in {".py", ".md", ".txt", ".cu", ".cuh", ".h"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if text:
            parts.append(f"\n\nFILE {path.name}\n{text}")
            size += len(text)
        if size >= limit_chars:
            break
    if not parts:
        raise RuntimeError(f"no usable corpus under {root}")
    return "".join(parts)


def exact_prompt_ids(tok, corpus: str, length: int, salt: str) -> list[int]:
    ids = tok.encode(corpus, add_special_tokens=False, truncation=True, max_length=length)
    hdr = tok.encode(f"Controlled benchmark sample {salt}. Read the following source "
                     f"archive.\n", add_special_tokens=False)
    q = tok.encode("\nGive one concise technical observation about the archive.",
                   add_special_tokens=False)
    body = max(0, length - len(hdr) - len(q))
    reps = (body + len(ids) - 1) // len(ids)
    out = hdr + (ids * reps)[:body] + q
    if len(out) > length:
        out = out[:length]
    elif len(out) < length:
        out += ids[: length - len(out)]
    assert len(out) == length
    return out


def one_request(port: int, model: str, ids: list[int], max_tokens: int) -> dict:
    """Stream one request, recording first-content and finish timestamps and chunk times."""
    payload = json.dumps({
        "model": model, "prompt": ids, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_key()}"})
    start = time.perf_counter()
    first = None
    chunk_times: list[float] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                if ch.get("text"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    chunk_times.append(now)
    end = time.perf_counter()
    return {
        "start": start, "first": first, "end": end,
        "chunk_times": chunk_times,
        "prompt_tokens": int(usage.get("prompt_tokens", len(ids))),
        "output_tokens": int(usage.get("completion_tokens", max_tokens)),
    }


def analyse_cell(port, model, tok, corpus, ctx, conc, max_tokens, out_len):
    """Run one (ctx, concurrency) cell with 1 Hz telemetry."""
    ids_list = [exact_prompt_ids(tok, corpus, ctx, f"ct-{ctx}-{conc}-{i}")
                for i in range(conc)]
    sampler = Sampler(port, interval=1.0)
    sampler.start()
    time.sleep(1.5)                      # a couple of idle samples for a baseline
    with cf.ThreadPoolExecutor(max_workers=conc) as pool:
        rs = list(pool.map(lambda ids: one_request(port, model, ids, max_tokens),
                           ids_list))
    time.sleep(1.0)
    sampler.stop()

    t0 = min(r["start"] for r in rs)
    firsts = [r["first"] or r["end"] for r in rs]
    ends = [r["end"] for r in rs]
    steady_start = max(firsts)           # all requests are decoding from here
    steady_end = min(ends)               # ... until the first one finishes
    steady = steady_end - steady_start
    total_out = sum(r["output_tokens"] for r in rs)

    win = sampler.window(steady_start, steady_end)
    # Full-run extremes as well, since Case A is about what happened BEFORE the window.
    full = sampler.window(t0, max(ends))
    # ITL proxy: inter-chunk intervals inside the steady window.
    itls: list[float] = []
    for r in rs:
        cs = [t for t in r["chunk_times"] if steady_start <= t <= steady_end]
        itls += [b - a for a, b in zip(cs, cs[1:])]

    return {
        "ctx": ctx, "concurrency": conc,
        "per_request": {
            "first_rel_s": [round(f - t0, 2) for f in firsts],
            "end_rel_s": [round(e - t0, 2) for e in ends],
            "output_tokens": [r["output_tokens"] for r in rs],
            "prompt_tokens": [r["prompt_tokens"] for r in rs],
        },
        "batch_wall_s": round(max(ends) - t0, 2),
        "steady_window_s": round(steady, 2),
        "steady_window_negative": steady <= 0,
        "steady_samples": win.get("samples", 0),
        "steady_running_max": win.get("vllm:num_requests_running__max"),
        "steady_waiting_max": win.get("vllm:num_requests_waiting__max"),
        "steady_kv_usage_max": win.get("vllm:kv_cache_usage_perc__max",
                                       win.get("vllm:gpu_cache_usage_perc__max")),
        "steady_preemptions_delta": win.get("vllm:num_preemptions_total__delta"),
        "steady_prompt_tokens_delta": win.get("vllm:prompt_tokens_total__delta"),
        "steady_generation_tokens_delta": win.get("vllm:generation_tokens_total__delta"),
        "steady_spec_iters_delta": win.get(
            "vllm:spec_decode_num_drafts_total__delta"),
        "steady_accepted_delta": win.get(
            "vllm:spec_decode_num_accepted_tokens_total__delta"),
        # NUMERATOR MUST COVER THE SAME INTERVAL AS THE DENOMINATOR. `total_out` counts the
        # whole request, including tokens emitted before the window opened, so dividing it by
        # the steady window inflated the rate (4K C2: 224 tok/s instead of 160). The
        # server's own generation-token delta inside the window is the correct numerator.
        # Use the SAMPLE SPAN as the denominator, not the full window: the counter delta
        # spans first-sample..last-sample, which is shorter than the window, so dividing by
        # `steady` under-reports (measured 91.5 tok/s vs ~119 for the same cell).
        "steady_sample_span_s": win.get("span_s"),
        "steady_tok_s": (round(win.get("vllm:generation_tokens_total__delta", 0.0)
                               / win["span_s"], 2)
                         if win.get("span_s") else None),
        "steady_tok_s_window_denom": (round(
            win.get("vllm:generation_tokens_total__delta", 0.0) / steady, 2)
            if steady > 0 else None),
        "steady_tok_s_naive_wrong": (round(total_out / steady, 2) if steady > 0 else None),
        "itl_p50_ms": (round(statistics.median(itls) * 1000, 2) if itls else None),
        "itl_p95_ms": (round(sorted(itls)[int(len(itls) * 0.95)] * 1000, 2)
                       if len(itls) > 1 else None),
        "itl_samples": len(itls),
        "full_running_max": full.get("vllm:num_requests_running__max"),
        "full_waiting_max": full.get("vllm:num_requests_waiting__max"),
        "full_waiting_capacity_max": full.get(
            "vllm:num_requests_waiting_by_reason[capacity]__max"),
        "full_waiting_deferred_max": full.get(
            "vllm:num_requests_waiting_by_reason[deferred]__max"),
        "steady_waiting_capacity_max": win.get(
            "vllm:num_requests_waiting_by_reason[capacity]__max"),
        "full_kv_usage_max": full.get("vllm:kv_cache_usage_perc__max"),
        "full_preemptions_delta": full.get("vllm:num_preemptions_total__delta"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--cells", default="4096:2,32768:2,126000:2,250000:2,250000:4")
    ap.add_argument("--output", type=int, dest="out_len", default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    corpus = build_corpus(args.corpus_root)

    rows = []
    for cell in args.cells.split(","):
        ctx_s, conc_s = cell.split(":")
        ctx, conc = int(ctx_s), int(conc_s)
        print(f"\n=== ctx={ctx} C{conc} (output={args.out_len}) ===", flush=True)
        try:
            r = analyse_cell(args.port, args.model, tok, corpus, ctx, conc,
                             args.out_len, args.out_len)
        except Exception as exc:  # noqa: BLE001
            print(f"  CELL FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            rows.append({"ctx": ctx, "concurrency": conc, "error": str(exc)[:200]})
            continue
        rows.append(r)
        print(json.dumps(r, indent=2), flush=True)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)

    print("\n=== SUMMARY TABLE ===")
    hdr = ("%8s %3s %10s %9s %9s %11s %11s %9s %8s" %
           ("ctx", "C", "wall_s", "steady_s", "run_max", "wait_max",
            "prompt_dtok", "steady_t/s", "itl_p50"))
    print(hdr)
    for r in rows:
        if r.get("error"):
            print("%8s %3s  ERROR %s" % (r["ctx"], r["concurrency"], r["error"][:40]))
            continue
        print("%8d %3d %10.1f %9.1f %9s %9s %11s %11s %9s" % (
            r["ctx"], r["concurrency"], r["batch_wall_s"], r["steady_window_s"],
            r.get("steady_running_max"), r.get("steady_waiting_max"),
            r.get("steady_prompt_tokens_delta"), r.get("steady_tok_s"),
            r.get("itl_p50_ms")))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
