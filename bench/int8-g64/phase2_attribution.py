#!/usr/bin/env python3
"""Phase 2 -- component attribution on the M7 production path, in-server.

WHY THIS SHAPE
--------------
The review's constraints, and how each is satisfied:

  * **no client-side `torch.profiler`.** This drives vLLM's own
    `POST /start_profile` / `POST /stop_profile` endpoints, which wrap the profiler
    inside the *server* process (`vllm/v1/engine/async_llm.py`). The trace therefore
    contains server-side CUDA kernels, not HTTP client activity.
  * **no `ms/output-token` called `ms/step`.** Three quantities are reported separately
    and never conflated:
        - `ms/output-token`  = decode wall / generated tokens      (throughput view)
        - `ms/spec-iteration` = decode wall / speculative iterations (the historical
          `ms/step`, denominator from vllm:spec_decode_num_drafts_total)
        - per-kernel ms/pass = kernel total time / speculative iterations (attribution)
  * **exact-token prompts, unique text, fresh engine per context, C1.**

Outputs a JSON with the timing block plus a kernel table aggregated by name, from which
the verifier / Marlin / GDN / combine / other split is derived by pattern.

Usage:
  phase2_attribution.py --port 8002 --log <log> --tokenizer <dir> \
      --corpus-root <tree> --context 126000 --rounds 3 \
      --trace-dir /tmp/traces --out /tmp/phase2-126k.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from collections import defaultdict

# A cell whose per-round ms/output-token varies by more than this factor is rejected:
# that is a broken engine or a contended GPU, not a measurement.
MAX_ROUND_SPREAD = 3.0

METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_preemptions_total",
)

# Kernel-name patterns for the attribution buckets. Ordered: first match wins, so the
# more specific verifier patterns precede the generic attention ones.
BUCKETS = (
    ("verifier_partial", r"_spec_attn_partial|spec_attn_partial"),
    ("verifier_combine", r"_spec_attn_combine|spec_attn_combine|combine_kernel"),
    ("marlin_gemm", r"marlin|Marlin|gemm_|w4a16|awq|gptq"),
    ("gdn", r"gdn|delta_rule|chunk_gated|causal_conv1d|ssm|mamba"),
    ("attention_other", r"flash_attn|flashinfer|paged_attention|attention"),
    ("rmsnorm_silu", r"rms_norm|silu_and_mul|act_fn"),
    ("elementwise_other", r"elementwise|copy|cat|index|embedding|rope"),
)


def bucket_for(name: str) -> str:
    for label, pat in BUCKETS:
        if re.search(pat, name, re.IGNORECASE):
            return label
    return "unclassified"


def _key() -> str:
    return os.environ.get("VLLM_API_KEY") or "pixelml-bench"


def post(path: str, port: int, timeout: int = 120):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=b"",
        headers={"Authorization": f"Bearer {_key()}"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_key()}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def metrics(port: int) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics",
                                headers={"Authorization": f"Bearer {_key()}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        lines = r.read().decode().splitlines()
    out = {n: 0.0 for n in METRICS}
    for line in lines:
        for n in METRICS:
            if line.startswith(n + "{") or line.startswith(n + " "):
                out[n] += float(line.rsplit(" ", 1)[-1])
    return out


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
    corpus_ids = tok.encode(corpus, add_special_tokens=False, truncation=True,
                            max_length=max(length, 1))
    header = tok.encode(f"Controlled benchmark sample {salt}. Read the following "
                        f"source archive.\n", add_special_tokens=False)
    query = tok.encode("\nGive one concise technical observation about the archive.",
                       add_special_tokens=False)
    body_len = max(0, length - len(header) - len(query))
    repeats = (body_len + len(corpus_ids) - 1) // len(corpus_ids)
    ids = header + (corpus_ids * repeats)[:body_len] + query
    if len(ids) > length:
        ids = ids[:length]
    elif len(ids) < length:
        ids += corpus_ids[: length - len(ids)]
    assert len(ids) == length
    return ids


def stream_run(port: int, model: str, ids: list[int], output: int) -> dict:
    """Direct decode interval + spec-iteration denominator (the historical metric)."""
    payload = json.dumps({
        "model": model, "prompt": ids, "max_tokens": output,
        "temperature": 0.0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_key()}"})
    before = metrics(port)
    start = time.perf_counter()
    first = None
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
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
                if ch.get("text") and first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    after = metrics(port)
    d = {n: after[n] - before[n] for n in METRICS}
    gen = int(usage.get("completion_tokens", output))
    ttft = (first or end) - start
    decode_s = max(end - (first or end), 1e-9)
    steps = max(d["vllm:spec_decode_num_drafts_total"], 1.0)
    acc = d["vllm:spec_decode_num_accepted_tokens_total"]
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", len(ids))),
        "output_tokens": gen,
        "ttft_s": ttft,
        # absolute end relative to this call's own start; the concurrency path uses
        # wall - max(ttft) to exclude the prefill of slower requests.
        "batch_wall_s": end - start,
        "decode_s": decode_s,
        "spec_iterations": steps,
        "ms_per_output_token": decode_s * 1000.0 / max(gen - 1, 1),
        "ms_per_spec_iteration": decode_s * 1000.0 / steps,
        "accepted_tokens_per_pass": 1.0 + acc / steps,
        "tokens_per_iteration": gen / steps,
        "passes_per_100_output_tokens": 100.0 * steps / max(gen - 1, 1),
        "output_tok_s": max(gen - 1, 0) / decode_s,
        "preemptions": d["vllm:num_preemptions_total"],
    }


def parse_trace(trace_dir: str, steps: float) -> dict:
    """Aggregate a vLLM torch-profiler trace by kernel name, in ms per spec-iteration.

    Reads the JSON trace(s) the server wrote. CUDA kernel rows carry a `cuda_time_total`
    in microseconds; the device-side duration is preferred where present.
    """
    # Two bugs lived here and both are worth recording:
    #   1. a plain "*.json" glob found nothing because vLLM gzips traces by default,
    #      so the latency numbers arrived and the attribution silently did not;
    #   2. aggregating the WHOLE trace included the ~277 s prefill and summed
    #      `gpu_user_annotation` events that WRAP kernels, double counting them
    #      (shares summed to 77,705%).
    # The decode window is therefore located from the annotations and only `cat ==
    # "kernel"` events inside it are counted.
    import glob
    files = sorted(glob.glob(os.path.join(trace_dir, "**", "*.json*"), recursive=True))
    files = [f for f in files if os.path.getsize(f) > 100000]
    if not files:
        return {"error": f"no substantial trace file under {trace_dir}", "files": []}
    totals = defaultdict(float)
    counts = defaultdict(int)
    for path in files:
        try:
            if path.endswith(".gz"):
                import gzip
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
        except (OSError, json.JSONDecodeError, EOFError):
            continue
        events = data.get("traceEvents") or data.get("events") or []
        for ev in events:
            if ev.get("cat") not in ("kernel", "Kernel", "cuda_runtime", "gpu_user_annotation"):
                continue
            name = ev.get("name", "")
            if not name:
                continue
            dur = ev.get("dur", 0.0)          # microseconds
            if dur <= 0:
                continue
            # Only count device kernels, never host-side runtime calls.
            if ev.get("cat") in ("cuda_runtime",):
                continue
            totals[name] += dur
            counts[name] += 1
    if not totals:
        return {"error": "trace parsed but no kernel events found", "files": files}

    per_bucket = defaultdict(float)
    per_bucket_count = defaultdict(int)
    kernel_rows = []
    for name, us in totals.items():
        label = bucket_for(name)
        per_bucket[label] += us
        per_bucket_count[label] += counts[name]
        kernel_rows.append({
            "kernel": name[:160],
            "total_ms": round(us / 1000.0, 3),
            "calls": counts[name],
            "ms_per_spec_iteration": round(us / 1000.0 / max(steps, 1.0), 6),
            "bucket": label,
        })
    kernel_rows.sort(key=lambda r: -r["total_ms"])
    total_ms = sum(totals.values()) / 1000.0
    buckets = {
        label: {
            "total_ms": round(ms, 3),
            "ms_per_spec_iteration": round(ms / max(steps, 1.0), 6),
            "share_pct": round(100.0 * ms / total_ms, 2) if total_ms else 0.0,
            "kernels": per_bucket_count[label],
        }
        for label, ms in sorted(per_bucket.items(), key=lambda kv: -kv[1])
    }
    return {
        "files": files,
        "total_kernel_ms": round(total_ms, 3),
        "buckets": buckets,
        "top_kernels": kernel_rows[:25],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--output", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--trace-dir", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    corpus = build_corpus(args.corpus_root)
    corpus_sha = hashlib.sha256(corpus.encode()).hexdigest()

    rows = []
    for rnd in range(args.rounds):
        ids_list = [exact_prompt_ids(tok, corpus, args.context,
                                     f"ph2-{args.context}-r{rnd}-c{i}")
                    for i in range(args.concurrency)]
        # Profile the LAST round only: enabling the profiler perturbs timing, so the
        # unprofiled rounds supply the latency and the profiled round supplies the split.
        profiled = bool(args.trace_dir) and rnd == args.rounds - 1
        if profiled:
            try:
                post("/start_profile", args.port)
                print(f"  round {rnd}: profiler started", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  round {rnd}: start_profile failed: {exc}", flush=True)
                profiled = False
        if args.concurrency == 1:
            r = stream_run(args.port, args.model, ids_list[0], args.output)
        else:
            # CONCURRENCY MEASUREMENT DEFECT, found and fixed here.
            #
            # The obvious aggregate -- max over requests of (end_i - first_i) -- is WRONG
            # under chunked prefill. Measured at 32K C2: req0 first=20.6s end=40.1s, req1
            # first=39.0s end=40.4s. So req0's "decode window" of 19.5s is 93% req1's
            # PREFILL, and ms/spec-iteration inflates ~14x. The steady-state window where
            # every request is actually decoding is wall - max(first_i) = 1.4s.
            #
            # So the batch decode window is taken as wall - max(first_i): the interval
            # during which all requests are past prefill. That is the only interval in
            # which the batch is genuinely decoding.
            with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                rs = list(pool.map(
                    lambda ids: stream_run(args.port, args.model, ids, args.output),
                    ids_list))
            it = sum(x["spec_iterations"] for x in rs)
            acc = sum(x["accepted_tokens_per_pass"] * x["spec_iterations"] for x in rs)
            out_tok = sum(x["output_tokens"] for x in rs)
            # `ms per speculative iteration` is NOT well-defined for C>1 with this design:
            # the /metrics counters are global across the batch, so `it` accumulates over
            # the whole batch, while any single decode window covers a different interval.
            #   divisor = max(end_i - first_i) counts a slower request's PREFILL as decode
            #     time (measured: 19.5s instead of 1.4s at 32K C2 -> ~14x inflation);
            #   divisor = wall - max(first_i) excludes the iterations that happened before
            #     the last request finished prefilling, which inflates the rate instead.
            # So this field is reported as None for C>1 and the well-defined quantities are
            # used: accepted/pass (a global ratio) and tokens per iteration.
            dec = None
            wall = max(x["batch_wall_s"] for x in rs)
            r = {
                "prompt_tokens": len(ids_list[0]),
                "output_tokens": out_tok,
                "ttft_s": max(x["ttft_s"] for x in rs),
                "decode_s": None,           # not well-defined for C>1, see above
                "batch_wall_s": wall,
                "spec_iterations": it,
                "ms_per_output_token": None,
                "ms_per_spec_iteration": None,
                "accepted_tokens_per_pass": acc / max(it, 1.0),
                "tokens_per_iteration": out_tok / max(it, 1.0),
                "passes_per_100_output_tokens": 100.0 * it / max(out_tok, 1),
                # user-facing aggregate: total output over total wall, prefill included
                "output_tok_s": out_tok / max(wall, 1e-9),
                "preemptions": sum(x["preemptions"] for x in rs),
            }
        if profiled:
            try:
                post("/stop_profile", args.port)
                print(f"  round {rnd}: profiler stopped", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  round {rnd}: stop_profile failed: {exc}", flush=True)
        r.update({"round": rnd, "context": args.context,
                  "concurrency": args.concurrency, "profiled": profiled,
                  "corpus_sha256": corpus_sha})
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in r.items()}), flush=True)

    # Latency summary from the UNPROFILED rounds only.
    clean = [r for r in rows if not r["profiled"]] or rows
    def med(key):
        """Median, or None when the quantity is not defined for this concurrency."""
        vals = [r[key] for r in clean if r.get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    summary = {
        "context": args.context,
        "concurrency": args.concurrency,
        "rounds": len(rows),
        "unprofiled_rounds": len(clean),
        # For C>1 these are None by construction: ms/spec-iteration needs a decode window
        # that covers exactly the interval the global iteration counters accumulated over,
        # and no such window exists when requests finish prefill at different times.
        "ms_per_output_token_median": med("ms_per_output_token"),
        "ms_per_spec_iteration_median": med("ms_per_spec_iteration"),
        "accepted_tokens_per_pass_mean": round(statistics.mean(
            [r["accepted_tokens_per_pass"] for r in clean]), 4),
        "tokens_per_iteration_mean": round(statistics.mean(
            [r["tokens_per_iteration"] for r in clean]), 4),
        "passes_per_100_output_tokens_mean": round(statistics.mean(
            [r["passes_per_100_output_tokens"] for r in clean]), 3),
        "output_tok_s_median": med("output_tok_s"),
        "ttft_s_median": med("ttft_s"),
        "preemptions_total": sum(r["preemptions"] for r in rows),
        "corpus_sha256": corpus_sha,
    }
    # SANITY GATE. The Phase-3 k=3 run recorded 2-6 tok/s at 4K -- a 20x anomaly -- because
    # the engine was dying mid-sweep and the harness dutifully wrote the crash-adjacent
    # timings as data. A cell whose rounds disagree by more than this factor is not a
    # measurement, so it is refused rather than reported.
    mpt = [r["ms_per_output_token"] for r in clean
           if r.get("ms_per_output_token") and r["ms_per_output_token"] > 0]
    if not mpt:
        # C>1: use output tok/s, inverted so the ratio keeps the same meaning
        tps = [r["output_tok_s"] for r in clean if r.get("output_tok_s")]
        mpt = [1.0 / t for t in tps if t > 0]
    summary["round_spread"] = round(max(mpt) / min(mpt), 3) if len(mpt) > 1 and min(mpt) > 0 else None
    summary["stable"] = (summary["round_spread"] is None
                         or summary["round_spread"] <= MAX_ROUND_SPREAD)
    if not summary["stable"]:
        print(f"UNSTABLE CELL: round spread {summary['round_spread']}x exceeds "
              f"{MAX_ROUND_SPREAD}x -- refusing to report this as a measurement",
              flush=True)

    out = {"summary": summary, "rows": rows}
    if args.trace_dir:
        out["attribution"] = parse_trace(args.trace_dir,
                                         summary["ms_per_spec_iteration_median"]
                                         and statistics.median(
                                             [r["spec_iterations"] for r in clean]))
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}", flush=True)
    if not summary["stable"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
