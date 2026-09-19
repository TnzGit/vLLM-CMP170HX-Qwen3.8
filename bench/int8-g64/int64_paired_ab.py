#!/usr/bin/env python3
"""Paired A/B for the int64 widening, on the HISTORICAL metric.

WHY THIS EXISTS
---------------
The `.to(tl.int64)` widening in `_spec_attn_partial` fixes a real correctness defect
(handover 56) but changes the address arithmetic of a hot kernel, so it may have a
performance cost. Phase 1's unpaired numbers suggested ~-3.8% at 4K C1 and ~-3.9% at
4K C2 -- right at the edge of this project's +-2-3% noise gate -- but those came from
`full_wall - independent_prefill_wall`, which is retired.

This measures the cost properly:

  * **paired and interleaved (ABBA)**, so drift cannot masquerade as a difference;
  * the **historical metric**: `ms/speculative iteration = decode_s * 1000 /
    vllm:spec_decode_num_drafts_total`, with `decode_s` the DIRECT first-content-chunk
    to stream-end interval (the definition recovered in docs/m7-historical-contract.md).
    No prefill subtraction anywhere;
  * same checkpoint, same prompt token IDs, same graph mode, same clocks for both arms;
  * acceptance reported alongside, because a throughput difference can come from
    acceptance rather than from the kernel.

The only difference between arms is the kernel source file, which the caller swaps
between rounds. This script does not swap it -- it takes a `--arm` label so the caller
controls the swap and the label cannot drift from the code.

Usage:
  int64_paired_ab.py --port 8002 --log <log> --tokenizer <dir> \
      --corpus-root <tree> --context 4096 --concurrency 1 \
      --arm old --rounds 5 --out /tmp/int64_ab.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import time
import urllib.request

METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_preemptions_total",
)


def get_json(url: str) -> dict:
    """GET with the serving key.

    Without the Authorization header this 401s against the lab units, which made the
    first ABBA run fail at its very first call -- twenty engine restarts, no data. The
    model name is also passed explicitly by the driver now, so this is only a fallback.
    """
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {_api_key()}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _api_key() -> str:
    import os
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


def metrics(base: str) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as r:
        lines = r.read().decode().splitlines()
    out = {n: 0.0 for n in METRICS}
    for line in lines:
        for n in METRICS:
            if line.startswith(n + "{") or line.startswith(n + " "):
                out[n] += float(line.rsplit(" ", 1)[-1])
    return out


def build_corpus(root, limit_chars: int = 8_000_000) -> str:
    """Identical rule to bench/context_ab.py, so the corpus matches the historical one."""
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
        raise RuntimeError(f"no usable corpus files under {root}")
    return "".join(parts)


def exact_prompt_ids(tok, corpus: str, length: int, salt: str) -> list[int]:
    """Identical rule to bench/context_ab.py::exact_prompt_ids."""
    corpus_ids = tok.encode(corpus, add_special_tokens=False, truncation=True,
                            max_length=max(length, 1))
    if not corpus_ids:
        raise RuntimeError("corpus tokenized to an empty sequence")
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


def stream_run(base: str, model: str, ids: list[int], output: int) -> dict:
    """The historical measurement: direct decode interval + spec-iteration denominator."""
    payload = json.dumps({
        "model": model, "prompt": ids, "max_tokens": output,
        "temperature": 0.0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_api_key()}"})
    before = metrics(base)
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
    after = metrics(base)
    d = {n: after[n] - before[n] for n in METRICS}
    gen = int(usage.get("completion_tokens", output))
    ttft = (first or end) - start
    decode_s = max(end - (first or end), 1e-9)
    steps = max(d["vllm:spec_decode_num_drafts_total"], 1.0)
    accepted = d["vllm:spec_decode_num_accepted_tokens_total"]
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", len(ids))),
        "output_tokens": gen,
        "ttft_s": ttft,
        "decode_s": decode_s,
        "spec_iterations": steps,
        "ms_per_spec_iteration": decode_s * 1000.0 / steps,
        "accepted_tokens_per_pass": 1.0 + accepted / steps,
        "accepted_tokens": accepted,
        "output_tok_s": max(gen - 1, 0) / decode_s,
        "preemptions": d["vllm:num_preemptions_total"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002")
    ap.add_argument("--model", default="")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--output", type=int, default=256)
    ap.add_argument("--arm", required=True, help="'old' or 'fixed' -- labels only")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--salt-prefix", default="ab")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    corpus = build_corpus(args.corpus_root)
    corpus_sha = hashlib.sha256(corpus.encode()).hexdigest()
    model = args.model or get_json(args.base.rstrip("/") + "/v1/models")["data"][0]["id"]

    rows = []
    for rnd in range(args.rounds):
        # distinct prompt IDs per round and per arm: same length, same corpus, but the
        # salt differs so a prefix-cache hit cannot carry across an arm swap.
        ids_list = [exact_prompt_ids(tok, corpus, args.context,
                                     f"{args.salt_prefix}-{args.arm}-r{rnd}-{i}")
                    for i in range(args.concurrency)]
        if args.concurrency == 1:
            r = stream_run(args.base, model, ids_list[0], args.output)
        else:
            with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                rs = list(pool.map(lambda ids: stream_run(args.base, model, ids,
                                                          args.output), ids_list))
            # Aggregate by summing numerators and denominators rather than averaging
            # per-request ratios, so one slow request cannot dominate the ratio.
            # Note for concurrency: the decode interval of the BATCH is what a
            # concurrent caller experiences, so ms/spec-iteration uses the max decode
            # wall over requests divided by the TOTAL iterations issued.
            it = sum(x["spec_iterations"] for x in rs)
            acc = sum(x["accepted_tokens"] for x in rs)
            dec = max(x["decode_s"] for x in rs)
            out_tok = sum(x["output_tokens"] for x in rs)
            r = {
                "prompt_tokens": len(ids_list[0]),
                "output_tokens": out_tok,
                "ttft_s": max(x["ttft_s"] for x in rs),
                "decode_s": dec,
                "spec_iterations": it,
                "ms_per_spec_iteration": dec * 1000.0 / max(it, 1.0),
                "accepted_tokens_per_pass": 1.0 + acc / max(it, 1.0),
                "accepted_tokens": acc,
                "output_tok_s": out_tok / max(dec, 1e-9),
                "preemptions": sum(x["preemptions"] for x in rs),
            }
        row = {"arm": args.arm, "round": rnd, "context": args.context,
               "concurrency": args.concurrency, "corpus_sha256": corpus_sha,
               "prompt_sha256": hashlib.sha256(
                   json.dumps(ids_list).encode()).hexdigest(), **r}
        rows.append(row)
        print(json.dumps(row), flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
