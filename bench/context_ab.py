#!/usr/bin/env python3
"""Deterministic single-stream context benchmark for kernel A/B work.

The prompt is sent as token IDs so its length is exact.  A varied corpus is
assembled from text/source files under ``--corpus-root`` and tokenized once;
``--salt`` changes the first block to defeat prefix-cache reuse between A/B
samples without changing the requested length.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_preemptions_total",
)


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def metrics(base: str) -> dict[str, float]:
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as response:
        lines = response.read().decode().splitlines()
    out = {name: 0.0 for name in METRICS}
    for line in lines:
        for name in METRICS:
            if line.startswith(name + "{") or line.startswith(name + " "):
                out[name] += float(line.rsplit(" ", 1)[-1])
    return out


def build_corpus(root: Path, limit_chars: int = 8_000_000) -> str:
    parts: list[str] = []
    size = 0
    for path in sorted(root.rglob("*")):
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


def exact_prompt_ids(tokenizer, corpus: str, length: int, salt: str) -> list[int]:
    # Truncate while tokenizing instead of materializing millions of tokens and
    # then slicing them.  Besides lowering host-memory use, this avoids the
    # misleading "sequence is longer than the model maximum" warning in long
    # context A/B runs.
    corpus_ids = tokenizer.encode(
        corpus,
        add_special_tokens=False,
        truncation=True,
        max_length=max(length, 1),
    )
    if not corpus_ids:
        raise RuntimeError("corpus tokenized to an empty sequence")
    header = tokenizer.encode(
        f"Controlled benchmark sample {salt}. Read the following source archive.\n",
        add_special_tokens=False,
    )
    query = tokenizer.encode(
        "\nGive one concise technical observation about the archive.",
        add_special_tokens=False,
    )
    body_len = max(0, length - len(header) - len(query))
    repeats = (body_len + len(corpus_ids) - 1) // len(corpus_ids)
    ids = header + (corpus_ids * repeats)[:body_len] + query
    if len(ids) > length:
        ids = ids[:length]
    elif len(ids) < length:
        ids += corpus_ids[: length - len(ids)]
    assert len(ids) == length
    return ids


def run(base: str, model: str, prompt_ids: list[int], output_tokens: int) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    before = metrics(base)
    start = time.perf_counter()
    first = None
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("text") and first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    after = metrics(base)
    delta = {name: after[name] - before[name] for name in METRICS}
    generated = int(usage.get("completion_tokens", output_tokens))
    ttft = (first or end) - start
    decode_s = max(end - (first or end), 1e-9)
    steps = max(delta["vllm:spec_decode_num_drafts_total"], 1.0)
    accepted = delta["vllm:spec_decode_num_accepted_tokens_total"]
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", len(prompt_ids))),
        "output_tokens": generated,
        "ttft_s": ttft,
        "decode_s": decode_s,
        "decode_tok_s": max(generated - 1, 0) / decode_s,
        "spec_steps": steps,
        "accepted_tokens": accepted,
        "tokens_per_step": 1.0 + accepted / steps,
        "ms_per_step": decode_s * 1000.0 / steps,
        "preemptions": delta["vllm:num_preemptions_total"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8002")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--output", type=int, default=256)
    parser.add_argument("--salt", default="a")
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    model = args.model or get_json(args.base.rstrip("/") + "/v1/models")["data"][0]["id"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    ids = exact_prompt_ids(tokenizer, build_corpus(args.corpus_root), args.context, args.salt)
    result = {"label": args.label, "context": args.context, "salt": args.salt}
    result.update(run(args.base, model, ids, args.output))
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
