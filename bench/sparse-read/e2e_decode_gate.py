"""Single-request long-context decode measurement for the sparse-read gate.

The server must already be running on the research port.  The same prompt file
must be reused for dense and sparse arms.  The script reports two separate
intervals:

1. streamed decode throughput: first content token -> stream completion, with
   the first token excluded from the numerator;
2. speculative counter interval: first /metrics snapshot midpoint -> final
   /metrics snapshot midpoint.  Counter-derived ratios use only counters from
   those same two snapshots.

This avoids combining request-total counters with a shorter decode window.

Example:
  python bench/sparse-read/e2e_decode_gate.py \
      --tag dense-250k --prompt-file ~/bench/sparse-read/ctx250k.txt \
      --contract ~/bench/sparse-read/ctx250k.contract.json

The first run creates the contract.  Later arms must match its prompt SHA256 and
server-reported prompt token count exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_preemptions_total",
)


def api_key(repo: Path) -> str:
    value = os.environ.get("VLLM_API_KEY", "")
    if value:
        return value
    for path in (
        repo / "api_key.txt",
        Path.home() / "qwen-serving" / "api_key.txt",
    ):
        try:
            return path.read_text().strip()
        except OSError:
            pass
    return ""


def request(url: str, key: str, payload: dict | None = None):
    headers = {}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=data, headers=headers)


def metric_snapshot(base: str, key: str) -> tuple[float, dict[str, float]]:
    t0 = time.perf_counter()
    body = urllib.request.urlopen(
        request(base + "/metrics", key), timeout=30
    ).read().decode()
    t1 = time.perf_counter()
    values: dict[str, float] = {}
    for line in body.splitlines():
        for name in COUNTERS:
            if line.startswith(name + " ") or line.startswith(name + "{"):
                values[name] = values.get(name, 0.0) + float(line.split()[-1])
    return (t0 + t1) / 2.0, values


def delta(a: dict[str, float], b: dict[str, float], name: str) -> float | None:
    if name not in a or name not in b:
        return None
    return b[name] - a[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompt-file", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--task",
        default=(
            "Continue by explaining the document's implementation details. "
            "Be concrete and use the document only; do not mention this benchmark."
        ),
    )
    ap.add_argument("--results-dir", type=Path, default=Path.home() / "bench/results")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    key = api_key(repo)
    base = f"http://127.0.0.1:{args.port}"

    raw = args.prompt_file.read_bytes()
    prompt_sha = hashlib.sha256(raw).hexdigest()
    document = raw.decode("utf-8")
    content = (
        "SPARSE-READ PERFORMANCE GATE.\n\n"
        + document
        + "\n\nCURRENT TASK:\n"
        + args.task
    )

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    req = request(base + "/v1/chat/completions", key, payload)
    request_begin = time.perf_counter()
    first_content = None
    counter_start = None
    usage: dict = {}
    pieces: list[str] = []

    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            event = json.loads(body)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if not text:
                    continue
                pieces.append(text)
                if first_content is None:
                    first_content = time.perf_counter()
                    counter_start = metric_snapshot(base, key)

    stream_end = time.perf_counter()
    counter_end = metric_snapshot(base, key)

    if first_content is None or counter_start is None:
        raise SystemExit("no streamed content; measurement invalid")
    if not usage:
        raise SystemExit("stream did not include usage; measurement invalid")

    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    if completion_tokens < 2:
        raise SystemExit("fewer than two completion tokens; decode interval invalid")

    contract = {
        "prompt_file": str(args.prompt_file),
        "prompt_sha256": prompt_sha,
        "prompt_tokens": prompt_tokens,
    }
    if args.contract.exists():
        expected = json.loads(args.contract.read_text())
        for key_name in ("prompt_sha256", "prompt_tokens"):
            if expected.get(key_name) != contract[key_name]:
                raise SystemExit(
                    f"prompt contract mismatch for {key_name}: "
                    f"expected={expected.get(key_name)!r} got={contract[key_name]!r}"
                )
    else:
        args.contract.parent.mkdir(parents=True, exist_ok=True)
        args.contract.write_text(json.dumps(contract, indent=2) + "\n")

    decode_span = stream_end - first_content
    decode_tps = (completion_tokens - 1) / decode_span

    c0_t, c0 = counter_start
    c1_t, c1 = counter_end
    counter_span = c1_t - c0_t
    drafts = delta(c0, c1, "vllm:spec_decode_num_drafts_total")
    accepted = delta(c0, c1, "vllm:spec_decode_num_accepted_tokens_total")
    preemptions = delta(c0, c1, "vllm:num_preemptions_total")

    ms_per_step = None
    tok_per_step = None
    if drafts is not None and drafts > 0 and counter_span > 0:
        ms_per_step = counter_span * 1000.0 / drafts
        if accepted is not None:
            tok_per_step = 1.0 + accepted / drafts

    output = "".join(pieces)
    result = {
        "tag": args.tag,
        "prompt": contract,
        "completion_tokens": completion_tokens,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "request_wall_s": stream_end - request_begin,
        "ttft_s": first_content - request_begin,
        "decode_interval": {
            "start": "first_content_token",
            "end": "stream_completion",
            "seconds": decode_span,
            "numerator_tokens": completion_tokens - 1,
            "tok_s": decode_tps,
        },
        "counter_interval": {
            "start_midpoint": c0_t,
            "end_midpoint": c1_t,
            "seconds": counter_span,
            "draft_steps_delta": drafts,
            "accepted_tokens_delta": accepted,
            "preemptions_delta": preemptions,
            "ms_per_spec_iteration": ms_per_step,
            "tok_per_step": tok_per_step,
        },
        "runtime_env_intent": {
            "VLLM_SPARSE_KV_READ": os.environ.get("VLLM_SPARSE_KV_READ"),
            "VLLM_SPARSE_KV_READ_TOKENS": os.environ.get(
                "VLLM_SPARSE_KV_READ_TOKENS"
            ),
            "VLLM_SPARSE_KV_READ_SINK_TOKENS": os.environ.get(
                "VLLM_SPARSE_KV_READ_SINK_TOKENS"
            ),
        },
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"sparse_read_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    print(
        f"SPARSE_GATE {args.tag} prompt={prompt_tokens} out={completion_tokens} "
        f"ttft={result['ttft_s']:.3f}s decode={decode_tps:.2f} tok/s "
        f"counter_ms/step={ms_per_step if ms_per_step is not None else 'null'} "
        f"tok/step={tok_per_step if tok_per_step is not None else 'null'} "
        f"preemptions={preemptions if preemptions is not None else 'null'}"
    )
    print(out_path)


if __name__ == "__main__":
    main()
