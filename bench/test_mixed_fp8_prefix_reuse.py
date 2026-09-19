"""A -> B -> A prefix-cache qualification for the mixed-FP8 profile.

The first request is cold. B keeps the same long document prefix but changes the
question. The final A repeats the first request exactly.  The test records API
reported cached tokens, TTFT, output hashes, and deterministic output parity.
"""

import hashlib
import json
import os
import sys
import time
import urllib.request


PORT = os.environ.get("PORT", "8002")
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")
CTX = int(os.environ.get("CTX_TOKENS", "10000"))
OUT = int(os.environ.get("OUT_TOKENS", "64"))
RUN_SALT = os.environ.get("RUN_SALT", "")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")

FILLER = (
    "The RTX 3090 has 24 GB of GDDR6X and 82 streaming multiprocessors. "
    "Memory bandwidth is 936 GB/s, which is what decode is bound by. "
)
TOK_PER_REPEAT = 44
DOCUMENT = f"Qualification run: {RUN_SALT}\n" + FILLER * max(
    1, round(CTX / TOK_PER_REPEAT)
)


def run(question: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"Document:\n\n{DOCUMENT}\n\n{question}",
            }
        ],
        "max_tokens": OUT,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=headers)
    started = time.perf_counter()
    first = None
    chunks: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            event = json.loads(body)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                text = choice.get("delta", {}).get("content")
                if text:
                    if first is None:
                        first = time.perf_counter()
                    chunks.append(text)
    ended = time.perf_counter()
    content = "".join(chunks)
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0) or 0,
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_s": (first or ended) - started,
        "total_s": ended - started,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "preview": content[:120],
    }


question_a = "Explain in two sentences why memory bandwidth matters for decoding."
question_b = "List three hardware facts from the document, one per line."
rows = [("A-cold", run(question_a)), ("B", run(question_b)), ("A-warm", run(question_a))]
for name, row in rows:
    print(
        f"{name:6s} prompt={row['prompt_tokens']} cached={row['cached_tokens']} "
        f"ttft={row['ttft_s']:.3f}s total={row['total_s']:.3f}s "
        f"out={row['completion_tokens']} sha256={row['sha256'][:16]}"
    )

cold = rows[0][1]
warm = rows[2][1]
parity = cold["sha256"] == warm["sha256"]
hit = warm["cached_tokens"] > 0
print(
    f"RESULT parity={parity} cache_hit={hit} "
    f"ttft_speedup={cold['ttft_s'] / max(warm['ttft_s'], 1e-9):.2f}x"
)

if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as f:
        json.dump({"ctx_tokens": CTX, "rows": rows, "parity": parity, "cache_hit": hit}, f, indent=2)

raise SystemExit(0 if parity and hit else 1)
