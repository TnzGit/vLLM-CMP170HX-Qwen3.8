# CMP 170HX 64GB deployment snapshot

This repository is the source/patch stack used by a LAN inference server on a
single NVIDIA CMP 170HX 64GB (SM80). It is published as a development baseline,
so changes can be reviewed here and then pulled back to the server.

## Reproducibility boundary

- Base package: vLLM `0.27.1`
- PyTorch: `2.13.0`
- CUDA runtime/toolkit: CUDA 13
- Deployed recipe base: upstream commit
  `69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c`
- All patches listed by `verify.sh`, plus both KVarN patches, passed the bundled
  installed-content verification on the deployment host.
- The Python environment, compiled wheel, model weights, caches, logs and API
  credentials are intentionally not committed.

## Hardware and operating envelope

- GPU: 1 x NVIDIA CMP 170HX 64GB
- Power limit: 180 W
- Temperature guard: 80 C
- API bind: `0.0.0.0:8002`, LAN only, no API key
- Served model name: `qwen3.8-27b`

## Model pair used for the snapshot

- Target: `Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4`
- Draft: `Qwen3.8-27B-DFlash2-W4A16`
- Speculator: DFlash2, 7 speculative tokens

Model weights are not part of this repository. Configure `MODEL` and `DRAFT`
with local absolute paths.

## Published launch profiles

`deploy/pixelml-vllm-8002.service.example` records the systemd environment used
for the 250K/KVarN/C4/vision snapshot. The launcher remains
`single-user/start_qwen.sh`; systemd only supplies its environment.

Important semantics:

- `MAX_SEQS` is an admission limit, not a guarantee that every admitted request
  can keep its entire maximum context resident simultaneously.
- Image tokens consume the same context budget as text tokens.
- The deployment intentionally leaves `VLLM_API_KEY` unset because it is exposed
  only inside a trusted LAN. Do not expose this configuration to the Internet.
- Do not commit model files, `.env`, `api_key.txt`, tokens, logs or virtual
  environments.

## Baseline measurements at 180 W

Correct usage-token accounting (`scripts/bench-usage.py`) is required; SSE event
counts are not token counts.

With BF16 KV, 64K, C1 and text-only on the same target/draft pair:

| Case | Result |
|---|---:|
| 256-token decode | 133.82 tok/s |
| 900-token decode | 126.78 tok/s |
| 6,603-token prefill | 1,868.7 tok/s |

With KVarN K4V2, 250K, C4 admission and vision enabled:

| Case | Result |
|---|---:|
| 256-token decode | 124.22 tok/s |
| 900-token decode | 107.82 tok/s |
| 6,603-token prefill | 1,767.7 tok/s |

The KVarN profile initialized a 1,978,593-token KV pool and reported theoretical
7.91x concurrency at 250,000 tokens per request. A real image request completed
successfully after startup.

## Validation before deployment

Run the repository verification against the target venv before replacing the
server environment:

```bash
PY=/path/to/venv/bin/python bash verify.sh --install
```

Then validate in an isolated port before changing the service:

1. health and model listing;
2. deterministic text generation;
3. one real image request if vision is enabled;
4. usage-token-counted C1 baseline;
5. requested concurrency and long-context tests;
6. CUDA errors, temperature, power and VRAM headroom.

