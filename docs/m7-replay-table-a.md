# Table A — M7 reconstructed-protocol replay

**The frozen historical numbers reproduce.** 4K is within 0.3%, and the residual drift
grows monotonically with context to +2.96% at 250K.

## Conditions

| dimension | value |
| --- | --- |
| service | `cmp170hx-mixed-fp8-full-256k-8002.service` (started, not rebuilt) |
| effective env | `CTX=cmp-mixed-fp8 MAX_LEN=262144 MAX_SEQS=4 SPEC_ATTN=1 VLLM_FP8_SPEC_VERIFY=1 VLLM_FP8_SPEC_FULL_CG=1 VLLM_SPEC_DECODE_ATTN_SEGMENTS=35` |
| path | mixed-FP8: static FP8 (E4M3) target KV, BF16 draft KV, FlashInfer target attention + split-KV q8 verifier, **NSEG 35** |
| drafter | DFlash2 W4A16, k=7 |
| checkpoint | `Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4` |
| graph | **FULL** (`VLLM_FP8_SPEC_FULL_CG=1`) |
| power | **180 W**, per the historical condition; **no clock lock** (`-rgc`) |
| harness | `bench/context_ab.py` (the historical harness), output 256, C1 |
| metric | `ms/step = decode_s * 1000 / vllm:spec_decode_num_drafts_total`, `decode_s` = direct first-content-chunk to stream-end |
| corpus | reconstructed from the repo tree; **not** token-identical to the historical corpus |
| `corpus_sha256` | `ebf41c9d04b01f4dac6372c3626ada16bab85d2de2e884b39f9c4ed19f1cab91` |
| engine | fresh per context; 3 rounds per context |

## Results

| ctx | round | actual clock | power | **ms/step** | tokens/step | decode tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | 1 | 1140 MHz | 44.6 W | 21.962 | 3.266 | 147.0 |
| 4K | 2 | 1470 MHz | 67.8 W | 22.254 | 2.965 | 133.2 |
| 4K | 3 | 1470 MHz | 68.7 W | 22.348 | 3.000 | 134.2 |
| 65K | 1 | 1140 MHz | 45.4 W | 28.392 | 3.403 | 116.6 |
| 65K | 2 | 1455 MHz | 75.7 W | 29.153 | 3.364 | 113.6 |
| 65K | 3 | 1455 MHz | 81.6 W | 29.346 | 2.629 | 89.6 |
| 126K | 1 | 1140 MHz | 49.4 W | 35.031 | 3.253 | 92.1 |
| 126K | 2 | 1455 MHz | 87.0 W | 35.717 | 3.083 | 85.0 |
| 126K | 3 | 1455 MHz | 89.6 W | 35.735 | 3.295 | 91.5 |
| 250K | 1 | 1140 MHz | 50.5 W | 47.775 | 3.269 | 68.4 |
| 250K | 2 | 1455 MHz | 92.2 W | 48.418 | 3.108 | 63.5 |
| 250K | 3 | 1440 MHz | 90.9 W | 48.329 | 3.316 | 66.8 |

## Table A — comparison with the frozen historical values

| ctx | frozen M7 | reconstructed (median) | mean | sd | **delta** |
| --- | --- | --- | --- | --- | --- |
| 4K | 22.315 | **22.254** | 22.188 | 0.202 | **−0.27%** |
| 65K | — | 29.153 | 28.964 | 0.505 | — |
| 126K | 35.230 | **35.717** | 35.494 | 0.402 | **+1.38%** |
| 250K | 46.941 | **48.329** | 48.174 | 0.348 | **+2.96%** |

Every context: `Xid 31 delta = 0`.

## Reading the result

**The replay reproduces the frozen M7 baseline**, and the residual is structured rather
than random:

- **4K is within 0.3%**, i.e. inside the run-to-run spread (sd 0.20 ms on 22.3 ms = 0.9%).
  At 4K the prompt is 4096 tokens of a 390k-token corpus, so the reconstructed corpus is
  effectively a fresh document and acceptance lands at 2.97–3.27 — the historical regime.
- **The drift grows monotonically with context**: −0.27% → +1.38% → +2.96%. A monotone
  trend in context length is not what measurement noise looks like; it points at something
  that scales with the prompt. Two candidates, neither yet discriminated:
  1. **corpus content**: at 250K the prompt is 64% of a corpus pass, so it covers far more
     of the reconstructed document than the historical corpus covered of its own. Different
     text → different acceptance → different iteration count. `tokens/step` is 3.11–3.32 at
     250K against an unknown historical value, so this is plausible but unproven;
  2. **clock behaviour**: round 1 ran at 1140 MHz and rounds 2–3 at 1455/1440 MHz, so the
     rounds are not measured at identical clocks. Power stayed well under the 180 W limit
     (44–92 W), so the drift is not the power cap engaging.
- **Round 1 is the fastest at every context**, and also has the highest `tokens/step`. Since
  `ms/step` is per speculative iteration it should be acceptance-insensitive, so this is
  most likely the lower clock also being the lightly-loaded state, or a warm-up effect —
  flagged rather than explained.

Per the review's guidance, a 1–3% residual at 126K/250K is **not** grounds to declare the
historical numbers wrong. The honest summary is: **the frozen baseline reproduces at 4K and
is within ~3% at 250K under a reconstructed corpus, with a monotone context-dependent
residual whose cause is not yet isolated.**

This is a **reconstructed-protocol replay**, not a token-for-token replay: the historical
corpus is gone.
