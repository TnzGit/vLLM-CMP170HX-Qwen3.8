# E38/E39 INT8-G64 vLLM integration (experimental)

## Scope

This is the smallest whole-model bridge for the E38/E39 INT8-G64 attention
kernel. It keeps the existing W4A16 target, DFlash2 drafter, scheduler, logical
block IDs and prefix-cache protocol. Only the target full-attention verify
backend uses the new cache dtype; sliding-window/GDN layers stay on their
existing path.

The logical payload per layer, per 64-token page, is

```text
K int8 codes:       4 × 64 × 256   = 65,536 B
V int8 codes:       4 × 64 × 256   = 65,536 B
K scales FP16:      4 × 64 × 4     =  2,048 B  (one per 64-dim group)
V scales FP16:      4 × 64 × 4     =  2,048 B
                                     --------
                                     135,168 B
```

## Page layout: what the allocator actually hands over

The original bridge assumed those four planes were **contiguous across the
whole cache** (`[all K][all V][all Kscales][all Vscales]`, page stride 65,536 B)
and rejected anything else. The allocator does not produce that: it pads every
physical page to its own stride — measured `1,777,664` at `max_model_len=32000`
— so the layout is **page-local**:

```text
[page][K 65,536][V 65,536][Kscales 2,048][Vscales 2,048][padding]
```

The two layouts are not interconvertible by reinterpretation: byte-sentinel
tests show that reading the padded storage as if it were contiguous lands on
page padding instead of the next page's codes, which is what produced the
recorded warmup failure
(`INT8-G64 requires VLLM_KV_CACHE_LAYOUT=HND; got strides=(1777664, 16384, 256, 1),
expected=(65536, 16384, 256, 1)`).

`g64_views()` therefore returns four **page-local views** that share the
allocator's page stride and differ only by in-page offset:

| view | in-page byte offset | element type |
| --- | --- | --- |
| K codes | +0 | int8 |
| V codes | +65,536 | int8 |
| K scales | +131,072 | fp16 |
| V scales | +133,120 | fp16 |

Overlapping or misaligned pages are still rejected rather than reinterpreted;
only the ABI changed, not the strictness. The E38 decode adapter is the
stride-aware port (`page_stride_adapter.cu` + `page_stride_kernel.cuh` +
`paged_kv_address_strided.cuh`), which derives each plane's page stride from the
views instead of folding the old compile-time 64-token constant. Measured cost
of the runtime `int64` page stride versus the compile-time constant: **0.87x –
1.01x** across batch 1/4 and split 4/16/32, i.e. no measurable penalty.

## Required runtime contract

```text
VLLM_KV_CACHE_LAYOUT=HND
--block-size 64
--attention-backend TRITON_ATTN
--kv-cache-dtype int8_g64
VLLM_SPEC_DECODE_ATTN=1                      # gates the E38 verify branch
VLLM_INT8_G64_SRC=/.../page_stride_adapter.cu
VLLM_INT8_G64_EXT_NAME=vllm_int8_g64_e38_strided
VLLM_INT8_G64_NINFER_ROOT=/.../ninfer-cmp170hx-reference
```

`VLLM_SPEC_DECODE_ATTN=1` is not optional: `_spec_attn_enabled()` reads it and
the E38 branch is gated on it, so with the variable unset the bridge is dead
code and verify silently falls back to a prefill-shaped Triton kernel.

The extension is built lazily in the test venv with `-maxrregcount=170`; the
accepted E38 resource check is 168 registers/thread, zero spill, and 49,088 B
shared memory. Production 8000 is not modified by this patch.

The isolated launcher is `int8g64-8002.service`, an experiment-side unit on the
lab host (not tracked here). It uses the same W4A16 target and DFlash2 draft as
the existing 8002 recipe, with `CTX=g64`, `MAX_SEQS=4`, `DFLASH_TOKENS=7`,
`G64_MAX_LEN=32000`, and no pinned `KV_MEM` so the 64-GiB card can size the pool
from `GPU_UTIL`. The unit is intentionally not enabled; start it only for an
experiment and stop it before restoring any other 8002 workload.

`G64_MAX_LEN` is capped by the allocation problem below, not by the model: at
`262144` the engine refuses to start at all.

On this SM80 card, the stock vLLM Triton backend does **not** expose native
FP8 KV (it rejects the mode on compute capability 8.0). Therefore the
requested “Triton-FP8” control must be recorded as the exact runnable control
mode (the repository's existing `int8_per_token_head` Triton route, or a
separately verified FlashInfer-FP8 route), rather than being labelled FP8 when
the runtime actually uses INT8. The A/B report must state this limitation and
compare identical model, prompts, seed, power limit, graph mode, and DFlash2
settings.

## Acceptance gates

1. KV spec/page bytes and allocation: no OOB, no under-budget allocation.
2. Prefill writer oracle: K/V dequantized error against the E39 CPU reference.
3. C1/C2/C4 fixed eight-query verify: finite output and deterministic replay.
4. CUDA Graph capture/replay at each fixed context length.
5. Locked 1350 MHz whole-model A/B against the same W4A16 checkpoint and
   DFlash2 configuration, with 4K/126K/250K and C1/C2/C4.
6. DFlash2 acceptance, usage-token decode tok/s, TTFT/prefill, peak VRAM,
   power/temperature, and fixed prompt/seed output comparison.

Any failed gate leaves this mode experimental and the patch unapplied to
production.

## Gate status after the 2026-09-17 integration

| gate | status | evidence |
| --- | --- | --- |
| 1 page bytes / no OOB | **fail (memory)** | pool 65,362 tokens vs 529,060 for bf16; 13.15x page padding, see below |
| 2 prefill writer oracle | pass | `test_merged_bridge.py`, `test_g64_batch34_prefill.py` (batch 1..4, `max_abs` 0.0125) |
| 3 C1/C2/C4 verify | pass | 4k C1/C2/C4 = 43.46 / 43.02 / 41.51 tok/s, `dup=False` |
| 4 CUDA Graph replay | pass | graph-mode startup completes with E38 engaged |
| 5 locked A/B 4K/126K/250K | **partial** | 4k measured both arms; 126k/250k blocked by gate 1; baseline arm unstable at C4 and 32k |
| 6 acceptance / quality | pass at 4k | 9/12 byte-identical vs bf16, 78.73% token prefix, mean abs dlogprob 0.00385 |

Allocation is the blocking gate. `max_page_size` comes from the 48 MambaSpec
(linear-attention) layers at 1,777,664 B. The bf16 attention page
(131,072 = 2^17) divides it (`1,835,008 = 2^18·7` at 65k max len), so the bf16
arm takes the **zero-waste block-scaling** path. The G64 page
(135,168 = 2^12·33) does **not** divide `1,777,664 = 2^13·217`, so the G64
attention layers are forced onto `page_size_padded` and waste 13.15x; the 5
sliding-window draft layers waste 52.6x. At `G64_MAX_LEN=262144` the engine
refuses outright: `114.23 GiB KV cache is needed ... (37.85 GiB available)`.

Fixing it is not a local change: scaling the G64 block from 64 to 448 would
break the 64-token page contract that `g64_views`, the writer and the E38
kernel all depend on. The correct fix is to stop unifying every layer onto the
Mamba page and give the linear-attention layers their own KV cache group with
their own page size.
