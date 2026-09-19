# Post-27B roadmap: evolve this repository into a generic CMP 170HX / SM80 Qwen inference fork

Last updated: 2026-09-15

This document is intentionally **not** part of the current Qwen3.8-27B implementation plan.

Finish the 27B work described in `handover.md` first. Stabilize it, validate it on the real CMP 170HX, and get the current mixed-FP8/DFlash2 path into a known-good state before performing the structural changes in this document.

The purpose of this roadmap is to make the repository reusable for future Qwen models on CMP 170HX / SM80, especially Qwen3.8-Flash-Next and related Qwen4-preview architectures, without turning the current 27B work into an unmaintainable pile of model-specific patches.

---

## 1. Long-term repository identity

The current repository name is model-specific:

```text
vLLM-CMP170HX-Qwen3.8
```

The long-term goal should be conceptually closer to:

```text
CMP170HX / SM80 Qwen inference fork
```

The important abstraction boundary is:

```text
hardware/runtime platform
        +
model profile
```

not:

```text
one giant patch stack for one checkpoint
```

The platform layer should capture reusable SM80/CMP behavior:

- vLLM runtime fixes and backports;
- SM80-safe address arithmetic;
- quantized KV infrastructure;
- hybrid KV-cache geometry;
- recurrent-state correctness;
- speculative-decode infrastructure;
- CUDA Graph accounting and qualification;
- Marlin / quantized-weight loading behavior;
- prefix-cache instrumentation;
- deployment and Xid/NVRM validation methodology.

Each model family should then provide its own profile and model-specific patch layer.

---

## 2. Do not genericize before the 27B target is finished

The current 27B branch is the first complete reference target. It is valuable precisely because it gives us a known model, known drafter, known hardware, known failure cases, and known performance numbers.

Before restructuring the repository, the current 27B work should reach the Definition of Done in `handover.md`:

- FP8 split-KV verifier math validated on the real CMP 170HX;
- FlashInfer FP8 speculative verify path validated;
- target FP8 / DFlash BF16 backend separation validated;
- 896/448 cache geometry validated;
- prefix-cache reconciliation validated;
- exact 85,514-token stress point qualified;
- eager mode qualified before CUDA Graph;
- no new Xid/NVRM faults under the stress matrix;
- performance A/B recorded.

Only after that should the repository be reorganized.

Reason: if model implementation and repository abstraction are changed simultaneously, regressions become impossible to localize.

---

## 3. What from the 27B work is genuinely reusable

Not every patch should become a generic core patch. Some are platform/runtime work; others are specifically tied to Qwen3.8-27B + DFlash2.

### 3.1 Strong candidates for the generic SM80 platform layer

These should be preserved in a hardware/runtime-oriented layer where possible.

| Area | Reuse value | Notes |
|---|---:|---|
| high physical block-ID `tl.int64` addressing | very high | generic protection against 32-bit page/stride overflow in large KV pools |
| accepted-token bounds / recurrent-state safety | very high | relevant to GDN/Mamba-like recurrent state under speculative decode |
| overlapping recurrent-state copy fixes | high | runtime correctness, not a 27B-specific modeling choice |
| accepted-token cross-program/cross-stream race fixes | high | speculative/recurrent runtime concern |
| explicit V2 CUDA Graph memory accounting | high | generic memory budgeting on SM80 |
| mixed physical-page infrastructure | high | generic hybrid-cache requirement |
| KV group/page instrumentation | high | useful for every hybrid architecture |
| prefix-cache hit diagnostics | high | generic validation infrastructure |
| SM80 staged Marlin repack / load hygiene | medium-high | useful when the selected quantized linear backend is Marlin |
| CPU Marlin repack fallback | medium-high | safety fallback, not necessarily default |
| benchmark/token-accounting harnesses | very high | model-independent infrastructure |
| Xid/NVRM stress methodology | very high | hardware-platform qualification |
| gradual context qualification | very high | generic long-context methodology |

### 3.2 Keep these in the Qwen3.8-27B profile layer

These should **not** be presented as generic SM80 behavior.

- DFlash2-specific launcher and checkpoint preparation.
- DFlash2 lookup-augmented drafting.
- DFlash2 sliding-window layer assumptions.
- the exact 5-layer DFlash sliding-window topology.
- the 896-token FP8 target / 448-token BF16 DFlash geometry.
- the current full-attention split-KV speculative verifier dispatch policy.
- DFlash2 acceptance/tuning values such as `DFLASH_TOKENS=7/15`.
- KVarN integration as currently written for this 27B path.
- any hard-coded Qwen3.8-27B head counts, layer counts or page arithmetic.

These can still be examples of how to use the generic infrastructure, but they must live under the model profile rather than the platform core.

---

## 4. Why Flash-Next is not just another Qwen3.8 checkpoint

Before implementing Flash-Next support, re-check the upstream model and vLLM state because development is moving quickly.

As of 2026-09-15, the official Qwen3.8-Flash-Next release describes:

```text
main model               ~125B parameters
N-gram / PLE embeddings   ~51B additional parameters
active per token          ~6B
architecture              hybrid GDN + QSA
residual                  Gated Residual
attention                 Qwen Sparse Attention (QSA)
embedding                 N-gram / PLE-style embedding
model class               Qwen4-preview / Qwen4Exp family
```

The important point is architectural, not the exact parameter count:

```text
Qwen3.8-27B path
    full attention + GDN/Mamba-style hybrid pieces
    + external DFlash2 speculative drafter

Qwen3.8-Flash-Next path
    GDN + QSA sparse attention
    + sparse MoE
    + large N-gram/PLE embedding table
    + built-in MTP-style speculation
```

Therefore the 27B attention implementation cannot simply be copied into Flash-Next.

Official model reference:

```text
https://github.com/QwenLM/Qwen3.8-Flash-Next
```

Current vLLM optimization tracking reference:

```text
https://github.com/vllm-project/vllm/issues/55922
```

These links are references only. Re-check their current state before starting Flash-Next work.

---

## 5. What the 27B work teaches us for Flash-Next

The strongest reusable asset is not the exact kernel code. It is the runtime architecture and debugging methodology.

### 5.1 Recurrent/GDN correctness is still relevant

Flash-Next uses GDN extensively, so the following chain remains highly relevant:

```text
speculative accepted-token count
        -> recurrent state index
        -> state block selection
        -> state copy/update
        -> prefix-cache resume
        -> async scheduling / CUDA Graph ordering
```

The exact kernels may differ, but every accepted-token-derived index and every state copy must be audited with the same fail-closed rules used in the 27B work.

Do not assume a newer model has eliminated this class of bugs.

### 5.2 Quantized KV methodology is reusable even when the attention kernel changes

For Qwen3.8-27B we learned to separate:

```text
cache allocation / write format
from
attention read / dequant path
```

That same reasoning applies to QSA.

A current vLLM Flash-Next RFC demonstrates exactly this pattern for FP8 QSA KV:

- quantized allocation already exists;
- FP8 K/V scales already exist;
- KV write path is already dtype-aware;
- the missing piece is the QSA read/dequant path.

Reference:

```text
https://github.com/vllm-project/vllm/issues/54426
```

Do not copy that patch blindly. Use it as an architectural reference and re-evaluate it against the vLLM revision selected for this fork.

### 5.3 Large-page / high-block-ID safety remains relevant

Sparse attention does not remove large physical block IDs. Long contexts and large pools can still drive page IDs high enough that 32-bit `block_id * stride` arithmetic becomes dangerous.

Audit every custom Triton/CUDA path for:

```text
physical block id
page id
slot id
byte/element stride products
scale-cache stride products
```

and promote address arithmetic to 64-bit where required.

---

## 6. Flash-Next changes the performance bottlenecks

The 27B project is heavily about attention/KV/speculative verification.

Flash-Next adds two major new bottleneck classes:

```text
1. sparse MoE weight execution / residency
2. QSA + indexer cache execution
```

### 6.1 The 64 GB capacity problem

A raw 125B main model at 4 bits is approximately:

```text
125B * 4 bits ~= 62.5 GB
```

before accounting for:

- quantization scales and metadata;
- router/expert metadata;
- recurrent state;
- QSA KV cache;
- indexer cache;
- CUDA workspaces;
- CUDA Graph pools;
- temporary load/repack buffers.

Therefore a single 64 GB CMP 170HX cannot be approached as simply:

```text
"quantize the 125B main model to W4 and load it"
```

That leaves essentially no runtime headroom.

The 51B N-gram/PLE embedding is a separate large object and should be expected to live off-GPU on a 64 GB target.

The eventual Flash-Next plan must include an explicit **weight residency strategy**, not only a KV-cache strategy.

### 6.2 N-gram / PLE offload should be treated as a first-class profile capability

The model was designed so the large embedding table can be offloaded to host memory and prefetched asynchronously.

This should become a model-profile capability, for example conceptually:

```text
profile.flash_next.ple_offload = true
profile.flash_next.ple_device = cpu
profile.flash_next.ple_prefetch = async
```

Do not mix PLE offload implementation into generic KV-cache code.

### 6.3 W4A16 MoE is a different problem from W4A16 dense Marlin

If a future Flash-Next checkpoint uses W4A16, do not assume the 27B Marlin path automatically solves it.

For dense layers, Marlin may still be useful.

For MoE experts the execution path may instead be:

```text
router
 -> selected experts
 -> fused MoE backend
 -> expert GEMM
```

Possible backends may include Triton, CUTLASS, FlashInfer/TRTLLM or another fused MoE implementation depending on the chosen vLLM revision.

The future task is therefore:

```text
find or implement an SM80-compatible W4A16 fused MoE path
```

not:

```text
force every expert through the existing dense Marlin code
```

Keep `marlin` as one backend capability, not as a global assumption.

---

## 7. Proposed repository architecture after the 27B work is complete

Do not perform this move while the current 27B PR is still being debugged.

A useful target structure would look roughly like:

```text
patches/
  core/
    sm80/
      address-safety/
      load-repack/
      cuda-graph/
    kv/
      mixed-pages/
      quantized-cache/
      prefix-cache/
    recurrent/
      accepted-token-safety/
      state-copy/
    spec/
      common/
  models/
    qwen38-27b/
      dflash2/
      full-attn/
      kvarn/
    qwen38-flash-next/
      qsa/
      gdn/
      moe/
      ple/
      mtp/

profiles/
  qwen38-27b/
    bf16-fast.env
    fp8-dflash2.env
    kvarn-250k.env
  qwen38-flash-next/
    README.md
    sm80-w4a16.env.example

bench/
  common/
    token-accounting/
    xid-stress/
    prefix-cache/
    context-ladder/
  qwen38-27b/
  qwen38-flash-next/

scripts/
  audit-platform.sh
  audit-profile.sh
  apply-patch-series.sh
  collect-run-metadata.sh

docs/
  platform-sm80.md
  qwen38-27b.md
  qwen38-flash-next.md
```

The exact directory names are less important than the separation between:

```text
platform capability
model integration
runtime profile
validation harness
```

---

## 8. Patch manifests should become capability-based

The current repository grew from a single-model recipe, so patch application is largely a linear series.

After 27B stabilization, move toward explicit manifests such as conceptually:

```text
core-sm80.series
core-hybrid-kv.series
qwen38-27b.series
qwen38-27b-dflash2.series
qwen38-flash-next.series
```

Each patch or patch group should state:

- target vLLM revision/range;
- required predecessor patches;
- hardware capability requirement;
- model-family requirement if any;
- runtime env gate;
- validation test that proves it is active and correct.

Avoid environment variable names that encode one model when the behavior is actually generic.

For example, over time prefer abstractions like:

```text
SM80_SPEC_VERIFY_FP8=1
HYBRID_KV_ALLOW_GROUP_BLOCK_GEOMETRY=1
RECURRENT_SPEC_ACCEPTED_BOUNDS=1
```

over names that imply a permanent DFlash-only implementation.

Do not rename existing working variables during the 27B qualification cycle. Introduce compatibility aliases during the genericization phase.

---

## 9. Model profiles should be declarative

The generic fork should avoid hard-coding model decisions deep inside shared launcher logic.

A model profile should declare things such as:

```text
weight format
attention backend
KV dtype
speculative method
recurrent-state mode
cache-group geometry
context ladder
CUDA Graph policy
CPU/offload policy
benchmark acceptance thresholds
```

For example, the current 27B mixed profile conceptually resolves to:

```text
target:
  weights: W4A16
  attention: FlashInfer
  kv: FP8 E4M3

speculator:
  method: DFlash2
  weights: W4A16
  attention: FlashAttention2
  kv: BF16

cache geometry:
  target: 896 logical tokens/page
  draft: 448 logical tokens/page
  equal physical page bytes
```

A future Flash-Next profile may instead resolve to something closer to:

```text
main model:
  weights: W4A16 or another SM80-compatible format
  recurrent: GDN
  sparse attention: QSA
  main KV: FP8 if qualified
  QSA indexer cache: model/backend-specific

MoE:
  backend: SM80-qualified fused MoE

PLE/N-gram:
  residency: CPU/offloaded
  prefetch: enabled if supported

speculation:
  method: built-in MTP
```

These are profile-level decisions, not global platform assumptions.

---

## 10. Testing should also be split into platform and model layers

### 10.1 Platform tests

These should run for every supported model when applicable:

- address-width / high-block-ID tests;
- large KV pool allocation test;
- CUDA Graph memory accounting;
- recurrent accepted-token bounds;
- overlapping state copy;
- prefix-cache hit accounting;
- cold/warm TTFT instrumentation;
- long-output soak;
- Xid/NVRM log scan;
- VRAM/power/clock/temperature capture;
- patch applicability/content audit.

### 10.2 Qwen3.8-27B profile tests

Keep the existing tests for:

- DFlash2 acceptance;
- FP8 split-KV verify parity;
- 896/448 cache geometry;
- DFlash prefix reconciliation;
- exact 85,514-token stress;
- KVarN profile if still maintained.

### 10.3 Flash-Next profile tests

When work begins, add new tests rather than overloading the 27B ones:

- QSA FP8 read/dequant correctness against BF16/dequant reference;
- QSA sparse block-selection correctness;
- QSA indexer cache correctness and dtype qualification;
- GDN recurrent-state speculative correctness;
- MTP greedy/sample parity;
- MoE expert-routing parity;
- W4A16 expert GEMM correctness;
- PLE/N-gram offload correctness;
- PLE prefetch latency A/B;
- model residency / CPU-GPU transfer accounting;
- long-context sparse-attention semantic retrieval.

---

## 11. Suggested Flash-Next implementation order on CMP 170HX

Do not start with maximum context or CUDA Graph.

Recommended order:

### Phase FN0 — upstream survey

Before writing code, record:

- current official Flash-Next architecture/config;
- current minimum/recommended vLLM version;
- active vLLM optimization PRs/issues;
- current QSA backend and indexer implementation;
- current MoE backend options on SM80;
- available W4A16 checkpoints/quantizers;
- current PLE/N-gram offload support.

Freeze the selected base revision before implementing anything.

### Phase FN1 — model residency feasibility

Prove that the model can fit operationally on 64 GB.

Do not proceed based only on raw 4-bit arithmetic.

Measure:

```text
resident GPU weights
quant metadata
workspaces
recurrent state
minimum KV pool
CUDA graph/eager overhead
host-offloaded objects
```

The first result may be that additional expert/weight offload is mandatory. That is a valid engineering result.

### Phase FN2 — eager, BF16/known-good attention correctness

Get a minimal correct eager configuration before introducing custom FP8 QSA reads or W4 kernels.

### Phase FN3 — W4A16 weight execution

Qualify dense and MoE weight paths separately.

Record which operators use:

```text
Marlin
fused MoE
Triton
CUTLASS
FlashInfer/TRTLLM
other
```

Do not assume one quant backend serves every operator class.

### Phase FN4 — QSA FP8 main KV

Reuse the methodology from the 27B FP8 verifier project:

```text
existing cache representation
 -> explicit dequant reference
 -> custom QSA read path
 -> correctness first
 -> performance second
```

Keep QSA indexer cache dtype separate from the main KV cache dtype.

### Phase FN5 — recurrent/GDN + MTP stress

Run the accepted-token/state-copy audit again against Flash-Next's actual recurrent kernels.

### Phase FN6 — PLE/N-gram offload

Qualify CPU residency and asynchronous prefetch independently.

Record:

- host RAM;
- pinned memory;
- PCIe transfer volume;
- overlap efficiency;
- TTFT impact;
- decode impact.

### Phase FN7 — prefix cache and long context

Only after the sparse attention, recurrent state and offload paths are stable.

### Phase FN8 — CUDA Graph

Same rule as 27B:

```text
eager correctness first
then CUDA Graph
```

### Phase FN9 — context ladder

Do not jump directly to the architectural maximum.

Use staged qualification and include semantic retrieval, not only allocation success.

---

## 12. Current upstream Flash-Next work that future engineers should re-check

As of 2026-09-15, vLLM's Flash-Next optimization tracker includes work around:

- QSA kernel optimization;
- separate QSA prefill/decode paths;
- FP8 E4M3 main QSA KV cache;
- FP8 QSA indexer cache;
- PLE/Engram offload;
- MoE optimization;
- CPU offload.

Reference:

```text
https://github.com/vllm-project/vllm/issues/55922
```

The open FP8 QSA KV RFC is:

```text
https://github.com/vllm-project/vllm/issues/54426
```

Future work should prefer consuming upstream implementations when they have become correct/stable instead of maintaining a permanent local fork of the same feature.

The purpose of this repository should be:

```text
SM80/CMP enablement + qualification + missing backports
```

not:

```text
fork everything forever
```

---

## 13. Definition of Done for the genericization phase

The repository should only be considered successfully genericized when all of the following are true:

1. The known-good Qwen3.8-27B profile still reproduces its validated results after the directory/config restructuring.
2. Core patches no longer encode Qwen3.8-27B layer counts, DFlash layer counts or 896/448 constants unless those constants live in the 27B profile.
3. Platform tests can run without selecting a specific model profile where technically applicable.
4. Model-specific patch series are separated from platform/core patch series.
5. Runtime profiles declare backend/KV/speculation choices rather than requiring launcher code edits.
6. Experimental patches are isolated from production `verify.sh`/install behavior unless their profile explicitly selects them.
7. Every patch group states its vLLM revision expectations and validation command.
8. The 27B profile is a first-class regression target, not a discarded legacy configuration.
9. A placeholder Flash-Next profile can exist without contaminating or changing the 27B runtime.
10. Documentation clearly distinguishes hardware/runtime capabilities from model-specific integrations.

---

## 14. Guidance to the agent finishing the 27B work

When the current `handover.md` work is complete:

1. Do **not** immediately rewrite the working 27B patch stack.
2. Tag or otherwise record the exact known-good 27B commit and benchmark evidence.
3. Create a new genericization branch from that known-good point.
4. Move/refactor one capability group at a time.
5. After each move, rerun the 27B regression profile.
6. Preserve compatibility aliases for existing environment variables until the new profile system is proven.
7. Only after the repository structure is stable should Flash-Next implementation begin.

The 27B target is the regression oracle for the genericization work.

If a refactor makes the repository look cleaner but loses a known 27B behavior or benchmark without explanation, the refactor is not complete.

---

## 15. Core design principle

The long-term project should be thought of as:

```text
CMP 170HX / SM80 inference platform
    |
    +-- memory/address safety
    +-- quantized weight loading
    +-- hybrid KV infrastructure
    +-- recurrent-state correctness
    +-- speculative-decode infrastructure
    +-- CUDA Graph accounting
    +-- prefix-cache observability
    +-- benchmark / Xid qualification
    |
    +-- model profiles
        |
        +-- Qwen3.8-27B
        |    +-- full attention
        |    +-- DFlash2
        |    +-- FP8/BF16 896/448 profile
        |    +-- KVarN profile
        |
        +-- Qwen3.8-Flash-Next
             +-- GDN
             +-- QSA
             +-- sparse MoE
             +-- PLE/N-gram offload
             +-- MTP
             +-- QSA/indexer KV profiles
```

The 27B project is the first validated model integration on that platform, not the final shape of the repository.
