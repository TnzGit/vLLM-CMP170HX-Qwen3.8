# Engineering handover: CMP 170HX / Qwen3.8 mixed-FP8 path

Last updated: 2026-09-16

This document is the handover for the work in `TnzGit/vLLM-CMP170HX-Qwen3.8` to reconstruct, validate, and eventually productionize the mixed-KV configuration demonstrated on a single 64 GB NVIDIA CMP 170HX (SM80 / GA100-class) with Qwen3.8-27B W4A16 + DFlash2.

The intended reader is an engineer taking over the branch without prior conversation context. Treat this file as the operational source of truth for what has been done, what has **not** been proven yet, and the order in which the remaining work should be performed.

---

## 1. Repository and branch state

Repository:

```text
https://github.com/TnzGit/vLLM-CMP170HX-Qwen3.8
```

Production/reference branch:

```text
main
```

Current engineering branch:

```text
work/cmp170hx-mixed-fp8
```

Draft PR:

```text
#1 CMP170HX mixed-FP8 speculative verify reconstruction
https://github.com/TnzGit/vLLM-CMP170HX-Qwen3.8/pull/1
```

Working branch base:

```text
c003cc73b4c34e418b29e074735107a0771c6ca1
```

The recorded recipe ancestry includes syv-ai commit:

```text
69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c
```

The intended vLLM base is:

```text
vLLM 0.27.1
PyTorch 2.13.0
CUDA 13
```

The branch must remain separate from `main` until the GPU qualification matrix in this document passes. Do not merge simply because the server boots.

Structural verifier optimization after the initial mixed-FP8 reconstruction is
tracked milestone-by-milestone in:

```text
docs/verifier-redesign-log.md
```

That append-only log is the current continuation point.  It freezes the active
runtime source, profiler boundary, rejection gates, and the next isolated
experiment so another agent can resume without conversation history.

---

## 2. Existing deployment baseline

The repository already records a known working CMP 170HX 64 GB deployment in:

```text
docs/cmp170hx-64gb-deployment.md
deploy/pixelml-vllm-8002.service.example
```

Recorded operating envelope:

```text
GPU            1 x NVIDIA CMP 170HX 64 GB
architecture   SM80
power limit    180 W
model          Qwen3.8-27B W4A16 target
speculator     DFlash2 W4A16, 7 draft tokens
```

Recorded BF16-KV C1 baseline at 180 W:

| Test | Baseline |
|---|---:|
| 256-token decode | 133.82 tok/s |
| 900-token decode | 126.78 tok/s |
| ~6,603-token prefill | 1,868.7 tok/s |

Recorded KVarN K4V2 / 250K profile:

| Test | Baseline |
|---|---:|
| 256-token decode | 124.22 tok/s |
| 900-token decode | 107.82 tok/s |
| ~6,603-token prefill | 1,767.7 tok/s |

These are negative-control/reference numbers. The mixed-FP8 work must not regress the existing production/KVarN route unintentionally.

---

## 3. Target architecture being reconstructed

The goal is an **asymmetric** target/drafter cache configuration rather than making the entire speculative stack FP8.

| Component | Target Qwen3.8-27B | DFlash2 drafter |
|---|---|---|
| weights | W4A16 | W4A16 |
| attention backend | FlashInfer | FlashAttention-2 |
| KV dtype | E4M3 FP8 | BF16 |
| logical block/page unit | 896 tokens | 448 tokens |
| desired physical page | 1,835,008 bytes | 1,835,008 bytes |

Conceptually:

```text
Qwen3.8 target
  W4A16
  FlashInfer
  FP8 E4M3 target KV
        |
        | multi-query speculative verify
        v
SM80 split-KV verifier
        ^
        |
DFlash2 drafter
  W4A16
  FlashAttention-2
  BF16 draft KV
```

The equal physical byte size of the target and draft pages is intentional. The logical tokens-per-page differ, so the implementation must preserve group-specific geometry rather than pretending there is one global token block size.

Do **not** implement this by globally setting `--block-size 896`.

---

## 4. Important conclusions from the source audit

Several parts initially believed to require private reverse engineering are already available in vLLM or the public syv-derived patch stack.

### 4.1 Draft-specific backend and KV dtype already exist conceptually

vLLM 0.27.1 has `SpeculativeConfig` support for draft-specific:

```text
attention_backend
kv_cache_dtype
```

and the DFlash loader can construct a draft-specific `VllmConfig` using those fields.

Therefore the correct implementation is to wire the launcher JSON to those fields, not invent another configuration mechanism.

Desired speculative config shape:

```json
{
  "method": "dflash",
  "model": "/path/to/draft",
  "num_speculative_tokens": 7,
  "attention_backend": "FLASH_ATTN",
  "kv_cache_dtype": "bfloat16"
}
```

while the target process itself uses:

```text
--attention-backend FLASHINFER
--kv-cache-dtype fp8
```

### 4.2 Mixed physical-page infrastructure already has a reference design

vLLM's KV-cache spec machinery already knows how to reconcile different natural page sizes in a number of cases. Later upstream work, especially vLLM PR #45181, is the reference design for DFlash mixed page sizes and block-stride-indexed views.

The remaining task is to make the exact target/draft/Mamba geometry work correctly through the scheduler, block tables, prefix cache, and attention backend constraints.

### 4.3 The high-physical-block-ID fix is known

The split-KV verifier must promote physical block IDs to 64-bit **before** multiplying by cache strides:

```python
blk = tl.load(...).to(tl.int64)
```

This prevents 32-bit address arithmetic overflow in large KV pools. The repository audit checks for this and it must remain covered by a high-block-ID regression.

### 4.4 SM80 FP8 verify should consume the normal vLLM FP8 representation

Do not invent a new FP8 cache format.

The intended static/per-layer path is:

```text
raw E4M3 cache value
    -> convert to BF16/FP32 in the verifier
    -> apply the existing per-layer K or V scale
    -> use the existing split-KV online-softmax/reduction path
```

This is distinct from the existing `int8_per_token_head` path, which has a scale per token/head.

### 4.5 Marlin load problems must not automatically be blamed on SM80

There is a CPU Marlin repack fallback and a staged GPU repack path in the broader stack. Later debugging showed that at least one severe CMP Xid-31 load failure was caused by an unlock/driver memory-map problem involving reserved framebuffer/WPR/GSP memory, not a generic SM80 Marlin arithmetic bug.

Prefer the staged GPU path when it is stable on the host. Keep CPU repack as a fallback/debug control rather than assuming it is mandatory.

---

## 5. Work already completed on `work/cmp170hx-mixed-fp8`

### 5.1 Engineering map

Created:

```text
docs/cmp170hx-mixed-fp8-engineering.md
```

This records the target architecture, reusable pieces, missing implementation, correctness gates, performance qualification, and long-context policy.

### 5.2 Prerequisite audit script

Created:

```text
scripts/audit-mixed-fp8-prereqs.sh
```

Purpose:

- verify expected repository patch files exist;
- verify split-KV block IDs are promoted to `tl.int64` by the normal patch
  series. The deployed 206 package was found to lack this cast even though the
  legacy verifier reported the patch as applied, so the repository patch and
  the dedicated audit are now the source of truth for this prerequisite;
- verify the installed vLLM exposes draft-specific `attention_backend` and `kv_cache_dtype`;
- verify the DFlash loader actually consumes those overrides;
- report whether private-style heterogeneous-page/full-CUDA-graph hooks are already present.

Run it from the repository root on the deployment machine:

```bash
PY=/opt/qwen38-vllm/venv/bin/python \
  bash scripts/audit-mixed-fp8-prereqs.sh
```

or, for a checkout-local venv:

```bash
bash scripts/audit-mixed-fp8-prereqs.sh
```

A `WARN` for unreconstructed private-style hooks is expected at this stage. A `FAIL` in the base DFlash/backend/dtype/int64 prerequisites must be resolved before doing FP8 runtime testing.

### 5.3 Static-scale FP8 split-KV kernel patch

Created:

```text
experimental/cmp170hx-mixed-fp8/patches/spec-decode-fp8-kv-sm80.patch
```

Intent:

- extend `vllm/v1/attention/ops/spec_decode_attn.py`;
- preserve the existing BF16 path;
- preserve the existing per-token-head INT8 path;
- add a mutually-exclusive static-scale FP8 path;
- load raw FP8 K/V;
- convert K/V to BF16 for tensor-core math;
- fold the layer-level K scale into attention score computation;
- fold the layer-level V scale into the value accumulation.

The implementation is intentionally kernel-only. It is separated from backend dispatch so math correctness can be tested before routing real model traffic through it.

**Status: written, not yet qualified on a CMP 170HX GPU.**

### 5.4 Experimental FlashInfer -> split-KV verify hook

Created:

```text
experimental/cmp170hx-mixed-fp8/patches/flashinfer-sm80-fp8-spec-verify.patch
```

Intent:

- leave normal FlashInfer prefill unchanged;
- leave normal single-token decode unchanged;
- for SM80 + E4M3 FP8 KV + pure multi-query speculative verify, route target verification through the split-KV Triton kernel;
- pass the target layer's existing static K/V scales;
- remain opt-in behind:

```bash
VLLM_SPEC_DECODE_ATTN=1
VLLM_FP8_SPEC_VERIFY=1
```

Current restrictions in the draft patch include no sliding window, no softcap, no ALiBi, no sinks, and NHD layout.

**Status: written, not yet runtime-qualified. The dispatch/gating code requires review on the actual installed vLLM tree before use.**

One specific item to review before enabling it: the patch documentation says DCP must be disabled, but the current draft gate should be checked carefully to ensure it is testing the actual DCP state rather than merely the existence of a combine callable. Do not treat this hook as production-ready until that is verified.

### 5.5 Direct FP8 kernel correctness test

Created:

```text
bench/test_spec_decode_fp8_sm80.py
```

This test deliberately bypasses FlashInfer dispatch. It creates a paged E4M3 K/V cache with known static scales, runs the new split-KV FP8 path, explicitly dequantizes the **same FP8 bytes** for a reference computation, and compares the outputs.

Current cases include:

```text
1.5K context, q=8
8K context, q=8
mixed 4097/1300 batch, q=5
32K context, q=8
```

Current acceptance threshold:

```text
max absolute error < 0.08
```

This tolerance is intended to cover BF16 tensor-core accumulation. It is not a model-quality/FP8 quantization comparison because the reference uses exactly the same quantized FP8 bytes.

### 5.6 Experimental runtime contract

Created:

```text
deploy/cmp170hx-mixed-fp8.env.example
```

This records the desired eventual configuration, including:

```text
Target: FLASHINFER + fp8
Draft:  FLASH_ATTN + bfloat16
VLLM_SPEC_DECODE_ATTN=1
VLLM_FP8_SPEC_VERIFY=1
```

The heterogeneous-page and full-CUDA-graph switches are deliberately recorded as disabled/not-yet-implemented.

### 5.7 Experimental patch order

Created:

```text
experimental/cmp170hx-mixed-fp8/series
```

Current experimental order:

```text
spec-decode-fp8-kv-sm80.patch
flashinfer-sm80-fp8-spec-verify.patch
```

These depend on the normal repository stack already having applied at least:

```text
spec-decode-attn.patch
spec-decode-int8-kv.patch
```

---

## 6. Experimental patch isolation

The Phase 0 repository-structure problem has been resolved. `verify.sh` still
loops over only the normal production directory:

```bash
for p in patches/*.patch; do
    ...
done
```

The mixed-FP8 files now live under:

```text
experimental/cmp170hx-mixed-fp8/
```

and are handled only by that directory's `install.sh`. Normal Docker builds and
`verify.sh --install` no longer require or apply them. The installer supports
`--dry-run`, `--apply`, `--check`, and reverse-order `--reverse` against an
explicit `--site` path or the vLLM imported by `PY`.

---

## 7. What has **not** been completed or proven

This section is the branch-creation snapshot and is preserved to show the
original risk register. Most 27B/SM80 items below were subsequently completed;
the authoritative current status is section 17.

The following are still open:

- the two new FP8 patches have not been applied and executed on the actual CMP 170HX host;
- patch applicability has not yet been proven against the exact installed vLLM 0.27.1 tree on that host;
- FP8 kernel correctness has not yet passed `bench/test_spec_decode_fp8_sm80.py` on SM80;
- FlashInfer runtime dispatch has not been exercised end-to-end;
- `single-user/start_qwen.sh` does not yet wire `DFLASH_ATTN_BACKEND` and `DFLASH_KV_CACHE_DTYPE` into `SPEC_CFG`;
- the target/draft 896/448 logical geometry is implemented behind an opt-in
  environment gate and still requires end-to-end qualification;
- Mamba checkpoint geometry is not yet aligned to the target 896-token unit;
- complete 448-token DFlash prefix pages are not yet qualified in common-prefix reconciliation;
- the equivalent of the reported `VLLM_FP8_SPEC_FULL_CG` behavior is not implemented/qualified;
- no new mixed-FP8 profile has passed 85,514-token stress;
- no 350K/500K/700K/1M mixed-FP8 profile is qualified;
- no claim should be made yet that the reconstructed path matches the reported ~169 tok/s short-context or 100K+-context performance.

---

## 8. Recommended next steps, in order

Do not skip phases. The main debugging risk is stacking cache geometry, FP8 dequantization, speculative verify, prefix caching, CUDA graphs, and large physical KV pools at the same time and then receiving an opaque Xid.

### Phase 0 — isolate the experimental patch layer

Goal: keep `main`/existing KVarN production semantics untouched.

Status: **complete on `work/cmp170hx-mixed-fp8`**. The experimental series now
lives under `experimental/cmp170hx-mixed-fp8/`, and the normal wildcard patch
installer no longer sees it.

Recommended change:

```text
experimental/cmp170hx-mixed-fp8/
  patches/
    heterogeneous-attn-pages-sm80.patch
    spec-decode-fp8-kv-sm80.patch
    flashinfer-sm80-fp8-spec-verify.patch
  series
  install.sh
```

or add an explicit gate such as:

```bash
CMP_MIXED_FP8_EXPERIMENTAL=1
```

that controls whether the extra patches are installed/verified.

Acceptance criteria:

- `bash verify.sh --install` on a normal production tree still passes without applying experimental patches;
- the experimental installer can independently dry-run/apply/reverse its three patches.

The two original FP8 verifier patches were also regenerated as valid unified diffs against
the deployed vLLM 0.27.1 source. The original handoff patches had malformed hunk
counts and could not be applied by `patch(1)`.

### Phase 1 — prove patch applicability on the deployment host

On a fresh test checkout/venv, do **dry-run first**.

Determine the installed vLLM package path:

```bash
PY=/opt/qwen38-vllm/venv/bin/python
SP=$($PY - <<'PY'
import inspect, pathlib, vllm
print(pathlib.Path(inspect.getfile(vllm)).resolve().parent)
PY
)
echo "$SP"
```

Run prerequisites:

```bash
PY="$PY" bash scripts/audit-mixed-fp8-prereqs.sh
```

Assuming the normal stack is already installed, dry-run the experimental layers in order:

```bash
experimental/cmp170hx-mixed-fp8/install.sh --dry-run --site "$SP"
```

Only if both dry-runs are clean:

```bash
experimental/cmp170hx-mixed-fp8/install.sh --apply --site "$SP"
experimental/cmp170hx-mixed-fp8/install.sh --check --site "$SP"
```

If any hunk fails, rebase the patch against the actual installed source. Do not use `--force` and do not manually ignore failed hunks.

The SM80 FlashInfer runtime patch must also opt the FP8 split-KV verifier into
the builder's speculative-as-decode threshold. With the stock threshold of one,
DFlash verification queries are classified as prefill and a runtime hook that
requires a decode-only batch is unreachable. The experimental patch now makes
that threshold change only when `VLLM_FP8_SPEC_VERIFY=1`, capability is exactly
SM80, and target KV dtype is FP8.

### Phase 2 — kernel correctness before model/runtime dispatch

Disable the runtime hook initially. Test only the math:

```bash
export VLLM_FP8_SPEC_VERIFY=0
$PY bench/test_spec_decode_fp8_sm80.py
$PY bench/test_spec_decode_fp8_sm80.py --high-block-id
```

SM80-specific implementation note: Triton cannot compile a native load of
PyTorch's NVIDIA `float8_e4m3fn` (`fp8e4nv`) on this architecture. The verifier
therefore takes a zero-copy `uint8` view of the same cache and explicitly
decodes E4M3FN bits before the BF16 tensor-core dot. This does not allocate a
second KV cache and does not change block-table or page geometry.

Required result:

```text
all cases OK
exit code 0
max absolute error < 0.08 in every case
```

After the standard cases pass, extend the test matrix before moving on:

```text
q = 5, 8, 16
context = 1.5K, 8K, 32K, 65K
batch = 1 and mixed-length batch
```

The `--high-block-id` case allocates a sparse-use FP8 pool and references block
32,780, whose first element lies above the signed int32 address boundary. This
is required to exercise the `tl.int64` address fix rather than merely checking
ordinary low block IDs. It needs about 4 GiB of free device memory.

After each long/high-block run:

```bash
dmesg -T | tail -n 200 | grep -Ei 'NVRM|Xid|CUDA|GPU' || true
```

Use the host's normal kernel-log access method if `dmesg` is restricted.

Any new Xid fails the phase even if Python happened to return output.

### Phase 3 — launcher wiring for asymmetric target/draft config

Modify `single-user/start_qwen.sh` so an explicit experimental profile can create:

```json
{
  "method": "dflash",
  "model": "$DRAFT",
  "num_speculative_tokens": 7,
  "attention_backend": "FLASH_ATTN",
  "kv_cache_dtype": "bfloat16"
}
```

while target args are:

```text
--attention-backend FLASHINFER
--kv-cache-dtype fp8
```

Prefer a new explicit profile name rather than silently changing current `CTX=fast`, `CTX=long`, or `CTX=huge` semantics. For example:

```text
CTX=cmp-fp8
```

is easier to audit than overloading `CTX=long`.

Acceptance criteria at this phase:

- launch log confirms target backend is FlashInfer;
- launch log confirms target KV is FP8;
- draft model construction confirms FA2/BF16 overrides;
- normal BF16/KVarN profiles are unchanged.

### Phase 4 — end-to-end eager correctness at ordinary context

Keep CUDA graphs out of the equation initially.

Start with:

```text
MAX_LEN=65536
MAX_SEQS=1
DFLASH_TOKENS=7
prefix cache OFF initially
full CUDA graph OFF
```

Run three controls with identical deterministic prompts:

1. target FP8, speculation OFF;
2. BF16-known-good DFlash2 control;
3. target FP8 + DFlash BF16 + new SM80 verify path.

Use greedy decoding:

```text
temperature=0
thinking disabled for simple parity probes where appropriate
```

Do not compare only the first few tokens. Generate at least 512 tokens for stress/parity cases.

The mixed-FP8 path must match the target's greedy result semantics. Speculative decoding must not change what the target would have emitted.

### Phase 5 — 896/448 cache geometry

Only after ordinary eager FP8 verification is correct should heterogeneous logical page geometry be enabled.

Desired invariants:

```text
target FP8 logical block = 896 tokens
DFlash BF16 logical block = 448 tokens
physical target page      = 1,835,008 bytes
physical draft page       = 1,835,008 bytes
```

Do not force a global 896 block size.

Implement this through KV-cache specs/group geometry, following the existing mixed-page/unification machinery and upstream PR #45181 as the design reference.

The experimental `heterogeneous-attn-pages-sm80.patch` now implements this as
a fixed-point logical block-size alignment before the normal page unifier. It
corrects the observed 880/448 startup failure without a Worker `as_strided`
shim: target becomes 896, draft remains 448, and neither attention page needs
physical padding. Validate the pure geometry first with:

```bash
VLLM_ALIGN_HETEROGENEOUS_ATTN_PAGES=1 \
  $PY bench/test_mixed_fp8_page_alignment.py
```

Add explicit startup logging or a diagnostic test that prints, per KV group:

```text
layer/group type
backend
KV dtype
logical block size
real page bytes
padded page bytes
scheduler/hash unit
```

Acceptance criteria:

- target group resolves to 896;
- DFlash group resolves to 448;
- physical page bytes match exactly;
- backend kernel page/block constraints remain legal;
- no scheduler assertion;
- no block-table indexing mismatch;
- 64K eager generation remains correct.

### Phase 6 — Mamba and prefix-cache geometry

Align the Mamba checkpoint/reconciliation geometry to the target 896-token unit and ensure complete 448-token DFlash pages participate correctly in prefix reconciliation.

This is separate from checkpoint lifetime/eviction-order fixes such as the syv Mamba checkpoint retention work. Geometry and lifetime must be tested independently.

Regression test:

1. choose an approximately 9K-10K prompt;
2. send it cold;
3. send the exact same prefix again;
4. record common-prefix hit tokens and TTFT.

Expected shape of a successful result, based on the reference experiment:

```text
cold request: normal prefill
warm request: real non-zero reconciled hit
warm TTFT: dramatically lower
```

The reference report saw an 8,960-token usable hit on a ~9,658-token repeated prompt. Do not require that exact number unless the geometry and prompt are identical, but a zero hit is a failure.

Also run an A -> B -> A pattern so cache reuse is tested after intervening traffic, not only immediate self-repeat.

### Phase 7 — high-block and long-context qualification

Increase context in controlled steps:

```text
32K
65K
85,514 exact known failure point
128K
256K
```

At every tier record:

```text
boot success
TTFT
decode tok/s
prefill tok/s
accepted tokens / speculative acceptance
GPU memory
power
SM clock
temperature
NVRM/Xid log state
```

At the exact 85,514 tier, repeat enough times to expose intermittent address issues. A useful target is at least 16 repetitions plus a smaller exact-boundary repeat set after any fix.

Use at least 512 generated tokens per stress request.

Test both:

```text
cold/non-cached layout
warm/prefix-cached layout
```

### Phase 8 — CUDA Graph qualification

Only after eager mode is stable.

Then reconstruct/enable the equivalent of the reported FP8 speculative full-graph path.

A/B:

```text
eager
vs
full CUDA Graph
```

Qualification requires:

- output parity;
- no graph-captured pointer lifetime errors;
- no late buffer reallocations;
- no request-finish/mid-generation crash with multiple requests;
- stable acceptance length;
- measured performance gain large enough to justify the complexity.

The existing split-KV partial buffers are intentionally fixed-size because captured CUDA graphs retain addresses. Do not introduce lazy growth after graph capture.

### Phase 9 — performance tuning

Only after correctness/stability.

Primary performance comparisons:

```text
BF16 known-good DFlash2 baseline
current KVarN 250K baseline
new mixed-FP8 eager
new mixed-FP8 full graph
```

Record at least:

- 256-token decode;
- 900-token decode;
- ~6.6K prefill;
- one long-context prefill;
- cold TTFT;
- repeated-prefix TTFT;
- accepted tokens per verify step;
- VRAM/power/clock/temperature.

Do not use SSE event count as token count. Use the repository's usage-token-aware benchmark tooling.

### Phase 10 — >256K and YaRN

Do not jump from a 64K success directly to 1M.

Suggested progression:

```text
350K -> 500K -> 700K -> 1M
```

For every tier qualify:

```text
memory fit
prefill completion
TTFT
decode stability
prefix reuse
semantic retrieval / needle-style checks
long-generation stability
Xid-free behavior
```

A configuration merely allocating enough KV memory for 1M is not a production-quality 1M context implementation.

---

## 9. Test matrix and pass/fail rules

### 9.1 Repository/install sanity

Command:

```bash
PY=/path/to/venv/bin/python bash verify.sh --install
```

The experimental series is isolated and is not included in this normal
production verification command.

Pass:

```text
0 FAIL entries
normal production patch stack intact
```

### 9.2 Mixed-FP8 prerequisite audit

```bash
PY=/path/to/venv/bin/python bash scripts/audit-mixed-fp8-prereqs.sh
```

Pass:

```text
base prerequisite checks = OK
private/unreconstructed hooks may WARN until implemented
```

### 9.3 FP8 verifier mathematical correctness

```bash
PY=/path/to/venv/bin/python
$PY bench/test_spec_decode_fp8_sm80.py
```

Pass:

```text
exit 0
all cases OK
max abs error < 0.08
no CUDA error
no Xid
```

### 9.4 Greedy semantic parity

For the same target checkpoint and prompt:

```text
speculation OFF
vs
mixed-FP8 DFlash2 ON
```

with:

```text
temperature=0
same sampling configuration
same chat template options
```

Pass:

```text
greedy target output remains semantically/exactly consistent with the target path
no repeatable corruption after prefix-cache hits
```

If byte-for-byte equality is not possible because of a changed prompt/template path, first eliminate that source of nondeterminism before blaming speculation.

### 9.5 Prefix-cache test

Use exact repeated prompt and A -> B -> A traffic.

Record:

```text
prompt token count
reported prefix hit/common prefix
cold TTFT
warm TTFT
output hash or deterministic response
```

Pass:

```text
non-zero meaningful reconciled hit
warm TTFT reduction
correct output
```

### 9.6 High-block-ID test

Build a large physical KV pool and deliberately map a request to blocks near the top of the pool.

Pass:

```text
correct output
no illegal memory access
no Xid 31
no int32 address overflow symptoms
```

### 9.7 Long-context soak

At each context tier, use long outputs and repeated runs.

Minimum useful stress shape:

```text
>= 512 generated tokens
multiple repetitions
both cold and cached
```

Pass:

```text
zero new Xid/NVRM faults
zero CUDA illegal-memory-access errors
zero silent output corruption
stable request completion
```

---

## 10. Logging and evidence to retain for every serious A/B

When running on the deployment host, save enough information that another engineer can reproduce the result.

Recommended bundle per test:

```text
git commit SHA
vLLM version
applied patch list
MODEL path/checkpoint identifier
DRAFT path/checkpoint identifier
full launcher environment
command/request JSON
server log
benchmark JSON/result
nvidia-smi snapshot
power limit
GPU clocks
GPU temperature
VRAM usage
kernel/NVRM Xid excerpt
```

For before/after performance claims, keep both result files. Do not report a speedup from two runs whose output length, context length, temperature, power limit, speculative settings, or sampling parameters differ.

---

## 11. Recommended commit structure for remaining work

Keep correctness and performance changes bisectable. A good target sequence is:

```text
1. build: isolate mixed-FP8 experimental patch layer
2. tests: qualify static FP8 split-KV verifier on SM80
3. dflash: wire draft-specific attention backend and KV dtype
4. spec: enable guarded FlashInfer FP8 verify dispatch on SM80
5. kv: implement target/draft heterogeneous logical geometry
6. prefix: align Mamba and DFlash prefix-cache geometry
7. tests: add high-block-ID and repeated-prefix regressions
8. graph: qualify FP8 speculative verify under full CUDA Graph
9. profiles: add 64K/128K/256K CMP mixed-FP8 profiles
10. docs: record measured CMP 170HX results
11. profiles: add experimental 350K/500K/700K/1M YaRN ladder
```

Avoid a single commit that combines allocator geometry, kernel math, launcher changes, prefix cache, and CUDA graphs.

---

## 12. Rollback strategy

Until production qualification, rollback should be trivial:

```text
main remains the known deployment branch
work/cmp170hx-mixed-fp8 remains experimental
```

For installed-package testing, prefer a disposable venv/container or preserve an untouched known-good venv. Reversing several overlapping patches in-place is more error-prone than rebuilding a clean test environment.

If testing in-place is unavoidable, at minimum record:

```bash
git rev-parse HEAD
python -c 'import vllm; print(vllm.__version__)'
```

and save the installed source tree or venv before applying the experimental layers.

If the GPU produces Xid 31/13 or becomes wedged, stop the test campaign and restore GPU/host health before interpreting subsequent benchmark results. Performance numbers after a faulted GPU state are not trustworthy.

---

## 13. Known risk areas

### 13.1 FlashInfer backend constraints

FlashInfer's supported kernel block sizes and layout constraints are not the same thing as the scheduler's logical/group block geometry. The 896/448 design must not be implemented by feeding an unsupported 896 page directly to a kernel that only accepts smaller native pages on SM80.

Keep these concepts separate:

```text
scheduler/hash logical unit
KV group logical block size
physical page stride / padded page
backend kernel-native page/block unit
```

### 13.2 CUDA graph pointer lifetime

The split-KV verifier's scratch/partial buffers must not be reallocated after graph capture. Any implementation that sizes them lazily after seeing a larger query block can create a captured graph that later reads freed memory.

### 13.3 Prefix-cache geometry versus lifetime

A prefix miss can come from two separate classes of problem:

```text
geometry/reconciliation mismatch
checkpoint lifetime/eviction ordering
```

Do not fix one by masking the other. Test immediate exact-repeat reuse first, then A -> B -> A eviction/reuse behavior.

### 13.4 CMP unlock/driver memory map

If Marlin repack or high-occupancy tests produce repeatable Xid 31, inspect the CMP unlock/driver memory map before declaring a vLLM kernel bug. Reserved framebuffer/WPR/GSP regions must not be accidentally exposed to the ordinary allocator.

### 13.5 Large-context claims

The target is not “make 1M allocate.” The target is a stack that remains correct, stable, performant enough, prefix-cache coherent, and semantically useful at the chosen context length.

---

## 14. Suggested first session for the next engineer

A good first session is deliberately boring and should produce evidence rather than new features.

```bash
# 1. Get the work branch
git fetch origin
git checkout work/cmp170hx-mixed-fp8
git pull --ff-only

# 2. Record environment
git rev-parse HEAD
nvidia-smi
venv/bin/python -c 'import torch,vllm; print(torch.__version__, vllm.__version__); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())'

# 3. Audit prerequisites
PY=$PWD/venv/bin/python bash scripts/audit-mixed-fp8-prereqs.sh

# 4. Experimental patches are isolated from the normal build chain.

# 5. Identify installed package path
PY=$PWD/venv/bin/python
SP=$($PY - <<'PY'
import inspect, pathlib, vllm
print(pathlib.Path(inspect.getfile(vllm)).resolve().parent)
PY
)

# 6. Dry-run the experimental patches in order
experimental/cmp170hx-mixed-fp8/install.sh --dry-run --site "$SP"

# 7. Apply only in a disposable/test environment if the dry-runs are clean
experimental/cmp170hx-mixed-fp8/install.sh --apply --site "$SP"
experimental/cmp170hx-mixed-fp8/install.sh --check --site "$SP"

# 8. Run kernel math test BEFORE enabling runtime dispatch
VLLM_FP8_SPEC_VERIFY=0 $PY bench/test_spec_decode_fp8_sm80.py

# 9. Check kernel logs before doing anything else
dmesg -T | tail -n 200 | grep -Ei 'NVRM|Xid|CUDA|GPU' || true
```

If step 8 does not pass, stop there and fix the kernel. Do not proceed to launcher wiring, 896/448 geometry, prefix caching, or CUDA graphs.

---

## 15. Definition of done for the reconstruction

The mixed-FP8 reconstruction can be considered ready to leave Draft only when all of the following are true:

- experimental patches are cleanly isolated/gated from the normal production install path;
- patch application is reproducible on the pinned vLLM version;
- static FP8 split-KV kernel passes direct numerical tests on CMP 170HX;
- target FP8 + speculation-off control is stable;
- target FP8 + DFlash BF16 eager path is greedy-correct;
- target/draft geometry resolves to the intended 896/448 logical layout with equal physical page bytes;
- repeated-prefix and A -> B -> A prefix-cache tests are correct and show real reuse;
- high physical block IDs are explicitly tested;
- 32K, 65K, exact 85,514, 128K, and 256K stress tiers complete without Xid/CUDA faults;
- CUDA Graph mode is separately qualified after eager mode;
- existing BF16 and KVarN production profiles still work;
- performance is measured with the same benchmark semantics as the recorded baseline;
- documentation records actual measured numbers rather than expected/report-derived numbers.

Only after that should the 350K -> 500K -> 700K -> 1M ladder be treated as the next project rather than part of the initial correctness reconstruction.

---

## 16. Related files

Start with these when resuming work:

```text
handover.md

docs/cmp170hx-mixed-fp8-engineering.md
docs/cmp170hx-64gb-deployment.md

scripts/audit-mixed-fp8-prereqs.sh
verify.sh

patches/spec-decode-attn.patch
patches/spec-decode-int8-kv.patch
experimental/cmp170hx-mixed-fp8/patches/spec-decode-fp8-kv-sm80.patch
experimental/cmp170hx-mixed-fp8/patches/flashinfer-sm80-fp8-spec-verify.patch
patches/hybrid-kv-groups-v2-cudagraph.patch
patches/hybrid-sw-block-promote.patch
patches/marlin-repack-staged-sm80.patch
experimental/cmp170hx-mixed-fp8/series

bench/test_spec_decode_attn.py
bench/test_spec_decode_fp8_sm80.py

deploy/cmp170hx-mixed-fp8.env.example
single-user/start_qwen.sh
```

When there is a disagreement between an old performance note and a fresh reproducible test on the actual CMP host, preserve the raw evidence and update this handover rather than silently changing the target.

---

## 17. 2026-09-15 CMP 170HX qualification update

This branch has now crossed the 64K correctness and CUDA Graph qualification
boundary on the real CMP 170HX host. It is still a Draft because 85K/128K/256K
and rollback-profile regressions remain outstanding.

### 17.1 Runtime and geometry now validated

```text
target model: Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4
draft model:  Qwen3.8-27B-DFlash2-W4A16
target attention/KV: FlashInfer / FP8 E4M3
draft attention/KV:  FlashAttention / BF16
speculation: DFlash2, 7 draft tokens
GPU: CMP 170HX 64GB, exact SM80, 180W limit
context: 65,536
max sequences: 4
prefix caching: enabled
API key: none
```

The heterogeneous-page fixed point is:

```text
target full attention: 896 tokens, 1,835,008 bytes
draft attention:       448 tokens, 1,835,008 bytes
Mamba align interval:  896 tokens
prefix hash unit:       448 tokens
```

The Mamba promotion changes the logical checkpoint interval only. Its physical
state storage size is unchanged. The complete experimental series applies
cleanly to the pinned normal-patch baseline; a disposable applied tree passed
`py_compile`, and `bench/test_mixed_fp8_page_alignment.py` directly verified
target=896, draft=448, Mamba=896 and equal attention page bytes.

### 17.2 Prefix-cache qualification

The original target=896/draft=448/Mamba=880 geometry either asserted during
hash resolution or produced no reusable common prefix. Promoting aligned Mamba
checkpoints to 896 and hashing at 448 fixes the geometry without changing KV
storage.

Measured A -> B -> A results:

```text
10K salted: cache hit 8,960; parity true; cold/warm TTFT 5.782/0.778 s; 7.44x
30K salted: cache hit 29,568; parity true; cold/warm TTFT 17.702/0.445 s; 39.74x
60K salted: cache hit 59,136; parity true; cold/warm TTFT 43.205/1.177 s; 36.69x
```

`bench/test_mixed_fp8_prefix_reuse.py` is the reproducible regression client.

### 17.3 Concurrency and long-context correctness

API behavior passed 12/12 smoke cases with the mixed verifier enabled. Eager
and graph runs completed mixed decode+prefill batches without output corruption,
preemption, CUDA illegal-memory access or new Xid.

Representative eager measurements:

```text
4K C1: decode 46.5 tok/s; 3.66 tok/step; 78.4 ms/pass
4K C2: decode 91.0 tok/s; 3.60 tok/step; 78.7 ms/pass
4K C4: decode 186.9 tok/s; 3.74 tok/step; 78.6 ms/pass
32K/512 C1: decode 45.7 tok/s; 0 preemptions
64K/512 C1: decode 44.8 tok/s; 0 preemptions
```

The direct static-FP8 split-KV verifier passed all GPU cases, including 65,536
KV tokens and multi-request/query shapes, with maximum absolute error <=0.00076.

### 17.4 CUDA Graph qualification

PIECEWISE captured seven graph sizes and passed API, prefix, C1/C4 and 32K/60K
tests. The guarded FULL capability advertisement is enabled only when all of
the following are true:

```text
VLLM_FP8_SPEC_VERIFY=1
VLLM_FP8_SPEC_FULL_CG=1
exact SM80
cache configuration is fp8/fp8_e4m3
attention storage is uint8 or float8_e4m3fn
```

Because the GDN backend advertises `UNIFORM_BATCH`, the final engine mode is
`FULL_AND_PIECEWISE`, not unconditional FULL. Startup captured both graph sets.
With a fixed prompt salt, identical sampling and identical acceptance, the
measured graph A/B was:

```text
                 PIECEWISE   FULL_AND_PIECEWISE   delta
4K C1 decode       118.1            127.8         +8.2%
4K C1 ms/pass       25.4             23.5         -7.5%
4K C4 decode       350.9            364.9         +4.0%
4K C4 ms/pass       38.5             37.0         -3.9%
```

`bench/conc_ladder.py --salt ...` now provides fixed-content A/B prompts so
proposal acceptance cannot silently invalidate graph comparisons.

Additional FULL_AND_PIECEWISE results:

```text
32K/512 C1: 104.1 decode tok/s; 3.38 tok/step; 32.3 ms/pass; TTFT 19.08 s
60K/512 C1:  87.9 decode tok/s; 3.66 tok/step; 41.2 ms/pass; TTFT 40.76 s
30K prefix: parity true; 29,568 cached; 39.74x TTFT speedup
API smoke: 12/12
```

The same guarded path also passed the next context tiers:

```text
85,514/512 C1: 74.8 decode tok/s; 3.72 tok/step; 49.5 ms/pass;
                 TTFT 65.17 s; KV peak 19.1%; 0 preemptions
126K/512 C1:    54.1 decode tok/s; 3.41 tok/step; 62.9 ms/pass;
                 TTFT 110.32 s; KV peak 27.1%; 0 preemptions
120K prefix:    119,168 cached; parity true; TTFT 103.38/1.51 s; 68.59x
250K/512 C1:    31.5 decode tok/s; 3.27 tok/step; 103.3 ms/pass;
                 TTFT 310.59 s; KV peak 51.6%; 0 preemptions
240K prefix:    239,232 cached; parity true; TTFT 290.89/2.54 s; 114.36x
```

The 256K profile allocated 1,126,218 KV tokens and reported theoretical 4.30x
concurrency at 262,144 tokens per request. These are capacity-planning numbers,
not a claim that four simultaneous 256K requests have completed a soak.

After every tier the service remained healthy and the kernel log contained no
Xid or illegal-memory-access report. The host is currently left on the guarded
256K FULL qualification service on port 8002. It exposes model id
`qwen3.8-27b` without an API key.

### 17.5 Remaining Draft blockers

Before declaring production-ready or removing Draft status:

1. run longer multi-request soak at the 128K/256K profiles and deliberate
   high-block-ID runtime traffic;
2. repeat exact 85,514 enough times to cover the original intermittent-fault
   history rather than treating one clean run as a soak;
3. verify the normal BF16 and KVarN profiles after this experimental series;
4. preserve and publish raw benchmark/log artifacts for the longer tiers;
5. decide whether the explicit FULL graph flag should remain experimental by
   default (recommended) or graduate into a shipped CMP profile.

## 18. Verifier redesign handover (2026-09-16)

The qualified 4-warp/NSEG35 Triton verifier remains unchanged.  The redesign
work is append-only and each experiment is preserved as a candidate patch plus
measured decision in `docs/verifier-redesign-log.md`.

Completed milestones:

| milestone | commit | result |
|---|---|---|
| V0 baseline freeze | `1231ccf` | verifier share and roofline frozen |
| V1 Triton stages 2/3 | `958cd6d` | rejected; shared-memory growth |
| V2 correctness gate | `030f941` | accepted test-only improvement |
| V3 8-warp/NSEG factorial | `f75a362` | rejected; one CTA/SM |
| V4 FP16 partial workspace | `e327b48` | rejected; 5-15% slower |
| V5 disable LICM | `dbb30ef` | rejected; no occupancy change |
| V6 explicit 3x16 rows | `d2f42c9` | rejected; 15-17% slower |
| V7-E0 CUDA scaffold | `5e76c2a` | build/ABI smoke accepted |
| V7 full correctness gate | `36bff72` | high-block-ID accepted |
| V7-E1 shared accumulator | `73ed877` | correct, scalar math 19-29x slower |
| V7-E2 BF16 WMMA | `e81c5fd` | correct; rejected, shared-feed path 9-20x slower |
| V7-E3a padded WMMA | `62d0375` | conflicts -3x; rejected, occupancy fell to 1 CTA |
| V7-E3b two-phase PV | `6e0d464` | correct, 2 CTA restored; ~5% long gain, still rejected |
| V7-E4a shared decode LUT | `0e09a1d` | correct, 2 CTA; 5.6-6.1% vs E2 long, scoreboard unchanged |
| V7-E4b direct vector load | `4c83b37` | correct; rejected, +13% at 4K and flat long-context |
| V7-E5 compact raw staging | `cc04e4e` | correct; -1.2/-2.9% at 4K/70K, flat long-context |
| V7-E6 bitwise FP8 decode | `c732fe9` | correct; 43-44% long gain, accepted isolated scaffold |
| V7-E7 direct global load | `b6980c6` | correct; rejected, 28-69% slower than staged E6 |
| V7-E8 dual staging | current milestone | correct; 1-2% long gain, +7.7% 4K, rejected |
| V7-E9b FP8x2-half bridge | current milestone | correct; -8.3% 4K but +2.8-3.0% long, rejected |

The service was deliberately stopped for isolated GPU testing.  Restore
`cmp170hx-mixed-fp8-full-256k-8002.service` only after the active experiment is
finished.  No rejected candidate is present in the active patch series or the
qualified service tree.

The next owner should use E6 four-phase staging as the performance baseline
but preserve E8's corrected NaN fail-closed contract. E9b proved the target
CUDA 13.0 toolkit has only FP8x2-to-half, not direct FP8x2-to-BF16: the bridge
improved 4K 8.3% but regressed 126K-250K 2.8-3.0% and is rejected. Next test
split-local page IDs/bases as an independent factorial.
Admission remains zero spill, two CTAs/SM, >=5% isolated gain at 126K/250K and
<=2% 4K regression before full-model/CUDA Graph A/B.
