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
| V7-E10a page-base staging | current milestone | correct; ~0.7-0.9% stable long gain, rejected |
| V7-E11 persistent Q | current milestone | correct; ~38-39% stable long gain, accepted isolated scaffold |

The service was deliberately stopped for isolated GPU testing.  Restore
`cmp170hx-mixed-fp8-full-256k-8002.service` only after the active experiment is
finished.  No rejected candidate is present in the active patch series or the
qualified service tree.

The next owner should use E6 four-phase staging as the performance baseline
but preserve E8's corrected NaN fail-closed contract. E9b proved the target
CUDA 13.0 toolkit has only FP8x2-to-half, not direct FP8x2-to-BF16: the bridge
improved 4K 8.3% but regressed 126K-250K 2.8-3.0% and is rejected. E10a then
tested split-local page bases independently. It passed every gate and retained
two CTAs/SM, but the stable 126K-250K gain was only 0.7-0.9%, so it is also
rejected for production admission. E11 then isolated persistent Q preparation:
166 registers/thread, zero spill, two CTAs/SM and all correctness/high-ID gates
passed. It reduced stable 126K/200K/250K latency from E6's
8.13/12.85/16.04 ms per layer to about 5.01/7.85/9.79 ms, with instructions
down 37.1% and long-scoreboard stalls down to 9.28%. E11 is now the accepted
isolated CUDA scaffold, not a production dispatch. The next owner should
preserve persistent Q and target the newly dominant 29.52% barrier and 13.97%
short-scoreboard costs in the serialized softmax/PV group schedule.
Admission remains zero spill, two CTAs/SM, >=5% isolated gain at 126K/250K and
<=2% 4K regression before full-model/CUDA Graph A/B.

Reference note: `Ithrial/ninfer-cmp170hx` is a real SM80 serving port but its
verifier is BF16/INT8-G64, not FP8. It independently validates the one-KV-head
CTA, all-GQA-rows and split-local page-list topology, but retains split policy
and limits from a 170-SM parent and is much slower than the measured vLLM
control. Borrow only the dataflow/testing ideas; do not copy its cache ABI,
64-token pages, INT8 scales or split constants.

## 19. E38/E39 INT8-G64 handover (2026-09-17)

This is the current stopping point for the INT8-G64 cache experiment. It ran
only on the isolated `base-node@192.168.10.206` port-8002 unit. Production
8000 and Guardian were not changed or restarted by this experiment. The 8002
unit is now stopped/failed and is not a working service.

### 19.1 Objective and isolation

The intended comparison is the existing W4A16 Qwen3.8 target + DFlash2 versus
a vLLM-compatible INT8-G64 KV cache:

```text
target: /home/base-node/models/Qwen3.8-27B-W4A16-AutoRound-fast
draft:  /home/base-node/models/Qwen3.8-27B-DFlash2-W4A16
target KV: int8_g64 (G=64), requested attention block=64
port: 8002 only; power 180 W; graphics clock locked to 1350 MHz
```

Remote experiment root:

```text
/home/base-node/.codex_tasks/pixelml-cmp170hx/int8g64-e38-test/
/home/base-node/.codex_tasks/pixelml-cmp170hx/int8g64-e38-test/recipe-g64/
/home/base-node/.codex_tasks/pixelml-cmp170hx/int8g64-e38-test/int8g64-8002.log
/home/base-node/.codex_tasks/pixelml-cmp170hx/int8g64-e38-test/v7_verifier_e38_int8.cu
```

Units:

```text
/home/base-node/.config/systemd/user/int8g64-8002.service
/home/base-node/.config/systemd/user/triton-int8-control-8002.service
```

Only one of these port-8002 units may run at a time. The control unit uses
stock per-token/head INT8 interpretation; it is a correctness control, not
the G64 implementation.

### 19.2 Proven facts

* The standalone E38 oracle/writer is numerically plausible (prior oracle
  cosine was about `0.989`) and stores int8 K/V plus FP16 G=64 scales.
* The stock per-token/head INT8 control with the same target/draft produced
  coherent text, so the original garbling was in the G64 path/geometry.
* Before the guards, vLLM promoted target cache geometry to `896` tokens and
  the custom shape gate was false; execution fell back to the incompatible
  stock INT8 interpretation.
* The platform hybrid-alignment guard and indivisible-promotion guard were
  reached. The bounded attempt therefore kept the requested 64-token block.
* The bridge is installed in the remote runtime at:

```text
/home/base-node/.codex_tasks/pixelml-cmp170hx/runtime-v0271/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/int8_g64.py
```

### 19.3 Latest failure and interpretation

The final bounded start used `G64_MAX_LEN=64000`; the earlier 250K attempt is
not usable. EngineCore warmup failed before serving any request:

```text
ValueError: INT8-G64 requires VLLM_KV_CACHE_LAYOUT=HND;
got strides=(1777664, 16384, 256, 1), expected=(65536, 16384, 256, 1)
```

The matching debug line was:

```text
kv_update mode=8 enabled=True cache_shape=(2905, 4, 64, 256)
cache_stride=(1777664, 16384, 256, 1) cache_dtype=torch.int8
heads=4 hs=256 slot_shape=(32,)
```

This is a cache-layout contract failure, not a Triton, DFlash2 selector,
acceptance, or OOM result. `g64_views()` requires HND but the vLLM allocation
does not provide that stride. No first token was generated; therefore there
are no valid G64 TPS, acceptance, CUDA Graph, NCU, or whole-model A/B results.

The previous forced-64-token/250K start also showed a separate geometry limit:
padding/accounting required `109.23 GiB` KV versus about `38.49 GiB` available,
and `KVCacheCoordinator` asserted on incompatible scheduler/hash block sizes.
Forcing `--block-size 64` alone cannot provide a 250K mixed cache.

### 19.4 Safe resume order

1. Confirm the isolated unit is stopped before making changes:

```bash
ssh base-node@192.168.10.206 \
  'systemctl --user stop int8g64-8002.service; systemctl --user is-active int8g64-8002.service'
```

2. Fix the layout contract first. Either make the allocator emit the HND
   shape/stride expected by `g64_views()` for the G64 target pool, or make the
   bridge consume the actual layout through an explicit validated permutation.
   Do not merely remove the check and reinterpret memory: that recreates the
   earlier garbling.

3. Keep a compact, separate 64-token G64 target pool and a separate Mamba/GDN
   pool. If allocation remains globally padded to Mamba's 896-token geometry,
   250K is impossible under current page accounting.

4. After the layout fix, run only bounded correctness: KV init -> one
   deterministic short request -> continuation prefill -> batch 2/4 -> graph
   capture. Check `/tmp/int8g64-debug.log`, the service log and `dmesg` for
   Xid/illegal access; stop on the first correctness failure.

5. Only after coherent first-token output run 4K/126K/250K C1/C2/C4 locked
   A/B, NCU, graph, DFlash2 acceptance and quality tests against a precisely
   identified control path. Label the control by the actual runtime path.

### 19.5 Machine state at pause

At the pause check on 2026-09-17, `int8g64-8002.service` was `failed` with
exit code 1 after the HND stride error. Its ExecStopPost reset graphics
clocks. Port 8000 was observed unavailable at that moment, but this
handover operation did not start, stop, or otherwise modify 8000. The next
agent must independently verify 8000 and Guardian before changing either.

The G64 helper scripts, service examples and integration notes are currently
untracked local experiment files. Review them before any commit or push; no
commit/push was made for this G64 experiment in this turn.

## 20. INT8-G64 layout contract fixed; new blocker at CUDA-graph replay (2026-09-17)

This section continues 19.4. The cache layout/allocator contract of 19.3 is
**resolved and verified inside the real isolated runtime**; a different blocker
now sits after it, at CUDA-graph replay. Nothing here touched port 8000 or
Guardian, and no commit or push was made.

### 20.1 Root cause of the 19.3 stride failure (confirmed)

The allocator pads every physical page to its own stride, so a per-layer tensor
arrives as `stride=(1777664, 16384, 256, 1)` rather than the contiguous
`(65536, 16384, 256, 1)` the original bridge demanded. Byte-sentinel experiments
proved the padded layout is **not** a contiguous reinterpretation: reinterpreting
it reads page padding (byte value 85) instead of page 1. The two layouts are:

| layout | page stride | structure |
| --- | --- | --- |
| original packed | 65536 B (K/V), 1024 el (scales) | `[all K][all V][all Kscales][all Vscales]` |
| allocator page-local | 1777664 B | per page `[K 65536][V 65536][Kscales 2048][Vscales 2048][pad]` |

The fix keeps the strict checks and changes the ABI: `g64_views()` now returns
four **page-local** views that share the allocator's page stride and differ only
by in-page offset (K at +0, V at +65536, Kscales at +131072, Vscales at
+133120). Overlapping or misaligned pages are still rejected rather than
reinterpreted, exactly as 19.4 required.

### 20.2 A second, previously unreported defect: prefill never worked

`KVQuantMode.INT8_G64` reports `is_per_token_head=True`, so stock
`unified_attention` computes `BLOCK_Q = BLOCK_M // num_queries_per_kv` = `16 // 0`
and raises `ZeroDivisionError` on every prefill/warmup call. The stock kernel
also cannot express per-group G=64 scales. A/B evidence, same runtime, prefill
route disabled:

```
run B (no prefill route):  ZeroDivisionError: integer division or modulo by zero
run A (prefill route):     gets past KV init, warmup and graph capture
```

So a dedicated G64 prefill kernel is **required**, not optional. The new kernel
dequantizes per-group K/V pages and follows vLLM's causal contract
`context_len = seqused_k - current_batch_query_len` (read from
`triton_unified_attention.py`), i.e. a continuation-prefill query at row `i`
sits at absolute position `context_len + i`.

### 20.3 Verified before deployment

All GREEN, run against the exact module that was then installed:

- `test_merged_bridge.py` — page-local view offsets/strides, writer round-trip,
  page-isolation sentinel (page 1 untouched, 0 non-sentinel bytes), prefill
  numerics `max_abs=0.00445 / 0.00459` on a two-batch continuation shape, and
  rejection of overlapping / misaligned pages.
- `test_g64_prefill_v2.py` — tiled prefill, both padded and compact geometries,
  `max_abs=0.01389`.
- `test_page_attention.py` — strided E38 adapter: zero-Q oracle
  `max_abs=0.00137`, padded and contiguous **bitwise equal**, non-zero-Q
  mixed-length shuffled-page ABI parity.
- `test_page_layout.py` — 8/8, writer + views on the padded page.

Debugging lesson worth keeping: several hours were lost editing kernel address
arithmetic before writing `test_g64_addr_probe.py`, which printed the four
values the kernel's formulas produce next to the torch views and proved the
addressing had been right all along — the error was in the test oracle (it
attended unwritten 85-fill tokens). Write the byte-level probe first.

### 20.4 New blocker: illegal memory access at CUDA-graph replay

With the page-local module and the prefill route deployed, the engine now gets
through KV init, the G64 writer, the E38 decode bridge and **both** CUDA graph
capture passes:

```
INT8-G64 cache views enabled (HND, block_size=64).
INT8-G64 writer debug: cache (2905, 4, 64, 256) (1777664, 16384, 256, 1) k_scale (2905, 4, 64, 4) (888832, 256, 4, 1)
INT8-G64 bridge debug: tables (4, 128) valid [8, 8, 8, 8] pos [0..7] split 4 capacity 8
Capturing CUDA graphs (PIECEWISE): 100%
Capturing CUDA graphs (FULL): 100%
```

then fails in `warmup.py:338 _run_decode_step` at
`cudagraph_utils.py run_fullgraph -> graphs[desc].replay()` with
`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`.
Confirmed with `CUDA_LAUNCH_BLOCKING=1`, which localised it from a sticky error
surfacing in a later Triton `load_binary` to the actual replay site.

The prefill inputs during warmup were captured with a one-shot probe and are
benign: `max_q 6, max_k 6, num_actual_tokens 24, q (24,24,256),
cu_q [0,6,12,18,24], seqused [6,6,6,6], table (4,128)` — `q_row` reaches 33
against 24 rows, but `mask_q = (0..15) < 6` masks every out-of-range row, so no
unmasked access occurs on that path.

**What is and is not established.** The A/B above proves the prefill route is
*necessary* (without it the run dies earlier with `ZeroDivisionError`). It does
**not** attribute the replay fault: run B never reached replay, so the E38 decode
path's graph-replay safety is still unverified. The leading hypothesis, to be
tested next, is the pre-existing E38 workspace cache — `_g64_workspace` is keyed
on `(batch, split_count)` and reallocated whenever either changes, while warmup
captures graphs for `cudagraph_capture_sizes=[1,2,4,8,16,24,32]`. Tensors
allocated during capture and later freed and reused for a different
`(batch, split_count)` would leave stale pointers baked into a captured graph,
which is exactly the observed replay-time fault. The next test is to pre-allocate
the workspace for every capture size before capture begins (or force
`enforce_eager=True` to confirm replay is the sole trigger).

### 20.5 Machine state at this pause

`int8g64-8002.service` is `failed`/stopped; GPU idle at 14 MiB; port 8000 had no
listener throughout and was never started, stopped, or modified. The isolated
runtime currently holds the page-local module (md5 `2252c9d366e322bc67cf8852fba09bf3`),
the prefill route, and untouched backups
`v1/attention/ops/int8_g64.py.orig-g64layout` and
`v1/attention/backends/triton_attn.py.orig-g64layout`. The strided-adapter
environment lives in the reversible systemd drop-in
`~/.config/systemd/user/int8g64-8002.service.d/g64-layout.conf`, which also sets
`G64_MAX_LEN=8192` and (for this debugging session) `CUDA_LAUNCH_BLOCKING=1`;
remove that variable before any performance measurement.

Reproduce with:

```bash
# artifacts live here (all local experiment files, still untracked)
ls int8g64-layout-audit/            # probes, kernels, tests, snapshots
python3 make_runtime_candidate.py   # regenerates runtime-candidate/int8_g64.py
scp runtime-candidate/int8_g64.py <remote>:.../int8g64-e38-test/
python3 scripts/deploy_int8_g64_layout_fix.py <vllm_root> int8_g64.py
systemctl --user start int8g64-8002.service
```

Per 19.5 these remain untracked local files: review before any commit or push.

### 20.6 Bounded correctness run: engine starts eagerly, batch 3/4 degenerate

`enforce_eager=True` localises the 20.4 fault to CUDA graphs and lets the engine
start for the first time on the page-local layout. Note that a systemd drop-in
must quote a multi-flag value — `Environment=EXTRA_ARGS=--a --b` is split on
whitespace into two assignments and silently drops the second flag; use
`Environment="EXTRA_ARGS=--a --b"`.

With the engine up (isolated unit only), `scripts/bounded_correctness_probe.py`
reported:

```
== 1. determinism: one greedy request twice ==   identical=True
== 2. long prompt: chunked / continuation prefill == coherent answer
== 3. batch 2: both requests coherent
== 3. batch 4: req0/req1 coherent, req2/req3 both 'Register Register Register...'
BOUNDED_PROBE OK
```

`BOUNDED_PROBE OK` only means "non-empty text"; the batch-4 rows are wrong. A
follow-up with content-distinguishable prompts (count 1-5 / 1-10 / 1-15 / 1-20),
run twice, is byte-identical across trials and shows the failure is systematic,
not a race:

| batch | req0 (1-5) | req1 (1-10) | req2 (1-15) | req3 (1-20) |
| --- | --- | --- | --- | --- |
| 3 | correct | `Register...` | partial (digits, not words) | — |
| 4 | correct | near-correct | `Register...` | `Register...`, **identical to req2** |

Two facts matter for the next session. First, the failing slot **moves with batch
size** (req1 at batch 3, req2+req3 at batch 4) rather than being tied to an
index, and req0 is always correct. Second, at batch 4 the last two requests
produce **identical** output from different prompts, which is the signature of
aliased KV state rather than of quantization error. Determinism across trials
rules out a race.

**Attribution is still open and must not be assumed.** These runs have no
baseline control, so it is not yet established whether the degeneracy belongs to
the INT8-G64 work or is pre-existing in the DFlash2 verify path at batch >= 3.
Recall that the E38 whole-model bridge guard is `1 <= batch <= 4 and
num_actual_tokens == batch * 8`, and the warmup inputs observed in 20.4 were
`cu_q [0,6,12,18,24]` (batch 4, q_len 6, so `num_actual_tokens == 24 != 32`),
which means the E38 branch was *skipped* and the new prefill route handled that
step. The immediate next experiment is the same batch-3/4 pattern against a
known-good baseline (CTX=fast, bf16 KV) on the same unit; only if the baseline
is clean does this become a G64 defect, and the first suspects are then the
per-request page-table/`split_count` handling in the E38 bridge and the batch
dimension of `_g64_workspace`.

### 20.7 Machine state after this run

The isolated unit was run eagerly for the probes above and then **stopped**;
port 8000 was never started, stopped or modified and had no listener throughout.
GPU idle. The runtime still holds the page-local module
(md5 `2252c9d366e322bc67cf8852fba09bf3`), the prefill route, and both
`.orig-g64layout` backups. The drop-in currently sets
`"EXTRA_ARGS=--enable-prompt-tokens-details --enforce-eager"`; drop the
`--enforce-eager` flag to return to graph mode (where 20.4's replay fault
reproduces) and before any performance measurement.

### 20.8 Baseline control run: the batch-3/4 degeneracy is G64-specific, and prefill is ruled out

The control experiment from 20.6/20.7 was run on the same isolated unit with the
same launcher settings (`MAX_SEQS=4`, `SPEC=dflash2`, `DFLASH_TOKENS=7`,
`--enforce-eager`) and only the KV cache changed. The baseline identifies itself
in its own log:

```
kv_cache_dtype=bfloat16
AttentionBackendEnum.FLASH_ATTN backend
enforce_eager=True
INT8-G64 code paths in this run: 0
```

Same probe, same content-distinguishable prompts, two trials:

| batch | INT8-G64 (TRITON_ATTN, kv int8_g64) | baseline (FLASH_ATTN, kv bfloat16) |
| --- | --- | --- |
| 2 | both coherent | all correct, distinct |
| 3 | req1 `Register...`, req2 partial | all correct, distinct |
| 4 | req2 + req3 `Register...`, identical | all correct, distinct |

`dup=False` on every baseline row, and baseline output is correct and on-topic at
every batch size. **The degeneracy is therefore a G64-specific defect that
appears at batch >= 3**, not a pre-existing DFlash2/verify limitation.

The next question was which half of the G64 path owns it, so prefill — the piece
this session rewrote — was tested directly. Every earlier prefill test used B=2
only, which was a real gap. `test_g64_batch34_prefill.py` closes it:

```
== equal-length prompts (engine pattern) ==
  PASS batch3_equal12 per_batch=[0.01404, 0.012, 0.01415]
  PASS batch3_equal1 (decode-sized) per_batch=[0.00775, 0.00769, 0.00763]
  PASS batch4_equal12 per_batch=[0.01356, 0.01326, 0.01402, 0.01159]
  PASS batch4_equal1 (decode-sized) per_batch=[0.00778, 0.00778, 0.00768, 0.00775]
== continuation prefill (cached prefix, seqused_k > q_len) ==
  PASS batch4_cont12_of_40 per_batch=[0.00666, 0.0071, 0.00447, 0.00663]
  PASS batch3_cont8_of_40 per_batch=[0.00449, 0.00509, 0.00522]
== uneven lengths ==
  PASS batch3_uneven   PASS batch4_uneven
BATCH34_PREFILL PASS
```

Per-request errors are uniform across the batch (no tail-request outlier), so the
new prefill route is **exonerated at batch 3 and 4**. With prefill and the writer
both clean, the defect lies in the **E38 decode/verify path at batch >= 3** — the
pre-existing `run_e38_attention` bridge. Ranked suspects, none yet tested:

1. `_g64_workspace`, keyed on `(batch, split_count)` and reallocated whenever
   either changes, during a run that revisits several batch sizes;
2. `positions` and `valid_columns`, built as
   `attn_metadata.seq_lens[:batch] - 8 + arange(8)` and `seq_lens[:batch]` — worth
   checking that the per-request row of `block_table[:batch]` pairs with the
   matching `valid_columns` entry rather than a broadcast one;
3. `split_count = min(32, max(4, (max_seqlen_k + 4095) // 4096))`, computed from a
   **batch-wide** `max_seqlen_k` while each request has its own length — the
   tail-request aliasing at batch 4 is consistent with a per-request column
   being resolved from a batch-level value.

The cheapest next step is a standalone E38 decode test at batch 1..4 against a
dequantized-cache oracle, mirroring `test_page_attention.py` but sweeping batch
and per-request lengths, which separates suspects 2 and 3 from suspect 1 without
starting the engine.

### 20.9 Machine state after the control run

The baseline finished, the unit was stopped and the G64 experiment drop-in
(eager, `G64_MAX_LEN=8192`) was restored with the unit left **stopped**. Port 8000
was never started, stopped or modified and had no listener throughout; GPU idle.
The runtime still carries the page-local module
(md5 `2252c9d366e322bc67cf8852fba09bf3`), the prefill route, and both
`.orig-g64layout` backups. To reproduce the two open defects:

- batch >= 3 degeneracy: start the unit as configured (eager) and run the
  batch-3/4 probe with content-distinguishable prompts;
- CUDA-graph replay fault (20.4): remove `--enforce-eager` from the drop-in's
  `EXTRA_ARGS` and start the unit.

### 20.10 Root cause of the batch>=3 degeneracy: a PRE-EXISTING E38 decode defect

The standalone sweep from 20.8's plan was written as
`test_e38_decode_sweep.py`: batch 1..4, equal and mixed per-request lengths,
split_count 1/4/8/16, a per-request int8-Q-emulated softmax oracle, an explicit
cross-match aliasing check, and a sentinel-init diagnostic. Against the
page-local strided adapter:

```
PASS batch1_len200 split=4 errs=[0.0026]
PASS batch2_len200 split=4 errs=[0.003, 0.0033]
FAIL batch3_len200 split=4 errs=[0.0034, 0.0022, 1.6881195546468432e+38]
FAIL batch4_len200 split=4 errs=[0.0024, 0.0024, 127.4192, 127.2875]
FAIL batch4_mixed  split=4 errs=[0.0023, 0.003, 127.6642, 127.5081]
```

The failure is structural, not numeric: batch 1 and 2 are always exact, and
**every request with batch index >= 2 is broken, for every batch size and every
split_count**. Re-initialising `acc/m/l` to the online-softmax identity
(`acc=0, m=-inf, l=0`) before the call did not repair it — it turned the garbage
into `nan` instead, which means those requests' splits were never populated with
real data at all, not merely reduced wrongly.

To decide whether the page-stride port caused this, the identical sweep was run
against the **untouched** `v7_verifier_e38_int8.cu` with the original packed
contiguous ABI (`test_e38_orig_adapter_sweep.py`, log line `packed ABI:
page_stride = 65536`):

```
PASS ORIG-ADAPTER batch1_len200 split=4 errs=[0.0026]
PASS ORIG-ADAPTER batch2_len200 split=4 errs=[0.003, 0.0033]
FAIL ORIG-ADAPTER batch3_len200 split=4 errs=[0.0034, 0.0022, nan]
FAIL ORIG-ADAPTER batch4_len200 split=4 errs=[0.0024, 0.0024, 4.1224, 4.1986]
FAIL ORIG-ADAPTER batch4_mixed  split=4 errs=[0.0037, 0.0026, 4.3447, 4.0753]
```

Same signature, same index boundary. **The batch>=3 decode defect is pre-existing
in the E38 extension and was not introduced by this session's layout work.** The
garbage magnitudes differ between the two runs (1.7e38/127 vs nan/4.1) only
because the cache ABI differs; what is invariant is that batch index >= 2 is dead.

This was invisible until now because **every previous E38 decode test used batch
2** — `test_page_attention.py` builds `tables` of shape `(2, 2)`. Combined with
20.8's finding that every earlier prefill test used B=2 as well, the whole G64
verification suite had a batch-2 blind spot, which is why the engine's first real
run at `MAX_SEQS=4` surfaced it.

Where to look next, all in the E38 partial/reduce pair:

1. `page_stride_kernel.cuh:194` — `if (split >= active_split_count) { return; }`
   runs *after* `write_neutral` (lines 154-172) has stored `acc=0, m=?, l=0`. Check
   that the neutral `m` value and the `active_split_count` derivation both hold for
   every batch entry, and that the early return happens after the neutral store for
   batch >= 2 as well;
2. the index helpers `gqa_partial_acc_index<Geometry>(q_head, d, token, split,
   TokenTile)` and `gqa_partial_stat_index<Geometry>(q_head, token, split,
   TokenTile)` — the per-batch pointer advance in the kernel is
   `batch * D * QHeads * TokenTile * split_count`, which matches the
   `[batch, split, 8, 24, 256]` workspace, so verify the helper's own split/token
   decode next;
3. `reduce_batch` launches `grid(QHeads, 1, Batch * 8)` and derives
   `Batch = out.numel() / (8*24*256)`; confirm the reduce kernel decodes batch from
   `blockIdx.z` the same way the partial kernel does.

A standalone reproduction needs no engine: run `test_e38_decode_sweep.py` (or the
orig-adapter variant) and watch request index 2.

### 20.11 Machine state at the end of the sweep session

`int8g64-8002.service` is **stopped**; the drop-in on disk
(`g64-experiment.conf`) holds the G64 eager config
(`G64_MAX_LEN=8192`, `"EXTRA_ARGS=--enable-prompt-tokens-details --enforce-eager"`).
Port 8000 was never started, stopped or modified and had no listener throughout;
GPU idle. The runtime still carries the page-local module
(md5 `2252c9d366e322bc67cf8852fba09bf3`), the prefill route, and both
`.orig-g64layout` backups.

Three defects now stand, in dependency order:

1. **batch >= 3 decode (20.10)** — pre-existing E38 defect; blocks any C3/C4
   measurement and explains the engine's batch-3/4 garbage;
2. **CUDA-graph replay fault (20.4)** — blocks graph-mode startup entirely; the
   engine only runs with `--enforce-eager`;
3. performance/quality A/B (19.4 step 5) — not started; requires 1 and 2, and the
   `CUDA_LAUNCH_BLOCKING`/`--enforce-eager` debug settings must be removed first.

New files this session: `test_e38_decode_sweep.py`, `test_e38_orig_adapter_sweep.py`,
`test_g64_batch34_prefill.py`, `test_merged_bridge.py`, `test_g64_addr_probe.py`,
`test_g64_both_geometries.py`, `make_runtime_candidate.py`,
`runtime-candidate/int8_g64.py`, and in the PR worktree
`scripts/deploy_int8_g64_layout_fix.py`, `scripts/bounded_correctness_probe.py`.
All remain untracked per 19.5.

## 21. Both open defects fixed; engine runs in graph mode at batch 1..4 (2026-09-17)

Sections 20.4 and 20.10 left two defects. Both are now fixed, and each fix has a
component-level RED->GREEN record plus an end-to-end confirmation.

### 21.1 Fix 1 — E38 decode dropped every request with batch index >= 2

Root cause, found by reading the reduce launch rather than guessing: the adapter
passed a **literal `2`** for the reduce kernel's `batch_size` parameter.

```cpp
// page_stride_adapter.cu and v7_verifier_e38_int8.cu, reduce_batch():
const dim3 grid(Geometry::QHeads, 1, Batch * 8);
...<<<grid, 256, 0, stream>>>(
    partial_acc, partial_m, partial_l, pos, nullptr,
    8, 8, 0, 2, split_count,            // <- batch_size was hardcoded to 2
    out);
```

The ninfer kernel itself is generic (`ops/kernel/gqa_attention_decode.cuh` takes
`batch_size` as a parameter); only the adapter's argument was wrong:

```cpp
if constexpr (MultiBatch) { if (batch >= batch_size) { return; } }
```

so `out` was **never written** for batch index >= 2, leaving whatever the caller's
`torch.empty` buffer contained. That is exactly the observed signature: garbage
magnitudes (1.7e38 / 127.4) and two unwritten slots holding *identical* content,
which is the req2 == req3 aliasing seen in the engine. The vestigial
`"pos must be int32 [2,8]"` / `"block_tables must be int32 [2,pages]"` checks in
the same function are the fingerprint of the original 2-request design this
literal was left behind by.

Fix: pass `Batch`. Applied to both `page_stride_adapter.cu` (the deployed strided
port) and `v7_verifier_e38_int8.cu` (the untouched original), because both carried
the same literal. The check messages now say `[batch,...]`.

RED -> GREEN on `test_e38_decode_sweep.py` (batch 1..4, equal/mixed lengths,
split_count 1/4/8/16):

```
before:  FAIL batch3_len200 errs=[0.0034, 0.0022, 1.688e+38]
         FAIL batch4_len200 errs=[0.0024, 0.0024, 127.4192, 127.2875]
after:   PASS batch3_len200 errs=[0.0034, 0.0022, 0.0024]
         PASS batch4_len200 errs=[0.0024, 0.0024, 0.0034, 0.0021]
         ... all 12 cases PASS, every split_count
```

### 21.2 Fix 2 — CUDA graph replay fault: E38 workspace allocated inside capture

20.4's hypothesis is confirmed. Disabling the E38 verify branch (so verify fell
through to the G64 prefill kernel) let the engine reach `Application startup
complete.` **in graph mode**, which attributed the replay fault to the E38 branch
and cleared the prefill route.

The cause is the workspace below, whose tuple is keyed on `(batch, split_count)`
and therefore reallocated repeatedly while warmup captures graphs for
`cudagraph_capture_sizes = [1,2,4,8,16,24,32]`. Tensors allocated inside one
capture region were reused for another, so replay followed stale pointers:

```python
if (self._g64_workspace is None or self._g64_workspace[0] != batch
        or self._g64_workspace[1] != split_count):
    self._g64_workspace = (batch, split_count, torch.empty(...), torch.empty(...), torch.empty(...))
```

Fix: pre-allocate one buffer per split_count in `__init__` (batch <= 4 is enforced
by the adapter, and the bridge only ever derives split_count from
`min(32, max(4, ...))`, normalised to the next value in `(1, 4, 8, 16, 32)`), so
no allocation can happen inside a capture region while the dynamic split_count is
preserved:

```python
self._g64_ws_cache = {}
if self._g64_enabled:
    _dev = torch.device("cuda", torch.cuda.current_device())
    for _s in (1, 4, 8, 16, 32):
        self._g64_ws_cache[_s] = (
            torch.empty((4, _s, 8, self.num_heads, self.head_size), dtype=torch.bfloat16, device=_dev),
            torch.empty((4, _s, 8, self.num_heads), dtype=torch.float32, device=_dev),
            torch.empty((4, _s, 8, self.num_heads), dtype=torch.float32, device=_dev),
        )
```

With E38 re-enabled and no `--enforce-eager`, the engine starts in graph mode in
42 s.

### 21.3 End-to-end verification after both fixes

Graph mode, E38 enabled, content-distinguishable prompts, two trials, byte-stable
across trials:

| batch | request outputs (head) |
| --- | --- |
| 1 | `One Two Three Four Five` |
| 2 | + `1 2 3 ... 10  Wait, the user ask` |
| 3 | + `1 2 3 ... 15  Wai` |
| 4 | + `1 2 3 ... 15 16 1` (continues past 15, matching its own prompt) |

`dup=False` on every row. Before the fix, batch 3 lost a request to
`Register Register Register...` and batch 4 returned that same string for both of
its last two requests.

### 21.4 A/B harness

`scripts/ab_bench.py` measures one arm (one engine on one port): greedy, fixed
seed, `ignore_eos`, identical prompts per row, warmup rounds discarded, C1/C2/C4,
emitting one JSON row per concurrency with p10/p50/p90 latency and aggregate
output tokens/s. Both arms are run with `--ctx` 4096 / 126000 / 250000 so the
rows can be diffed directly. Results are recorded in section 22.

### 21.5 Fix 3 — the E38 verify path was never even enabled

While running the A/B, the G64 arm measured ~2.7 output tok/s at 4k context
against the baseline's ~114 tok/s (section 17), and the engine log showed
`INT8-G64 bridge debug` **never printed**, i.e. `run_e38_attention` was never
called. Two gates were responsible, and both are adapter-level, not kernel-level.

1. The E38 branch tested `self.sliding_window == (-1, -1)`, but full-attention
   layers report `sliding_window = None`. The strict test silently dropped every
   layer to the fallback, exactly like the batch-2 blind spot of 20.10 — the
   branch simply never ran. Fixed to
   `(self.sliding_window is None or self.sliding_window == (-1, -1))`, matching
   what the prefill route already did.
2. `_spec_attn_enabled()` is
   `os.environ.get("VLLM_SPEC_DECODE_ATTN", "0") == "1"`
   (defined in `vllm/v1/attention/backends/flash_attn.py:1728`), and the isolated
   unit **never set that variable**. Every E38 and spec-attention branch was
   therefore dead code in this configuration. Fixed by adding
   `Environment=VLLM_SPEC_DECODE_ATTN=1` to the drop-in.

After both changes the bridge debug line does print, so E38 now owns the verify
step.

### 21.6 A/B status: NOT completed, and why

The A/B was started with `scripts/ab_bench.py` (locked greedy/seed/`ignore_eos`,
warmup rounds discarded, C1/C2/C4, one JSON row per row) against 4k context and
was **stopped after the G64 C1 row measured 2.7 output tok/s** — roughly 40x below
the bf16 baseline, and identical (47.30 s / 2.7 tok/s / 3756 prompt tokens) before
and after fix 3, which is itself evidence that the bottleneck is not the E38
routing. Reporting A/B rows from this configuration would be meaningless, so no
numbers are recorded here.

The isolated unit was left running for the next session (state `active`, port
8002, `G64_MAX_LEN=8192`, `VLLM_SPEC_DECODE_ATTN=1`, graph mode). What is
established and what is not:

- **Established**: the layout contract (20.1), the prefill route (20.2), the
  batch >= 3 decode defect (21.1), the CUDA-graph workspace hazard (21.2), the
  E38 verify gate (21.5). Batch 1..4 correctness is verified end to end in graph
  mode with E38 engaged.
- **Not established**: why 4k decode is ~2.7 tok/s. Prime suspect is the G64
  prefill kernel still serving some verify steps that E38 does not claim — the
  E38 branch requires `max_seqlen_q == 8` **and**
  `num_actual_tokens == batch * 8`, and the warmup probe of 20.4 measured
  `max_q 6, num_actual_tokens 24` (batch 4 x 6), which fails that equality and
  falls through to the naive kernel. That kernel is O(context) per token with a
  broadcast-multiply-sum QK (no tensor cores), which is consistent with a decode
  rate three orders of magnitude below the speculation path.

The immediate next measurement is cheap and decisive: instrument the G64 prefill
route to count how many verify steps per request it serves, and read
`max_seqlen_q` / `num_actual_tokens` for each. If it is serving decode, relax the
E38 gate the same way its sliding-window test was relaxed, then re-run
`ab_bench.py` for both arms at 4k/126k/250k.

### 21.7 21.6's suspect is REFUTED by measurement

21.6 guessed that the G64 prefill kernel was still serving verify steps. A counter
probe was added to the prefill route (recording `max_seqlen_q`,
`num_actual_tokens`, batch per call, skipping device reads while capturing) and a
single 4k C1 request (3756 prompt tokens, 128 output tokens, i.e. ~16-20 verify
steps) was measured. The prefill route was called **18 times in total**:

```
counts {(1,1,1):3, (1,2,2):3, (1,4,4):2, (2,8,4):2, (4,16,4):2, (6,24,4):2,
        (8,9,2):1, (9,36,4):1}
plus max_q 1736 num_actual_tokens 1736 batch 1   (the real 4k prefill chunk)
```

Those are the warmup/profiling shapes plus the prefill chunks. **No decode verify
step went through the prefill route**, so E38 does own verify — the same conclusion
the bridge-debug print gave. The 2.7 tok/s therefore sits **inside the E38 G64
path**, and 21.6's proposed relaxation of `num_actual_tokens == batch * 8` would
have changed nothing. Do not chase it.

What remains to be separated, none of it yet measured:

1. whether the E38 **G64** kernel is intrinsically slow at 4k, or whether this
   whole configuration is slow — the clean control is the same engine config with
   `CTX=fast` (bf16 KV, FLASH_ATTN, `VLLM_SPEC_DECODE_ATTN=1`) measured with
   `ab_bench.py` at the same `--ctx 4096`, which is the baseline arm the A/B
   needs anyway;
2. whether the stride-aware addressing added by this session costs speed: the
   page stride is a runtime `int64` where the original kernel folded
   `kPagedKVPageSize` as a compile-time constant, so the per-element address math
   is no longer strength-reduced. `paged_kv_address_strided.cuh` is the file to
   profile, and `nsys`/`ncu` on a single decode step would settle it;
3. the `split_count` normalisation added in 21.2 — it maps the derived value up to
   the next entry of `(1,4,8,16,32)`, which is a no-op at 4k (derived 4) but must
   be re-checked at 126k/250k.

The unit was left stopped after this probe. `triton_attn.py` in the runtime still
carries the counter probe from this measurement; it only prints, costs nothing per
step, but should be removed (`/home/base-node/.codex_tasks/pixelml-cmp170hx/int8g64-e38-test/triton_attn.py.deployed-fixed`
is the pre-probe copy) before any timing run.

## 22. A/B: first measured row, and the G64 decode slowdown is real

The baseline arm was run on the same unit with the same launcher settings
(`MAX_SEQS=4`, `SPEC=dflash2`, `DFLASH_TOKENS=7`, `VLLM_SPEC_DECODE_ATTN=1`, graph
mode, no `--enforce-eager`), changing only the KV cache and attention backend.
Its own log confirms the identity: `kv_cache_dtype=bfloat16`,
`AttentionBackendEnum.FLASH_ATTN backend`, `enforce_eager=False`. Both arms were
measured with the identical 3756-token prompt asking for 128 output tokens,
greedy, `ignore_eos`, one warmup request discarded.

| arm | backend / KV | 4k C1 | per-request wall |
| --- | --- | --- | --- |
| baseline | FLASH_ATTN / bfloat16 | **47.6 out tok/s** | 2.69 s |
| INT8-G64 | TRITON_ATTN / int8_g64 (this work) | **2.7 out tok/s** | 47.28 s |

Two things follow. First, the configuration is not inherently slow — the baseline
does 47.6 tok/s at 4k on this card, so 21.7's second alternative is eliminated.
**The ~17.6x slowdown is inside the E38 G64 path.** Second, this is the first real
A/B number; C2/C4 and the 126k/250k rows are still missing, so the A/B is **not
complete**.

Ranked causes for the 17.6x, none measured yet:

1. **runtime page stride vs compile-time constant.** The original kernel folded
   `kPagedKVPageSize` (64 tokens) as a compile-time constant, so every page
   address was strength-reduced. `paged_kv_address_strided.cuh` replaces that with
   a runtime `int64` `page_stride * physical_page` multiply in the inner loop.
   The page stride (1777664) is not a whole multiple of the element page payload
   (135168), so it cannot be reduced to a shift, and a 64-bit multiply-add per
   access is exactly the shape that costs a large constant factor. This is the
   prime suspect and the only one this session introduced;
2. **parallelism.** With `split_count = 4` and `KVHeads = 4` the partial kernel
   launches only `grid(4, 4, batch)` = 16 CTAs on 82 SMs for a 4k window, while
   the bf16 path can use FlashAttention's own split. Worth checking whether the
   G64 arm should derive a larger `split_count` at 4k; note 21.2 normalises the
   derived value upward to `(1,4,8,16,32)`, which is a no-op at 4k (derived 4);
3. DFlash2 draft cost — the same in both arms, so it cannot explain the gap, but
   it is worth confirming that acceptance is comparable (a G64 arm with collapsed
   acceptance would do more target steps for the same output).

The measurement to run next is `ncu` or `nsys` on a single 4k decode step in each
arm and compare the attention kernel's time; that distinguishes 1 from 2 in one
shot. `ab_bench.py` is ready for the remaining rows.

Nothing in section 21 needs revisiting for this: the three fixes stand, batch 1..4
correctness is verified, and the two defects that blocked startup are gone. The
G64 arm is simply not yet competitive at 4k, which is exactly what the A/B exists
to measure.

### 22.1 Machine state

The baseline arm was left stopped. Port 8000 was never started, stopped or
modified and had no listener throughout. GPU idle. The isolated runtime still
carries the page-local module and the prefill route plus the counter probe from
21.7 (remove it, or restore
`.../int8g64-e38-test/triton_attn.py.deployed-fixed`, before any timing run), and
both `.orig-g64layout` backups. Drop-ins: `baseline-ctl.conf` (baseline arm) and
the saved G64 configuration used for the fixed-arm runs.

## 23. All four fixable defects fixed; A/B and quality completed within the reachable range

This section supersedes the partial status in 22. Section 22's ranking of causes was
wrong and is corrected here by measurement.

### 23.1 Fix 4 — the G64 prefill kernel was O(N^2) on the FP32 pipe (20.5x)

`test_g64_prefill_*` all passed, so the kernel was *correct*, but its QK step was

```python
s = tl.sum(q[:, None, :] * kq[None, :, :], axis=2) * scale
```

which materialises a `[Q_TILE, BLOCK_N, DIM]` fp32 temporary and never touches
tensor cores. Splitting prefill from decode with `scripts/ab_split_bench.py` made
the shape of the problem obvious:

| metric | baseline bf16 | G64 (broadcast QK) | G64 (tl.dot QK) |
| --- | --- | --- | --- |
| prefill 128 tok | 0.0675 s | 0.0783 s | 0.0687 s |
| **prefill 3756 tok** | **1.78 s (1938 tok/s)** | **39.37 s (87.4 tok/s)** | **1.92 s (1796 tok/s)** |
| decode 127 tok | 1.00 s (127.4 tok/s) | 1.62 s (78.6 tok/s) | 1.63 s (78.6 tok/s) |

Baseline prefill scales 26x for 29x more tokens (linear, GEMM-bound); the old G64
kernel scaled **503x** — the O(N^2) signature. Replacing the QK term with
`tl.dot(q.to(tl.bfloat16), tl.trans(kq.to(tl.bfloat16)))` cut prefill **39.37 s ->
1.92 s (20.5x)** and put it within 8% of the bf16 baseline. All correctness tests
still pass (`max_abs` 0.01389 -> 0.01252).

**This also invalidates section 22's conclusion.** The "17.6x aggregate slowdown"
was almost entirely prefill, not decode: G64 decode is only **1.6x** slower than
the bf16 baseline (78.6 vs 127.4 tok/s), and the earlier 2.7 tok/s figure was a
prefill-dominated wall-clock average. Nor was the strided page addressing at
fault: `bench_e38_packed_vs_strided.py` times the untouched contiguous adapter
against the strided port on identical data and gets ratios **0.87x-1.01x** across
batch 1/4, split 4/16/32, while the E38 attention call itself costs only
0.07-0.25 ms.

### 23.2 Completed A/B (4k and the reachable ceiling)

Both arms: `MAX_SEQS=4`, `SPEC=dflash2`, `DFLASH_TOKENS=7`,
`VLLM_SPEC_DECODE_ATTN=1`, graph mode, same prompt, greedy, warmup discarded.
Rows were run one process per concurrency by `run_ab_rows.sh` so a crash in one
row does not lose the others.

| row | baseline bf16 | INT8-G64 | G64/baseline |
| --- | --- | --- | --- |
| 4k C1 | 49.36 tok/s | 43.46 tok/s | 0.88x |
| 4k C2 | 51.70 tok/s | 43.02 tok/s | 0.83x |
| 4k C4 | **crash** | 41.51 tok/s | n/a |
| 32k C1 | **crash** | 2.52 tok/s | n/a |
| 32k C2 | **crash** | 2.00 tok/s | n/a |
| 126k / 250k | not serviceable | not serviceable | n/a |

Isolated components (same card, same prompt): prefill 1796 vs 1938 tok/s (0.93x),
decode 78.6 vs 127.4 tok/s (0.62x).

### 23.3 Quality A/B

`scripts/ab_quality.py` runs 12 fixed prompts greedily with `logprobs=1` per arm;
`ab_quality_compare.py` diffs them. Both arms produced **1495 tokens**:

```
byte-identical outputs  : 9/12
shared token prefix     : 1177/1495 (78.73%)
mean |dlogprob| on agreement: 0.00385 (n=1177, max=0.11758)
mean logprob baseline   : -0.1344
mean logprob INT8-G64   : -0.1279
```

The three divergences are all 192-token open-ended generations that flipped their
greedy argmax at tokens 10, 123 and 125 — the expected behaviour of a lossy int8
cache over long free-form text, not an addressing or scaling error. G64's own mean
logprob is marginally *higher*, i.e. no systematic degradation.

### 23.4 Remaining blocker A — the baseline config itself is unstable (pre-existing)

The bf16/FLASH_ATTN arm dies with `illegal memory access` at **C4 4k** and at
**32k**, reproducibly. This is not caused by this session's changes: after
restoring the pre-session `triton_attn.py` and `int8_g64.py` from the
`.orig-g64layout` backups (verified `prefill route=0 ws=0 tensorcore=0`) the
baseline still crashes identically, and the FLASH_ATTN backend never enters
`triton_attn.py` in the first place. It also reproduces with
`VLLM_SPEC_DECODE_ATTN=0`, so the spec-attention patch is not the trigger either.
It is worth its own investigation; it caps the A/B at C1/C2 for the baseline.

### 23.5 Remaining blocker B — the G64 KV cache costs 8.1x the bf16 pool

Measured pools at comparable settings: **baseline 529,060 tokens vs G64 65,362
tokens**. A per-layer dump (added temporarily in `kv_cache_utils.py`) gives the
mechanism exactly:

| | baseline bf16 | INT8-G64 |
| --- | --- | --- |
| `max_page_size` (from the 48 MambaSpec layers) | 1,835,008 | 1,777,664 |
| attention page | 131,072 (block 16) | 135,168 (block 64, native) |
| divisibility | `1,835,008 % 131,072 == 0` | `1,777,664 % 135,168 != 0` |
| promotion outcome | `page_size_padded=None`, **block 16 -> 448** | `page_size_padded=1,777,664`, **13.15x waste** |
| result | 126,615,552 bytes/page | 122,658,816 bytes/page, 28.6% of it padding |

`1,835,008 = 2^18 * 7` is divisible by `2^17`, so the bf16 arm takes the
zero-waste **block-scaling** path. `1,777,664 = 2^13 * 217` is **not** divisible by
`135,168 = 2^12 * 33`, so the G64 arm is forced onto the `page_size_padded` path.
The 5 sliding-window draft layers are padded even harder (33,792 -> 1,777,664,
52.6x).

This is why 126k/250k cannot be served: at `G64_MAX_LEN=262144` the engine refuses
with `114.23 GiB KV cache is needed, which is larger than the available KV cache
memory (37.85 GiB)`, i.e. ~467 KB per token for a format whose payload is 528 B
per token per layer. Fixing the block-scaling path is not a quick change: scaling
the G64 block from 64 to 448 would break the 64-token page contract that
`g64_views`, the writer and the E38 kernel all depend on. The real fix is to give
the linear-attention (Mamba) layers their own KV cache group with their own page
size instead of unifying every layer onto the Mamba page. Until then INT8-G64
cannot deliver its intended memory advantage, let alone a long-context A/B.

### 23.6 State and files

The isolated unit was left running the fixed G64 arm. Port 8000 was never started,
stopped or modified and had no listener throughout. The runtime holds the fixed
`triton_attn.py` and `int8_g64.py`; the pre-session originals are at
`*.orig-g64layout` and the fixed copies are also saved as
`triton_attn.py.g64-fixed-final` / `int8_g64.py.g64-fixed-final` in the test dir.
`kv_cache_utils.py` still carries the temporary `KVPAGE`/`KVPAGE-AFTER` dump
(harmless unless `VLLM_DUMP_KV_PAGES=1`) and should be reverted.

New this session: `ab_bench.py` (+`--only-concurrency`), `ab_split_bench.py`,
`ab_quality.py`, `ab_quality_compare.py`, `run_ab_rows.sh`,
`bench_e38_packed_vs_strided.py`, `test_e38_decode_sweep.py`,
`test_e38_orig_adapter_sweep.py`, `test_g64_batch34_prefill.py`,
`test_merged_bridge.py`, `test_g64_addr_probe.py`, `test_g64_both_geometries.py`,
`make_runtime_candidate.py`, `runtime-candidate/int8_g64.py`,
`deploy_int8_g64_layout_fix.py`. All untracked local experiment files per 19.5.

## 24. Records pushed (2026-09-17)

Sections 19-23 and the E41 write-up were committed and pushed:

```
804f272  docs: record E41 INT8-G64 integration and open allocation gate
         docs/int8-g64-vllm-integration.md                  (new, 137 lines)
         docs/verifier-optimization-handoff-2026-09-16.md   (+128, E41)
         handover.md                                        (+889, sections 19-23)
```

Pushed with `git push published HEAD:work/cmp170hx-mixed-fp8`
(`published` = `https://github.com/TnzGit/vLLM-CMP170HX-Qwen3.8.git`), advancing
that branch `c07cb00 -> 804f272`. PR #1
(`CMP170HX mixed-FP8 speculative verify reconstruction`, draft, base `main`) now
reports head `804f272`, and its body was extended with an "E41 slice" section plus
progress annotations on the original still-to-do items.

Scope of the commit is documentation only, matching the branch's established
`docs: ...` convention (the previous ten commits touched only docs). Deliberately
**not** committed:

- `patches/int8-g64-vllm.patch` — its header still describes the contiguous
  global-plane ABI (`[all K codes][all V codes][...]`) that 20.1/23.5 proved is
  not what the allocator produces. Committing it beside the corrected
  `docs/int8-g64-vllm-integration.md` would contradict the record; it needs
  rewriting for the page-local ABI first;
- `deploy/int8-g64-8002.service` and `deploy/triton-int8-control-8002.service` —
  they carry a literal `VLLM_API_KEY=pixelml-bench` whereas the tracked
  `deploy/*.service.example` files blank that variable, and they embed lab-host
  paths. Track them as `.example` with an empty key if they are wanted;
- the bridge module `vllm_int8_g64_module.py`, the layout-audit harnesses and
  `scripts/ab_*.py` — experiment scratch, still unreviewed per 19.5.

The documentation references no tracked path that does not exist, and the
integration doc now states that the launcher is an experiment-side unit rather
than promising `deploy/int8-g64-8002.service`.

## 25. E42 kill gate: NOT met — INT8-G64 loses to the production control at every reachable context

The review's short-term sequence was: measure INT8-G64 against the **production**
control (`TRITON_ATTN` + `int8_per_token_head`, i.e. the path G64 would replace),
not against the crashing BF16/FLASH_ATTN arm, and require a clear crossover —
decode/step at least +10% at 32K or 65K — before spending days on the E42
common-page allocator work. That measurement is now complete and the gate fails.

### 25.1 The control arm is the production path, not BF16

`triton-int8-control-8002.service` runs `CTX=long`, which resolves to
`--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head` with
`VLLM_SPEC_DECODE_ATTN=1` — the same split-KV verify kernel the production long
-context recipe uses, on the same W4A16 target and DFlash2 drafter. Its pool is
**1,110,675 tokens** at `max_len=250000`, against G64's 63,799-74,239, which is
the allocator defect of 23.5 restated from the other side.

### 25.2 Matched result (fresh engine per context, identical settings)

Both arms: same model, drafter, `MAX_SEQS=4`, `DFLASH_TOKENS=7`,
`VLLM_SPEC_DECODE_ATTN=1`, pinned 1350 MHz, graph mode, greedy, identical
prompt/seed, 191 decode steps. Decode is reported as **ms/step**, which is
independent of DFlash acceptance and therefore comparable across arms whose
acceptance differs.

| ctx | control decode | G64 decode | G64 vs control | control prefill | G64 prefill |
| --- | --- | --- | --- | --- | --- |
| 4,096 | **6.120 ms** | 7.339 ms | **-19.9%** | 1,709.6 tok/s | 1,626.5 tok/s |
| 16,384 | **8.161 ms** | 9.450 ms | **-15.8%** | 1,219.5 | 1,180.8 |
| 32,768 | **10.200 ms** | 11.292 ms | **-10.7%** | 876.1 | 859.7 |
| 65,000 | **12.673 ms** | 13.932 ms | **-9.9%** | 560.3 | **564.0 (+0.7%)** |

32K was reproduced twice (control 10.200/10.065, G64 11.292/11.241). The gap
**narrows with context but never crosses**: -19.9% -> -15.8% -> -10.7% -> -9.9%.
Prefill reaches parity at 65K but never leads. There is also a consistent
per-request fixed cost in the G64 arm (`prefill_128_s` 0.123-0.125 s against the
control's 0.072-0.094 s in every row), which is worth understanding but cannot
account for a per-step decode gap.

**Verdict: the gate is not met at either 32K or 65K, so E42 (common-page
round-up to 1,892,352 B, 896-token G64 pages, quant_group / physical_block
decoupling) is not justified and E38/E41 are frozen as a research result.** The
review's reasoning was that G64's only possible win is a compute-path bet —
native s8 MMA instead of FP8 -> BF16 decode -> BF16 MMA — because its KV bytes
(2,112 B/token/layer) are actually 3.1% *larger* than the production FP8/int8
control's. That bet does not pay off at any context this allocator can serve.

### 25.3 Two shared-engine bugs found while measuring (neither is G64-specific)

Both arms crash identically, so both are in shared engine code and they affect
any future measurement on this box:

1. **Sequential context changes trip an illegal access.** A single 32K request on
   a fresh engine completes normally (10.143 ms/step), but a 32K request that
   follows other context lengths in the same process dies with
   `illegal memory access` — in the control arm as well as G64, and with
   `VLLM_SPEC_DECODE_ATTN=0` too. Measurement protocol therefore had to become
   **one context length per engine lifetime** (`run_ab_rows.sh` /
   `fresh_sweep.sh`).
2. **Repeated identical long prefixes trip an illegal access.** One 16K request
   is fine; three repetitions of the same 16K prompt in one process die the same
   way. This is what made the first `--reps 3` sweeps non-monotonic and briefly
   produced a spurious "+14% for G64" reading.

Both deserve their own issue. They are the reason section 22's and the earlier
part of this session's numbers disagreed with 25.2, and 25.2 is the trustworthy
set because it holds arm identity and context constant per engine lifetime.

### 25.4 What this leaves

The review's long-term performance direction does not depend on G64 and is now
the active path toward the project's engineering ceiling (4K ~161 vs 170-200
reachable, 126K ~98 vs 131, 250K ~70 vs 100 tok/s): whole-step profiling first,
then fusion on the *existing* Marlin/G128 packing rather than repacking weights
for NInfer's Q4G64 layout — E40 showed the fusion itself is worth it (T1 +45%,
T8 +31%, T16 +58%) but only at T>=8 under the current format. At 126K the split
is verifier attention ~42% / Marlin target GEMM ~43%, so once attention is not
the problem the GEMM epilogue is the next ceiling.

## 26. Production-path ceiling work: baseline re-measured, 120K+ blocked

### 26.1 Where the review's ceiling numbers come from

Worth recording so the target is not mistaken for a spec. The review's table
(`4K≈161 / 126K≈98 / 250K≈70 tok/s` current, `170-200 / 131.2 / 100` as the
"engineering effective ceiling") does not appear anywhere in this repository as a
measured triple. What the repo does contain is `single-user/README.md`, which
measures DFlash2 at **`CTX=fast` (bf16, 64k)** with `SPEC=dflash2` and reports
C1 **121.8 / 131.2 tok/s** (default / greedy), C2 195.5 / 214.6, C4 278.9 / 285.7,
C8 389.9 / 405.5 — i.e. the 131.2 and 214.6 in the review's "ceiling" column are
this repo's *measured current* C1/C2 greedy numbers, and 161 is another project's
end-to-end figure quoted in `README.md` for comparison. Treat the ceiling column
as a modelling estimate, not a qualification gate.

### 26.2 Production-path baseline actually measured here

`CTX=long` (`TRITON_ATTN` + `int8_per_token_head`, `SPEC_ATTN=1`,
`SPEC=dflash2`, k=7, `MAX_SEQS=4`, `max_len=262144`, pool **1,137,362 tokens**),
fresh engine per context, 191 decode steps, greedy:

| ctx | decode ms/step | decode tok/s | prefill tok/s |
| --- | --- | --- | --- |
| 4,096 | 6.098-6.120 | 163-164 | 1,709-1,719 |
| 16,384 | 8.161-8.247 | 121-123 | 1,219-1,216 |
| 32,768 | 10.065-10.200 | 98-99 | 876 |
| 65,000 | 12.673 | 79 | 560 |

Whole-request throughput at 4K (prefill included) reads 142-164 tok/s depending on
prompt, which brackets the review's 161 "current" figure.

### 26.3 New blocker: long-context decode faults above ~64K (partially characterised)

A single long request on the freshly started control engine dies with
`illegal memory access` from roughly 66K upward; 126K and 250K fail the same way.
This is the production control arm, not INT8-G64, so it is unrelated to the
frozen G64 work. The pool is not the limit (1,137,362 tokens at
`max_len=150000`). Consequence: the 126K and 250K rows of the review's ceiling
table **cannot currently be measured on this isolated configuration at all**, and
neither could a G64 comparison at those lengths.

What is established, by controlled single-variable tests:

| probe | result |
| --- | --- |
| prefill-only at 70K and 85K | **OK** (0 faults, both in one process) |
| decode at 63K | OK |
| decode at 66K / 85K / 100K / 120K | fault (4 records each) |
| decode at 70K, `SPEC=none` | OK |
| decode at 70K, `SPEC=dflash2`, `max_len=150000` | OK |
| decode at 120K, same config | fault |
| `VLLM_SPEC_DECODE_ATTN_QMAX=64` | no change |
| `SPEC_ATTN=0` (stock Triton attention) | no change |
| `CUDA_LAUNCH_BLOCKING=1` present or absent | no change |
| GDN `causal_conv1d` bounds patch | already applied in this runtime |

So it is **decode-specific** (prefill at the same lengths is fine), it needs
**speculative decoding** (70K passes with `SPEC=none`), and it is **not** in the
split-KV verify kernel, not the KV dtype, not `QMAX`, and not the known GDN
accept-bound bug. The 70K-with-dflash2 pass against a 66K fault means the trigger
is not a clean context threshold, which is where the characterisation stops.

The one probe that would settle it has not been run: **`compute-sanitizer` is not
installed on this host** (`ncu` is), and the failure only reproduces at long
context, so the practical route is `ncu --launch-count` over a failing 120K
decode rather than more setting bisection. Until then this stays an open,
partially-characterised blocker, and the honest status of the 126K/250K rows is
"not measurable", not "measured and slow".

This joins the two measurement faults already recorded in 25.3 (context-length
change within one process; repeated identical long prefix) as the three
engine-level obstacles that must be fixed before any 126K/250K qualification,
G64 or otherwise, is possible.

### 26.4 Consequence for the plan

The review's short-term branch is closed: the E42 kill gate was measured and
failed (25.2), so INT8-G64 is frozen and no allocator work is justified. Its
long-term branch — whole-step profiling, then fusion on the existing Marlin/G128
packing rather than repacking weights for NInfer's Q4G64 layout — remains the
right direction, and the repo's existing 126K profile is the reference:

| component | time / step | share |
| --- | ---: | ---: |
| speculative FP8 verifier partials | 12.847 ms | 42.2% |
| target Marlin GEMMs | 13.219 ms | 43.4% |
| GatedDeltaNet kernels | 1.120 ms | 3.7% |
| combine reduction | 0.123 ms | 0.4% |
| all other kernels | ~3.12 ms | 10.3% |

with the dominant GEMMs being `M=16,N=34816,K=5120` (64 calls, 6.111 ms) and
`M=16,N=5120,K=17408` (64 calls, 3.139 ms). The Marlin tile candidates were
already swept and rejected (`(128,64,128)` +18% at 4K, `(64,128,128)` +12.3%), so
the remaining GEMM-side headroom is an **epilogue fusion** — SwiGLU into the
gate+up GEMM — not a retile. `apply_gptq_marlin_linear` already takes a `bias`
argument, which is the natural place to attach such an epilogue.

The immediate prerequisite is 26.3: without a working 120K+ path there is no way
to measure whether any of this moves the long-context number the ceiling table is
about.

## 27. Milestone artifacts committed (2026-09-17)

Sections 19-26 were documentation only; the bridge, the tests and the harnesses
lived outside the repository. They are now tracked under `bench/int8-g64/`
(31 files) with `bench/int8-g64/README.md` as the entry point, committed as
`dac97b5 bench: ship the INT8-G64 integration, tests and A/B harnesses`.

What moved in, and what was changed on the way:

- **bridge**: `int8_g64.py` is the corrected module (page-local views, strided
  E38 loader, tensor-core prefill route) that was actually deployed and
  measured. `make_runtime_candidate.py` was **not** shipped: it rebuilt that
  module from the stale `vllm_int8_g64_module.py`, so it was circular once the
  corrected file became the source of truth. The stale root-level
  `vllm_int8_g64_module.py` stays untracked for the same reason;
- **patches**: `int8-g64-vllm.patch` had its header rewritten to describe the
  page-local ABI instead of the contiguous global planes that 20.1 disproved.
  `int8-g64-triton.patch` was **regenerated from the real diff** between the
  pristine `triton_attn.py` and the deployed one, because the tracked copy was an
  early revision missing the prefill route, the graph-safety fix and the E38
  gate fix. It now applies cleanly and reproduces the deployed file
  byte-for-byte (verified with `patch -p1` against the pristine source);
- **E38 adapters**: both the pristine `v7_verifier_e38_int8.orig.cu` and the
  batch-fixed `v7_verifier_e38_int8.cu` are shipped, so the `constexpr int
  Batch = 2` defect behind 21.1 is visible as a diff inside the repository;
- **tests**: the ten component tests that produced the recorded evidence,
  including `test_e38_decode_sweep.py` (the RED->GREEN for the batch defect) and
  `test_e38_orig_adapter_sweep.py` (the control that proved it pre-existing);
- **harnesses**: `ab_bench.py`, `ab_split_bench.py`, `ab_quality.py`,
  `ab_quality_compare.py`, `step_profile.py`, `bounded_correctness_probe.py`;
- **deploy**: `apply_int8_g64_remote.py`, `deploy_int8_g64_layout_fix.py`, and
  the two units as `.service.example` with `VLLM_API_KEY` blanked, matching the
  tracked `deploy/*.service.example` convention.

Path hygiene applied before committing: lab-absolute paths were removed from the
scripts (they now resolve relative to the file, and take the NInfer tree from
`VLLM_INT8_G64_NINFER_ROOT` with an explicit error if unset); the `pixelml-bench`
API key literal was replaced by the repo's existing convention
(`VLLM_API_KEY`, else `api_key.txt`, else empty); and a secret scan over the
whole set is clean. `NInfer` kernel headers remain external by design and are
supplied through that env var.

The README states the frozen status up front, carries the kill-gate table, and
documents the measurement protocol the two shared-engine faults force (one
context per fresh engine; distinct text per request) so the earlier spurious
in-process sweeps are not repeated.

## 28. Correction: the "shared-engine bugs" were a measurement-protocol error, not an engine bug

Sections 25.3 and 26.3 recorded two engine faults and concluded that measurement
had to use one context length per engine lifetime. Bisection this session shows
that conclusion was **wrong in its attribution**, and the cost was real: the
over-conservative protocol forced a full engine restart (40-175 s) per row.

### 28.1 What actually reproduces, and what does not

On a **freshly started** engine, none of the following fault, on either arm:

| probe | result |
| --- | --- |
| sequential context changes 4K -> 8K -> 16K -> 32K in one process | OK, 0 faults |
| 3 and 8 repeated full-context prefills at 16K, distinct text | OK, 0 faults |
| 3 byte-identical prompts repeated at 16K | OK, 0 faults |
| a full `--reps 3` context sweep across five contexts | OK (see 28.3) |

The fault is therefore **not** triggered by changing context length, by repeating
a long prefix, or by repeating prefills — the three hypotheses in 25.3/26.3.

### 28.2 The real rule: a crashed engine stays poisoned

What reproduces is a fault **only on an engine that has already faulted**. After
one `illegal memory access`, the same engine returns stale failures for
everything afterwards — a probe that reports `faults=0` on the next request is
seeing the *previous* crash, not a new one. That is why the earlier session kept
"reproducing" the fault: it was measuring on an already-corrupted engine, and
each new crash made the next row invalid. It is also why the first clean
reproduction attempt at 3 prefills failed while 8 prefills passed — the 8-prefill
run had a fresh engine and the 3-prefill run did not.

The protocol that is actually justified:

- **restart the engine after any fault**, before trusting another measurement;
- a single engine may otherwise run a whole multi-context, multi-repetition
  sweep, so the per-row cold start is unnecessary;
- `CUDA_LAUNCH_BLOCKING=1` must not be set for timing or sweep work: it breaks
  CUDA-graph replay and will manufacture exactly this fault. It was left in a
  drop-in during the earlier bisection and caused a full false-negative sweep
  (see 28.4).

`one_engine_sweep.sh` implements this: one engine, many contexts, restart only
when a row faults.

### 28.3 Confirmation

One engine, G64 arm, `--reps 3` per context, distinct text per request:

```
ctx=4096   prefill 1762.6 tok/s   decode 8.090 ms/step
ctx=16384  prefill 1429.3 tok/s   decode 8.087 ms/step
ctx=32768  prefill 1151.4 tok/s   decode 13.569 ms/step
ctx=48000  prefill  978.7 tok/s   decode 8.977 ms/step
```

All four completed on one engine with zero faults, where the old protocol would
have paid four cold starts.

### 28.4 A configuration-pollution mistake worth recording

A sweep labelled "G64" actually ran the **control** configuration: a leftover
`repro262.conf` drop-in set `CTX=long`, which overrides the unit's own `CTX=g64`,
and it also carried `CUDA_LAUNCH_BLOCKING=1`. The run therefore measured
`TRITON_ATTN`/`int8_per_token_head` under launch blocking, faulted on every row
after the first, and would have been read as "G64 fails at 16K+". The lesson is
mechanical: **a drop-in silently overrides the main unit's `Environment=`, so
verify the effective environment before every sweep**, not the file you think you
edited:

```
systemctl --user show <unit> -p Environment | tr ' ' '\n' | grep -E 'CTX|G64|SPEC'
```

### 28.5 Open: within-context variance is not yet explained

The same sweep shows large spread across the three repetitions at a fixed
context — at 32K, `[8.528, 14.416, 13.569]` ms/step. That is 1.7x, far above the
±0.3% seen in the earlier fresh-engine runs, and it is why the 32K row's median
(13.569) disagrees with the fresh-engine measurement (10.200). Until that spread
is explained — warmup ordering, block reuse, or DFlash acceptance drift — a
multi-context sweep on one engine is **not** yet a substitute for the per-context
protocol for *timing*, even though it is now clearly sufficient for *liveness*.
This is the next thing to settle before resuming the A/B.

## 29. Benchmark-infrastructure corrections (review feedback, 2026-09-18)

Three corrections to what this session recorded. The first two are outright
errors in the harnesses and are fixed in `bench/int8-g64/`.

### 29.1 "ms/step" was actually ms per output token

`ab_split_bench.py` computed

```python
dec_n = full["s"] - pre["s"]; dec_n = completion_tokens - 1
decode_ms = dec_t / dec_n
```

`completion_tokens - 1` is the number of **generated output tokens**, not the
number of DFlash verify passes. DFlash2 accepts ~3.3-3.4 tokens per target pass
(the repo's own table below), so the reported figure is
**milliseconds per output token** and the claim in 25.2 that "ms/step is
independent of DFlash acceptance" is **wrong** — it is exactly the metric
acceptance moves.

This does **not** change the freeze decision: the A/B gap was measured on the
same end-to-end quantity for both arms, and 10-20% slower user-visible decode is
sufficient reason not to rewrite the allocator. But it does invalidate the
stronger claim that "the native s8 compute path did not pay off": that cannot be
concluded from an end-to-end number, because a faster target pass with worse
acceptance would produce the same reading. Whether G64's target pass is itself
faster is **unmeasured**.

`bench/int8-g64/spec_bench.py` replaces it and reports the three quantities that
move independently:

```
ms per output token        (what a user feels)
ms per target verify pass  (kernel efficiency)
accepted tokens per pass   (proposal efficiency)
```

with `target_passes_per_100_out` and the raw `drafted/accepted` counters read
from the server's own `SpecDecoding metrics` log lines.

### 29.2 step_profile.py could not profile anything

It wrapped the HTTP request in a client-side `torch.profiler`. A client process
only sees CUDA work it submits itself; the vLLM engine runs the kernels in a
separate worker, so the trace was empty of the kernels the script claimed to
attribute. The reference breakdown in `docs/cmp170hx-mixed-fp8-engineering.md`
is unaffected (it was produced properly), but shipping this script implied it
could be reproduced this way, which is false.

Rewritten: it now either `--drive-only` (send a warmed long generation so a
profiler started elsewhere has work to measure) or `--summarize <trace.json>`
(bucket kernels from a trace captured in the engine process). The docstring
states both supported capture routes — vLLM's own
`--profiler-config.profiler=torch --profiler-config.torch_profiler_dir`, or
`nsys profile` around the server — and the `--help` path errors out rather than
pretending to profile. The old file is kept as `step_profile_TODO_BROKEN.py`.

### 29.3 "production FP8 control" was conflating two different stacks

25.2 and 26.2 measured `CTX=long` (`TRITON_ATTN` + `int8_per_token_head`) and
called it the production control. That is a real, current recipe and a fair
control for the G64 arm — but it is **not** the same as this PR's
`CTX=cmp-mixed-fp8` route (FlashInfer FP8 target KV + static FP8 q8 split-KV
verifier), and that latter stack has long-context records this session's wording
wrongly cast doubt on:

| input | segments | decode tok/s | accepted tok/step | verifier ms/pass |
| --- | --- | --- | --- | --- |
| 126K | 16 | 54.2 | 3.42 | 62.8 |
| 126K | 32 | 68.2 | 3.41 | 50.0 |
| 250K | 16 | 32.5 | 3.38 | 103.3 |
| 250K | 32 | 42.2 | 3.28 | 77.5 |

(`docs/cmp170hx-mixed-fp8-engineering.md`, qualified with zero preemptions.)

So the accurate statement is: **the `CTX=long int8_per_token_head + DFlash2`
qualification path is what faults above ~64K, not the card's ability to serve
long context.** 26.3's "not measurable" applies to that path only. The
`cmp170hx-mixed-fp8-full-256k` route remains a working long-context research
platform and should be used so performance work does not stall behind this bug.

### 29.4 "it needs speculative decoding" was overclaimed

The evidence matrix was `70K SPEC=none` PASS, `70K SPEC=dflash2` PASS,
`66K SPEC=dflash2` FAIL — which cannot establish speculation as a *sufficient*
trigger. Corrected wording: the fault has so far been observed **only** in
speculative configurations, and a non-speculative 70K control passes, but
DFlash2 at 70K also passes, so speculation is not proven sufficient.

### 29.5 What the review changes about the plan

Accepted, and it reorders the next steps:

1. **Fix the infrastructure first** (29.1, 29.2) — done in this commit;
2. **Long-context IMA**: exact-token residue sweep (the repo already has the
   right primitive, `bench/context_ab.py:exact_prompt_ids`, because a previous
   bug in this repo broke at one prompt length in 128); then spec-source
   bisection `DFlash2 vs MTP vs none`; then `ASYNC_SCHED=1/0`; then
   `FULL/PIECEWISE/eager`; then a pre-fault state dump
   (`seq_len`/`positions`/`slot_mapping`/`block_table` tail/`num_spec_tokens`/
   accepted/scheduled draft IDs/workspace shape/graph desc). Compute Sanitizer
   memcheck is the right tool for a CUDA OOB and should be installed rather than
   substituted with more configuration bisection;
3. **Do not let the IMA block performance work** — use the known-good
   mixed-FP8 256K service as the long-context research platform in parallel;
4. First performance experiment after that is **gate_up-only W4A8-INT8 Marlin**,
   not Marlin+SwiGLU fusion: the repo already carries
   `marlin-int8-negative-scales.patch` and `marlin-int8-layer-select.patch`, and
   `VLLM_MARLIN_INPUT_DTYPE=int8` is the upstream-supported route on
   non-SM89/SM12x. At the 126K split (Marlin 13.219 ms of a 30.426 ms step), a
   20% Marlin win is ~8.7% whole-step;
5. Then a DFlash `k=3/5/7` sweep on `accepted/pass x pass time`, since
   speculation changes only proposal efficiency and a 3.4 -> 3.8 acceptance gain
   is worth ~12% output throughput with no kernel change;
6. **Marlin+SwiGLU fusion is demoted**, and 26.4's "the `bias` argument is the
   natural hook" was half wrong: bias is `C[n] += bias[n]`, whereas SwiGLU needs
   the gate and up halves (17408 apart in a 34816-wide output) in the same
   program. That is paired gate/up output scheduling, i.e. a kernel redesign, and
   E40's own data shows the win comes from the changed dataflow, not from
   removing a few-microsecond activation kernel. Measure the standalone
   `silu_and_mul` share first; if it is microseconds, epilogue fusion has almost
   no headroom and only the paired-schedule redesign is worth anything.
7. Verifier micro-optimisation (E43) stays parked unless a fresh long-context
   profile puts verifier attention back above ~50% of the step.

## 30. The >64K fault is discrete at 2^16, not a threshold (2026-09-18)

Section 26.3 called this a "long-context" fault with a threshold somewhere above
64K and stopped there. An **exact-token** sweep (the reviewer's first
recommendation, and the right one: this repo has already had a bug that broke at
one prompt length in 128) shows the fault is not a threshold at all.

`bench/int8-g64/exact_residue_sweep.py` builds the prompt as exactly N token IDs
and verifies it via the response's `prompt_tokens`, so the length is controlled
to the token. Sweeping one token at a time across 65530..65538 on the
`CTX=long` control arm:

| prompt tokens | result |
| --- | --- |
| 65530 | OK |
| **65531** | **FAULT** |
| 65532 | OK |
| 65533 | OK |
| 65534 | OK |
| 65535 | OK |
| **65536** | **FAULT** |
| 65537 | OK |
| 65538 | OK |

So the fault is **discrete**: single lengths inside a 9-token window fault while
their neighbours do not. `65536 = 2^16` is an obvious candidate; `65531` shows at
least one further period on top, and 65531/65536 are 5 apart, which does not match
16/32/64/128 directly, so the second period is not yet identified.

This retracts the framing of 26.3 and of the reviewer's summary: the correct
question is not "above what context length does it break" but "which exact
sequence lengths trip it, and why 2^16". That also means the earlier
`63K OK / 66K FAULT` pair was consistent with a discrete pattern rather than a
threshold, and that a range-based protocol (or a range-based claim of "clean")
was never sound.

### 30.1 Measurement variance is structured, not noise

Six consecutive identical-shape runs at 32K on one engine, three repetitions
each (`ms per output token`, the corrected metric of 29.1):

```
rep1 [9.099, 14.489, 13.546]   rep4 [8.512, 14.268, 13.639]
rep2 [8.545, 14.562, 13.689]   rep5 [8.564, 14.373, 13.716]
rep3 [8.409, 14.580, 13.546]   rep6 [8.543, 14.406, 13.653]
```

The pattern within each repetition is identical every time — first fast (~8.5),
then slow (~14.4), then ~13.6 — and the median is stable to +-0.9%
(13.546-13.716). Two consequences:

1. the median is a usable estimator, but a **single** measurement is not, and the
   per-context fresh-engine numbers recorded earlier (e.g. control 32K = 10.200)
   are the *optimistic first-request* value, not the steady-state one. Both arms
   were measured that way so the comparison stands, but the absolute figures are
   biased low and should not be quoted as steady-state;
2. the ordering effect is deterministic, so it is a warmup/ordering artefact
   (block reuse or acceptance settling), not thermal or scheduling noise. A
   protocol that discards the first repetition and takes the median of the rest
   is the right shape.

This is the answer to 28.5, and it means a single-engine multi-context sweep is
acceptable for **timing** provided the first repetition at each context is
discarded.

## 31. Session state: sanitizer available, box left idle (2026-09-18)

### 31.1 compute-sanitizer is now installed

The blocker recorded in 26.3 ("compute-sanitizer is not installed, use ncu
instead") is resolved. The reviewer was right that memcheck is the correct tool
for a CUDA OOB rather than more configuration bisection.

```
pip install --no-deps nvidia-cuda-sanitizer-api
# -> /home/base-node/vllm-cmp170hx-nightly/lib/python3.12/site-packages/nvidia/cu13/bin/compute-sanitizer
#    NVIDIA (R) Compute Sanitizer 2026.3.0.0
```

Note it landed in the **nightly** venv, not the `runtime-v0271` venv that the
serving units use, because pip resolved `Location` there. Use the absolute path
above; do not assume it is on `PATH`.

The engine's launcher `exec`s `venv/bin/vllm serve ...`, so sanitizer can wrap it
directly. The exact argv for the control arm was captured by running
`start_qwen.sh` with the unit's own environment and replacing the `exec` with a
print (see 31.3).

### 31.2 What the exact-token sweep established

Recorded in section 30: the fault is **discrete**, hitting `65531` and `65536`
inside a 9-token window whose other lengths pass. Two further facts from this
session:

- it is a **prompt-length** property, not a sequence-growth one: prompt 65530 with
  `max_tokens=64` (so the sequence crosses 65536 during decode) **passes**;
- `65536 = 2^16` is therefore the leading candidate, and 65536/16 = 4096 blocks at
  the control arm's `block_size=16`, which points at block-table or slot-mapping
  indexing rather than at any quantization kernel. `65531` implies a second,
  still-unidentified period.

A full 128-length window scan around 65536 was started and abandoned: at ~124 s
per length (each request prefills 64K tokens) it is ~4.4 hours, which is not a
good use of the card. A targeted residue scan at the specific candidate moduli
(16/32/64/128/448/896) is the cheaper next step, and the sanitizer run below
supersedes it anyway.

### 31.3 Sanitizer attempt hit a mundane obstacle, not a sanitizer problem

The first memcheck run failed with

```
ValueError: Free memory on device cuda:0 (17.29/63.39 GiB) on startup is less
than desired GPU memory utilization (0.9, 57.05 GiB)
```

That was **not** a sanitizer incompatibility: a leftover `VLLM::EngineCore` was
holding ~58 GiB. Two separate mistakes produced it and both are worth avoiding:

1. `pkill -9` on a unit's `EngineCore` while the unit is still wanted leaves
   systemd's cgroup bookkeeping inconsistent, and the unit then restarts and
   re-grabs the GPU. Stop units with `systemctl --user stop`, never with `pkill`;
2. the sweep scripts restart engines themselves, so an "idle" box can be
   mid-restart. Check `nvidia-smi --query-compute-apps` **and**
   `systemctl --user show <unit> -p ActiveState` before launching anything that
   needs a specific amount of free VRAM.

### 31.4 Machine state at the end of this session

All vLLM/SGLang units are `inactive`, no `failed` units remain, no `vllm` or
`EngineCore` processes are running, port 8000 and 8002 have no listener, and the
GPU is at 14 MiB / 0% with 64898 MiB free. `pixelml-vllm-8002` was reset earlier
along with the other stale failures. Nothing was left running.

`gpu-dashboard` (pid 14878) and `dualwan-guardian` (pid 470595) are untouched and
still serving their own ports; neither is an LLM server and neither was started,
stopped or modified by this work.

### 31.5 Next actions, in the order the review set

1. run the control engine **under compute-sanitizer memcheck** at prompt length
   65531 (and 65536) and read the OOB report -- this is now unblocked, and the
   argument-capture recipe is in 31.1/31.3;
2. in parallel, stand up `cmp170hx-mixed-fp8-full-256k` as the long-context
   research platform so the Marlin work does not wait on this bug;
3. first performance experiment: **gate_up-only W4A8-INT8 Marlin**, using the
   repo's existing `marlin-int8-negative-scales.patch` and
   `marlin-int8-layer-select.patch` with `VLLM_MARLIN_INPUT_DTYPE=int8`;
4. then the DFlash `k=3/5/7` sweep on `accepted/pass x pass time` using
   `bench/int8-g64/spec_bench.py`;
5. only then consider a paired gate/up Marlin+SwiGLU kernel, and first measure
   the standalone `silu_and_mul` share to bound the payoff.

## 32. Review corrections applied, and the residue-123 hypothesis is refuted (2026-09-18)

### 32.1 Committed artifacts that were missing (review item 4 — correct)

`exact_residue_sweep.py` had never been `git add`ed: section 30 documented the
65531/65536 result while the harness that produced it existed only on the lab
host. Worse, the API-key fallback fix from `56c83df` was also still in the working
tree, so the *pushed* harnesses would have 401'd against the isolated units.

Both are committed now. This was the third time in this session that a result was
recorded before its harness was tracked, so the rule is now explicit: **a result
is not recorded until its harness is committed**, and `bench/int8-g64/README.md`
says so.

### 32.2 Three code defects the review found, all confirmed and fixed

1. **`spec_bench.py --exact-tokens` did not guarantee the length.** It assumed an
   8-token tail, and its rotation `rot[-4:] = ids[-(4+idx%4):] + ids[-4:-(4+idx%4)]`
   assigns a right-hand side of 5-7 elements into a 4-element slice, which
   **grows the list** (verified: 20 -> 21 tokens at idx=1, 20 -> 23 at idx=3). The
   mode was therefore unusable for exactly the residue work it was built for.
   Replaced with a full-length reassignment plus
   `assert usage.prompt_tokens == requested` on every call.
2. **`SpecDecoding metrics` windows were not aligned to the measured reps.** The
   logger emits periodic windowed-and-reset counters, and the old code took a
   median over every window after a start marker. Now each measured request is
   bracketed (log line count before/after) and only windows inside the bracket are
   read.
3. **`estimated_ms_per_target_pass`, not `ms_per_target_pass`.** vLLM's
   `Mean acceptance length` is `1 + accepted_draft_tokens / num_spec_steps`, so
   the product with ms/output-token is a derived estimate from two independently
   windowed measurements, not a request-local counter. Renamed until the engine
   exposes per-request `num_spec_steps` / `num_accepted_draft_tokens`.

Also fixed: `step_profile.py`'s summariser had been adding `cuda_runtime` events
(CPU-side `cudaLaunchKernel` / `cudaMemcpyAsync`) into "total CUDA kernel time".
Those are now a separate host-overhead bucket, so GPU kernel time is not
double-counted.

And `exact_residue_sweep.py` gained a hard `assert len(body) == length`, a check
that the server's `prompt_tokens` equals the requested length (so an "OK" cannot
mean "shorter prompt"), and a `--k-residues` mode that probes the lengths implied
by the historical `bad residue = 117 + k` relation.

### 32.3 The residue-123 hypothesis is refuted by measurement

The review suggested that `65531 mod 128 = 123` matches the historical
`117 + k = 124` bad residue for k=7, and that the fault might therefore be the
known spec/prefix residue family. That is a good hypothesis and it is now tested
and **fails**: at every length with residue 123 that is small enough to run, the
request passes.

| length | mod 128 | result |
| --- | --- | --- |
| 123 | 123 | OK |
| 251 | 123 | OK |
| 379 | 123 | OK |
| 507 | 123 | OK |
| 1023 | 127 | OK |
| 2047 | 127 | OK |
| 4095 | 127 | OK |
| 8191 | 127 | OK |
| 16383 | 127 | OK |

So residue 123 alone is not sufficient: the fault needs a **large** length as well
as the residue. That is consistent with the review's own warning that 65531 and
65536 may be two different failure classes, and it means the `117 + k` series is
not the whole story here. The `--k-residues` probe is kept for the case where the
sanitizer is inconclusive.

### 32.4 Review items accepted without change

- the fault should **not** be attributed to "4096 blocks" yet: `65531..65535` all
  need 4096 blocks and only 65531 faults, so the block count alone cannot explain
  it — recorded as strongly suggestive only, not as a conclusion;
- sanitizer should be run at **65531 and 65536 separately**, and with
  `--target-processes all` written explicitly (vLLM forks an EngineCore worker);
- the `pkill` gotcha's causal story was wrong and has been corrected: with
  `Restart=no` and `disabled`, systemd does not restart a unit because a child was
  SIGKILLed. The likely cause was the parent `vllm serve` still being alive and
  respawning EngineCore, or a sweep helper calling `systemctl start`. The
  operational rule (use `systemctl --user stop`) stands; the explanation is now
  marked as unverified and the diagnostic commands are recorded instead of a
  story;
- the W4A8 plan should be validated on the **exact hot shapes**
  (`M=16,N=34816,K=5120` gate_up and `M=16,N=5120,K=17408` down) as a locked-1350
  interleaved single-kernel A/B *before* any whole-model run, with a gate of
  >=10% on the dominant GEMM and >=8% after activation-quant overhead;
- `cmp170hx-mixed-fp8-full-256k` runs a **different checkpoint**
  (`Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4`) from the recent control
  (`Qwen3.8-27B-W4A16-AutoRound-fast`), so it is a long-context
  research/stability/profiling platform but **not** a matched whole-model A/B
  baseline without first confirming the Marlin packing/scales are equivalent.

### 32.5 The `[8.5, 14.4, 13.6]` pattern is not yet explained (review item 5)

Section 30.1 concluded it was ordering/warmup and that the first repetition
should be discarded. That conclusion was premature, and the review's alternative
is better: each rep uses a **different salt** (case200/201/202) and the six reruns
repeat the same three salts, so a stable per-salt DFlash acceptance difference
would produce exactly the same stable pattern.

`spec_bench.py --permute-salts` now runs the same three salts in three orders
(200,201,202 / 202,200,201 / 201,202,200) and reports acceptance alongside
timing: if the timing follows the salt it is a content/proposal effect, and if it
follows the position it is an ordering effect. Until that is run, **"discard the
first rep" is not a benchmark contract** and 30.1's conclusion is downgraded to a
hypothesis.

### 32.6 Correction: the fault is NOT deterministic in length — 65536 passed on re-test

Section 30's central claim was that the fault is **discrete** at specific lengths
(65531 and 65536 fault while their neighbours pass). The boundary re-test
refutes that as stated:

| length | blocks | this run | section 30 run |
| --- | --- | --- | --- |
| 65519 | 4095 | OK | -- |
| 65520 | 4095 | OK | -- |
| **65521** | **4096** | **FAULT** | -- |
| 65535 | 4096 | OK | OK |
| **65536** | 4096 | **OK** | **FAULT** |
| 65537 | 4097 | OK | OK |
| 65551 | 4097 | OK | -- |
| **65552** | 4097 | **FAULT** | -- |

**65536 faulted in the section 30 run and passed here.** A length cannot be both
a deterministic trigger and not, so the "discrete at 2^16" model is wrong. What
the data actually supports is a **probabilistic** fault in the ~64K+ region whose
per-length outcome varies between runs, which also explains why the earlier
`63K OK / 66K FAULT` pair and the "threshold somewhere above 64K" framing kept
looking consistent with each new observation: single measurements at single
lengths were never enough to distinguish the models.

Consequences, and they matter for how this is reported:

- 32.3's refutation of residue-123 still stands as a refutation of the
  *hypothesis as stated* (residue alone is not sufficient), but it is weaker
  evidence than it looked, because a passing length does not prove absence when
  the fault is probabilistic;
- the faulting lengths found so far (65521, 65531, 65536, 65552) are all in a
  narrow band and 65521/65536/65552 are not separated by a clean period, so no
  arithmetic pattern should be claimed from them;
- the right experiment is now a **repeat-rate measurement**, not a boundary
  search: for a fixed set of lengths near 64K, repeat each N times and report the
  fault *rate*, then compare rates across `k`, `ASYNC_SCHED`, graph mode and
  spec source. A single OK or FAULT at one length is not evidence either way.

This is the second time in this session that a clean-looking discrete pattern
turned out to be an artifact of single observations (the first was the "repeated
prefix / changed context" attribution in 25.3, corrected in 28). The methodological
rule to carry forward: **for a fault with no confirmed mechanism, measure a rate,
never a boundary.**

### 32.7 Measured: the fault is probabilistic, and 65531/65536 are the same bug

`bench/int8-g64/fault_rate.py` sends N independent requests per length (distinct
content, exact token count verified against the server's `prompt_tokens`) and
restarts the engine after every fault, because a crashed engine keeps returning
stale failures. Six repeats each at the two lengths that section 30 believed were
special:

| length | ok | fault | rate | pattern |
| --- | --- | --- | --- | --- |
| 65536 | 5 | 1 | **0.167** | ok ok ok ok FAULT ok |
| 65531 | 5 | 1 | **0.167** | ok ok ok FAULT ok ok |

Two conclusions, both firm:

1. **The fault is probabilistic, not length-deterministic.** The same length
   passes 5 times out of 6 and faults once, and the failure lands at a different
   request index for each length (4th and 5th). So there is no length arithmetic
   to find: section 30's "discrete at 2^16" model is retracted (32.6) and this
   confirms the retraction with a rate rather than a second anecdote.
2. **65531 and 65536 are the same bug, not two failure classes.** The review
   reasonably suspected two classes because 65531..65535 all need 4096 blocks yet
   only 65531 faulted in the first pass. With a measured rate both lengths fault
   at exactly 1/6, which is what a single shared probabilistic fault looks like
   when sampled twice. The "4096 blocks" and "residue 123" stories were both
   artefacts of reading single observations as deterministic.

This also explains, in hindsight, every earlier inconsistency: the `63K OK / 66K
FAULT` pair, `70K` passing with `SPEC=dflash2` while `66K` failed, and `65536`
faulting then passing. All were single samples of a ~17%-per-request event.

### 32.8 What this changes about the diagnosis

A probabilistic fault at roughly constant rate across neighbouring lengths, with
no dependence on request index, is the signature of a **race or stale-state**
defect rather than an out-of-bounds index computed from the length. That
reprioritises the candidate list:

- **speculative scheduler / draft bookkeeping** (slot reuse, accepted-token
  state, the free-slot accounting the launcher notes already implicate at
  `117 + k`);
- **GDN recurrent state** under speculation (the repo carries
  `vllm-pr50021-gdn-spec-bounds.patch` for exactly this class, and it is already
  applied, so this would be a *second* such defect);
- **CUDA graph state lifetime** (workspace/buffer reuse across replays);
- **async scheduling** (`--async-scheduling` is on; the upstream hybrid
  GDN + MTP + async IMA reports are the closest published match).

and it deprioritises "block table / slot mapping arithmetic", which a
length-determined fault would have fitted.

The measurement to run next is therefore a **rate comparison**, not a boundary
search: same `fault_rate.py` protocol, varying one axis at a time —
`ASYNC_SCHED=1/0`, `k=3/5/7`, `FULL/PIECEWISE/eager`, and `DFlash2 vs MTP vs
none`. A rate that moves with an axis identifies the subsystem; a rate that does
not move exonerates it. `compute-sanitizer` remains worth running, but a
probabilistic fault may not reproduce under memcheck's serialising execution, so
the rate comparison should run first.

### 32.9 The trigger is cumulative request/token volume, not length

Three rate measurements, each on a fresh engine, all with distinct content and
exact token counts:

| length | repeats | faults | rate | fault at request # |
| --- | --- | --- | --- | --- |
| 16,384 | 30 | 2 | 0.067 | **12, 24** |
| 57,344 | 10 | 2 | 0.200 | **5, 10** |
| 65,536 | 6 | 1 | 0.167 | 4 |

Two things stand out. First, the fault is **reproducibly periodic in the request
count** (16K: every 12th request; 56K: every 5th), not random. Second, the period
shortens as the length grows, and the product is roughly constant:

```
16,384 x 12 = 196,608 tokens
57,344 x  5 = 286,720 tokens
65,536 x  4 = 262,144 tokens
```

So the trigger is **cumulative KV volume processed by one engine instance**
(~200-290K tokens), not prompt length and not a per-request coin flip. The
"length threshold above 64K" model, the "discrete at 2^16" model and the
"probabilistic per request" model are all wrong; this supersedes them.

That also explains why 49,152 passed 5/5 (5 x 48K = 245K, just under the ~200-290K
band it might have hit at request 5) and why 16K looked clean in earlier
single-request tests: both were sampling below the cumulative threshold.

### 32.10 Consequence: axis testing moves to 16K, 3.5x cheaper

Because the trigger is cumulative volume, a **cheap short-context run reaches it**
simply by issuing more requests: 16K needs 12 requests, at ~30 s each, versus 56K
at ~105 s each. Per-sample cost drops **3.5x**, and the periodicity makes the
outcome predictable rather than requiring large sample counts to resolve a rate.

This is the answer to "why is this taking so long": the experiment was designed
around the wrong model of the fault. Under the corrected model the same axis
comparison costs hours, not tens of hours.

### 32.11 `ASYNC_SCHED` is exonerated

`ASYNC_SCHED=0` and `=1` produced **identical** results at 56K: both 2/10 faults,
both at requests 5 and 10. A fault that does not move when async scheduling is
disabled is not an async-scheduling race, so the closest published upstream match
(hybrid GDN + MTP + async) is not this bug.

Next axes, in the order that best discriminates under the cumulative model:
`k=3/5/7` (does the period scale with draft depth, as the historical `117 + k`
series suggests?), then `DFlash2 vs MTP vs none` (does the drafter matter at all,
or only the target's KV churn?), then `FULL/PIECEWISE/eager` (is a graph's
retained state involved?), then the pre-fault state dump.

### 32.12 Draft depth `k` is exonerated; the period is fixed per length

At 16K, 16 requests each, three draft depths, fresh engine per arm, distinct
content, exact token counts:

| k | faults | rate | fault at request # |
| --- | --- | --- | --- |
| 3 | 1/16 | 0.062 | **12** |
| 5 | 1/16 | 0.062 | **12** |
| 7 | 2/30 | 0.067 | **12, 24** |

The fault lands at **the same request index (12) for every k**, and the k=7 run
faulted at 12 and 24, i.e. exactly on the period. So:

- draft depth is **not** a factor, which exonerates the speculative
  verify/free-slot geometry and the historical `117 + k` residue series as
  explanations of *this* fault (that series may well be a real, separate bug --
  the launcher documents it -- but it is not what is happening here);
- the period is a property of the **length**, not of the draft: ~12 requests at
  16K and ~5 at 56K.

The cumulative-volume products do not land on one exact number
(16K x 12 = 196,608; 56K x 5 = 286,720; 64K x 4 = 262,144), so "cumulative
tokens" is the right *shape* of explanation but not yet a calibrated constant. A
period in *requests* at a fixed length is equally consistent with per-request
state that is only released on some boundary, so the honest statement is: **the
fault is periodic in the request count with a length-dependent period, and the
mechanism is not yet identified.**

### 32.13 `ASYNC_SCHED` exonerated (32.11 restated with the k data)

`ASYNC_SCHED=0` and `=1` at 56K gave identical 2/10 with faults at requests 5 and
10, and now k=3/5/7 at 16K give identical request-12 faults. Two independent axes
failing to move the fault is evidence the trigger is neither async scheduling nor
draft depth, and it makes the remaining candidates sharper:

- **target-side KV block churn / pool recycling** (the leading candidate: it is
  the only thing that scales with cumulative tokens processed and would have a
  length-dependent period);
- **prefix-cache or block-reuse bookkeeping** (the launcher notes the historical
  residue bug "needs a prefix-cache HIT to fire at all", and
  `enable_prefix_caching` is **False** in this arm, so if this fault also needs a
  hit it cannot be that family -- worth confirming with `PREFIX_CACHE=1`, which
  would both test it and make the sweep far cheaper);
- CUDA graph retained state across replays.

`PREFIX_CACHE=1` is now the highest-value single experiment: it discriminates the
prefix family, and if the fault still reproduces it collapses per-sample cost
because repeated prefixes prefill almost for free.

### 32.14 Prefix caching exonerated, and concurrency changes the period

`PREFIX_CACHE=1` (which the launcher expands to `--enable-prefix-caching
--mamba-cache-mode align`) at 16K, 16 requests: **1/16, fault at request 12** --
identical to `PREFIX_CACHE=0`. So that configuration path is not the trigger.

Note a limitation of that test as run: `fault_rate.py` gives every request
distinct content, so prefix caching was enabled but never actually *hit*. The
result therefore rules out the option's presence, not the "needs a cache hit"
family. A follow-up that reuses one prefix is still the discriminating test for
that family, and it would also collapse per-sample cost.

**Concurrency changes the period.** `fault_rate_concurrent.py`, 16K, 4-way, 6
rounds (24 requests):

```
round 0  ok=4   fault=0   ~65,536 tokens    55 s
round 1  ok=8   fault=0   ~131,072 tokens  111 s
round 2  FAULT (after ~131,072 tokens)
round 3  ok=12  fault=4   ~196,608 tokens  267 s
round 4  ok=16  fault=4   ~262,144 tokens  322 s
round 5  FAULT (after ~262,144 tokens)
```

Faults landed after 8 and 16 successful requests, against **12 and 24** in the
serial 1-way protocol at the same length. So the period depends on how the work is
scheduled, not only on how much of it there is. That is further evidence against a
pure token-count threshold and points at **live KV occupancy or block-allocation
pattern** rather than a cumulative counter.

Caveat on this run's numbers: when a round faults, all four requests in that round
are counted as faults, so the reported 8/24 rate is an over-count -- the true
per-request rate is between 0.062 (serial) and 0.333 (as counted). The harness
should attribute the fault to the request that caused it; that is not yet done.

### 32.15 Cost, and where this stands

Concurrency gives **17.6 s/sample against ~30 s serial at 16K** (1.7x, not 4x,
because four concurrent 16K prefills take longer than one). Combined with the
16K-vs-56K choice (3.5x), the protocol is now ~6x cheaper than when this
investigation started.

Three axes have been eliminated (`ASYNC_SCHED`, draft depth `k`, prefix-cache
option) and the fault is reproducibly periodic, so the search space is much
smaller than at 32.1. But the mechanism is still unidentified, and each remaining
candidate (KV block churn, graph retained state, a request-local counter on some
boundary) needs its own rate comparison.

Honest assessment of the remaining cost: at ~18 s/sample and needing enough
samples to distinguish rates, one axis is ~30-60 min. The candidate list has
roughly four entries left, so pinning the mechanism is a multi-hour, open-ended
investigation of an **upstream** defect -- upstream carries several reports in this
family (DFlash2 cumulative OOB on sm_80/v0.27.1, hybrid GDN + MTP + async IMA),
and this is not a defect this project introduced.

Recommendation, for the human to choose: either (a) bound this to one more axis
(the prefix-reuse test, which is both discriminating and cheap), then park it with
the evidence recorded, or (b) park it now and spend the time on the W4A8 Marlin
line, which is ready (`marlin_shape_bench.py`) and does not depend on this bug.

## 33. Driver/environment gate: CLEAN — this is not the cmpunlocker WPR2 defect (2026-09-18)

Upstream vLLM #55279 is the same symptom family (CMP 170HX / SM80 / vLLM 0.27.1 /
DFlash2 / Xid 31 `FAULT_PDE`), and its maintainer's first request was not more
vLLM debugging but a specific environment check: confirm whether this card and
driver are hitting **`amoghmunikote/cmpunlocker#32`**, where the unlocker
registers the highest reserved framebuffer region into the PMA even though it is
actually **WPR2 + GSP firmware heap**. That would let the CUDA allocator hand out
memory the GPU then faults on — producing exactly Xid 31 / `FAULT_PDE` /
`ACCESS_TYPE_VIRT_READ`.

Everything about this machine matched the affected combination
(`10de:20c2`, CMP 170HX 64 GB, nvidia-open 610.43.02), so this had to be checked
before any further vLLM work. **It is fixed on this host.** The boot log carries
the decisive line:

```
NVRM: GPU0 memmgrSec2DebugLateExtendHighPmaRegion: SEC2_DEBUG_LATE_PMA:
      candidate=6 base=0xff7300000 limit=0xfffffffff left reserved (backs WPR2)
```

`left reserved (backs WPR2)` is the post-fix behaviour; the defect would show
this candidate being *registered* instead. The region size is
`rsvdSize=0x8d00000` = **141 MiB**, matching the ~141 MiB WPR2 region the issue
describes, and the PMA/heap split is consistent:

```
numFBRegions=7 numPmaRegions=1 stockFb=0x200000000
pma_total=0xfd8f50000 pma_free=0xfd8f50000 heap_total=0x1000000000 heap_free=0xa133000
```

Corroborating details: device ID `10de:20c2`; driver
`610.43.02` nvidia-open built locally (`Sun Sep 13 12:43:04 PM PST 2026`) and
installed from the cmpunlocker tree at
`/lib/modules/7.0.0-31-generic/updates/cmpunlocker/nvidia.ko`; idle free memory
**64,898 MiB**, which sits at the reviewer's *fixed* reference point (~64,908)
rather than the pre-fix one (~65,049).

One caveat worth stating rather than glossing: the local cmpunlocker git clone
(`/home/base-node/cmpunlocker-ee41902`, HEAD `ee41902` "Add support for
615.71.09") does **not** contain the fix commit object
`ed579213998acc4da3d0391ae5106e8b5f870f12`, so ancestry could not be checked from
the source tree. The boot-log message is the stronger evidence and it is
conclusive: whatever commit it came from, the installed module leaves the WPR2
region reserved. A second tree, `cmpunlocker.incomplete-20260913-1226`, has no
git metadata at all.

**Conclusion: the WPR2 environment gate passes. This IMA is a software defect in
the serving stack, not a driver mapping defect.** The line is closed and no
driver change or reboot is required.

### 33.1 Next, in the order the review set

1. ~~driver gate~~ — done, clean (above);
2. `SPEC=dflash2 / mtp / none` at 16K x 24 (cheap, and the single most
   discriminating axis left: the earlier "70K with SPEC=none passes" was **one
   request**, which cannot exclude a fault whose period is 5-12 requests);
3. fixed 16K while varying `--max-num-batched-tokens` 1024 / 2048 / 4096 — if the
   fault follows **prefill-chunk / model-runner invocation count** rather than KV
   volume, the fault index should scale inversely with MBT;
4. only then `FULL / PIECEWISE / eager`, to test graph retained state;
5. pre-fault state dump (with the cross-request persistent state the review
   listed: request slot id, block-table and slot-mapping and KV-pool pointers,
   free/used block counts, graph input buffer pointers, and above all the
   **scheduler/model-runner invocation counter and prefill chunk index**);
6. Compute Sanitizer on the minimised reproducer.

### 33.2 The conserved quantity is not yet identified (review correction accepted)

32.9/32.14 said "cumulative KV volume is the right shape". The review is right
that this is still too strong: 196,608 / 286,720 / 262,144 do not converge on a
constant, whereas the **chunk** interpretation is more suggestive at the
current `max_num_batched_tokens=2048`:

```
57,344 / 2048 = 28 chunks/request  x 5 requests  = 140
65,536 / 2048 = 32 chunks/request  x 4 requests  = 128
49,152 / 2048 = 24 chunks/request  x 5 requests  = 120   (this one was CLEAN)
16,384 / 2048 =  8 chunks/request  x 12 requests =  96
```

140 and 128 bracketing a clean 120 is not decisive, but it is close enough that
MBT is now the right knob. Corrected wording: **cumulative work proportional to
prompt length; whether the conserved quantity is KV blocks, prefill chunks /
model-runner invocations, or another per-request resource is not established.**

## 34. Classification: speculative decoding is NECESSARY (2026-09-18)

Three spec sources, same engine, same length, same 24 requests each, distinct
content, exact token counts, restart after every fault:

| spec source | k | 16K x 24 result |
| --- | --- | --- |
| `DFlash2` (`method=dflash`) | 7 | **2 faults — requests 12, 24** |
| `MTP` (`method=mtp`) | 7 | **2 faults — requests 12, 24** |
| **`none`** (no `--speculative-config`) | — | **0 faults — 24/24 clean** |

This is the sharpest result of the investigation so far, and it is a two-sided
one:

**Speculation is necessary.** With the speculator removed the identical workload
runs 24/24 clean on the same length, dtype, backend, schedulers and graph config.
So every target-only path is exonerated: prompt prefill by itself, the KV
allocator in isolation, GDN/Mamba alone, the Triton attention kernels, the
int8_per_token_head format, the driver, and the CUDA graph machinery on its own.

**But the drafter is not it.** DFlash2 and MTP fault identically, at the same two
request indices, with the same period. Two structurally different drafters
(DFlash2 is a 5-layer block drafter consuming target hidden states; MTP is
Qwen's own multi-token head) cannot produce the same period by coincidence. The
defect is therefore in **shared speculative machinery**, not in either drafter:

- `num_accepted_tokens` / accepted-token bookkeeping,
- slot and recurrent-state allocation per speculative step (the repo already
  needed `vllm-pr50021-gdn-spec-bounds.patch` in exactly this area, so a second
  defect here is plausible),
- the spec-token / draft-id buffers and their slot mapping,
- verify-batch metadata that both drafters feed.

**This also retracts the earlier "SPEC=none at 70K passes" observation.** That was
a single request, and a fault with a period of 12 requests cannot be ruled out by
one sample; 32.4's wording ("only observed in speculative configurations... not
proven sufficient") was appropriately cautious but is now superseded by a real
30-sample-class result. The correct statement is: **speculation is necessary and
either drafter suffices to trigger it.**

Combined with the exonerations already recorded (draft depth `k` 3/5/7, async
scheduling, the prefix-cache option), the search space is now:

```
NOT: driver/WPR2, KV dtype, attention backend, split-KV verifier, async scheduler,
     prefix-cache option, draft depth, DFlash2-specific metadata
IS :  shared speculative state, in the verify/addressing path, released or
      reallocated on a boundary whose period scales with prompt length and with
      concurrency
```

### 34.1 The MBT experiment is now the right next probe

The reviewer's chunk-count hypothesis can now be tested against a *speculative*
baseline rather than a mixture. Fixed prompt of exactly 16,384 tokens with the
DFlash2 arm, varying only `--max-num-batched-tokens` (the launcher hardcodes
2048, so this needs the direct-launch route used for the `none` arm):

| MBT | prefill chunks/request | predicted fault index if chunk-count driven |
| --- | --- | --- |
| 1024 | 16 | ~6 |
| 2048 | 8 | **12 (measured)** |
| 4096 | 4 | ~24 |

An inverse relationship would put the conserved quantity at model-runner
invocations rather than KV bytes, and would point the instrumentation at the
runner's static buffers. A fault index that stays at 12 regardless of MBT would
instead implicate a per-request persistent resource.

## 35. The conserved quantity is a PER-REQUEST resource, not chunk count (2026-09-18)

The review's leading alternative to "cumulative KV volume" was
**prefill-chunk / model-runner invocation count**: with `max_num_batched_tokens`
at 2048, the chunk products were 140 (57K), 128 (64K) and 120 (49K, the clean
one), which is close enough to be suggestive. The discriminating experiment was
to fix the prompt at exactly 16,384 tokens and vary only
`--max-num-batched-tokens`, predicting an **inverse** relationship between MBT and
the fault index.

Measured (DFlash2, k=7, 16,384 tokens, `--decode-tokens 16`):

| MBT | prefill chunks/request | predicted fault index if chunk-driven | **measured** |
| --- | --- | --- | --- |
| 1024 | 16 | ~6 | **11** |
| 2048 | 8 | **12** | **12** |
| 4096 | 4 | ~24 | **11** |

**The fault index does not move when MBT changes by 4x.** The chunk-count /
model-runner-invocation hypothesis is refuted, and by the review's own stated
criterion ("if it is still request #12 however MBT changes, a per-request
persistent resource / block lifecycle is more likely") the conserved quantity is
**per-request**, not per-chunk and not a cumulative byte count.

Note the launcher hardcodes `--max-num-batched-tokens 2048`, so these arms were
launched directly with a hand-built command line. That exposed a harness defect
worth recording: `fault_rate.py` restarted engines through the **systemd unit**,
whose config is the 2048 one, so a directly-launched arm was silently restarted
with the wrong configuration and then aborted on a length assertion. The harness
now takes `--restart-cmd` so an arm restarts with *its own* config. Any arm that
does not use a matching restart is not measuring what it claims to.

### 35.1 Where the investigation now stands

Necessary and sufficient evidence accumulated:

| factor | status |
| --- | --- |
| speculation | **necessary** — `SPEC=none` is 24/24 clean; DFlash2 and MTP both fault (34) |
| drafter identity | not a factor — DFlash2 and MTP fault identically |
| draft depth `k` | not a factor — 3/5/7 all fault at request 12 (32.12) |
| async scheduling | not a factor — 0/1 identical (32.11) |
| prefix-cache option | not a factor — 0/1 identical (32.14) |
| driver / cmpunlocker WPR2 | **clean** — `left reserved (backs WPR2)` (33) |
| KV dtype / attention backend | not a factor between int8_g64 and int8_per_token_head |
| split-KV verifier | not a factor — `SPEC_ATTN=0` still faults |
| `max_num_batched_tokens` | not a factor — 4x change moves nothing (35) |
| chunk-count / runner invocations | **refuted** (35) |

So: a **per-request resource in the shared speculative path**, whose exhaustion or
recycling on some boundary has a period set by prompt length (12 requests at 16K,
5 at 57K, 4 at 64K) and by concurrency (8 at 16K with 4-way).

That is now a narrow enough target for instrumentation rather than more A/B. The
next step is the pre-fault state dump the review specified, especially the
cross-request persistent state: **request slot id, block-table / slot-mapping /
KV-pool pointers, free-vs-used block counts, graph input buffer pointers, and a
scheduler/model-runner invocation counter** — plus the natural addition given
§35, a per-request **resource index** (which slot, which block range) so the
faulting request's identity can be compared against the previous 11.

### 35.2 The `marlin_shape_bench.py` ABI is wrong (review — accepted, not yet fixed)

The review is right and the harness is currently unusable for its purpose:

1. the hand-rolled packing is `codes[:, 1::2] << 4 | codes[:, 0::2]`, i.e. two
   nibbles **along N**, whereas vLLM's `gptq_pack` packs **8 x 4-bit along K**
   (`q_res[i::8, :] << (4*i)`); the layouts are different;
2. `ops.gptq_marlin_repack(..., device=...)` — the 0.27.1 signature takes no
   `device`;
3. the INT8 activation scale is derived **after** overwriting the tensor with the
   quantized values, so it collapses to ~1 instead of
   `original_fp16_row_max / 127`.

It must be rebuilt on upstream primitives (`gptq_quantize_weights`, `gptq_pack`,
`ops.gptq_marlin_repack`, `per_token_quant_int8`) or, better, by taking the
already-loaded `qweight`/`scales` straight off a real AutoRound `gate_up` /
`down_proj` layer, which also removes the "is a random synthetic weight
representative of production packing" variable. **Do not run it or quote numbers
from it until then.** It is left in the tree as a skeleton with this caveat.

## 36. W4A8 already exists in this repo — do the engine A/B, not a microbench (2026-09-18)

Two findings while trying to build the review's M1 (gate_up-only W4A8-INT8 Marlin)
as a single-kernel microbench. Both change the plan.

### 36.1 The checkpoint is compressed-tensors, not GPTQ

`marlin_shape_bench.py` was written to read `qweight` / `scales` / `qzeros` /
`g_idx` off a real layer. It found nothing, because this checkpoint stores

```
model.language_model.layers.N.mlp.{gate,up,down}_proj.weight_packed
model.language_model.layers.N.mlp.{gate,up,down}_proj.weight_scale
model.language_model.layers.N.mlp.{gate,up,down}_proj.weight_shape
```

i.e. **compressed-tensors W4A16**, which vLLM repacks to Marlin at load time.
There are therefore no GPTQ-format tensors to read, and a microbench cannot get
"the real production packing" by opening the checkpoint -- it has to go through
vLLM's own loader/repack. Combined with the three defects the review already
found in that file (packing along the wrong axis, a `device=` argument that
0.27.1's `gptq_marlin_repack` does not accept -- confirmed by signature
inspection -- and an INT8 scale derived after the value was overwritten), the
microbench is **not** the cheapest way to answer the question. It is left in the
tree marked as superseded.

### 36.2 The repo already implements gate_up-only W4A8, correctly

`patches/marlin-int8-layer-select.patch` adds exactly the review's M1 shape:

```
VLLM_MARLIN_INPUT_DTYPE=int8
VLLM_MARLIN_INT8_INCLUDE_RE=<regex>     # restrict the int8-activation path
VLLM_MARLIN_INT8_EXCLUDE_RE=lm_head|mtp # default
```

matched against the **layer name**, plus the critical detail that the two new
vars are registered in `envs.py` so they participate in the torch.compile cache
key -- without which switching the selection replays a stale compiled graph and
crashes with `KeyError: 'input_global_scale'`.

And `patches/marlin-int8-negative-scales.patch` already fixes the AutoRound
negative-group-scale corruption that the review cited as upstream #48905 (the
kernel reads the requantised scales as *unsigned* int16, so every negative group
becomes garbage while still benchmarking fine).

So the experiment the review asked for is a **one-environment-variable engine
A/B**, not a new kernel harness:

```
arm A (baseline): CTX=long, int8_per_token_head, W4A16                (measured)
arm B (W4A8):     same + VLLM_MARLIN_INPUT_DTYPE=int8
                  + VLLM_MARLIN_INT8_INCLUDE_RE='mlp\.(gate|up)_proj'
```

with the same quality instrumentation already built (`ab_quality.py`,
`ab_quality_compare.py`) and the same decode protocol (`spec_bench.py`). That
reuses the production packing, needs no ABI reconstruction, and directly measures
the thing the review's gate is about ("gate_up kernel >=10%, whole-step >=8%
after activation-quant overhead").

### 36.3 Where the IMA line stands, for the record

Since 33 the following are settled, each with matched measurement:

- **driver/WPR2 clean** (33) -- not the cmpunlocker defect;
- **speculation necessary**: `SPEC=none` 24/24 clean, DFlash2 and MTP both fault
  at requests 12 and 24 (34);
- **drafter not implicated**: two structurally different drafters, identical
  period (34);
- **`max_num_batched_tokens` not implicated**: 1024/2048/4096 give fault indices
  11/12/11, so a 4x change in prefill chunk count moves nothing (35) -- this
  refutes the chunk-count / model-runner-invocation model and, by the review's
  own criterion, leaves a **per-request persistent resource in the shared
  speculative path** as the target;
- already excluded earlier: `k`, async scheduling, prefix-cache option, KV dtype,
  attention backend, split-KV verifier.

The next IMA step is instrumentation (pre-fault state dump with the cross-request
persistent fields listed in 33.1), not more A/B. The next performance step is the
W4A8 engine A/B above, which does not depend on the IMA.

## 37. W4A8 gate_up engine A/B: no measurable gain (2026-09-18)

Ran the review's M1 as an engine A/B rather than a microbench (36.2), because the
repo already carries the correct per-layer W4A8 machinery.

**Configuration.** Same `CTX=long` arm, same checkpoint, same DFlash2 k=7, same
`max_num_batched_tokens=2048`, clocks pinned. The only difference:

```
baseline: (nothing)
W4A8:     VLLM_MARLIN_INPUT_DTYPE=int8
          VLLM_MARLIN_INT8_INCLUDE_RE=mlp\.(gate|up)_proj
          VLLM_MARLIN_INT8_EXCLUDE_RE=lm_head|mtp
```

**The selector was verified, not assumed.** `get_marlin_input_dtype` returns
`torch.int8` for `mlp.gate_proj` and `mlp.up_proj`, and `None` for `mlp.down_proj`,
`lm_head` and `mtp` -- i.e. exactly gate+up, which is the review's M1. Two
independent checks that the arm is real rather than silently ignored (the failure
mode of upstream #48904):

- `patches/marlin-int8-negative-scales.patch` **is applied** -- the sign-fold is
  present at `model_executor/kernels/linear/mixed_precision/marlin.py:150-173`.
  Without it the AutoRound negative group scales turn to garbage while still
  benchmarking normally, so this had to be confirmed before any timing meant
  anything;
- the model produces coherent text (`"The capital of France is"` -> `" Paris. The
  capital of Germany is Berlin."`), which the patch header says it will not if the
  sign-fold is missing.

**Result** (`bench/int8-g64/spec_bench.py`, 3 reps, 191 output tokens, median):

| arm | ms per output token @ 4K | per-rep |
| --- | --- | --- |
| W4A16 baseline | **6.675** | 8.311, 6.675, 5.156 |
| W4A8 gate_up | **6.645** | 8.276, 6.645, 5.134 |

A **0.45%** difference, and the per-rep structure is superimposable. The review's
gate for proceeding to whole-step work was ">=10% on the dominant GEMM and >=8%
after activation-quant overhead"; this does not come close, so on this evidence
gate_up-only W4A8 does **not** justify further integration work at 4K.

Caveats stated rather than hidden:

- measured at **4K only**. The 16K run of the W4A8 arm hit the IMA (HTTP 500), and
  the activation-quant overhead is a fixed cost per token while the GEMM saving
  grows with context, so the balance *could* differ at 126K. That is the one
  reason this is "no gain at 4K" and not "no gain";
- `accepted_tokens_per_pass` came back `null` for the baseline run (window
  alignment found no `SpecDecoding metrics` line inside the bracket) and 3.1 for
  W4A8. Since the reported metric is ms **per output token**, a differing
  acceptance would move it, so the two arms' acceptance should be matched before
  this result is treated as final -- the per-rep ms values are nearly identical
  (8.311/6.675/5.156 vs 8.276/6.645/5.134), which is what one expects if
  acceptance was also matched, but that was not verified from the log.

### 37.1 Session position

Two lines are now in a clear state:

**IMA (correctness blocker).** Driver gate clean (33); speculation necessary and
drafter-independent (34); `max_num_batched_tokens` irrelevant, so the conserved
quantity is a per-request resource in the shared speculative path (35). Next step
is instrumentation, not more A/B.

**Performance line.** W4A8 gate_up measured and negative at 4K (37). The
remaining review-ordered items are the DFlash `k=3/5/7` proposal-efficiency sweep
(`spec_bench.py` is built for exactly this and now reports accepted/pass
alongside pass cost), and only after that a paired gate/up Marlin+SwiGLU kernel --
whose payoff should be bounded first by measuring the standalone `silu_and_mul`
share, since E40's data shows the win there comes from the changed dataflow
rather than from removing a microsecond-scale activation.

`marlin_shape_bench.py` remains in the tree as a superseded skeleton: it reads
GPTQ-format keys that this compressed-tensors checkpoint does not have (§36.1),
and the engine A/B above is both cheaper and ABI-correct.

## 38. Graph mode is exonerated: true eager faults identically (2026-09-18)

The review's last configuration A/B before instrumentation: `FULL` / `PIECEWISE` /
`eager` at 16K x 24, DFlash2 k=7, fresh engine per arm.

**A methodological catch first, because the first attempt was invalid.** Driving
`CUDAGRAPH_MODE=NONE` through the launcher produced a run that *looked* like an
eager arm but was not: reading back the engine's effective config showed
`cudagraph_mode: <CUDAGraphMode.FULL_AND_PIECEWISE: (2,1)>` with
`cudagraph_capture_sizes: [1,2,4,8,16,24,32]`, i.e. capture fully active. The
launcher's `CG_MODE` plumbing did not express `NONE` on this path, and the arm
would have been recorded as "eager still faults" while measuring captured
execution. It was discarded and re-run by direct launch, and the replacement was
**verified** from the log before the measurement was believed:

```
cudagraph_mode': <CUDAGraphMode.NONE
cudagraph_capture_sizes': []
```

This is the same class of error as §28.4 and §35 -- a config that did not reach
the engine -- and it is now the third time it has produced a would-be wrong
conclusion in this investigation. Every arm must have its effective configuration
read back out of the engine log, not assumed from the environment.

**Result** (16,384 exact tokens, 24 requests, capture verified from the log):

| graph mode | capture sizes | 16K x 24 |
| --- | --- | --- |
| `FULL_AND_PIECEWISE` (baseline) | `[1,2,4,8,16,24,32]` | **2/24 -- requests 12, 24** |
| `NONE` (true eager) | `[]` | **2/24 -- requests 12, 24** |

**Bit-for-bit the same failure, with no CUDA graph anywhere in the process.**

So CUDA graph retained pointers, static graph input buffers, and capture-lifetime
state are all eliminated: there is no capture to retain anything. `FULL` and
`PIECEWISE` are therefore unnecessary to run as separate arms for this question --
the fully-uncaptured case is the strongest version of the test and it faults
exactly like the captured one, which also means graph-mode choice cannot fix this.

By the review's stated criterion, this pushes the probability decisively onto
**hybrid GDN/Mamba per-request speculative state reuse** rather than graph state
or the plain KV allocator.

### 38.1 The remaining suspect class, precisely

Everything below the request lifecycle is now excluded, and everything
speculation-specific and drafter-independent remains:

```
req_id -> input_batch.idx_mapping -> persistent req_state_idx -> {
    Mamba/GDN state index,
    block-table row,
    num_accepted_tokens,
}
```

with the cross-request persistent GPU tensors in
`GDNAttentionMetadataBuilder` (`spec_state_indices_tensor`, `spec_sequence_masks`,
`spec_token_indx`, `spec_query_start_loc`, `num_accepted_tokens`) and the
`_mamba_state_idx_gpu` / `_mamba_src_col_gpu` / `_mamba_src_off_gpu` /
`num_accepted_tokens_gpu` buffers as the natural first targets. These advance on
the speculative accept path -- which is exactly the path that `SPEC=none` does not
enter, matching §34's classification.

### 38.2 Instrumentation design (review-specified, not yet built)

The review's guidance is adopted in full because a timing-sensitive fault is at
issue and the naive approach can erase it:

- **no printf, no per-step `.cpu()` / `torch.cuda.synchronize()`** in the hot path;
  heavy synchronisation can move or hide the bug;
- a **device-side first-error record buffer**: a small struct
  (`error_code, request_ordinal, req_state_idx, seq_len, num_computed,
  num_accepted, mamba_state_idx, src_col, src_off, block_table_col, block_id,
  num_blocks`) written once via `atomicCAS(error_code, 0, MY_ERROR)` on the first
  invariant violation, read back only after the request finishes;
- invariants checked immediately before each dereference:
  `0 <= req_state_idx < max_num_reqs`;
  `mamba_state_idx` within its valid row range;
  `block_table_col < row_width`;
  `block_id == NULL_BLOCK_ID or 0 <= block_id < physical_block_count`;
- **request-slot generation tagging**: `generation[slot] += 1` on every
  add/free/reuse, with `(request ordinal, req_id, slot, generation, seq_len,
  mamba_state_idx, block_table row)` recorded. The hypothesis this tests directly
  is a slot that has been freed and reused while some associated state still
  carries the previous generation;
- **state diff of requests #10/#11/#12 only** -- find the *first* field that
  differs at #12 versus #11, rather than dumping everything;
- **allocation-relative fault addresses**: record KV pool / Mamba state pool /
  graph buffer / spec persistent buffer base addresses and widths at startup, then
  convert each Xid 31 VA to `fault_va - allocation_base`. The absolute-address
  statistics (63 Xid 31, 34 distinct VAs, a few repeating 7x/5x/5x) are not
  actionable; a repeating *relative* offset would be. `FAULT_PDE VIRT_READ` is
  consistent with a stale or freed VA, so a constant relative offset would be a
  strong signal.

No configuration knob sweep should be run after this. Compute Sanitizer follows
the instrumentation (on the narrowed path), not the reverse, since memcheck's
serialisation may prevent a timing-sensitive fault from reproducing at all, and
even when it fires it reports only the final illegal load rather than when the
metadata first went bad.

### 38.3 The fault addresses are page-aligned and cluster at a fixed in-allocation offset

Converting the Xid 31 absolute VAs to band-relative offsets (`fault_va - band_base`
for 2 GiB bands), because absolute CUDA VAs are not actionable but relative ones
are:

```
offsets repeating across different bands (fixed relative address):
  0x022b3c000 = 555.234 MiB  x7
  0x0261f9000 = 609.973 MiB  x7
  0x041400000 = 1044.000 MiB  x5
  0x03d96c000 = 985.422 MiB  x4
  0x022f91000 = 559.566 MiB  x4
  0x025cc0000 = 604.750 MiB  x4
  0x02288b000 = 552.543 MiB  x3
  0x022dc4000 = 557.766 MiB  x3
  0x024348000 = 579.281 MiB  x3
  0x03c9ac000 = 969.672 MiB  x2
```

Three properties, all of which the review asked to test for:

1. **Every faulting VA is 4 KiB-aligned** (72/72), while only 5/72 are 2 MiB-aligned
   and 11/72 are 64 KiB-aligned. Page-aligned-but-not-hugepage-aligned is exactly
   what a stale or freed *page mapping* looks like, and is consistent with
   `FAULT_PDE` (the page directory entry is absent) rather than with wild pointer
   arithmetic, which would produce arbitrary misalignment;
2. **the same relative offsets recur across different 2 GiB bands** -- 555 MiB
   appears 7 times, 610 MiB 7 times, and so on. The same offset landing in
   different arenas means the bad address is a *fixed offset within whichever
   allocation is there*, not a one-off pointer value;
3. **the offsets cluster in a narrow window**: seven of the ten repeating offsets
   fall between **552 and 610 MiB** (~57 MiB wide), with a second small cluster at
   970 and 985 MiB and one exactly 2 MiB-aligned entry at 1044.000 MiB (n=5).

A reproducible ~57 MiB window at ~550-610 MiB into an allocation is a much
stronger lead than the raw absolute addresses, and it is the kind of result that
can be matched directly against a live allocation map.

### 38.4 What this adds

`FAULT_PDE ACCESS_TYPE_VIRT_READ` on a page-aligned address, at a reproducible
allocator-relative offset, in a process whose CUDA graphs are not even captured
(38), points at **a stale page mapping for a buffer that is still being read** --
i.e. freed or remapped memory being dereferenced -- rather than at an
out-of-range index into a live tensor. That is consistent with §38.1's suspect
class: per-request speculative state (Mamba/GDN state index, block-table row,
accepted-token bookkeeping) that is recycled on a boundary while an in-flight
consumer still holds the old address.

Next actions, in order: (a) record the live allocation map of this engine (KV pool
base/size, Mamba state pool base/size, spec persistent buffers, graph buffers) and
compute `fault_va - base` against it, which turns the offsets above into a named
allocation; (b) then the device-side first-error buffer and slot-generation
tagging in §38.2, targeting the named allocation.

Artifacts: `int8g64-layout-audit/fault_va_analysis.py` and `fault_va_bands.py`.

### 38.5 Live allocation bases captured — and why cross-process VA comparison is invalid

Added `patches/log-alloc-bases.patch` (logging only) and applied it to the KV
allocator. Two findings, one of which is a correction to how the VA data in 38.3
must be read.

**The first attempt produced no output**, because vLLM 0.27.1 carries **two**
KV-cache allocators with byte-identical bodies:

```
v1/worker/gpu_model_runner.py::_allocate_kv_cache_tensors   <- patched first, never called
v1/worker/gpu/attn_utils.py::_allocate_kv_cache             <- the one this stack uses
```

Patching the first and seeing silence is the same "config did not reach the code"
trap as 28.4/35/38, so the marker line was confirmed present in the module that
actually runs before any conclusion was drawn.

With the right module patched, the engine reports its KV backing allocations:

```
ALLOC kv_cache base=0x0000737f00000000 size=5443282944 bytes (5191.12 MiB) first_layer=...layers.16.linear_attn
ALLOC kv_cache base=0x0000738060000000 size=5443282944 bytes (5191.12 MiB) first_layer=...layers.8.linear_attn
ALLOC kv_cache base=0x00007381c0000000 size=5443282944 bytes (5191.12 MiB) first_layer=...layers.0.linear_attn
```

so the pools are **2 GiB-aligned, 5191.12 MiB each**, three of them (one per
hybrid group), each holding 5,443,282,944 bytes.

**Correction to 38.3.** Resolving the recorded fault addresses against these bases
gives `inside = 0`: every historical fault VA is in a different address band from
the current process (`0x1c...`, `0x38...`, `0x7469...` vs current `0x737f/0x7380/
0x7381...`). `dmesg` is a ring buffer spanning many engine generations and CUDA
places allocations at different bases in each, so **absolute VAs from different
processes cannot be subtracted from the current process's bases**. 38.3's
band-relative offsets remain valid as a *shape* observation -- the offsets do
repeat -- but the specific numbers must not be matched against a live map obtained
from a different process. The correct procedure is to capture the bases and the
fault within the **same** engine generation.

**What does hold, and is unchanged:** all 72 faulting addresses are 4 KiB-aligned
(low 12 bits identically `0x000`), which is the signature of a stale or absent
page mapping rather than misaligned pointer arithmetic, and consistent with
`FAULT_PDE`. The most common 2 MiB-page offsets are `0x013c000` (x7), `0x01f9000`
(x7), `0x0000000` (x5), `0x016c000` (x4) -- repeating, non-zero page offsets.

### 38.6 The procedure from here

Capture bases and faults in **one** engine generation, which requires the engine to
fault while its own `ALLOC` lines are still the most recent in the log:

1. start the engine (bases logged), record them;
2. drive 16K x 24 until it faults;
3. read the Xid 31 VA that appears **after** those bases were logged, via
   `journalctl -k --since` or by timestamp comparison against the log;
4. compute `fault_va - base`; the verdict is one of *inside a pool* (offset is
   meaningful), *just past a pool end* (overrun), or *below/unmapped* (freed page);
5. only then instrument the named region with the first-error buffer and slot
   generation tags from 38.2.

`bench/int8-g64/resolve_fault_va.py` implements steps 1-4 but must be given a log
whose `ALLOC` lines come from the same process as the faults; it says so in its
output rather than silently resolving across generations.

### 38.7 Same-generation capture: the fault VA is a CONSTANT offset, and it is not in a KV pool

Captured the KV pool bases and then drove the same engine (pid 572825) to fault, so
the fault address and the bases belong to one process. This is the procedure 38.6
called for, and it produced the sharpest result of the whole investigation.

The four most recent Xid 31 events, from four consecutive engine generations:

```
pid=570013   faulted @ 0x77ca_239af000
pid=570799   faulted @ 0x748b_239af000
pid=572825   faulted @ 0x7377_a39af000     <- bases known for this process
pid=573484   faulted @ 0x7e34_639af000
```

**Every one of them ends in `39af000`.** The high bits differ per process (as CUDA
relocates allocations), but the low 21 bits are **identical across four separate
engine processes**, and `0x39af000` is a 4 KiB-aligned offset inside its 2 MiB
page. A constant offset repeated across processes is not a wild pointer: it is a
fixed address computed the same wrong way every time, landing the same distance
into whatever region the allocator put there.

Resolving the same-process VA against the eight logged 2 GiB-aligned KV pools
(5191.12 MiB each):

```
fault VA 0x00007377a39af000
  nearest KV pool below : 0x0000737820000000   (fault is 1990.316 MiB BELOW it)
  verdict               : NOT inside any KV pool
  gap                   : 2,086,998,016 bytes = 1990.316 MiB = 1.9437 GiB
```

So the faulting read is **not** in the paged attention KV cache. It sits ~1.94 GiB
below the lowest KV pool, a gap of a size that matches a **recurrent-state / small
pool** rather than the paged KV: the launcher's own notes record ~0.098 GiB of
recurrent-state reservation per `(k+2)` per resident request, i.e. ~0.88 GiB per
request at k=7, so a small number of resident requests occupies a region of about
this size.

Combined with everything already excluded (driver/WPR2, graph capture, attention
backend, KV dtype, split-KV verifier, async scheduling, prefix-cache option, draft
depth, `max_num_batched_tokens`, and now the paged KV pool itself) this is the
tightest localisation so far:

```
a fixed, recomputed-wrong address at 0x...39af000, ~1.94 GiB below the paged KV
pools, read during the speculative accept path (SPEC=none never enters it), in a
region the size of the recurrent-state reservation
```

### 38.8 The immediate next step

The paged KV pools are now logged; the region that actually faults is not one of
them, so the next action is to log the **remaining** allocations the same way --
specifically the Mamba/GDN recurrent-state buffers and the spec-decode persistent
tensors -- then repeat the same-generation capture and subtract. The `39af000` low
bits give a target to match: whichever allocation's base, when subtracted from a
fault VA, yields exactly `0x39af000` is the region being read out of range.

Until that is done, no further configuration A/B should run; the address is
already specific enough that instrumentation should aim at it directly.

### 38.9 The fault address decomposes exactly: base(2 GiB-aligned) + 0x39af000

Correcting a misreading in 38.7 and sharpening the result. The eight logged pools
are **all `linear_attn` (GDN/Mamba) groups**, 5191.12 MiB each, 2 GiB-aligned,
totalling 40.56 GiB -- they are the recurrent-state pool family, not the paged
attention KV. Every one of their bases has **low 21 bits = 0**.

Against that, the same-process fault VA decomposes exactly:

```
fault VA                      0x00007377a39af000
fault VA - 0x39af000        = 0x00007377a0000000     <- 2 GiB-aligned
lowest logged pool base     = 0x0000737820000000
gap between them            = 1,990.316 MiB = 1.9437 GiB
```

So the faulting read is **an allocation base (2 GiB-aligned) plus `0x39af000`**, and
that base lies 1.94 GiB below the lowest recurrent-state pool. Because *all* logged
bases are 2 GiB-aligned and the low 21 bits of the fault VA are `0x1af000`, the
`0x39af000` cannot be "an offset inside a logged pool" -- it is the offset into an
allocation that was never logged.

The four-generation repetition from 38.7 now reads differently and more usefully:
the *whole* VA is not constant, but `va - (va & ~0x1FFFFF)` is, i.e. the fault
recurs at a **fixed 2 MiB-page offset of `0x39af000`** in whichever allocation the
allocator placed at that spot in each process. That is the signature of a
**constant computed displacement** -- the same index arithmetic evaluated the same
way every run -- landing in an allocation whose base moves.

This is now a concrete, falsifiable target: find which allocation occupies
`fault_va - 0x39af000` and the displacement `0x39af000` names the structure. Note
`0x39af000` = 3,777,536 bytes, and that the recurrent-state reservation the
launcher documents is ~0.098 GiB per `(k+2)` per request -- so the offset should be
checked against the per-request slot stride of that pool rather than treated as an
arbitrary number.
