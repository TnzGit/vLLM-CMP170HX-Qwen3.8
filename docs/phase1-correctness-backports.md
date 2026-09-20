# Phase 1 — two upstream correctness backports

Both are minimal, independently revertible, and touch disjoint files.

| | 1A | 1B |
| --- | --- | --- |
| upstream | #51812 | #54282 |
| commit | `2edd7e3` | `9d6acf0` |
| file | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | `v1/spec_decode/llm_base_proposer.py` |
| patch | `patches/gdn-51812-gate-index-select.patch` | `patches/draft-noise-independence-54282.patch` |
| kind | **P0 correctness** — silently updates the wrong GDN recurrent state | **robustness / reproducibility** — NOT distribution-correctness (see §1B semantics) |
| perf intent | none | none |

## 1A — GDN gate gather (#51812)

**Invariant.** `mixed_qkv` is permuted before the recurrent step
(`mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)`), so `q/k/v` are in permuted
token order and the gate tensors `a`/`b` must be permuted by the **same** mapping.

**What v0.27.1 was missing.** Two call sites passed the unpermuted `a`/`b`:

| call site | `q/k/v` | `a`/`b` before |
| --- | --- | --- |
| speculative | `query_spec` (gathered) | `a`, `b` — **not gathered** |
| pure-decode | `query_non_spec` (gathered) | `a`, `b` — **not gathered** |

The prefill branch already gathered (`a_non_spec = a.index_select(0, non_spec_token_indx)`),
so this is a consistency repair inside one function. The pure-spec case needs no gather
because then `mixed_qkv_spec == mixed_qkv`.

**Failure mode.** The kernel consumes whatever it is handed, so nothing raises — the GDN
recurrent state is silently updated with the wrong rows. That is why this is P0 even though
it changes no performance.

**Not the same defect as `vllm-pr50021-gdn-spec-bounds.patch`**, which bounds
`spec_state_indices` / `num_accepted_tokens` arithmetic. Both are required; neither
subsumes the other.

**Regression** (`bench/int8-g64/test_gdn_gate_gather.py`). Constructs the ordering the
invariant is about — a **prefill row before a speculative row** — because that is when the
two mappings disagree:

```
spec_token_indx     = [7, 8, 9, 10, 11, 12, 13, 14]
non_spec_token_indx = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17]
correct vs pre-fix gather identical? False
mean relative difference: 1.4536    max absolute: 4.6917
```

A **145%** separation means the test is discriminating, not a vacuous pass. It also asserts
both call sites gather.

**Engine-level mixed C2 correctness smoke** (`docs/b1a-mixed-c2-raw.txt`). Comparison
standard stated explicitly rather than "looks fine": speculation is exact under rejection
sampling, so **greedy output must not change when a request is batched with another**.

```
prompt_tokens alone long=9002 short=5
prompt_tokens mixed long=9002 short=5
long  greedy identical (alone vs mixed C2): True
short greedy identical (alone vs mixed C2): True
xid_delta=0
```

## 1B — draft noise independence (#54282)

**Invariant.** Draft proposal noise must not come from a process-global stream.

**What v0.27.1 did.**

```python
q = empty_exponential_noise_like(probs, use_fp64_gumbel)
q.exponential_()          # global default generator, unseeded
# upstream comment: "# TODO(woosuk): Consider seeds."
```

Two real consequences: the draft stream is shared process-wide, so **concurrent requests
perturb each other's draft tokens**; and an unseeded request's drafts are not reproducible.

**Fix.** Draw from a dedicated generator seeded at `DRAFT_NOISE_SALT = 1 << 30`, far above
the request-seed range. The target/residual path is untouched.

### A claim I made and then had to withdraw

I first wrote the patch comment to say the shared stream "biases the accepted-token
distribution". **The measurement does not support that**, so the comment was corrected
before commit. Reading the source carefully shows why:

```python
# rejection_sampler.generate_uniform_probs
uniform_probs[start:end].uniform_(generator=generator)      # per-request SEEDED
# rejection_sampler.sample_recovered_tokens
q[i].exponential_(generator=generator)                       # per-request SEEDED
```

The target and residual paths **already** use per-request seeded generators, so they never
shared a stream with the draft. A 200k-trial chi-square over vocab 16 found **no bias in
either regime** (`bench/int8-g64/test_draft_noise_independence.py`). The honest scope is
therefore the narrower one above: process-global sharing and non-reproducibility.

The test also had a **false positive** of its own: it printed a success message on a
`0.00 > 0.00` comparison. That logic was replaced with a guard that reports
"NO BIAS DETECTED" and explains why the test cannot demonstrate a distributional bias at
all. Both the code comment and the test now state what was actually measured.

### Semantics, formally corrected (review item 6)

An earlier framing — inherited from how upstream #54282 is described — was that the v0.27.1
speculative path suffered a **rejection-distribution bias** because draft and target shared a
Gumbel stream. **That does not apply to this code path, and the claim is withdrawn.**

What the source actually shows, and what the 200k-trial test confirmed:

| party | randomness source in v0.27.1 | consequence |
| --- | --- | --- |
| draft proposal | process-global **unseeded** generator (`q.exponential_()`) | shares one stream process-wide |
| target rejection + residual resampling | **per-request seeded** generators (`generate_uniform_probs`, `sample_recovered_tokens`) | already independent of the draft |

So the two sides were never drawing from the same stream for the same request, and the
chi-square test found **no bias before or after** the change. The backport therefore must
**not** be described as fixing a distributional-correctness P0.

Its actual value, which is real but narrower:

- draft noise comes from a **dedicated stream** rather than a process-global one;
- **seeded reproducibility** of drafts for an unseeded request;
- **isolation between concurrent requests** — one request's draw count can no longer perturb
  another's draft tokens;
- unrelated global RNG traffic cannot change the proposal.

**This patch is complete and will not be extended.** The Phase 1B performance check already
showed −0.58% at M7 126K C1 (inside noise), so there is nothing further to tune.

By contrast **#51812 remains a clear P0 correctness fix**, because it feeds the `a`/`b` gate
of the **wrong token** into the GDN recurrent update — a silent state corruption rather than
a stream-hygiene issue.

### Verification

| check | result |
| --- | --- |
| greedy repeat identical | True |
| greedy seeded == greedy unseeded | True (the greedy path draws no noise, so the patch must be a no-op) |
| temp>0 produces output | True |
| temp>0 seeded repeat identical | True |
| temp>0 differs from greedy | True |
| draft draw independent of global-stream traffic | True |
| control: global stream IS order-dependent | True (so the check is not trivially true) |
| both vLLM installs patched | True |
| Xid delta | 0 |

**Coverage.** The generic draft path and the DFlash2 selector both reach
`compute_probs_and_sample_next_token` (`DFlashProposer` subclasses
`SpecDecodeBaseProposer` and does not override it), so one edit covers both.

### Performance sanity (M7 126K C1, 1350 MHz / 180 W)

| round | ms/spec-iteration | accepted/pass |
| --- | --- | --- |
| 1 | 35.224 | 2.56 |
| 2 | 35.478 | 3.47 |
| 3 | 35.790 | 2.35 |

Median **35.478** vs the B2 Path-A baseline **35.685** = **−0.58%**. Below the 1% threshold,
so per instruction this is not pursued further. `xid_delta = 0`.

Note acceptance swings 2.35–3.47 while `ms/spec-iteration` stays 35.2–35.8: per-iteration
cost and speculative efficiency are separate quantities, which is the whole point of
reporting both.

## Independent revertability, tested rather than asserted

The two patches touch disjoint files. Apply both to a pristine tree, revert **only** 1B,
confirm 1A survives; revert **only** 1A, confirm both files return **byte-identical** to
pristine (`diff -q` clean for both). Verified on the host.

## Deployment note (this cost real risk)

The M7 production path loads `mixed-fp8-test-site` via `PYTHONPATH`, and that tree carries
**its own copies** of both patched files — byte-identical to the runtime's pre-fix versions.
A fix applied only to `runtime-v0271` **does not reach M7**. Both fixes were applied and
verified in **both** trees. See `docs/m7-historical-contract.md`.
