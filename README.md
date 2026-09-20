# Qwen3.8-27B on NVIDIA CMP 170HX (SM80, 64 GiB)

**English** | [简体中文](README.zh-CN.md)

A vLLM 0.27.1 research-and-production fork for running **Qwen3.8-27B + DFlash2 speculative decoding on a single NVIDIA CMP 170HX**.

The current mainline performance path is the **M7 mixed-FP8 verifier**: FP8 target KV, BF16 draft KV, a custom SM80 split-KV speculative verifier, heterogeneous target/draft cache geometry, and FULL CUDA Graph execution.

> **Project status (2026-09-20):** the M7 path has been reconstructed, replayed against its historical freeze, compared against the repaired INT8 alternate path, and promoted to the recommended long-context production path for this hardware/runtime. The large research ledger is archived in PR #1; only the qualified slices are merged into `main`.

This README documents the **CMP 170HX path**. The repository still contains inherited RTX 3090 / batch / KVarN / Docker assets from the upstream project; those remain useful as alternate experiments, but they are **not** the configuration behind the results below.

---

## TL;DR

For this exact stack:

| profile | context | DFlash `k` | `MAX_SEQS` | effective attention block | use when |
|---|---:|---:|---:|---:|---|
| **short** | ≤ 32K | **5** | **4** | 816 | short/medium interactive requests where acceptance is high |
| **long** | ≥ 48K | **3** | **1** | 800 | long-context / 126K / 250K service |

The crossover is around **48K**. Do **not** implement an in-engine dynamic-`k` switch: `k` changes the derived cache/page geometry (800/816/832 for k=3/5/7), so it is an engine-level service profile in this fork.

For 126K and 250K, `MAX_SEQS=1` is not merely a latency preference. In measured service A/B it was better than 2 on **every measured axis**, including total makespan.

---

## Hardware and software contract

The qualified numbers in this repository are for:

| component | value |
|---|---|
| GPU | **NVIDIA CMP 170HX**, SM80, 64 GiB |
| formal benchmark clock | **1350 MHz locked** |
| power limit | **180 W** |
| vLLM | **0.27.1** |
| PyTorch | **2.13** |
| CUDA | **13.x** |
| Triton | **3.7.1** |
| target model used for final qualification | `Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4` |
| drafter | Qwen3.8 DFlash2 W4A16 |
| target attention/KV | FlashInfer + **FP8 E4M3 KV** |
| draft attention/KV | FlashAttention + **BF16 KV** |
| speculative verifier | custom SM80 split-KV, q≤8 / GQA6 / D256 |
| verifier segmentation | **NSEG=35** → 140 CTAs across 140 SMs |
| graph mode | **FULL** |
| maximum qualified model length | **262144** |

The exact checkpoint matters. Performance and acceptance results should not be compared across different target checkpoints as if they were path-only measurements.

---

## Why this fork exists

The mainline vLLM 0.27.1 path was not enough for this card/workload:

- Qwen3.8 combines **16 full-attention layers + 48 GDN layers**.
- The target uses 24 query heads, 4 KV heads, GQA=6, head dimension 256.
- Long-context speculative verification repeatedly scans a very large paged KV cache.
- A 64 GiB KV allocation can exceed 2 GiB address arithmetic even while the block table itself remains int32.
- The best-performing target and draft KV formats are different, so the allocator must support heterogeneous attention page geometry.
- The verifier must remain graph-stable and fast for q≤8 rather than behave like a generic dense-attention backend.

The M7 path is the result of turning those hardware/runtime constraints into an explicit SM80 execution path rather than treating CMP 170HX as a generic CUDA device.

---

## Current production architecture

```text
Qwen3.8-27B W4A16 target
        |
        |  FlashInfer target attention
        |  FP8 E4M3 target KV cache
        v
custom split-KV speculative verifier (SM80)
        |
        |  q <= 8
        |  Hq=24 / Hkv=4 / GQA=6 / D=256
        |  NSEG=35
        |  FP8 LUT decode
        |  page-local block-table carry
        |  int64 physical-block addressing before stride multiplication
        v
DFlash2 W4A16 drafter
        |
        |  FlashAttention
        |  BF16 draft KV
        v
FULL CUDA Graph replay
```

The directory name `experimental/cmp170hx-mixed-fp8/` is retained for history. The M0→M7 series inside it is now the **qualified production performance path in this fork**.

---

## Installation

### 1. Create the vLLM 0.27.1 environment

The repository patches are pinned to vLLM 0.27.1.

A convenient dependency set is the one used by the repository Docker build:

```bash
git clone https://github.com/TnzGit/vLLM-CMP170HX-Qwen3.8
cd vLLM-CMP170HX-Qwen3.8

python3.12 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r docker/requirements.txt
```

Confirm:

```bash
venv/bin/python - <<'PY'
import vllm, torch, triton
print("vllm", vllm.__version__)
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

The intended result is vLLM 0.27.1 and an SM80 GPU.

### 2. Apply the normal patch stack

```bash
SP=$(venv/bin/python - <<'PY'
import inspect, pathlib, vllm
print(pathlib.Path(inspect.getfile(vllm)).resolve().parent)
PY
)

for p in patches/*.patch; do
    echo "== $p"
    patch -p1 -d "$SP" < "$p"
done
```

This stack now includes the important correctness hardening merged during the final qualification:

- int64 physical block addressing in the split-KV verifier;
- Qwen GDN gate/token alignment backport (#51812);
- dedicated draft RNG stream / reproducibility hardening (#54282 invariant).

### 3. Apply the CMP 170HX mixed-FP8 M0→M7 series

First dry-run it:

```bash
bash experimental/cmp170hx-mixed-fp8/install.sh --dry-run
```

Then apply:

```bash
bash experimental/cmp170hx-mixed-fp8/install.sh --apply
```

Audit the installed package:

```bash
bash scripts/audit-mixed-fp8-prereqs.sh
bash experimental/cmp170hx-mixed-fp8/install.sh --check
```

The series is listed explicitly in:

```text
experimental/cmp170hx-mixed-fp8/series
```

### 4. Provide the target and draft checkpoints

Set host-specific paths:

```bash
export MODEL=/path/to/Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4
export DRAFT=/path/to/Qwen3.8-27B-DFlash2-W4A16
```

The final numbers in this README use those checkpoint families. If you substitute another W4A16 target, treat it as a new qualification point: acceptance can change even when kernel cost does not.

---

## Launching the qualified profiles

The common mixed-FP8 contract is:

```bash
export SPEC=dflash2
export CTX=cmp-mixed-fp8

export CMP_TARGET_ATTN_BACKEND=FLASHINFER
export CMP_TARGET_KV_DTYPE=fp8
export DFLASH_ATTN_BACKEND=FLASH_ATTN
export DFLASH_KV_CACHE_DTYPE=bfloat16

export SPEC_ATTN=1
export VLLM_SPEC_DECODE_ATTN_SEGMENTS=35
export VLLM_FP8_SPEC_VERIFY=1
export VLLM_FP8_SPEC_FULL_CG=1
export VLLM_ALIGN_HETEROGENEOUS_ATTN_PAGES=1
export CUDAGRAPH_MODE=FULL

export LOOKUP=0
export PREFIX_CACHE=1
```

For benchmark reproduction on the CMP host:

```bash
sudo nvidia-smi -pl 180
sudo nvidia-smi -lgc 1350,1350
```

### Short profile — ≤32K

```bash
export MAX_LEN=32768
export DFLASH_MAX_LEN=$MAX_LEN
export DFLASH_TOKENS=5
export MAX_SEQS=4
export PORT=8000

bash single-user/start_qwen.sh
```

### Long profile — ≥48K, including 126K/250K

```bash
export MAX_LEN=262144
export DFLASH_MAX_LEN=$MAX_LEN
export DFLASH_TOKENS=3
export MAX_SEQS=1
export PORT=8000

bash single-user/start_qwen.sh
```

For research/qualification, this project conventionally used port **8002** so the production service on 8000 was never touched.

### Single-profile fallback

If operational simplicity matters more than extracting the last few percent on short prompts:

```text
k=3
MAX_SEQS=1
```

is the conservative single-profile choice. It gives up the workload-dependent short-context acceptance advantage (about 8% at 4K in the measured corpus) but is structurally cheapest per speculative iteration at every measured context.

---

## Measured performance

### M7 historical replay

The historical M7 freeze was replayed using the reconstructed historical protocol (180 W, no forced clock lock, FULL graph, NSEG=35):

| context | frozen M7 ms/spec-iteration | reconstructed | delta |
|---:|---:|---:|---:|
| 4K | 22.315 | **22.254** | **-0.27%** |
| 126K | 35.230 | **35.717** | **+1.38%** |
| 250K | 46.941 | **48.329** | **+2.96%** |

This is close enough to treat the original M7 result as reproduced. The remaining 1–3% long-context drift is not interpreted as evidence that the historical numbers were wrong.

### M7 vs repaired INT8 path

Same checkpoint, same prompt IDs, FULL graph, 1350 MHz / 180 W:

| context | M7 mixed-FP8 ms/iter | repaired INT8 ms/iter | INT8 penalty |
|---:|---:|---:|---:|
| 4K | **22.306** | 24.065 | +7.9% |
| 65K | **29.258** | 46.438 | +58.7% |
| 126K | **35.685** | 68.827 | +92.9% |
| 250K | **48.292** | 115.408 | +139.0% |

The acceptance rates are close except at 250K, so the widening gap is principally a **target verifier iteration-cost** effect, not a drafter-efficiency story.

The repaired INT8 path remains useful as a correctness-complete alternate/fallback, but it is not the long-context performance default.

### Production k crossover

C1, fresh engine per cell, exact-token unique corpus, 1350 MHz / 180 W:

| context | k=3 tok/s | k=5 tok/s | k=7 tok/s | production choice |
|---:|---:|---:|---:|---|
| 4K | 123.49 | **134.68** | 129.67 | **k=5** |
| 32K | 134.64 | **147.02** | 137.62 | **k=5** |
| 48K | **91.68** | 91.57 | 88.58 | **k=3** |
| 65K | **100.05** | 90.64 | 94.56 | **k=3** |
| 126K | 89.55 | 88.18 | **90.21** | k=3 operationally; k=7 lead is only 0.7%, inside spread |
| 250K | **64.90** | 58.68 | 59.69 | **k=3** |

Important: output tok/s is acceptance-dependent. The acceptance-insensitive structural metric, `ms/spec-iteration`, favors **k=3 at every context**. k=5 wins the short cells only because its higher acceptance more than pays for its more expensive pass.

Full data and caveats: [docs/production-service-profile.md](docs/production-service-profile.md).

---

## Long-context concurrency policy

A major result of the qualification is that **more admitted long-context requests are worse on this single GPU**.

Service-policy A/B:

| context | policy | makespan | useful aggregate | first completion | slowest request |
|---:|---|---:|---:|---:|---:|
| 126K | **MAX_SEQS=1** | **215.8 s** | **4.75 tok/s** | **107.6 s** | **84.2 tok/s** |
| 126K | MAX_SEQS=2 | 219.8 s | 4.66 tok/s | 218.4 s | 4.4 tok/s |
| 250K | **MAX_SEQS=1** | **561.9 s** | **1.82 tok/s** | **278.1 s** | **68.7 tok/s** |
| 250K | MAX_SEQS=2 | 589.8 s | 1.74 tok/s | 589.8 s | 1.7 tok/s |

At both 126K and 250K, `MAX_SEQS=2` is worse even on total makespan. A single long request already saturates the GPU; admitting a second ultra-long prefill mostly introduces interference and consumes KV headroom.

The prefill token budget is **not** the lever here: a 4× sweep of the budget changed steady-state ITL by only ~0.4%.

At 250K the engine can physically host at most **two resident sequences** in the measured configuration, but production still uses **one**, because two performs worse.

See [docs/maxseqs-service-policy-ab.md](docs/maxseqs-service-policy-ab.md).

---

## M0 → M7 optimization history

The successful performance chain is intentionally preserved as original commits:

| milestone | main change | 4K ms/pass | 126K ms/pass | 250K ms/pass |
|---|---|---:|---:|---:|
| M0 | mixed-FP8 qualified baseline, 16 segments | 23.100 | 62.800 | 103.300 |
| M1 | 16 → 32 verifier segments | 23.000 | 50.000 | 77.500 |
| M2 | E4M3FN → BF16 LUT decode | 22.500 | 41.700 | 61.200 |
| M3 | reduce spill / repeated block-ID work | ~22.6 | ~38.4 | 54.200 |
| M4 | q≤8 / GQA6 specialization | 22.500 | ~37.2 | ~51.5 |
| M5 | page-local block-table carry | 22.500 | 37.200 | 51.600 |
| M6 | read-only-cache LUT | 22.500 | 36.300 | 50.000 |
| **M7** | **NSEG=35 / 140-CTA resident-wave alignment** | **22.315** | **35.230** | **46.941** |

The foundation and M1–M7 were merged as PRs **#2** and **#3** without squashing the research commits.

---

## Correctness fixes that matter

### 1. Large-KV int32 address overflow

The long-context illegal-memory-access root cause was:

```python
blk = tl.load(...)              # int32 block id
k_ptr = k_ptr + blk * stride_kb # may overflow int32 before pointer arithmetic
```

With a ~5.44 GiB KV pool and ~898 KiB block stride, a valid physical block id can make `blk * stride_kb` exceed signed 32-bit range and wrap negative.

Fix:

```python
blk = tl.load(...).to(tl.int64)
```

The address arithmetic closed exactly against the observed Compute Sanitizer fault displacement.

Post-fix qualification passed with zero new Xid 31 events across:

- DFlash2 16K: 120/120
- DFlash2 65K: 12/12
- DFlash2 126K: 8/8
- MTP 16K: 24/24
- C4 16K: 32/32
- FULL graph 16K: 24/24

A paired 4K ABBA measured only **+0.64% C1 / +0.27% C2**, below the material-regression gate.

### 2. Qwen GDN gate/token alignment (#51812)

vLLM 0.27.1 gathered Q/K/V rows for mixed speculative batches but could pass the original, unpermuted GDN gate rows `a/b` into the recurrent update.

That does not necessarily crash; it can silently update the wrong recurrent state.

The backport gathers `a/b` with the same `spec_token_indx` / `non_spec_token_indx` mapping as Q/K/V. It is merged in `main` as a P0 correctness hardening.

### 3. Draft RNG isolation (#54282 invariant)

The v0.27.1 draft proposal path used the process-global default RNG for exponential noise. This makes proposal draws sensitive to unrelated/concurrent global RNG traffic.

The backport gives draft proposals a dedicated generator.

**Scope correction:** in this v0.27.1 path, target/rejection/residual sampling already used per-request seeded generators, and a 200k-trial test found no shared-noise distribution bias. This patch is therefore **reproducibility / isolation hardening**, not evidence that the old path produced a biased target distribution.

---

## Greedy-equivalence caveat

Cross-k byte identity proves consistency **inside this speculative path**; it does not prove that a block-shaped speculative verification forward is token-exact with an ordinary target-only q_len=1 greedy forward.

Upstream vLLM issue [#54928](https://github.com/vllm-project/vllm/issues/54928) reports Qwen3.8 examples where both paths are deterministic but diverge near a logit tie because the multi-position verifier forward and single-token target forward choose different argmax tokens.

This repository has **not** established that the final W4A16 + M7 CMP170HX stack exhibits the same behavior. Therefore any future claim of target-only greedy equivalence must compare, on the exact production checkpoint/runtime:

- target-only emitted token;
- verifier target-logit argmax;
- speculative emitted token;
- target-only top-1/top-2 margin around the first divergence.

Do not diagnose such a divergence as RNG or cache corruption without that evidence.

---

## Profiling status and why the current optimization cycle is closed

Fresh current-head attribution:

| component | 126K | 250K |
|---|---:|---:|
| target Marlin GEMMs | ~48.8% | ~36.4% |
| verifier | ~35.9% | **~52.2%** |
| remaining kernels | balance | balance |

The verifier is still the largest 250K target, but the cheap structural ideas have been exhausted:

- tensor-core softmax denominator: **11–14% slower**;
- two-level FP32 PV accumulation: killed before implementation by the register gate;
- production verifier: **252 registers/thread, 0 spill**, exactly 2 resident CTAs/SM;
- projected FP32 long-term accumulator: ~300 regs/thread > SM80 limit;
- register-release audit found no credible ≥16-reg release without introducing a new memory/dataflow cost;
- fused DFlash2 grouped convolution is too small a fraction of the step to matter;
- prior E21–E42 experiments already cover many additional verifier designs.

A hypothetical 3% verifier win at 250K is only about **1.5% whole-step**. Further verifier work now requires a structural rewrite rather than another local tweak.

The current recommendation is therefore to **ship the qualified M7 path instead of continuing micro-optimization**.

---

## Benchmark rules

The final qualification uncovered the same measurement defect three separate times: a ratio whose numerator and denominator covered different time intervals.

The permanent rules are in [docs/benchmark-rules.md](docs/benchmark-rules.md). The most important ones are:

1. Every ratio must state the interval covered by numerator and denominator.
2. If those intervals do not match, report `null` or an explicit bound — not a plausible estimate.
3. Concurrent decode must use a true steady-state window:
   ```text
   steady_start = max(first_output_i)
   steady_end   = min(finish_i)
   ```
4. Global server counters must not be divided by per-request time windows.
5. Runtime facts come from the **parsed/running engine**, not from launcher intent.
6. One meaningful context/config cell → fresh engine.
7. Exact-token, unique prompts for formal long-context work.
8. Report Xid **delta**, not absolute dmesg counts.
9. A crashed engine is poisoned; restart before measuring again.
10. Standalone microbench, kernel attribution, C1 decode throughput, and service-policy measurements are different categories and must stay labeled separately.

---

## Repository map

| path | purpose |
|---|---|
| `patches/` | base vLLM 0.27.1 patch stack, including final correctness hardening |
| `experimental/cmp170hx-mixed-fp8/` | qualified CMP mixed-FP8 M0→M7 series; name retained for history |
| `single-user/start_qwen.sh` | launcher used by the CMP profiles |
| `scripts/audit-mixed-fp8-prereqs.sh` | targeted base/install contract audit |
| `deploy/` | systemd/env examples from qualification runs |
| `bench/` | kernel, path, concurrency, k-sweep and regression harnesses |
| `docs/production-service-profile.md` | authoritative measured service-profile decision |
| `docs/benchmark-rules.md` | permanent benchmark methodology |
| `docs/cmp170hx-mixed-fp8-engineering.md` | mixed-FP8 engineering notes |
| `docs/gotchas.md` | implementation/runtime gotchas |
| `handover.md` | full chronological investigation record |
| `batch/`, `kvarn/`, Docker assets | inherited/alternate paths; not the primary CMP M7 production recommendation |

The `deploy/cmp170hx-mixed-fp8-*.service.example` files are qualification snapshots. Some intentionally retain historical k/max-seq values used in those runs. For current production policy, use **docs/production-service-profile.md** and the profile values in this README.

---

## GitHub history / closure

The research was deliberately kept separate from the production history:

- **PR #1** — closed, archived full research ledger; **do not merge**.
- **PR #2** — merged mixed-FP8 foundation / M0.
- **PR #3** — merged M1–M7 successful optimization series with original commit history preserved.
- **PR #4** — closed as superseded; its int64 widening is already contained in #2.
- **PR #5** — merged Qwen GDN gate/token correctness backport.
- **PR #6** — merged draft RNG isolation / reproducibility hardening.
- **PR #7** — merged production profiles and benchmark rules.

Negative post-M7 research remains documented rather than deleted so the same paths are not rediscovered and re-run later.

---

## Known boundaries

- The benchmarked production path is **hardware-specific**: exact SM80 CMP 170HX behavior matters.
- The final profile numbers are **checkpoint- and workload-dependent**, especially DFlash acceptance.
- `k=5` short-context advantage is acceptance-driven; `k=3` is structurally cheaper per verify iteration everywhere.
- 250K C4 is not a valid four-way steady-decode benchmark on the measured pool.
- Long-context `MAX_SEQS=1` is an evidence-based service policy for this single-GPU stack, not a generic vLLM recommendation.
- The Docker image in this repository applies the inherited normal patch stack and KVarN path, but it does **not** automatically apply the CMP mixed-FP8 M0→M7 series. For the qualified CMP path, use the host/venv install procedure above or extend the image deliberately.
- Do not assume a launcher/environment file proves the runtime path. Verify the final engine config and, when necessary, `/proc/<engine-pid>/environ`.

---

## Further reading

- [Production service profile](docs/production-service-profile.md)
- [Benchmark rules](docs/benchmark-rules.md)
- [Long-context MAX_SEQS A/B](docs/maxseqs-service-policy-ab.md)
- [CMP mixed-FP8 engineering notes](docs/cmp170hx-mixed-fp8-engineering.md)
- [Gotchas](docs/gotchas.md)
- [Full handover / research chronology](handover.md)

---

## License and upstream lineage

This repository retains the original project's code, scripts and license while adding the CMP 170HX / Qwen3.8 specialization described above.

The focus of this fork is reproducible engineering on a pinned vLLM 0.27.1 stack. It is intentionally **not** a claim that these hardware-specific patches should replace current upstream vLLM behavior on other GPUs.
