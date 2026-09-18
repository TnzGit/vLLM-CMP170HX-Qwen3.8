# CUDA illegal memory access in vLLM 0.27.1 long-context speculative decoding (SM80 / CMP 170HX)

A self-contained bug report: environment, exact reproduction, every measurement
taken so far, what has been **ruled out with evidence**, and the specific
questions that are still open. Written to be handed to someone who has not seen
the investigation.

---

## 0. RESOLVED — root cause and fix

**Status: fixed and verified.** The sections below are the investigation as it stood
before the cause was known; they are kept because the ruled-out list and the
measurement traps are the useful part of the record.

**Cause.** An **int32 multiply overflow** in `_spec_attn_partial`
(`v1/attention/ops/spec_decode_attn.py`). The kernel builds its K/V addresses as

```python
blk    = tl.load(bt_ptr + req * stride_bt + pos // BLOCK_SIZE, mask=k_ok, other=0)
k_ptrs = k_ptr + blk[:, None] * stride_kb + slot[:, None] * stride_ks + kvh * stride_kh + d[None, :]
k      = tl.load(k_ptrs, mask=k_ok[:, None], other=0.0)          # line 109
```

`blk` is loaded from an **int32** block table and `stride_kb` is a byte stride. This
configuration's KV pool is **5,444,206,592 bytes, i.e. greater than 2\*\*31**, so for
any block id above roughly **2,390 of ~6,059** (~40% of valid ids) the product
`blk * stride_kb` exceeds 2\*\*31, Triton evaluates the multiply in int32, and the
result **wraps negative**. The address then lands *below* the pool.

`compute-sanitizer --tool memcheck` located it, and the arithmetic closes exactly:

```
========= Invalid __global__ read of size 1 bytes
=========     at _spec_attn_partial+0x7160 in spec_decode_attn.py:109
=========     Access to 0x7a8a039af040 is out of bounds
=========     and is 2,086,997,952 bytes before the nearest allocation
              at 0x7a8a80000000 of size 5,444,206,592 bytes

2**31 - 2,086,997,952 = 60,485,696 = 0x39af040   <- the fault address's low bits
```

That is the signature of a signed int32 wrap. The GPU raises Xid 31 / MMU `FAULT_PDE`
on the unmapped read.

**Why it looked like a "long context" bug.** It needs a large pool *and* a high block
id to be referenced, which needs enough KV -- so short contexts and small caches never
reach it. Only the speculative verify kernel does this arithmetic, so a drafter is
required (any drafter: DFlash2 and MTP faulted identically). The block id itself is
**valid**, which is why every host-side guard reported clean.

**Fix** (`patches/spec-decode-attn-int64-block-id.patch`) -- one operand widening,
which is what the GDN path in the same codebase already does, with a comment naming
this exact failure mode:

```python
blk = tl.load(bt_ptr + req * stride_bt + pos // BLOCK_SIZE,
              mask=k_ok, other=0).to(tl.int64)
```

**Verification** (zero Xid 31 *delta* asserted on every leg, because `dmesg` retains
all earlier runs' faults):

| leg | requests | result | Xid delta |
| --- | --- | --- | --- |
| DFlash2 16K (10x the 12-request fault period) | 120 | 120/120 OK | 0 |
| DFlash2 65K | 12 | 12/12 OK | 0 |
| DFlash2 126K | 8 | 8/8 OK | 0 |
| MTP 16K | 24 | 24/24 OK | 0 |
| C4 (4-way) 16K | 32 | 32/32 OK | 0 |
| FULL graph mode 16K | 24 | 24/24 OK | 0 |

The C4 leg previously faulted after ~8 requests; it now completes 32 requests /
524,288 tokens clean. `VLLM_SPEC_DECODE_ATTN=0` is no longer needed as a workaround.

The upstream-relevant summary is two sentences: *`_spec_attn_partial` multiplies an
int32 block id by a byte stride that can exceed 2\*\*31 for a KV pool larger than
2 GiB, and Triton wraps the product negative. `v1/worker/mamba_utils.py` already
widens the same operand for the same reason; this kernel does not.*

---

## 1. Summary

A vLLM 0.27.1 server serving `Qwen3.8-27B-W4A16` + `DFlash2` drafter on an
NVIDIA CMP 170HX (SM 8.0) dies with `CUDA error: an illegal memory access was
encountered` after serving **a period-dependent number of requests** at
long-context sizes. The GPU reports **Xid 31, MMU Fault, `FAULT_PDE
ACCESS_TYPE_VIRT_READ`**.

The fault is **not** a length threshold, **not** deterministic per length, and
**not** a per-request coin flip. It is **periodic in the request count with a
length-dependent period**, and the period also changes when requests are made
concurrent:

| context length | protocol | fault lands at request # | period |
| --- | --- | --- | --- |
| 16,384 | 1 at a time | 12, 24 | 12 |
| 57,344 | 1 at a time | 5, 10 | 5 |
| 65,536 | 1 at a time | 4 (and not at all in another run) | ~4 |
| 16,384 | 4 concurrent | 8, 16 | 8 |

Three axes have been eliminated with matched A/B measurements: **draft depth
`k`**, **async scheduling**, and the **prefix-cache option**. Three different
attention backends / KV dtypes all fault, which points away from the attention
kernels and KV formats and at **shared engine state**.

Mechanism unidentified. That is the question being asked.

---

## 2. Environment

```
GPU          NVIDIA CMP 170HX, 64 GiB, compute capability 8.0 (SM80)
Driver       610.43.02
CUDA         13.0 (toolkit shipped inside the venv under nvidia/cu13)
PyTorch      2.13.0+cu130
Triton       3.7.1
vLLM         0.27.1 (patched; see §7)
Model        Qwen3.8-27B-W4A16-AutoRound-fast (compressed-tensors W4A16, Marlin)
Drafter      Qwen3.8-27B-DFlash2-W4A16 (W4A16, `method=dflash`, k=7)
OS           Ubuntu 24.04.4
```

Clocks are pinned to 1350 MHz and power to 180 W for all measurements.

---

## 3. The configuration that faults

The "control" arm, which is the current `CTX=long` recipe:

```
vllm serve <model> \
  --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8002 \
  --gpu-memory-utilization 0.90 --max-model-len 150000 --max-num-seqs 4 \
  --api-server-count 1 --language-model-only \
  --attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head \
  --mamba-ssm-cache-dtype float16 --async-scheduling \
  --max-num-batched-tokens 2048 \
  --speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prompt-tokens-details
```

Observed at startup: `GPU KV cache size: 1,110,675 tokens` (also 1,137,362 with
`max-model-len=262144`), `max_num_batched_tokens=2048`,
`enable_prefix_caching=False`, `block_size` = vLLM default (not overridden).

An illegal access with this symptom has been observed under **three different
attention/KV configurations**. Be careful how much this proves, because the
strength of the evidence differs per row:

| attention backend | KV dtype | observation | same mechanism? |
| --- | --- | --- | --- |
| `TRITON_ATTN` | `int8_per_token_head` | the periodic fault measured in §6 (control above) | this is the characterised one |
| `TRITON_ATTN` | `int8_g64` (custom) | faulted identically in matched A/B against the control, same trigger profile | very likely the same |
| `FLASH_ATTN` | `bfloat16` | illegal access at C4 long prompts and at 32K, reproduced on a pristine pre-session runtime tree | **unproven** — same symptom, different trigger profile (concurrency-driven), mechanism not verified |

So it is not obviously one backend's kernel and not one KV format. Two things
*are* firmly excluded on the characterised configuration: the custom split-KV
verify kernel (setting `SPEC_ATTN=0`, falling back to stock Triton attention,
still faults) and the split-KV segment cap (`VLLM_SPEC_DECODE_ATTN_QMAX=64`
changes nothing).

---

## 4. Symptom

Engine log (sticky error surfaces at an unrelated later sync):

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
  ... vllm/v1/worker/gpu/model_runner.py line 1702 in shutdown
  ... torch/accelerator/__init__.py line 270 in synchronize
```

The useful evidence is in `dmesg`, not the Python traceback:

```
NVRM: Xid (PCI:0000:01:00): 31, pid=<n>, name=VLLM::EngineCor, channel 0x00000004,
      intr 00000000. MMU Fault: ENGINE GRAPHICS GPC3 GPCCLIENT_T1_0 faulted @
      0x1c_261f9000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ
```

`FAULT_PDE` = the page directory entry for the accessed virtual address is not
present, i.e. a kernel dereferenced an **unmapped** address. Over the
investigation window (17.5 h):

- **63 Xid 31 events**, 9 in the last hour, 29 in the last 6 h, all
  `VLLM::EngineCor`;
- **34 distinct fault addresses**, the most frequent repeating 7×, 5×, 5×, 4×;
- Xid 43 (GPU stopped processing) appears 11× around the same periods.

Note for anyone reading `dmesg`: it is a ring buffer with **632 Xid 13 events**
on this host ("Graphics SM Warp Exception: Out Of Range Address"). Those are
**historical** — last one 33.7 h before the analysis, **zero in the last 6 h** —
and belong to an earlier unrelated session. Do not attribute them to this bug.

---

## 5. Reproduction

Requires the engine above, the model/drafter, and a tokenizer. Requests use
**exact** token lengths (the server's own `usage.prompt_tokens` is asserted
against the requested length) and **distinct content per request**.

```bash
# 16K, 30 requests, sequential, ~15 min on the CMP 170HX
python bench/int8-g64/fault_rate.py \
  --port 8002 --log <server.log> \
  --tokenizer <model dir> \
  --lengths 16384 --repeats 30 --decode-tokens 16 --tag repro

# expected: ok ... fault at request 12, restart, ok ... fault at request 24
```

Cost is dominated by prefill, which is the reason the protocol matters:

| context | prefill | ~s per sample (1-at-a-time) | ~s per sample (4 concurrent) |
| --- | --- | --- | --- |
| 16,384 | ~30 s | 30 | 17.6 |
| 57,344 | ~105 s | 105 | not measured |
| 65,536 | ~121 s | 121 | not measured |

**Critical protocol requirement:** after any fault the engine must be
**restarted**. A crashed engine keeps returning stale failures for every
subsequent request, which will look like a much higher fault rate and will
corrupt any measurement that continues on it.

---

## 6. Measurements

All runs: distinct content per request, exact token count verified, engine
restarted after every fault, clocks pinned.

### 6.1 Period is set by length (sequential, one request at a time)

| length | repeats | faults | fault at request # |
| --- | --- | --- | --- |
| 16,384 | 30 | 2 | **12, 24** |
| 16,384 | 16 | 1 | **12** |
| 49,152 | 5 | 0 | — |
| 57,344 | 10 | 2 | **5, 10** |
| 65,536 | 6 | 1 | 4 |

`16,384 × 12 = 196,608`, `57,344 × 5 = 286,720`, `65,536 × 4 = 262,144`. The
products are the same order of magnitude but **not equal**, and §6.3 shows the
period moves when concurrency changes at a *fixed* length — so "cumulative
tokens" is the right shape of explanation but is not a calibrated constant.

### 6.2 A length that faulted once passed on a later run

| length | run A | run B |
| --- | --- | --- |
| 65,536 | **FAULT** | **OK** |
| 65,531 | FAULT | OK |
| 65,521 | — | FAULT |
| 65,552 | — | FAULT |

This is what killed the "deterministic per-length" model. Single observations at
single lengths are not evidence.

### 6.3 Concurrency changes the period

`16K, 4 requests in flight, 6 rounds`:

```
round 0  ok=4   fault=0   ~65,536 tokens    55 s
round 1  ok=8   fault=0   ~131,072 tokens  111 s
round 2  FAULT (after ~131,072 tokens)
round 3  ok=12  fault=4   ~196,608 tokens  267 s
round 4  ok=16  fault=4   ~262,144 tokens  322 s
round 5  FAULT (after ~262,144 tokens)
```

Faults after **8 and 16** successful requests, against **12 and 24** for the
same length one-at-a-time. The trigger therefore depends on **how the work is
scheduled**, not only on how much of it there is — which argues for live KV
occupancy or block-allocation pattern rather than a simple cumulative counter.

(Caveat: this harness counts every request in a faulted round as a fault, so its
8/24 rate is an over-count. The per-request attribution is not implemented yet.)

### 6.4 Residue at small lengths: no effect

The repo has a documented, separate, historical bug with a prompt-length residue
(`bad residue = 117 + k`, i.e. 124 mod 128 for k=7). At 65,531 tokens,
`65531 mod 128 = 123`, which looked like it might be that family. It is not:
every small length with residue 123 passes.

| length | mod 128 | result |
| --- | --- | --- |
| 123, 251, 379, 507 | 123 | OK |
| 1023, 2047, 4095, 8191, 16383 | 127 | OK |

So residue alone is not sufficient — a large length is also required.

---

## 7. What has been ruled out (matched A/B, one variable at a time)

| axis | change | result | conclusion |
| --- | --- | --- | --- |
| **draft depth `k`** | 3 vs 5 vs 7, 16K, 16 req | all **1/16, fault at request 12** | k is not a factor; exonerates speculative verify geometry and the `117+k` residue series for *this* bug |
| **async scheduling** | `ASYNC_SCHED=0` vs `1`, 56K, 10 req | **identical**: both 2/10, both at requests 5 and 10 | not an async-scheduling race |
| **prefix cache** | `PREFIX_CACHE=0` vs `1` (adds `--enable-prefix-caching --mamba-cache-mode align`), 16K, 16 req | **identical**: both 1/16, both at request 12 | that option path is not the trigger |
| **KV dtype / attention backend** | int8_g64 vs int8_per_token_head (matched A/B) | both fault with the same trigger profile | not a KV-format defect; the bf16/FLASH_ATTN arm also faults but with a different trigger profile and its mechanism is unverified (see §3) |
| **split-KV verify kernel** | `SPEC_ATTN=0` | still faults | not the custom verify kernel |
| **split-KV segment cap** | `VLLM_SPEC_DECODE_ATTN_QMAX=64` | no change | not the segment cap |
| **launch synchronisation** | `CUDA_LAUNCH_BLOCKING=1` present vs absent | no change | timing-only; fault is not masked by it |
| **GDN accept-bound patch** | `vllm-pr50021-gdn-spec-bounds.patch` | **already applied** in this runtime | this is not a missing upstream GDN bound fix |

Repo patches present in the runtime that are relevant to the implicated area:
`hybrid-kv-groups-v2-cudagraph.patch`, `hybrid-sw-block-promote.patch`,
`spec-decode-attn.patch`, `spec-decode-int8-kv.patch`,
`vllm-pr50021-gdn-spec-bounds.patch`, `dflash2-backport.patch`,
`dflash2-lookup-drafting.patch`.

---

## 8. Current hypotheses (not yet tested)

Ranked by how well they fit "periodic in request count, period depends on length
**and** on concurrency, `FAULT_PDE` on a virtual read":

1. **Live KV block occupancy / block-pool recycling.** The only candidate that
   naturally scales with cumulative work *and* changes when requests overlap.
   A freed-and-reused block whose slot mapping or block table is stale would
   produce exactly a virtual read of an unmapped address, at a rate set by how
   many blocks have been recycled.
2. **CUDA graph retained state across replays.** `--compilation-config` captures
   with `max_cudagraph_capture_size=32`; state owned by a captured graph that is
   reallocated or freed between captures/replays leaves stale pointers. Not yet
   separated from (1) — the `FULL` / `PIECEWISE` / `eager` comparison has not
   been run.
3. **GDN / Mamba recurrent-state bookkeeping under speculation.** The repo
   already needed `vllm-pr50021-gdn-spec-bounds.patch` for a defect in this area,
   so a second defect is plausible. Not discriminated yet: the
   `DFlash2` vs `MTP` vs `none` comparison has not been run.
4. **A request-local counter that is only reset on some boundary** (e.g. a
   per-engine or per-block-pool refcount), which would explain a period in
   *requests* rather than in tokens.

---

## 9. What has been attempted but not completed

- **Compute Sanitizer.** `nvidia-cuda-sanitizer-api` (2026.3.0) is now installed
  on the host, so `compute-sanitizer --tool memcheck` is available. It has **not
  yet produced a report**: the one attempt failed at engine init because a
  leftover `VLLM::EngineCore` held ~58 GiB, not because of any sanitizer
  incompatibility. Since the fault is periodic rather than deterministic, a
  sanitizer run may not reproduce it at all (memcheck serialises execution),
  which is itself a consideration.
- **Nsight Compute** is installed; Nsight Systems is not.
- **Pre-fault state dump.** The obvious next diagnostic — capture
  `seq_len / positions / slot_mapping max / block_table tail / num_spec_tokens /
  accepted tokens / scheduled draft ids / workspace shape / graph desc` for the
  step immediately before a fault — has **not** been done.

---

## 10. Measurement traps found the hard way (please don't repeat these)

These cost real time and produced three wrong conclusions before being caught:

1. **`dmesg` is not session-scoped.** 632 Xid 13 events on this host are
   historical. Check timestamps against uptime before attributing anything.
2. **A single observation at a single length proves nothing** here. The fault
   rate is roughly 1-in-6 to 1-in-12 per request, so `OK` and `FAULT` at the same
   length on different runs are both expected. Measure a **rate**, or exploit the
   known **period**, never a boundary.
3. **A crashed engine poisons everything after it.** Every subsequent request
   returns stale failures. Always restart before the next sample.
4. **Whole-request throughput mixes prefill and decode.** Isolate them; at 64K
   the prefill alone is ~121 s and will dominate any aggregate number.
5. **`ms per output token` is not `ms per verify step`.** DFlash2 accepts ~3.3-3.4
   tokens per pass, so a per-step figure must be derived (and labelled as
   derived) from acceptance, not read off a wall-clock subtraction.
6. **A client-side `torch.profiler` cannot see a server process's kernels.** It
   records nothing useful. Profile inside the engine worker, or use `nsys`.
7. **`pkill` on an engine's `EngineCore` is not how to stop a unit**; it leaves
   the parent alive and the GPU re-occupied. Use `systemctl --user stop`, and
   check both `systemctl --user show <unit> -p ActiveState` **and**
   `nvidia-smi --query-compute-apps` before assuming the box is idle.
8. **Uncommitted harnesses.** Twice, a result was written down while the script
   that produced it existed only on the lab host. Every number in §6 comes from
   `bench/int8-g64/{fault_rate,fault_rate_concurrent,exact_residue_sweep}.py`,
   which are committed.

---

## 11. Specific questions for the reader

1. Is a **periodic** illegal access with period set by *context length* and by
   *concurrency* more consistent with a stale-block/slot-mapping defect, a
   CUDA-graph retained-state defect, or something else? What would discriminate
   them most cheaply?
2. `FAULT_PDE ACCESS_TYPE_VIRT_READ` on a **repeating** address (7×, 5×, 5× the
   same few addresses across separate engine processes). Does the repetition of
   the same virtual address across *processes* tell us anything specific — e.g.
   a fixed-size allocation whose offset is deterministic rather than a random
   wild pointer?
3. Given that **three** attention backends and KV dtypes all fault, is there a
   shared allocation that all three paths still use — KV block pool, block
   table, slot mapping, GDN state, spec-decode buffers, or the CUDA graph's
   static input buffers — that would be the natural first place to instrument?
4. Is there a known vLLM 0.27.1 issue matching this shape? Upstream reports that
   look related but may or may not be the same: DFlash2 deterministic cumulative
   OOB on sm_80/v0.27.1 (engine dies after many decode steps), and hybrid GDN +
   MTP + async-scheduling IMA. Note `ASYNC_SCHED` was ruled out here.
5. What is the most informative **single** next experiment? The current
   shortlist is (a) `FULL` / `PIECEWISE` / `eager`, (b) `DFlash2` vs `MTP` vs
   `none`, (c) the pre-fault state dump, (d) memcheck on a reproducing run.
6. Is there a plausible mechanism by which the period would be **12 requests at
   16K** but **5 at 57K** and **8 at 16K with 4-way concurrency**? A formula that
   fits those three points would be very convincing.

---

## 12. Artifacts

In `bench/int8-g64/` of this repository:

| file | purpose |
| --- | --- |
| `fault_rate.py` | sequential rate/period measurement, exact tokens, restarts on fault |
| `fault_rate_concurrent.py` | same with N requests in flight (cheaper per sample) |
| `exact_residue_sweep.py` | exact-token sweep with a `--k-residues` fingerprint mode |
| `spec_bench.py` | separates ms/output-token, ms/target-pass, accepted/pass |
| `marlin_shape_bench.py` | single-kernel W4A16 vs W4A8-INT8 A/B on the hot shapes |

Handover sections 25-32 in `handover.md` record the investigation including the
three retracted models (length threshold, discrete 2^16, residue 123).
