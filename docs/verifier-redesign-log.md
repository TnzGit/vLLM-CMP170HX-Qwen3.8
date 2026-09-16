# CMP 170HX mixed-FP8 verifier redesign log

This is the append-only handover log for structural verifier work on
`work/cmp170hx-mixed-fp8`.  Every milestone must leave enough evidence for a
new agent to reproduce, reject, or continue the experiment without relying on
chat history.

## Milestone V0 — freeze the qualified baseline

**Date:** 2026-09-16  
**Parent commit:** `333f651`  
**Runtime:** vLLM 0.27.1, PyTorch 2.13.0, CUDA 13, SM80 CMP 170HX 64 GiB  
**Service:** `cmp170hx-mixed-fp8-full-256k-8002.service`  
**Power:** 180 W peak profile; 175 W separately qualified  
**Model:** Qwen3.8-27B W4A16 target + DFlash2 W4A16, seven draft tokens  
**Cache:** FP8 E4M3 target KV + BF16 draft KV, 896/448-token logical pages  
**Graph:** FULL CUDA Graph  
**Concurrency/capacity:** C4 / 262144 tokens  
**Verifier:** q<=8, GQA=6 specialization; 35 segments; 32-token KV tiles

### Objective

Freeze the exact code and measurement boundary before structural changes.  V0
does not change code generation or runtime behavior.

### Actual runtime entry points

The deployed test-site source is:

```text
vllm/v1/attention/backends/flash_attn.py
  _spec_attn_run()

vllm/v1/attention/ops/spec_decode_attn.py
  _spec_attn_partial_q8_g6_fp8()
  _spec_attn_combine()
  SpecDecodeAttention.run()
```

The repository representation of the active extension is the ordered series in
`experimental/cmp170hx-mixed-fp8/series`.  Do not patch only the remote
test-site and leave the series behind.

### Qualified kernel structure

For the production q<=8/GQA=6 case, one CTA owns one
`(request, kv_head, segment)` and computes all 48 useful query rows as 32+16.
Each 32-token K/V tile is loaded once and reused by both row groups.  Splitting
query rows across CTAs would duplicate the dominant long-context K/V traffic and
is not an acceptable first redesign.

The kernel retains FP32 online-softmax maxima/normalizers, FP16 running output,
read-only-cache E4M3FN LUT loads, page-local block-table carry, and int64
physical block IDs.  Partial outputs use graph-stable `part_o/part_m/part_l`
buffers and a separate segment combine kernel.

### Frozen performance and profiler boundary

The deterministic exact-token harness reports the following 180 W step
latencies.  Decode tok/s is acceptance-sensitive; `ms/step` is the primary
structural metric.

| input | qualified ms/step | representative decode tok/s |
|---:|---:|---:|
| 4K | 22.315 | ~161 |
| 126K | 35.230 | ~98 |
| 250K | 46.941 | ~70 |

At 126K, a shape-aware CUDA profile assigned 30.426 ms of GPU kernel time per
step as follows:

| component | ms/step | share |
|---|---:|---:|
| FP8 verifier partials | 12.847 | 42.2% |
| target Marlin GEMMs | 13.219 | 43.4% |
| GatedDeltaNet | 1.120 | 3.7% |
| combine | 0.123 | 0.4% |
| other | ~3.12 | 10.3% |

The verifier partial kernel used about 250 registers/thread, no local spill,
43,008 bytes of shared memory, about 284 GB/s measured memory throughput, and
54.75% no-eligible-warp cycles.  The grid is 35 segments x 4 KV heads = 140
CTAs, matching 70 SMs x two resident CTAs.

### Correctness gates carried forward

Every candidate must preserve:

- exact 256-code E4M3FN semantics, with the two NaN encodings fail-closed to 0;
- explicit-dequantization agreement for q=5/8/16/64;
- 895/896/897 boundaries with `--block-size 896`;
- int64 addressing before physical-block stride multiplication;
- mixed KV lengths and high physical block IDs;
- graph-stable workspace addresses;
- zero preemptions, CUDA errors and new Xid/NVRM events in qualified runtime A/B.

Known test gaps before claiming production readiness are mixed query lengths,
q=6/7 special-path edges, a production-896 high-block-ID case, and automated
CUDA Graph capture/replay parity.  These should be closed alongside the first
candidate that survives isolated performance testing.

### V1 decision

The first experiment is a source-controlled q8/GQA6 pipeline variant with
Triton `num_stages=2` and, only if viable, `num_stages=3`.  It must keep cache
layout, tile size, grid, workspace, accumulation and global K/V byte traffic
unchanged.  Merely setting `num_stages` is not evidence of pipelining: generated
code must be inspected for `cp.async` or an equivalent overlapped load schedule.

V1 stops and is rejected if any of the following occurs:

- K/V global bytes increase;
- local spill appears or registers/shared memory prevent two resident CTAs;
- 126K and 250K isolated-kernel gain is below about 5%;
- 4K whole-model step latency regresses by more than 2%;
- correctness, graph replay, preemption or kernel-log gates fail.

### Handover command boundary

Before changing runtime code, confirm:

```bash
git switch work/cmp170hx-mixed-fp8
git status --short --branch
systemctl --user is-active cmp170hx-mixed-fp8-full-256k-8002.service
curl -fsS http://127.0.0.1:8002/health
```

The active 8002 service is the reference baseline.  Candidate code must be
installed in a separate test-site or guarded by a candidate-only environment
variable, and the baseline path must remain recoverable without rebuilding.

