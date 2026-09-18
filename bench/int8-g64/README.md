# INT8-G64 integration and its A/B (frozen)

Everything needed to reproduce the INT8-G64 work recorded in `handover.md`
sections 19-26 and `docs/verifier-optimization-handoff-2026-09-16.md` E41/E42,
plus the measurements that **froze** it.

## Status: frozen as a research result, not a production candidate

The E42 kill gate was measured and **failed**. Against the production control
(`TRITON_ATTN` + `int8_per_token_head`, `SPEC_ATTN=1`, DFlash2 k=7), decode at
matched settings, 191 steps, fresh engine per context, ms/step (so DFlash
acceptance cannot confound it):

| ctx | control | INT8-G64 | G64 vs control |
| --- | --- | --- | --- |
| 4,096 | **6.120 ms** | 7.339 ms | **-19.9%** |
| 16,384 | **8.161 ms** | 9.450 ms | **-15.8%** |
| 32,768 | **10.200 ms** | 11.292 ms | **-10.7%** |
| 65,000 | **12.673 ms** | 13.932 ms | **-9.9%** |

The gap narrows with context but never crosses. The reason is structural: G64's
KV bytes are 2,112 B/token/layer against the control's 2,048, i.e. **3.1% larger**,
so G64's only possible advantage is its compute path (native s8 MMA instead of
FP8 -> BF16 decode -> BF16 MMA), and that does not pay off at any context this
allocator can serve. The planned common-page round-up (E42: `14 x 135,168 =
1,892,352 B`, 896-token G64 pages) was therefore **not** implemented.

Quality was fine (9/12 prompts byte-identical against bf16, 78.73% shared token
prefix, mean `|dlogprob|` 0.00385) — the freeze is a performance decision, not a
correctness one.

## Layout

| file | role |
| --- | --- |
| `int8_g64.py` | the bridge; installed as `vllm/v1/attention/ops/int8_g64.py` |
| `int8-g64-vllm.patch` | adds the `int8_g64` cache dtype + KV-cache spec |
| `int8-g64-triton.patch` | routes prefill, fixes graph-safety, E38 gate, debug counter |
| `page_stride_adapter.cu` | stride-aware E38 decode adapter (the deployed one) |
| `page_stride_kernel.cuh`, `paged_kv_address_strided.cuh` | its kernel + address helper |
| `v7_verifier_e38_int8.orig.cu` | pristine upstream adapter, kept for the diff |
| `v7_verifier_e38_int8.cu` | same file with the `batch_size` defect fixed |
| `page_local_candidate.py` | shim re-exporting the corrected views for the tests |
| `int8_g64_prefill_v2.py` | the G64 prefill kernel on its own (tensor-core QK) |
| `*service.example` | isolated 8002 units, `VLLM_API_KEY` blanked |

### The page ABI, which an earlier revision got wrong

The allocator pads every physical page to its own stride, so the layout is
**page-local**, with the four planes inside each page:

```
[page][K codes 65,536][V codes 65,536][K scales 2,048][V scales 2,048][pad]
```

The four views share the allocator's page stride and differ only by in-page
offset (K +0, V +65,536, K scales +131,072, V scales +133,120). Byte-sentinel
probes show the padded storage **cannot** be read as contiguous global planes —
that misreading is what produced the original warmup failure
(`got strides=(1777664, 16384, 256, 1), expected=(65536, 16384, 256, 1)`).
Overlapping or misaligned pages are still rejected; only the ABI changed, not
the strictness.

## Required environment

```
VLLM_KV_CACHE_LAYOUT=HND
--block-size 64 --attention-backend TRITON_ATTN --kv-cache-dtype int8_g64
VLLM_SPEC_DECODE_ATTN=1                    # gates the E38 verify branch
VLLM_INT8_G64_SRC=<this dir>/page_stride_adapter.cu
VLLM_INT8_G64_EXT_NAME=vllm_int8_g64_e38_strided
VLLM_INT8_G64_NINFER_ROOT=<NInfer src tree containing ops/kernel/>
```

`VLLM_SPEC_DECODE_ATTN=1` is not optional: `_spec_attn_enabled()` reads it, so
without it the E38 path is dead code and verify silently falls back to the
prefill-shaped kernel.

## Tests

Each needs a GPU and `VLLM_INT8_G64_NINFER_ROOT` for the ones that build the
extension. They are component-level and do not start a server.

```
python test_page_layout.py            # writer + views on the padded page, 8 cases
python test_merged_bridge.py int8_g64.py        # the exact deployed artifact
python test_g64_batch34_prefill.py int8_g64.py  # prefill at batch 3 and 4
python test_g64_prefill_v2.py         # tiled prefill, both page geometries
python test_g64_addr_probe.py         # kernel addressing vs torch views
python test_g64_both_geometries.py    # padded and compact layouts agree
python test_e38_decode_sweep.py int8_g64.py page_stride_adapter.cu <NInfer>
python test_e38_orig_adapter_sweep.py # the same sweep on the untouched adapter
python test_page_attention.py page_stride_adapter.cu <NInfer>
python bench_e38_packed_vs_strided.py # strided vs compile-time page addressing
```

`test_e38_decode_sweep.py` and `test_e38_orig_adapter_sweep.py` are the two that
found the `batch_size` defect: before the fix they fail with
`batch3 errs=[..., 1.688e+38]` and `batch4 errs=[..., 127.4, 127.3]`; after it,
all cases pass across batch 1..4 and split_count 1/4/8/16.

## Whole-model harnesses

```
python ab_split_bench.py --port 8002 --tag g64 --ctx 32000 --reps 3
python ab_bench.py --port 8002 --tag g64-4k --ctx 4096
python ab_quality.py --port 8002 --tag g64 --out quality-g64.json
python ab_quality_compare.py quality-baseline.json quality-g64.json
python step_profile.py --port 8002 --tag prod-126k --ctx 126000
```

**Measurement protocol matters here, and two engine bugs force it:**

1. changing context length within one engine lifetime trips an
   `illegal memory access`; use **one context per fresh engine**;
2. repeating the same long prefix in one process trips the same fault; give each
   request distinct text (`ab_split_bench.py` does).

Both reproduce on the production control arm and are unrelated to INT8-G64, but
they invalidate in-process context sweeps. An early sweep run this way produced a
spurious "+14% for G64" reading that the fresh-engine protocol overturned.

`step_profile.py` reports per-kernel attribution in the same buckets as the
reference profile in `docs/cmp170hx-mixed-fp8-engineering.md` (126K: verifier
partials 42.2% / target Marlin GEMMs 43.4% / GatedDeltaNet 3.7% / other 10.3%).

## Open blocker that outlives this work

Long-context **decode** faults with `illegal memory access` above roughly 64K on
this configuration, in the production control arm as well as here: prefill at the
same lengths is fine, and 70K passes with `SPEC=none`. It is not the split-KV
verify kernel, not the KV dtype, not `QMAX`, and not the known GDN accept-bound
bug (that patch is applied). `compute-sanitizer` is not installed on the lab
host; `ncu` is. Until it is fixed the 126K and 250K rows are **not measurable**,
which is why the ceiling comparison above stops at 65K. See handover 26.3.

## Deployment

```
python apply_int8_g64_remote.py <vllm_root>          # compatibility patches
python deploy_int8_g64_layout_fix.py <vllm_root> int8_g64.py
```

Both back up every file they touch as `*.orig-g64layout` and refuse to guess
anchors. Production port 8000 and Guardian were never started, stopped or
modified during any of this work; the isolated units are 8002 only.
