# V7 prototype handoff

Status: V7-E3a BF16 WMMA padded-shared-layout candidate passed correctness but
was rejected for throughput and occupancy.  It is an isolated layout-only
follow-on to E2 and is not connected to production dispatch.  The workstation
that authored this source has no CUDA device; all measurements below came from
the isolated remote test directory.

Files:

- `v7_verifier.cu` — standalone PyTorch C++ extension with fixed SM80 partial
  and combine kernels.
- `../../../bench/test_v7_cuda_prototype.py` — JIT build, reference check,
  boundary cases, mixed lengths, and int32/int64 index variants.
- `build_and_smoke.sh` — convenience wrapper for the bench.
- `README.md` — geometry, interface, build command, and limitations.

V7-E3a candidate layout in `v7_verifier.cu`:

- persistent all-row FP16 shared accumulator `[48, 256]`, 24,576 B;
- one decoded BF16 K/V tile physically `[2, 32, 264]`, 33,792 B; only the
  first 256 elements of each row are logical K/V data;
- one BF16 Q/P pack physically `[16, 264]`, 8,448 B; logical Q is 256-wide
  and P is repacked densely as `[16, 32]` with `ld=32`;
- one FP32 score/PV temporary `[16, 256]`, 16,384 B;
- total dynamic shared: 83,200 B;
- `m/l` retained in registers, with warp-0 lanes 0..15 owning three row-pack
  states each;
- `part_o/m/l` published once per segment after all tiles, preserving the ABI.

E3a WMMA mapping:

- QK uses two N16 tiles per 16-row pack: row-major BF16 Q with logical
  `[16,256]` and physical `ld=264`, and col-major BF16 K with logical
  `[256,32]` viewed from physical token-major rows with `ld=264`;
- warp 0 lanes 0..15 perform the 32-score causal online softmax, write BF16
  P using compact `ld=32`, and save exact FP32 alpha in the Q/P buffer tail;
- PV uses four warps, each owning four N16 output tiles: row-major P `[16,32]`
  with `ld=32` times row-major logical V `[32,256]` with physical `ld=264`,
  with FP32 temporary stores followed by parallel alpha fusion into the
  all-row FP16 accumulator;
- each tile decodes raw FP8 K/V through the supplied LUT with the full CTA and
  synchronizes before WMMA; there is no K/V double buffer.

Historical E2 baseline: the unpadded candidate compiled to 88 registers/thread,
zero local spill, 81,920 B dynamic shared memory and two CTAs/SM.  Its complete
correctness gate passed after fixing P's store stride from Q's 256 columns to
compact 32 columns.  The largest ordinary error was 0.003906 and the 4-GiB
high-block-ID case was 0.031250.

It nevertheless failed performance admission: 4K/70K/126K/200K/250K were
488.2/8,546.7/15,269.0/24,167.6/30,208.5 us per layer.  At 126K, NCU measured
the same 6,048,768 tensor instructions as Triton but only 0.82% tensor-pipe
activity, 190.54 M shared-load bank conflicts, 51.20% long-scoreboard stalls
and 1.079 B executed instructions.  Triton measured 17.21%, 2.02 M, 12.09%
and 126.0 M respectively.  This is a shared-layout/scalar-feed failure, not
missing HMMA lowering.

Measured E1 resources were 126 registers/thread, zero stack/spill, 81,920 bytes
dynamic shared and two active CTAs/SM.  The full gate passed with maximum error
0.0625, but latency was 19-29x worse than Triton because QK/PV were scalar.

E3a compiled to 79 registers/thread and zero spill, and its complete
correctness/high-block-ID gate passed.  Padding reduced shared-load bank
conflicts from 190.54 M to 63.51 M, but 83,200 B crossed the residency cliff:
only one CTA/SM remained.  Latency regressed to
882.3/13,220.8/23,600.3/37,307.3/46,681.1 us at
4K/70K/126K/200K/250K.  Tensor-pipe activity fell further to 0.53%.

E3b handoff checklist:

1. Keep padded Q/K/V, but recover at least 1,280 B from the FP32 PV temporary
   so total dynamic shared is <=81,920 B and two CTAs/SM return.
2. A practical isolated design is a 224-column FP32 PV phase followed by a
   32-column tail phase; measure the added barriers rather than assuming they
   are free.
3. Repeat full correctness, five-context latency and the same NCU counter set.
4. Only after two-CTA padded layout improves should page carry, vectorized raw
   loads, shared/read-only LUT and P-fragment reuse be combined.

Known risks:

- WMMA BF16 accumulation and online softmax ordering passed the E3a gate, but
  every temporary-layout change must repeat it before performance testing.
- E3a proved that 83,200 B leaves only one CTA/SM on this host.  E3b must not
  accept that occupancy loss even though the single-CTA allocation is legal.
- The raw FP8 LUT decode and each Q/P pack are reloaded for every tile.  NCU
  proved on E2 that this preparation starves the tensor pipe; E3a intentionally
  does not change that path.
- The launcher rejects non-contiguous caches and requests the required
  dynamic shared-memory carveout; WMMA is compiled only for SM80.
- `part_o/m/l` are the established flattened workspace contract, but the
  prototype does not allocate or own that workspace and does not hook the
  active vLLM series.
- A real high physical block-ID test requires a sparse/large KV allocation;
  the bench only exercises the int64 code path with ordinary physical IDs.

Measured E0 resources were 48 registers/thread, 81,920 bytes dynamic shared,
zero local bytes and two active CTAs/SM.  The 895/896/897/4097 correctness
smoke passed at max absolute error 0.000977.  These are historical E0/E1
results and do not qualify the new E2 WMMA candidate.

`bench/test_v7_cuda_prototype.py --full --high-block-id` subsequently passed
q=6/7, mixed queries, 8K/32K/65K KV and physical block ID 2341.  The synthetic
4.00-GiB two-cache case had max absolute error 0.031250 (<0.08).

No production files, active-series entries or qualified test-site files were
changed.
