# V7 prototype handoff

Status: E1 compiled and passed correctness on the CMP 170HX, but was rejected
for throughput.  The workstation that authored the source has no CUDA device;
all measurements below came from the isolated remote test directory.

Files:

- `v7_verifier.cu` — standalone PyTorch C++ extension with fixed SM80 partial
  and combine kernels.
- `../../../bench/test_v7_cuda_prototype.py` — JIT build, reference check,
  boundary cases, mixed lengths, and int32/int64 index variants.
- `build_and_smoke.sh` — convenience wrapper for the bench.
- `README.md` — geometry, interface, build command, and limitations.

E1 candidate layout in `v7_verifier.cu`:

- Q shared: scaled BF16 raw `uint16` for both BF16 and FP16 callers,
  24,576 B, matching the qualified Triton conversion;
- persistent FP16 shared accumulator `[48, 256]`, 24,576 B;
- double-buffered raw K/V staging: 32,768 B;
- total dynamic shared: 81,920 B;
- `m/l` retained in per-warp lane-zero register state;
- `part_o/m/l` published once per segment after all tiles, preserving the ABI.

Measured E1 resources are 126 registers/thread, zero stack/spill, 81,920 bytes
dynamic shared and two active CTAs/SM.  The full gate passed with maximum error
0.0625, but latency was 19-29x worse than Triton because QK/PV are scalar.

Review/qualification checklist on the CMP host:

1. `python bench/test_v7_cuda_prototype.py --build-only`
2. `python bench/test_v7_cuda_prototype.py --verbose`
3. Inspect `ext.resources()` (or add a one-line call in the bench) for
   registers, dynamic shared memory, and active CTAs/SM.
4. Repeat with `compute-sanitizer --tool memcheck` around the bench if the
   environment permits it.
5. Compare Nsight occupancy/register/shared-memory behavior only after the
   correctness cases pass.

Known risks:

- The candidate uses explicit BF16 LUT conversion and FP16 running `part_o`
  updates to mirror the existing static-FP8 q8 path; tensor-core instruction
  selection and numerical ordering can differ from Triton's `tl.dot`.
- K/V are double-buffered and accumulator state is on chip, but the current QK
  and PV loops do not use tensor cores.  WMMA is mandatory before another
  performance qualification.
- `cp.async` assumes SM80 and 16-byte-aligned contiguous D rows.  The launcher
  rejects non-contiguous caches and requests the required dynamic shared-memory
  carveout.
- `part_o/m/l` are the established flattened workspace contract, but the
  prototype does not allocate or own that workspace and does not hook the
  active vLLM series.
- A real high physical block-ID test requires a sparse/large KV allocation;
  the bench only exercises the int64 code path with ordinary physical IDs.

Measured E0 resources were 48 registers/thread, 81,920 bytes dynamic shared,
zero local bytes and two active CTAs/SM.  The 895/896/897/4097 correctness
smoke passed at max absolute error 0.000977.  This is still the global-workspace
correctness scaffold, not the on-chip E1 optimization.

`bench/test_v7_cuda_prototype.py --full --high-block-id` subsequently passed
q=6/7, mixed queries, 8K/32K/65K KV and physical block ID 2341.  The synthetic
4.00-GiB two-cache case had max absolute error 0.031250 (<0.08).

No production files, active-series entries or qualified test-site files were
changed.
