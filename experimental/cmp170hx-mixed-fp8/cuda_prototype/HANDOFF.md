# V7 prototype handoff

Status: E0 source/build/ABI smoke passed on the CMP 170HX.  The workstation
that authored the prototype has no CUDA device, but the isolated remote build
used Torch 2.13.0+cu130 and CUDA 13.0.88 successfully.

Files:

- `v7_verifier.cu` — standalone PyTorch C++ extension with fixed SM80 partial
  and combine kernels.
- `../../../bench/test_v7_cuda_prototype.py` — JIT build, reference check,
  boundary cases, mixed lengths, and int32/int64 index variants.
- `build_and_smoke.sh` — convenience wrapper for the bench.
- `README.md` — geometry, interface, build command, and limitations.

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
- Although K/V are double-buffered and all four partial warps participate, the
  current online state is deliberately read/written through global `part_o/m/l`
  once per tile.  This is a correctness scaffold, not the final E1 throughput
  implementation; moving state on-chip is the next optimization step.
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
