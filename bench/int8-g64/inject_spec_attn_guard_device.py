#!/usr/bin/env python3
"""Device-side first-error guard for `_spec_attn_partial`.

WHY THIS EXISTS (and why memcheck is not enough)
------------------------------------------------
Filtered `compute-sanitizer --tool memcheck` on this kernel does **not** yield a
location for the fault: the run ends with

    Internal Sanitizer Error: The Sanitizer failed to handle a hardware exception

i.e. the MMU-level `FAULT_PDE` is a hardware exception memcheck cannot turn into a
per-access violation report (handover 54). Host-side guards are also exhausted --
the GDN state index (1919 checks), the block table, block ids and negative ids on
active columns (785 checks each) are all clean -- so the bad address is computed
*inside* the kernel from arguments that are individually valid.

That leaves a device-side check as the only way to see it, which is what the original
review proposed: a small first-error record buffer written with atomics, no printf and
no per-step sync, read back only after the request finishes.

WHAT IT CHECKS
--------------
Every quantity the kernel turns into an address, checked immediately before use:

  1. `blk < num_blocks` -- the block id read from the block table, against the number
     of physical pages (`key_cache.shape[0]`);
  2. the computed `k`/`v` element offsets stay within the tensors' `numel` -- this is
     the check host-side guarding cannot do, and it catches a wrong *stride* argument
     even when every index is valid;
  3. the partial-buffer store index `pidx < part_n` -- the store side of the kernel,
     which is the review's leading suspect because `pidx` is pure in-kernel arithmetic.

Records maxima rather than first-incident details, so a single `atomic_max` per slot
suffices: no ordering requirement, no branch, no synchronisation, and the recorded
values are the worst case seen. Slots:

    [0] error flags (bit 0 blk, bit 1 k/v range, bit 2 pidx)
    [1] max blk seen
    [2] max |k offset| overshoot   (0 if none)
    [3] max |v offset| overshoot   (0 if none)
    [4] max pidx seen
    [5] num_blocks, [6] part_n, [7] max kv_len

Usage:
  inject_spec_attn_guard_device.py <vllm>/v1/attention/ops/spec_decode_attn.py
  # then allocate the buffer and pass it; the injector prints the call-site edit.
"""
from __future__ import annotations

import ast
import pathlib
import sys

MARKER = "SPEC_ATTN_DEV_GUARD"
ANCHOR = "        grid = (num_reqs * ntile, Hkv, self.nseg)\n"

# Allocate the debug buffer on the instance, so it survives for the process lifetime
# and can be read after a request without any per-step copy.
ALLOC_ANCHOR = "        self.part_l = torch.empty(n, dtype=torch.float32, device=device)\n"
ALLOC = '''
        # SPEC_ATTN_DEV_GUARD: 8 int32 slots for in-kernel first-error recording.
        # Read after a run with: att.dbg.cpu().tolist()
        self.dbg = torch.zeros(8, dtype=torch.int32, device=device)
'''

# Kernel-side checks. `num_blocks`, `part_n`, `k_numel`, `v_numel` and `dbg_ptr` are
# appended to the kernel signature by the second replacement below.
CHECKS_KV = '''
        # SPEC_ATTN_DEV_GUARD: bounds before dereference.
        _bmax = tl.max(blk, 0)
        if _bmax >= num_blocks:
            tl.atomic_max(dbg_ptr + 0, 1)
            tl.atomic_max(dbg_ptr + 1, _bmax)
        _koff = tl.max(blk * stride_kb + (BLOCK_SIZE - 1) * stride_ks
                       + (Hkv - 1) * stride_kh + (D - 1))
        _voff = tl.max(blk * stride_vb + (BLOCK_SIZE - 1) * stride_vs
                       + (Hkv - 1) * stride_vh + (D - 1))
        if _koff >= k_numel:
            tl.atomic_max(dbg_ptr + 0, 2)
            tl.atomic_max(dbg_ptr + 2, _koff - k_numel + 1)
        if _voff >= v_numel:
            tl.atomic_max(dbg_ptr + 0, 4)
            tl.atomic_max(dbg_ptr + 3, _voff - v_numel + 1)

'''

CHECKS_STORE = '''
    # SPEC_ATTN_DEV_GUARD: the partial store index is pure in-kernel arithmetic
    # (((req*Hq + hrow)*QMAX + ri)*NSEG + seg) -- the review's leading suspect.
    _pmax = tl.max(pidx, 0) + 1
    if _pmax > part_n:
        tl.atomic_max(dbg_ptr + 0, 8)
        tl.atomic_max(dbg_ptr + 4, _pmax - part_n)


def _noop():
    pass


'''


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    src = path.read_text()
    if MARKER in src:
        print(f"{path.name}: device guard already injected")
        return

    # 1. allocate the debug buffer
    if ALLOC_ANCHOR not in src:
        print(f"{path.name}: ALLOC ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(2)
    src = src.replace(ALLOC_ANCHOR, ALLOC_ANCHOR + ALLOC.lstrip("\n"), 1)

    # 2. extend the kernel signature and insert the checks before the k/v load and
    #    before the partial store.
    sig_old = "    NTILE: tl.constexpr, QUANT: tl.constexpr,\n):"
    sig_new = ("    NTILE: tl.constexpr, QUANT: tl.constexpr,\n"
               "    num_blocks, part_n, k_numel, v_numel, dbg_ptr,\n):")
    if sig_old not in src:
        print(f"{path.name}: SIGNATURE ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(2)
    src = src.replace(sig_old, sig_new, 1)

    kv_old = "        k = tl.load(k_ptrs, mask=k_ok[:, None], other=0.0)"
    if kv_old not in src:
        print(f"{path.name}: K/V ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(2)
    src = src.replace(kv_old, CHECKS_KV + kv_old, 1)

    store_old = "    pidx = ((req * Hq + hrow) * QMAX + ri) * NSEG + seg\n"
    if store_old not in src:
        print(f"{path.name}: STORE ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(2)
    src = src.replace(store_old, store_old + CHECKS_STORE, 1)

    ast.parse(src)
    path.write_text(src)
    print(f"{path.name}: device guard injected and validates")
    print("\nNow pass the buffer at the launch (a second injector handles this), "
          "and read `att.dbg` after a run.")


if __name__ == "__main__":
    main()
