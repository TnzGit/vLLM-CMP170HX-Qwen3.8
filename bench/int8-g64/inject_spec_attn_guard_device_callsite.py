#!/usr/bin/env python3
"""Second half of the device-side guard: pass the debug buffer and bounds at launch.

`inject_spec_attn_guard_device.py` adds the checks inside `_spec_attn_partial` and
extends its signature; this adds the matching arguments at the call site in
`SpecDecodeAttention.run`, plus a helper to read the recorded maxima after a run.

Apply both, then after a request read:

    att.dbg.cpu().tolist()
    # [0] flags (1=blk, 2=k range, 4=v range, 8=pidx)   [1] max blk
    # [2] max k overshoot   [3] max v overshoot   [4] max pidx overshoot
    # [5] num_blocks        [6] part_n            [7] reserved

A zero flags word with the request having completed means the kernel's address
arithmetic stayed in range for every launched program, which would rule the kernel
out and move the search back to the metadata feeding it.

Usage:
  inject_spec_attn_guard_device_callsite.py <vllm>/v1/attention/ops/spec_decode_attn.py
"""
from __future__ import annotations

import ast
import pathlib
import sys

MARKER = "SPEC_ATTN_DEV_CALLSITE"
ANCHOR = "            num_warps=warps, num_stages=1,\n        )\n"

ARGS = '''            # SPEC_ATTN_DEV_CALLSITE: bounds + debug buffer for the in-kernel guard.
            num_blocks=key_cache.shape[0],
            part_n=self.part_o.shape[0],
            k_numel=key_cache.numel(),
            v_numel=value_cache.numel(),
            dbg_ptr=self.dbg,
'''


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    src = path.read_text()
    if MARKER in src:
        print(f"{path.name}: call site already injected")
        return
    if ANCHOR not in src:
        print(f"{path.name}: CALL-SITE ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(2)
    out = src.replace(ANCHOR, ARGS + ANCHOR, 1)
    ast.parse(out)
    path.write_text(out)
    print(f"{path.name}: call site injected and validates")


if __name__ == "__main__":
    main()
