#!/usr/bin/env python3
"""Phase 4B -- generate an isolated tensor-core-denominator variant of the M7 verifier.

WHY GENERATED RATHER THAN HAND-WRITTEN
--------------------------------------
The variant must differ from the production kernel in exactly one respect (how the softmax
denominator is reduced) or the measurement means nothing. Hand-copying a 120-line Triton
kernel invites silent transcription drift, so this script reads the production source and
performs a **verified textual substitution**, asserting that each anchor matched exactly
once. If an anchor is missing the script fails instead of emitting a wrong kernel.

THE CHANGE
----------
Production reduces the softmax denominator on the generic FP32 path:

    l0 = l0 * a0 + tl.sum(p0, 1)

while the numerator already uses tensor cores via `tl.dot`. The variant expresses the same
row-sum as a matmul against a ones-vector, so it runs on the tensor cores:

    l0 = l0 * a0 + _rowsum_tc(p0, TILE)

KNOWN RISK, STATED UP FRONT
---------------------------
`tl.dot` with BF16 inputs quantizes `p` (which is in [0,1] after `exp`) to BF16, so the
denominator loses precision. The gate includes an oracle-correctness check precisely because
this is expected to be the failure mode if the variant fails.

The output is written to a separate module so the production file is never touched and the
variant can be imported, or simply not imported, on a per-run basis.

Usage:
  ph4b_generate_variant.py <production spec_decode_attn.py> <out variant.py>
"""
from __future__ import annotations

import ast
import pathlib
import sys

HELPER = '''

@triton.jit
def _rowsum_tc(p, TILE: tl.constexpr):
    """Row-sum of `p` evaluated on the tensor cores as `p @ ones`.

    `tl.dot` requires N >= 16 on SM80, so the ones-vector is padded to 16 columns and only
    column 0 is used. The other 15 columns are computed and discarded -- deliberately, since
    the point is to move the reduction onto the tensor cores, not to be efficient in FLOPs.
    """
    ones = tl.full([TILE, 16], 1.0, tl.bfloat16)
    acc = tl.dot(p.to(tl.bfloat16), ones).to(tl.float32)
    # Triton does not support constexpr scalar indexing (`acc[:, 0]`), so column 0 is
    # selected with a mask and summed. Only column 0 carries the row-sum; the other 15
    # columns are the same value (all ones) and are discarded here.
    col = tl.arange(0, 16)
    return tl.sum(tl.where(col[None, :] == 0, acc, 0.0), 1)

'''

# (anchor, replacement) pairs. Each anchor must occur exactly once.
EDITS = [
    # Insert the helper just before the production kernel it modifies.
    ("@triton.jit\ndef _spec_attn_partial_q8_g6_fp8(",
     HELPER + "\n@triton.jit\ndef _spec_attn_partial_q8_g6_fp8("),
    # Row group 0.
    ("        l0 = l0 * a0 + tl.sum(p0, 1)",
     "        l0 = l0 * a0 + _rowsum_tc(p0, TILE)"),
    # Row group 1.
    ("        l1 = l1 * a1 + tl.sum(p1, 1)",
     "        l1 = l1 * a1 + _rowsum_tc(p1, TILE)"),
]


def main() -> None:
    src_path = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])
    src = src_path.read_text()

    if "_rowsum_tc" in src:
        print("source already contains _rowsum_tc; refusing to double-apply", file=sys.stderr)
        sys.exit(2)

    for anchor, repl in EDITS:
        n = src.count(anchor)
        if n != 1:
            print(f"ANCHOR MATCHED {n} TIMES (expected 1):\n  {anchor[:80]!r}",
                  file=sys.stderr)
            sys.exit(2)
        src = src.replace(anchor, repl, 1)

    ast.parse(src)                      # refuse to emit something that cannot import
    out_path.write_text(src)

    # Report exactly what changed, so the diff is auditable without reading the file.
    print(f"variant written: {out_path}")
    print(f"  helper inserted : _rowsum_tc (p @ ones on tensor cores, N padded to 16)")
    print(f"  substitutions   : 2  (l0 and l1 denominators)")
    print(f"  untouched       : every load, mask, exp, scale, store and the combine kernel")
    for line in src.splitlines():
        if "_rowsum_tc" in line and "def " not in line:
            print(f"    {line.strip()}")


if __name__ == "__main__":
    main()
