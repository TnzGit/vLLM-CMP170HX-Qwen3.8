#!/usr/bin/env python3
"""Phase 4G -- read-only audit: can the M7 verifier's 252 registers be reduced?

Structural fact established in Phase 4C:
    252 regs/thread, 0 spill, 2 resident CTAs/SM, only 1,024 regs of headroom per SM.
So any optimisation that ADDS per-thread state is infeasible; future verifier work must
FREE registers first. This audit asks whether that is realistic.

METHOD: compile-only. Generate structurally simplified variants of the production kernel,
compile them, and read ptxas's register count. No GPU execution and no production change --
the variants are deliberately incomplete (they compute fewer query rows) because the question
is what each part COSTS in registers, not whether the variant is correct.

Live-range inventory of the production kernel (from source):
  qs0  [32, 256] bf16   loaded once before the tile loop, live across the WHOLE scan
  qs1  [16, 256] bf16   same
  acc0 [32, 256] fp16   accumulator, live across the whole scan
  acc1 [16, 256] fp16   same
  m0/l0 [32] fp32, m1/l1 [16] fp32, plus per-tile s0/p0 [32,32] fp32 softmax temporaries
  k, v [32, 256] bf16   per-tile, LUT-decoded
"""
from __future__ import annotations

import ast
import pathlib
import sys

# Each variant is a verified textual deletion from the production source.
VARIANTS = {
    # Remove the 16-row group entirely: measures what the second row group costs.
    "rows32_only": [
        # drop qs1 load
        ("""    r1 = tl.arange(0, 16) + 32
    ri1 = r1 // 6
    rg1 = r1 % 6
    ok1 = ri1 < q_len
    qp1 = kv_len - q_len + ri1
    qptr1 = q_ptr + (q_start + ri1)[:, None] * stride_qt + (kvh * 6 + rg1)[:, None] * stride_qh + d[None, :]
    qs1 = (tl.load(qptr1, mask=ok1[:, None], other=0.0) * scale).to(tl.bfloat16)
""", ""),
        ("""    m1 = tl.full([16], float("-inf"), tl.float32)
    l1 = tl.zeros([16], tl.float32)
    acc1 = tl.zeros([16, D], tl.float16)
""", ""),
        ("""        s1 = tl.dot(qs1, tl.trans(k)).to(tl.float32) * static_k_scale
        s1 = tl.where(k_ok[None, :] & (pos[None, :] <= qp1[:, None]) & ok1[:, None], s1, float("-inf"))
        mn1 = tl.maximum(m1, tl.max(s1, 1))
        ms1 = tl.where(mn1 == float("-inf"), 0.0, mn1)
        p1 = tl.exp(s1 - ms1[:, None])
        a1 = tl.exp(tl.where(m1 == float("-inf"), float("-inf"), m1 - ms1))
        l1 = l1 * a1 + tl.sum(p1, 1)
        acc1 = (
            acc1.to(tl.float32) * a1[:, None]
            + tl.dot((p1 * static_v_scale).to(tl.bfloat16), v).to(tl.float32)
        ).to(tl.float16)
        m1 = mn1
""", ""),
        ("""    h1 = kvh * 6 + rg1
    pi1 = ((req * Hq + h1) * QMAX + ri1) * NSEG + seg
    tl.store(part_o_ptr + pi1[:, None] * D + d[None, :], acc1, mask=ok1[:, None])
    tl.store(part_m_ptr + pi1, m1, mask=ok1)
    tl.store(part_l_ptr + pi1, l1, mask=ok1)
""", ""),
    ],
}


def main() -> None:
    src_path = pathlib.Path(sys.argv[1])
    src = src_path.read_text()
    out_dir = pathlib.Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, edits in VARIANTS.items():
        v = src
        ok = True
        for anchor, repl in edits:
            n = v.count(anchor)
            if n != 1:
                print(f"{name}: anchor matched {n}x, skipping", file=sys.stderr)
                ok = False
                break
            v = v.replace(anchor, repl, 1)
        if not ok:
            continue
        try:
            ast.parse(v)
        except SyntaxError as exc:
            print(f"{name}: generated code does not parse ({exc}); skipping", file=sys.stderr)
            continue
        (out_dir / f"{name}.py").write_text(v)
        print(f"wrote {out_dir / (name + '.py')}")


if __name__ == "__main__":
    main()
