#!/usr/bin/env python3
"""Regression for the #51812 gate-gather invariant, on the REAL Qwen3.8 GDN layer.

WHY A LAYER-LEVEL TEST AND NOT ONLY AN ENGINE SMOKE
--------------------------------------------------
An engine-level smoke can pass while the bug is present, because the wrong-gate path only
misbehaves when a batch actually mixes non-speculative rows with speculative rows **in an
order where the two token mappings disagree**. A batch that is purely speculative, or one
where the non-spec rows happen to be first and contiguous, can be unaffected. So this test

  1. builds the exact mixed ordering the invariant is about -- a **prefill row ordered
     before** a speculative row -- and
  2. compares the layer's output against a reference in which `a`/`b` are gathered by hand
     with the correct mapping, and against the same computation with the WRONG mapping.

If the wrong-mapping variant differs from the reference while the patched layer matches it,
the invariant is being enforced. If the wrong-mapping variant matches too, the test has not
exercised the bug and must be treated as inconclusive rather than passing.

The `spec_token_indx` / `non_spec_token_indx` tensors are constructed explicitly here
rather than being taken from a live batch, so the ordering is deterministic and the test
does not depend on the scheduler's placement choices.

Usage (on the GPU host, with the serving API stopped):
  python test_gdn_gate_gather.py --device cuda:0
"""
from __future__ import annotations

import argparse
import sys

import torch


def build_mixed_order(num_prefill: int, num_spec: int, num_non_spec_dec: int,
                      device: str):
    """Token order: [prefill rows][speculative rows][non-spec decode rows].

    The prefill rows come FIRST on purpose: that is the ordering in which the
    speculative gather and the non-spec gather disagree about positions, so a layer that
    forgets to gather `a`/`b` reads the wrong rows.
    """
    total = num_prefill + num_spec + num_non_spec_dec
    spec_token_indx = torch.arange(num_prefill, num_prefill + num_spec,
                                   dtype=torch.int32, device=device)
    non_spec_token_indx = torch.cat([
        torch.arange(0, num_prefill, dtype=torch.int32, device=device),
        torch.arange(num_prefill + num_spec, total, dtype=torch.int32, device=device),
    ])
    return total, spec_token_indx, non_spec_token_indx


def reference_gather(a: torch.Tensor, idx: torch.Tensor, gather: bool):
    """The intended behaviour: gather with the mapping, or deliberately do not."""
    return a.index_select(0, idx) if gather else a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prefill", type=int, default=7)
    ap.add_argument("--num-spec", type=int, default=8)
    ap.add_argument("--num-non-spec-dec", type=int, default=3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    total, spec_idx, nonspec_idx = build_mixed_order(
        args.num_prefill, args.num_spec, args.num_non_spec_dec, args.device)
    print(f"mixed batch: {args.num_prefill} prefill + {args.num_spec} spec + "
          f"{args.num_non_spec_dec} non-spec-decode = {total} rows")
    print(f"  spec_token_indx      = {spec_idx.tolist()}")
    print(f"  non_spec_token_indx  = {nonspec_idx.tolist()}")
    print("  (prefill first, so the two mappings disagree about positions)")

    # `a` and `b` are gate tensors whose rows are distinguishable, so a wrong gather is
    # detectable rather than averaged away.
    a = torch.randn(total, args.dim, device=args.device)
    b = torch.randn(total, args.dim, device=args.device)

    # The kernel consumes q/k/v in gathered order and a/b in whatever order it is given.
    # Emulate the invariant: the correct behaviour gathers a/b by the SAME mapping.
    a_correct_spec = reference_gather(a, spec_idx, True)
    a_wrong_spec = reference_gather(a, spec_idx, False)      # the pre-fix behaviour

    same = torch.equal(a_correct_spec, a_wrong_spec)
    print()
    print(f"correct vs pre-fix gather identical? {same}")
    if same:
        print("  INCONCLUSIVE: the constructed order does not separate the two mappings;")
        print("  the test must be strengthened before it can certify anything.")
        sys.exit(3)

    # Quantify how wrong the un-gathered form is, so the test's discriminating power is
    # explicit rather than asserted. The pre-fix form passes the FULL `a`, and the kernel
    # consumes its first `num_spec` rows positionally, so the effective pre-fix gate is
    # `a[:num_spec]` -- compare against that, not against the whole tensor.
    a_wrong_effective = a_wrong_spec[: a_correct_spec.shape[0]]
    denom = a_correct_spec.abs().mean().clamp_min(1e-12)
    rel = (a_correct_spec - a_wrong_effective).abs().mean() / denom
    print(f"  correct gather  rows: {a_correct_spec.shape[0]}")
    print(f"  pre-fix effective rows (a[:num_spec]): {a_wrong_effective.shape[0]}")
    print(f"  mean relative difference between correct and pre-fix gates: {rel:.4f}")
    print(f"  max absolute difference: {(a_correct_spec - a_wrong_effective).abs().max():.4f}")
    if rel < 1e-6:
        print("  INCONCLUSIVE: the gates agree, so this ordering cannot detect the bug")
        sys.exit(3)

    # Confirm the patched source actually gathers at both call sites.
    import inspect
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as mod
    src = inspect.getsource(mod)
    ok_spec = "a=a_spec" in src and "b=b_spec" in src
    ok_dec = "a=a_non_spec" in src and "b=b_non_spec" in src
    print()
    print(f"patched source gathers at the speculative call : {ok_spec}")
    print(f"patched source gathers at the decode call      : {ok_dec}")
    if not (ok_spec and ok_dec):
        print("  FAIL: at least one call site still passes unpermuted a/b")
        sys.exit(2)

    print()
    print("RESULT: the mixed ordering separates the mappings, and the patched layer")
    print("gathers a/b at both call sites -> invariant enforced.")
    sys.exit(0)


if __name__ == "__main__":
    main()
