#!/usr/bin/env python3
"""Apply the #51812 gate-gather backport to a vLLM install, then emit the real patch.

The two edits are applied by exact string replacement rather than by a hand-written
unified diff: hand-computed hunk headers have already gone wrong once in this project
(`patches/log-alloc-bases.patch`, "malformed patch"), and a generated diff is guaranteed
to match the file it came from.

Usage:
  apply_gdn_51812.py <vllm package dir> [--emit-patch <out.diff>]
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

TARGET = "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
MARKER = "Gather the gate tensors with the SAME token mapping"

# --- edit 1: the speculative call -------------------------------------------------
SPEC_OLD = """        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
"""

SPEC_NEW = """        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            # Gather the gate tensors with the SAME token mapping used for q/k/v above.
            # `query_spec` came from mixed_qkv.index_select(0, spec_token_indx), so `a`
            # and `b` must be selected identically; passing them unpermuted silently
            # updates the wrong GDN recurrent state (upstream #51812). When the batch is
            # purely speculative, mixed_qkv_spec == mixed_qkv and no gather is needed.
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                a_spec = a
                b_spec = b
            else:
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a_spec,
                    b=b_spec,
"""

# --- edit 2: the pure-decode call --------------------------------------------------
DEC_OLD = """        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
"""

DEC_NEW = """        elif attn_metadata.num_decodes > 0:
            # Same mapping requirement as the speculative branch: `query_non_spec` was
            # gathered, so `a`/`b` must be too. `a_non_spec` is already computed above
            # when spec_sequence_masks is set, and equals `a` otherwise.
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a_non_spec,
                    b=b_non_spec,
"""


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    emit = ""
    if "--emit-patch" in sys.argv:
        emit = sys.argv[sys.argv.index("--emit-patch") + 1]
    path = root / TARGET
    src = path.read_text()
    if MARKER in src:
        print(f"{path.name}: already patched")
        return

    for old, new, label in ((SPEC_OLD, SPEC_NEW, "spec"),
                            (DEC_OLD, DEC_NEW, "decode")):
        if old not in src:
            print(f"{path.name}: {label} ANCHOR NOT FOUND -- aborting", file=sys.stderr)
            sys.exit(2)
        src = src.replace(old, new, 1)

    ast.parse(src)          # refuse to write something that will not import
    path.write_text(src)
    print(f"{path.name}: #51812 gate gather applied (spec + decode call sites)")

    if emit:
        # Generate the patch from the edit itself, so headers cannot be wrong.
        before = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{TARGET}"],
            capture_output=True, text=True)
        if before.returncode != 0:
            print("  (not a git tree; skipping patch emission)", file=sys.stderr)
            return
        import difflib
        diff = difflib.unified_diff(
            before.stdout.splitlines(keepends=True),
            src.splitlines(keepends=True),
            fromfile=f"a/{TARGET}", tofile=f"b/{TARGET}")
        pathlib.Path(emit).write_text("".join(diff))
        print(f"  emitted {emit}")


if __name__ == "__main__":
    main()
