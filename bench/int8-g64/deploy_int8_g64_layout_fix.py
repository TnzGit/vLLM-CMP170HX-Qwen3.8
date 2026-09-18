#!/usr/bin/env python3
"""Install the audited page-local INT8-G64 bridge into the ISOLATED runtime.

Fail-closed by construction: it only touches the vLLM package under --vllm-root,
backs up every file it edits (``*.orig-g64layout``), refuses to guess anchors,
and never touches a running service, the production tree, or a model directory.

What it does
  1. installs the merged bridge as ``v1/attention/ops/int8_g64.py``
  2. routes INT8-G64 prefill to the dedicated Triton kernel in ``triton_attn.py``
     (the stock unified_attention kernel cannot express per-group G=64 scales)
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

PREFILL_BRANCH = '''        # E38/E39 INT8-G64 prefill route.  The stock unified_attention kernel
        # treats this mode as per-token-head (zero queries-per-kv) and cannot
        # express per-group G=64 scales, so prefill runs on the dedicated Triton
        # kernel over the page-local views instead.
        if (
            getattr(self, "_g64_enabled", False)
            and k_scale_cache is not None
            and v_scale_cache is not None
            and attn_metadata.causal
            and self.alibi_slopes is None
            and self.sinks is None
            and not self.logits_soft_cap
            and (self.sliding_window is None or self.sliding_window == (-1, -1))
            and mm_prefix_range_tensor is None
            and attn_metadata.rswa_prefix_lens is None
            and output_scale is None
        ):
            from vllm.v1.attention.ops.int8_g64 import unified_attention_g64

            unified_attention_g64(
                q=query[:num_actual_tokens],
                k_cache=key_cache,
                v_cache=value_cache,
                k_scale=k_scale_cache,
                v_scale=v_scale_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                block_table=block_table,
                softmax_scale=self.scale,
                max_seqlen_q=max_seqlen_q,
                causal=attn_metadata.causal,
            )
            return output

'''

ANCHOR = "        unified_attention(\n            q=query[:num_actual_tokens],"
MARKER = "E38/E39 INT8-G64 prefill route"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vllm_root", type=Path)
    ap.add_argument("module", type=Path, help="audited bridge candidate")
    args = ap.parse_args()

    root: Path = args.vllm_root
    target = root / "v1/attention/ops/int8_g64.py"
    attn = root / "v1/attention/backends/triton_attn.py"
    for path in (target, attn):
        if not path.is_file():
            raise SystemExit(f"missing expected file: {path}")

    backup = target.with_name(target.name + ".orig-g64layout")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"backed up {target.name} -> {backup.name}")
    shutil.copy2(args.module, target)
    print(f"installed {args.module.name} -> {target} ({sha256(target)})")

    text = attn.read_text()
    if MARKER in text:
        print("prefill route already present; nothing to patch")
    else:
        if ANCHOR not in text:
            raise SystemExit("prefill anchor not found in triton_attn.py; refusing to guess")
        attn_backup = attn.with_name(attn.name + ".orig-g64layout")
        if not attn_backup.exists():
            shutil.copy2(attn, attn_backup)
            print(f"backed up {attn.name} -> {attn_backup.name}")
        attn.write_text(text.replace(ANCHOR, PREFILL_BRANCH + ANCHOR, 1))
        print(f"patched {attn} ({sha256(attn)})")
    print("deploy complete")


if __name__ == "__main__":
    main()
