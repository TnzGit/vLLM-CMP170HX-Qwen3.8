#!/usr/bin/env python3
"""Inject a first-error state-index bound check into the GDN speculative path.

WHERE AND WHY
-------------
`qwen_gdn_linear_attn.py` receives `spec_state_indices_tensor` from the attention
metadata and uses it as a **cache-line index** into the conv/recurrent state pool:

    causal_conv1d_update(
        conv_state,
        conv_state_indices=spec_state_indices_tensor[:, 0],
        max_query_len=spec_state_indices_tensor.size(-1), ...)

and the Triton kernel then computes
`conv_state_ptr + coord * stride_conv_state_seq` with **no**
`coord < num_cache_lines` bound on the reads that consume it (handover 45.6). The
bound is available: `num_cache_lines == conv_state.size(0)`, i.e.
`self.kv_cache[0].size(0)` in this file.

The guard is placed here rather than in the kernel because:
  * the value being validated is already a tensor the host can read cheaply at a
    coarse checkpoint, so no GPU sync is needed inside a hot kernel;
  * a timing-sensitive fault can be moved or hidden by in-kernel printing/sync;
  * the FIRST bad value is what matters, and that is easier to capture before the
    kernel than after the eventual illegal access.

The check runs once per forward call, compares against the *real* pool row count, and
on the first violation writes a JSON record and raises. It is removed by reverting
this patch; nothing else about the layer changes.

Usage:
  python inject_state_index_guard.py <vllm>/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
"""
from __future__ import annotations

import ast
import pathlib
import sys

MARKER = "GDN_STATE_GUARD"

# Inserted immediately after `spec_state_indices_tensor` is read from attn_metadata,
# so it observes the exact tensor the kernels will consume.
ANCHOR = "        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501\n"

INJECT = '''
        # GDN_STATE_GUARD: validate the cache-line index before any kernel uses it.
        # num_cache_lines == conv_state.size(0); the kernel dereferences
        # conv_state_ptr + coord * stride_conv_state_seq with no upper bound on the
        # reads that consume this value, so an out-of-range entry here becomes an
        # unmapped read tens of MiB away. First-error only: no sync, no printf loop.
        if spec_state_indices_tensor is not None:
            try:
                import json as _json, os as _os
                _ncl = int(self.kv_cache[0].size(0))
                _bad = (spec_state_indices_tensor < 0) | (
                    spec_state_indices_tensor >= _ncl
                )
                # NULL_BLOCK_ID is a deliberate sentinel, not a violation.
                from vllm.v1.attention.backends.utils import NULL_BLOCK_ID as _NULL
                _bad = _bad & (spec_state_indices_tensor != _NULL)
                if bool(_bad.any()):
                    _w = _bad.nonzero()
                    _rec = {
                        "kind": "state_index_out_of_range",
                        "value": int(spec_state_indices_tensor[_w[0][0], _w[0][1]]),
                        "row": int(_w[0][0]), "col": int(_w[0][1]),
                        "num_cache_lines": _ncl,
                        "num_spec_plus1": int(spec_state_indices_tensor.size(-1)),
                        "bad_count": int(_bad.sum()),
                        "shape": list(spec_state_indices_tensor.shape),
                        "num_actual_tokens": int(
                            getattr(attn_metadata, "num_actual_tokens", -1)),
                        "null_sentinel": int(_NULL),
                    }
                    _p = _os.environ.get("GDN_GUARD_OUT",
                                         "/tmp/gdn_state_guard.json")
                    if not _os.path.exists(_p):
                        with open(_p, "w", encoding="utf-8") as _f:
                            _json.dump(_rec, _f, indent=2)
                    print(f"GDN_STATE_GUARD VIOLATION: {_json.dumps(_rec)}",
                          flush=True)
                    raise RuntimeError(f"GDN_STATE_GUARD: {_rec}")
            except RuntimeError:
                raise
            except Exception as _e:
                print(f"GDN_STATE_GUARD setup error (ignored): {_e}", flush=True)
'''


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    src = path.read_text()
    if MARKER in src:
        print(f"{path.name}: already injected")
        return
    if ANCHOR not in src:
        print(f"{path.name}: ANCHOR NOT FOUND -- aborting", file=sys.stderr)
        sys.exit(2)
    out = src.replace(ANCHOR, ANCHOR + INJECT, 1)
    ast.parse(out)  # refuse to write something that will not import
    path.write_text(out)
    print(f"{path.name}: guard injected and validates")


if __name__ == "__main__":
    main()
