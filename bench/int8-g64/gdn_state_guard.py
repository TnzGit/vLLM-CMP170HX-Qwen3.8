#!/usr/bin/env python3
"""Device-side first-error probe for GDN speculative state indices.

WHY
---
`v1/attention/backends/gdn_attn.py` builds the GDN state index as the physical
block id:

    spec_state_indices_tensor = block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]

and `model_executor/layers/mamba/ops/causal_conv1d.py` then dereferences it as

    conv_states_base = conv_state_ptr + (conv_states_input_coord * stride_conv_state_seq)

with the consuming load masked by `idx_feats < dim` **only** -- no
`conv_states_input_coord < num_cache_lines` bound (handover 45.6; the same bound
*is* present ~120 lines later in both kernels, and on STEP 2 of the same kernel).

`stride_conv_state_seq` is a recurrent-state row, i.e. MiB-scale, so one bad block
id lands tens of MiB outside the pool, in unmapped VA -- which is what the Xid 31
`FAULT_PDE` records show (handover 45).

WHAT THIS DOES
--------------
Checks the invariant `0 <= state_id < num_cache_lines` on the *host* side, once per
request, immediately after the tensor is built and before it is consumed. This is
deliberately NOT in-kernel:

  * no `printf`, no per-step `.cpu()` inside a kernel: a timing-sensitive fault can
    be moved or hidden by synchronisation, and the invariant is available on the CPU
    because `spec_sequence_masks_cpu` is already a CPU tensor at this point;
  * on the first violation it records once and (optionally) raises, so the *first*
    bad value is caught rather than the eventual illegal access;
  * it also checks `1 <= num_accepted_tokens <= k+1` and monotonic
    `spec_query_start_loc`, which are the other two invariants the review named.

INSTALLATION
------------
Injected as a logging/assert-only patch at the derivation site, so it observes the
exact tensor the kernel will use:

    python inject_state_index_guard.py <vllm>/v1/attention/backends/gdn_attn.py

The record is written to `/tmp/gdn_state_guard.json` (first violation only).
"""
from __future__ import annotations

import json
import os

RECORD_PATH = os.environ.get("GDN_GUARD_OUT", "/tmp/gdn_state_guard.json")


def check_state_indices(block_table_tensor, masks_cpu, num_spec, num_cache_lines,
                        num_reqs=None, request_ordinal=None):
    """Validate the slice the kernel is about to dereference.

    Returns the validated tensor unchanged. Raises GDNStateIndexError on the first
    violation so the fault is caught at its producer rather than at the eventual
    illegal access.
    """
    ncols = num_spec + 1
    try:
        sel = block_table_tensor[masks_cpu, :ncols]
    except Exception:
        return None
    try:
        import torch
        if not isinstance(sel, torch.Tensor):
            return sel
        bad = (sel < 0) | (sel >= num_cache_lines)
        if bool(bad.any()):
            where = bad.nonzero()
            row, col = int(where[0][0]), int(where[0][1])
            record = {
                "kind": "state_index_out_of_range",
                "value": int(sel[row, col]),
                "row": row,
                "col": col,
                "num_cache_lines": int(num_cache_lines),
                "num_spec": int(num_spec),
                "num_reqs": None if num_reqs is None else int(num_reqs),
                "request_ordinal": request_ordinal,
                "bad_count": int(bad.sum()),
                "tensor_shape": list(sel.shape),
            }
            _write_once(record)
            raise GDNStateIndexError(record)
    except GDNStateIndexError:
        raise
    except Exception:
        # Never let the guard itself break a run.
        pass
    return sel


def check_accepted_tokens(num_accepted, k, request_ordinal=None):
    """`1 <= num_accepted_tokens <= k+1` (the review's second invariant)."""
    try:
        import torch
        if not isinstance(num_accepted, torch.Tensor):
            return
        bad = (num_accepted < 1) | (num_accepted > k + 1)
        if bool(bad.any()):
            idx = int(bad.nonzero()[0][0])
            record = {
                "kind": "num_accepted_tokens_out_of_range",
                "value": int(num_accepted[idx]),
                "index": idx,
                "k": int(k),
                "expected": f"1..{k + 1}",
                "request_ordinal": request_ordinal,
            }
            _write_once(record)
            raise GDNStateIndexError(record)
    except GDNStateIndexError:
        raise
    except Exception:
        pass


def check_query_start_loc(qsl):
    """`spec_query_start_loc` must be monotonically non-decreasing."""
    try:
        import torch
        if not isinstance(qsl, torch.Tensor) or qsl.numel() < 2:
            return
        d = qsl[1:].to(torch.int64) - qsl[:-1].to(torch.int64)
        if bool((d < 0).any()):
            i = int((d < 0).nonzero()[0][0])
            record = {"kind": "query_start_loc_not_monotonic",
                      "index": i, "values": [int(qsl[i]), int(qsl[i + 1])]}
            _write_once(record)
            raise GDNStateIndexError(record)
    except GDNStateIndexError:
        raise
    except Exception:
        pass


class GDNStateIndexError(RuntimeError):
    pass


def _write_once(record: dict) -> None:
    """First violation wins; never overwrite, never block."""
    if os.path.exists(RECORD_PATH):
        return
    try:
        with open(RECORD_PATH, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"GDN_STATE_GUARD VIOLATION: {json.dumps(record)}", flush=True)
    except Exception:
        pass
