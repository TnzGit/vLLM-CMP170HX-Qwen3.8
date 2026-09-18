#!/usr/bin/env python3
"""v3 SPEC_ATTN guard: negative block ids, on ACTIVE columns only.

v2 checked `block_id >= num_blocks` across **all** columns. Two gaps remain:

  * it never checked for a **negative** id. A negative id multiplied by the
    MiB-scale page stride produces a *negative* displacement -- an address below the
    pool -- which is precisely the kind of address the earlier Xid 31 records show
    (they sit below the pools, handover 45.2). This is the one bound still untested;
  * it checked padding columns too. Padding holds 0 here, so it was harmless, but
    restricting to **active columns** is both more precise and avoids a false
    positive if padding ever holds a sentinel.

Active columns for request r are `0 .. ceil(kv_len_r / BLOCK_SIZE) - 1`, since the
kernel indexes `pos // BLOCK_SIZE` for `pos < kv_len_r` (`kv_len = seqused_k`). The
mask is built per request rather than from a global maximum so a request with a short
scan is not blamed for a column it never reads.

Counts every invocation so a null result stays provable. One sync for the final
`.any()`, which is safe: the fault reproduces under `CUDA_LAUNCH_BLOCKING=1`
(handover 47), so it is deterministic addressing, not a timing race.

Usage:
  python inject_spec_attn_guard_v3.py <vllm>/v1/attention/ops/spec_decode_attn.py
"""
from __future__ import annotations

import ast
import pathlib
import sys

MARKER = "SPEC_ATTN_GUARD v3"
ANCHOR = "        grid = (num_reqs * ntile, Hkv, self.nseg)\n"

INJECT = '''
        # SPEC_ATTN_GUARD v3: negative block ids on active columns. v2 covered
        # `>= num_blocks` on all columns but never `< 0`, and a negative id times
        # the MiB-scale page stride lands BELOW the pool -- the shape the Xid 31
        # records actually show (handover 45.2).
        try:
            import json as _json, os as _os
            import torch as _torch
            _cp = _os.environ.get("SPEC_ATTN_GUARD_COUNT",
                                  "/tmp/spec_attn_guard_count.txt")
            try:
                _n = int(open(_cp).read().strip() or 0) if _os.path.exists(_cp) else 0
            except Exception:
                _n = 0
            with open(_cp, "w") as _cf:
                _cf.write(str(_n + 1))

            _bs = int(key_cache.shape[1])
            _nb = int(key_cache.shape[0])
            # Positive control: SPEC_ATTN_GUARD_FORCE_NB=0 makes every non-negative
            # id "out of range", proving the detection path is live. The counter
            # alone only proves the block was entered.
            if _os.environ.get("SPEC_ATTN_GUARD_FORCE_NB"):
                _nb = int(_os.environ["SPEC_ATTN_GUARD_FORCE_NB"])
            _w = int(block_table.shape[1])
            _nr = int(num_reqs)
            _bt = block_table[:_nr]
            _kv = seqused_k[:_nr].to(_torch.int64)
            _active = (_kv + _bs - 1) // _bs                 # cols per request
            _cols = _torch.arange(_w, device=_bt.device)[None, :]
            _amask = _cols < _active[:, None]                # active columns only
            _bad = ((_bt < 0) | (_bt >= _nb)) & _amask
            _nactive = int(_amask.sum().item())
            _neg = int(((_bt < 0) & _amask).sum().item())

            if bool(_bad.any()):
                _wi = _bad.nonzero()
                _r, _c = int(_wi[0][0]), int(_wi[0][1])
                _rec = {"kind": "block_id_out_of_range_active",
                        "value": int(_bt[_r, _c]), "row": _r, "col": _c,
                        "is_negative": bool(int(_bt[_r, _c]) < 0),
                        "num_blocks": _nb, "bad_count": int(_bad.sum()),
                        "negative_count": _neg,
                        "active_columns_total": _nactive,
                        "bt_width": _w, "block_size": _bs, "num_reqs": _nr,
                        "kv_len_max": int(_kv.max().item()) if _nr else 0}
                _p = _os.environ.get("SPEC_ATTN_GUARD_OUT",
                                     "/tmp/spec_attn_guard.json")
                if not _os.path.exists(_p):
                    with open(_p, "w", encoding="utf-8") as _f:
                        _json.dump(_rec, _f, indent=2)
                print(f"SPEC_ATTN_GUARD VIOLATION: {_json.dumps(_rec)}", flush=True)
        except Exception as _e:
            print(f"SPEC_ATTN_GUARD setup error (ignored): {_e}", flush=True)

'''


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    src = path.read_text()
    if MARKER in src:
        print(f"{path.name}: v3 already injected")
        return
    if ANCHOR not in src:
        print(f"{path.name}: ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(2)
    out = src.replace(ANCHOR, INJECT + ANCHOR, 1)
    ast.parse(out)
    path.write_text(out)
    print(f"{path.name}: v3 guard injected and validates")


if __name__ == "__main__":
    main()
