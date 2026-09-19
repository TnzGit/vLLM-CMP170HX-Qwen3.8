#!/usr/bin/env python3
"""Inject the v2 SPEC_ATTN guard: BOTH hazards from handover 49.1.

v1 dropped the block-table **column** bound to avoid a per-step sync. That was the
wrong trade for this fault: the fault reproduces under `CUDA_LAUNCH_BLOCKING=1`
(handover 47), so it is a deterministic addressing bug rather than a timing race, and
synchronising cannot mask it. v2 therefore checks both:

  1. `columns_needed = ceil(max_kv_len / block_size) > block_table.shape[1]`
     -- the kernel indexes `pos // BLOCK_SIZE` up to `kv_len`, so if `seqused_k`
     exceeds what the table covers, the read walks past the row;
  2. `block_id >= num_blocks` -- the id read from the table is dereferenced
     unbounded.

Counts every invocation so a null result is provable.

Usage:
  python inject_spec_attn_guard_v2.py <vllm>/v1/attention/ops/spec_decode_attn.py
"""
from __future__ import annotations

import ast
import pathlib
import sys

MARKER = "SPEC_ATTN_GUARD v2"
ANCHOR = "        grid = (num_reqs * ntile, Hkv, self.nseg)\n"

INJECT = '''
        # SPEC_ATTN_GUARD v2: both hazards from handover 49.1, including the
        # block-table column bound that v1 dropped to save a sync. Syncs are safe
        # here because the fault reproduces under CUDA_LAUNCH_BLOCKING=1
        # (handover 47), i.e. it is deterministic addressing, not a timing race.
        try:
            import json as _json, os as _os
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
            _w = int(block_table.shape[1])
            _rows = int(block_table.shape[0])
            _nr = int(num_reqs)
            _kv = int(seqused_k[:_nr].max().item()) if _nr else 0
            _need = (_kv + _bs - 1) // _bs if _kv else 0

            _rec = None
            if _need > _w:
                _rec = {"kind": "block_table_column_overrun",
                        "max_kv_len": _kv, "block_size": _bs,
                        "columns_needed": _need, "bt_width": _w,
                        "num_reqs": _nr,
                        "tokens_covered_by_table": _w * _bs,
                        "deficit_tokens": _kv - _w * _bs}
            elif _nr > _rows:
                _rec = {"kind": "block_table_row_overrun",
                        "num_reqs": _nr, "bt_rows": _rows}
            else:
                _bt = block_table[:_nr]
                _bad = _bt >= _nb
                if bool(_bad.any()):
                    _wi = _bad.nonzero()
                    _rec = {"kind": "block_id_out_of_range",
                            "value": int(_bt[_wi[0][0], _wi[0][1]]),
                            "row": int(_wi[0][0]), "col": int(_wi[0][1]),
                            "num_blocks": _nb, "bad_count": int(_bad.sum()),
                            "bt_width": _w, "max_kv_len": _kv, "num_reqs": _nr}

            if _rec is not None:
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
        print(f"{path.name}: v2 already injected")
        return
    if ANCHOR not in src:
        print(f"{path.name}: ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(2)
    out = src.replace(ANCHOR, INJECT + ANCHOR, 1)
    ast.parse(out)
    path.write_text(out)
    print(f"{path.name}: v2 guard injected and validates")


if __name__ == "__main__":
    main()
