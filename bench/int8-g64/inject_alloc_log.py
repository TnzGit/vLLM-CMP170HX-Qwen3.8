#!/usr/bin/env python3
"""Inject the ALLOC logging hook into whichever KV-cache allocator the engine uses.

There are two implementations with identical bodies in vLLM 0.27.1:
  v1/worker/gpu_model_runner.py::_allocate_kv_cache_tensors
  v1/worker/gpu/attn_utils.py::_allocate_kv_cache
The runner used by this stack calls the attn_utils one, so patching only the first
produces no output (which is what happened on the first attempt).

Logging only: no allocation, layout or lifetime is changed.
"""
import ast
import pathlib
import sys

MARKER = "ALLOC kv_cache base="
ANCHOR = "            kv_cache_raw_tensors[layer_name] = tensor\n"

INJECT = '''
        # syv patch: log every distinct backing allocation so a fault address can
        # be converted to an allocation-relative offset (see
        # patches/log-alloc-bases.patch). Logging only -- no allocation, layout or
        # lifetime change, so this cannot alter whether the fault reproduces.
        try:
            _seen = {}
            for _name, _t in kv_cache_raw_tensors.items():
                _ptr = _t.data_ptr()
                if _ptr in _seen:
                    continue
                _seen[_ptr] = _name
                logger.info(
                    "ALLOC kv_cache base=0x%016x size=%d bytes (%.2f MiB) "
                    "end=0x%016x first_layer=%s",
                    _ptr, _t.numel(), _t.numel() / (1024 * 1024),
                    _ptr + _t.numel(), _name,
                )
        except Exception:
            pass
'''


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    src = path.read_text()
    if MARKER in src:
        print(f"{path.name}: already patched")
        return
    if ANCHOR not in src:
        print(f"{path.name}: ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(2)
    # Insert after the loop body, at the same indent as the loop's `for`.
    out = src.replace(ANCHOR, ANCHOR + INJECT, 1)
    ast.parse(out)  # refuse to write something that does not compile
    path.write_text(out)
    print(f"{path.name}: injected and validates")


if __name__ == "__main__":
    main()
