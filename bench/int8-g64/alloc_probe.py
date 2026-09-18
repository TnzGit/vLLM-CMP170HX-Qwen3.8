#!/usr/bin/env python3
"""List every CUDA allocation in the engine process with base addresses.

The fault address decomposes as `<2 GiB-aligned base> + 0x39af000` (handover
38.9), and that base lies ~1.94 GiB below the lowest recurrent-state pool, so the
owning allocation is one that `patches/log-alloc-bases.patch` does not cover. This
enumerates everything, so the base can be named.

Installed as a sitecustomize hook in the ENGINE process (a client process sees
none of these). Two outputs:

  * `torch.cuda.memory_snapshot()` -- every caching-allocator segment, which covers
    anything allocated through torch (workspaces, scratch, persistent buffers that
    were created with torch.zeros/empty);
  * a summary of which segment would contain `<base> + 0x39af000` for the observed
    fault, printed as a table so the match is mechanical.

Set ALLOC_PROBE_DELAY_S to wait for model load before dumping (default 0 = dump at
process start, which only catches early allocations).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

FAULT_OFFSET = 0x39AF000


def dump(out_path: str) -> None:
    import torch

    rows = []
    try:
        snap = torch.cuda.memory_snapshot()
    except Exception as exc:  # noqa: BLE001
        print(f"alloc_probe: snapshot failed: {exc}", flush=True)
        return
    for seg in snap:
        addr = seg.get("address")
        size = seg.get("total_size") or seg.get("size") or 0
        if addr is None:
            continue
        blocks = seg.get("blocks") or []
        active = sum(b.get("size", 0) for b in blocks
                     if b.get("state") == "active_allocated")
        rows.append({
            "address": addr,
            "address_hex": f"0x{addr:016x}",
            "size": size,
            "size_mib": round(size / 2**20, 3),
            "active": active,
            "active_mib": round(active / 2**20, 3),
            "n_blocks": len(blocks),
        })
    rows.sort(key=lambda r: -r["size"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"alloc_probe: {len(rows)} segments -> {out_path}", flush=True)
    for r in rows[:25]:
        print(f"  base {r['address_hex']}  size {r['size_mib']:10.3f} MiB  "
              f"active {r['active_mib']:10.3f} MiB  blocks {r['n_blocks']}",
              flush=True)

    # Which segment, if any, has an internal offset of exactly FAULT_OFFSET such
    # that base + FAULT_OFFSET is a plausible fault address? Report every segment
    # whose extent could contain such an address.
    print("\nsegments that could own a fault at base + 0x39af000:", flush=True)
    hit = False
    for r in rows:
        lo = r["address"] + FAULT_OFFSET
        hi = r["address"] + r["size"]
        if lo < hi:
            hit = True
            print(f"  {r['address_hex']} + 0x39af000 = 0x{lo:016x}  "
                  f"(inside, size {r['size_mib']:.3f} MiB)", flush=True)
    if not hit:
        print("  none -- the owning allocation is not a torch segment; it is "
              "allocated outside the torch allocator (e.g. by vLLM directly)",
              flush=True)


def _delayed(out_path: str, delay: float) -> None:
    time.sleep(delay)
    dump(out_path)


def install() -> None:
    out = os.environ.get("ALLOC_PROBE_OUT", "/tmp/alloc_probe.json")
    delay = float(os.environ.get("ALLOC_PROBE_DELAY_S", "0"))
    if delay > 0:
        threading.Thread(target=_delayed, args=(out, delay), daemon=True).start()
        print(f"alloc_probe: armed, dumping in {delay}s to {out}", flush=True)
    else:
        dump(out)


if __name__ == "__main__":
    install()
    sys.exit(0)
