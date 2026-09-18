#!/usr/bin/env python3
"""Quantify where a fault VA sits relative to the torch caching allocator's map.

THIS SCRIPT CANNOT ESTABLISH USE-AFTER-FREE. Read this before quoting its output.

An earlier version concluded from a large hole that the address "was never mapped by
this process". That is **too strong**, and handover 45 withdraws it. What the data
supports is narrower:

  * `torch.cuda.memory_snapshot()` sees only the **PyTorch caching allocator's**
    segments. It is blind to raw `cudaMalloc`, `cuMemCreate`/`cuMemMap`, third-party
    CUDA libraries, some custom-op workspaces, pinned/UVA mappings, and any lazy
    allocation made after the dump. A hole in this map is a hole *in this map*;
  * Xid 31 `FAULT_PDE` reliably tells us only that the final VA had no valid page
    directory entry **at the moment of the access**. Both of these produce that:
      - a stale pointer to freed/unmapped memory, and
      - a live tensor base plus a bad block/state index multiplied by a large
        stride, computing an address outside the allocation and into unmapped VA.
    Only the first is use-after-free, and nothing here distinguishes them.

Since the state/KV addressing in this stack is full of `base + id * stride`, the
second explanation is at least as likely, and it needs no lifetime defect at all.

So this script reports geometry (containment, distance to the nearest segment, hole
size) and explicitly refrains from a verdict. Use it to say "the read landed outside
the allocator's map"; do not use it to say "the pointer was freed".

Usage: verify_unmapped.py <alloc_probe.json> <fault_va_hex>
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    segs = json.load(open(sys.argv[1], encoding="utf-8"))
    va = int(sys.argv[2], 16)
    segs = sorted(segs, key=lambda s: s["address"])

    print(f"segments: {len(segs)}")
    if not segs:
        print("empty map -- cannot conclude anything", file=sys.stderr)
        sys.exit(2)

    lo = segs[0]["address"]
    hi = max(s["address"] + s["size"] for s in segs)
    print(f"mapped span: 0x{lo:016x} .. 0x{hi:016x}  ({hi / 2**40:.3f} TiB)")
    print(f"fault VA   : 0x{va:016x}")
    print(f"within span: {lo <= va <= hi}")
    print()

    below = [s for s in segs if s["address"] + s["size"] <= va]
    above = [s for s in segs if s["address"] > va]
    inside = [s for s in segs if s["address"] <= va < s["address"] + s["size"]]

    print(f"segments containing the VA : {len(inside)}")
    print(f"segments entirely below it : {len(below)}")
    print(f"segments entirely above it : {len(above)}")
    print()

    if inside:
        s = inside[0]
        print(f"CONTAINED in base 0x{s['address']:016x} size {s['size_mib']:.3f} MiB")
        print(f"  offset {(va - s['address']) / 2**20:.4f} MiB")
        return

    if below:
        b = max(below, key=lambda s: s["address"] + s["size"])
        gap_below = (va - (b["address"] + b["size"])) / 2**20
        print(f"nearest segment below: base 0x{b['address']:016x} "
              f"size {b['size_mib']:.3f} MiB, ends {(va - (b['address'] + b['size'])) / 2**20:.3f} MiB below the VA")
    else:
        print("no segment below the VA")

    if above:
        a = min(above, key=lambda s: s["address"])
        print(f"nearest segment above: base 0x{a['address']:016x} "
              f"size {a['size_mib']:.3f} MiB, starts {(a['address'] - va) / 2**20:.3f} MiB above the VA")

    if below and above:
        b = max(below, key=lambda s: s["address"] + s["size"])
        a = min(above, key=lambda s: s["address"])
        hole = (a["address"] - (b["address"] + b["size"])) / 2**20
        print(f"\nenclosing hole size: {hole:.3f} MiB")
        print("outside the torch caching allocator's map (hole size is reported for "
              "context only -- it does NOT distinguish a freed pointer from a bad "
              "index computed into unmapped VA, and it cannot see non-torch "
              "allocations at all)")


if __name__ == "__main__":
    main()
