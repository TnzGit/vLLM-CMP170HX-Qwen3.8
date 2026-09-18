#!/usr/bin/env python3
"""Verify that a fault VA really is unmapped, rather than merely unmapped *in the
torch segment list*.

A strong claim ("the address is in unmapped space") should not rest on a map that
might be incomplete. Two things can make it incomplete:

  * the probe dumped the process at a moment when some allocation did not yet
    exist (it is dumped 120 s after start, after model load but before the run);
  * torch.cuda.memory_snapshot() only reports the caching allocator's segments --
    anything reserved by cuMemCreate/cuMemMap directly, or by a library outside
    torch, would be absent.

This script quantifies the claim from the map itself: where the VA sits relative to
the nearest segment on each side, and how large the enclosing hole is. A VA in a
multi-GiB hole bounded by large segments is convincingly unmapped; a VA a few MiB
above a segment could simply be a missing small allocation.

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
        print("=> a hole of this size bounded by large segments is strong evidence "
              "the address was never mapped by this process"
              if hole > 1024 else
              "=> the hole is small enough that a missing small allocation could "
              "explain it; do not over-claim")


if __name__ == "__main__":
    main()
