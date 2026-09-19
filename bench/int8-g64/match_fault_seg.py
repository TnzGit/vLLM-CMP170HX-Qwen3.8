#!/usr/bin/env python3
"""Decide which engine allocation owns a given Xid 31 fault VA.

The point of the whole allocation-map exercise: with the map of the *same*
process, a fault VA can be resolved against real allocations.

CORRECTION (handover 45): this previously assumed the VA was
`<2 GiB-aligned base> + 0x39af000`. That is arithmetically false, so the script no
longer subtracts a fixed constant. It reports containment, distance to the nearest
allocation on each side, and -- because the code's own addressing is
`base + block_id * stride` -- it also reports, for each allocation, the offsets
that a small integer multiple of a plausible stride would produce, which is the
form a real explanation must take.

Verdicts, in order of usefulness:

  INSIDE <seg>      the read was inside a live allocation -> offset is meaningful
  PAST END of <seg> the read ran off the end -> an overrun of that structure
  BELOW <seg>       the address is below a live allocation -> a freed/unmapped page

Usage:
  match_fault_seg.py <alloc_probe.json> <fault_va_hex> [<fault_va_hex> ...]
"""
from __future__ import annotations

import json
import sys

FAULT_OFFSET = 0x39AF000


def main() -> None:
    segs = json.load(open(sys.argv[1], encoding="utf-8"))
    vas = [int(x, 16) for x in sys.argv[2:]]
    if not vas:
        print("give at least one fault VA in hex", file=sys.stderr)
        sys.exit(2)

    for va in vas:
        print(f"fault VA 0x{va:016x}  (segments in map: {len(segs)})")
        inside = [s for s in segs if s["address"] <= va < s["address"] + s["size"]]
        if inside:
            for s in inside:
                off = (va - s["address"]) / 2**20
                print(f"  INSIDE   base 0x{s['address']:016x} "
                      f"size {s['size_mib']:.3f} MiB  offset {off:.4f} MiB")
        else:
            below = sorted((s for s in segs if s["address"] <= va),
                           key=lambda s: -s["address"])
            if below:
                s = below[0]
                over = (va - (s["address"] + s["size"])) / 2**20
                print(f"  PAST END of base 0x{s['address']:016x} "
                      f"size {s['size_mib']:.3f} MiB, over end by {over:.4f} MiB")
                print(f"           (offset from that base would be "
                      f"{(va - s['address']) / 2**20:.4f} MiB)")
            else:
                print("  BELOW every mapped segment")

        # Report stride-product offsets instead of a subtracted constant: the
        # addressing in this stack is base + id * stride, so a real explanation
        # will be N * stride for a small N.
        print("  candidate (offset, implied id at a few strides):")
        if inside or below:
            host = inside[0] if inside else below[0]
            off = va - host["address"]
            for stride_mib in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 57.6836, 64.0):
                stride = int(stride_mib * 2**20)
                if stride:
                    print(f"    offset {off / 2**20:12.4f} MiB / {stride_mib:7.4f} MiB "
                          f"stride = id {off / stride:12.3f}")
        print()


if __name__ == "__main__":
    main()
