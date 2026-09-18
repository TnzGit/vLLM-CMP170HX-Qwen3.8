#!/usr/bin/env python3
"""Decide which engine allocation owns a given Xid 31 fault VA.

The point of the whole allocation-map exercise: the fault VA decomposes as a
2 GiB-aligned base plus a constant 0x39af000 (handover 38.9), and with the map of
the *same* process the base can finally be named rather than guessed.

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

        base_guess = va - FAULT_OFFSET
        exact = [s for s in segs if s["address"] == base_guess]
        print(f"  va - 0x39af000 = 0x{base_guess:016x}  "
              f"is a known base: {'YES -> ' + str(exact[0]['size_mib']) + ' MiB segment' if exact else 'no'}")
        print()


if __name__ == "__main__":
    main()
