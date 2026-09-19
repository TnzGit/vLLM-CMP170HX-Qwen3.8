"""CORRECTION (handover 45): the `base + 0x39af000` premise below is WITHDRAWN --
that decomposition is arithmetically false. Superseded by match_fault_seg.py
plus fault_va_invariants.py."""
#!/usr/bin/env python3
"""Match the constant fault displacement 0x39af000 against the engine's segments.

The fault address is `<base> + 0x39af000` where `base` is 2 GiB-aligned and lies
below the recurrent-state pools (handover 38.9). The engine's full segment list was
dumped by alloc_probe.py, so the owning allocation can be identified mechanically:

  for every segment S: is  S.base + 0x39af000  inside S?
  and separately:     is  S.base + 0x39af000  where a fault was actually seen?

The second form matters more: several segments can contain their own
`base + offset`, so the discriminator is whether that computed address *matches an
observed fault address's low bits and band*.
"""
import argparse
import json
import re
import subprocess
import sys

FAULT_OFFSET = 0x39AF000


def observed_faults() -> list[int]:
    raw = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True,
                         errors="ignore").stdout
    return sorted({int(m.group(1).replace("_", ""), 16)
                   for m in re.finditer(r"faulted @ 0x([0-9a-f_]+)", raw)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="alloc_probe.py output")
    ap.add_argument("--log", default="", help="engine log, to read the same process's bases")
    args = ap.parse_args()

    segs = json.load(open(args.json, encoding="utf-8"))
    print(f"loaded {len(segs)} segments")
    faults = observed_faults()
    print(f"observed distinct fault VAs (all generations): {len(faults)}")

    # Which segments could produce a fault at their own base + FAULT_OFFSET?
    # Note: we do NOT require the fault to have been observed inside THIS process,
    # because dmesg holds many generations; we report the shape and let the reader
    # correlate. What we CAN do is check which candidate bases have plausible
    # low-bit/band structure.
    print(f"\nsegments whose own base + 0x{FAULT_OFFSET:x} falls inside themselves:")
    candidates = []
    for s in sorted(segs, key=lambda r: -r["size"]):
        base = s["address"]
        size = s["size"]
        probe = base + FAULT_OFFSET
        if base <= probe < base + size:
            candidates.append((base, size, probe))
    for base, size, probe in candidates[:20]:
        print(f"  base 0x{base:016x}  size {size / 2**20:10.3f} MiB  "
              f"base+off = 0x{probe:016x}")

    # The stronger test: the fault VA seen in the SAME process generation must
    # equal some segment base + FAULT_OFFSET exactly.
    print(f"\nexact matches: observed fault VA == segment_base + 0x{FAULT_OFFSET:x}")
    bases = {s["address"] for s in segs}
    hit = 0
    for v in faults:
        b = v - FAULT_OFFSET
        if b in bases:
            hit += 1
            s = next(x for x in segs if x["address"] == b)
            print(f"  fault 0x{v:016x} == 0x{b:016x} + 0x{FAULT_OFFSET:x}   "
                  f"size {s['size'] / 2**20:.3f} MiB   "
                  f"active {s['active_miB' if 'active_miB' in s else 'active_mib']:.3f} MiB")
    if not hit:
        print("  none in this generation (expected: the segments dumped are from a "
              "process that had not faulted yet, and dmesg holds other generations)")

    # Fall back to the structural statement, which is generation-independent:
    # how many segments are 2 GiB-aligned, and what sizes recur?
    print("\n2 GiB-aligned segments (candidate pool bases), by size:")
    from collections import Counter
    two_gib = Counter()
    for s in segs:
        if s["address"] % (1 << 31) == 0:
            two_gib[round(s["size"] / 2**20, 3)] += 1
    for size_mib, n in two_gib.most_common(12):
        print(f"  size {size_mib:10.3f} MiB  x{n}")


if __name__ == "__main__":
    main()
