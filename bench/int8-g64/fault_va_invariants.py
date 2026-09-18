#!/usr/bin/env python3
"""Analyse Xid 31 fault addresses: what is actually invariant, and against what.

CORRECTION (handover §45). An earlier version of this analysis claimed the fault
address decomposes as `<2 GiB-aligned base> + 0x39af000` and used that as the
anchor for matching allocations. Both halves were wrong:

  * `0x39af000` is 60,485,632 bytes = **57.6836 MiB**, not the 3,777,536 bytes
    written in §38.9 (that decimal is 0x39a400 -- a transcription slip);
  * `fault_va - 0x39af000` is **not** 2 GiB-aligned. For
    `0x7377a39af000 - 0x39af000 = 0x7377a0000000`, which sits 512 MiB past a
    2 GiB boundary.

What the observed addresses *actually* share is a **2 MiB-page offset of
`0x1af000`**, i.e. `mod 2 MiB == 0x01af000`, and 4 KiB alignment. They do *not*
share a 2 GiB-relative displacement: three of the four below have `mod 2 GiB ==
0x0239af000` and one has `0x0639af000`.

So this script makes no assumption about an owning base. It reports:

  * the invariants that hold across every observed fault address;
  * the page-offset histogram, which is the statistic that has actual support;
  * how the addresses distribute across 2 GiB and 2 MiB boundaries, so a future
    reader can see the real structure rather than a claimed decomposition.

Usage:  fault_va_invariants.py [--dmesg]
        (run on the GPU host; reads dmesg via sudo -n)
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter

PAGE_4K = 1 << 12
PAGE_2M = 1 << 21
BAND_2G = 1 << 31


def fault_addresses() -> list[int]:
    raw = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True,
                         errors="ignore").stdout
    return [int(m.group(1).replace("_", ""), 16)
            for m in re.finditer(r"faulted @ 0x([0-9a-f_]+)", raw)]


def main() -> None:
    vas = fault_addresses()
    if not vas:
        print("no Xid 31 fault addresses found in dmesg")
        sys.exit(0)

    uniq = sorted(set(vas))
    print(f"Xid 31 events: {len(vas)}   distinct VAs: {len(uniq)}\n")

    # --- invariants with actual support -------------------------------------
    aligned4k = sum(1 for v in vas if v % PAGE_4K == 0)
    print(f"4 KiB aligned   : {aligned4k}/{len(vas)}"
          f"{'  <- invariant' if aligned4k == len(vas) else ''}")

    off2m = Counter(v % PAGE_2M for v in vas)
    print("\n2 MiB-page offset histogram (the real common structure):")
    for off, n in off2m.most_common(8):
        print(f"  0x{off:07x}  x{n}")

    mod2g = Counter(v % BAND_2G for v in vas)
    print("\nmod 2 GiB histogram (NOT shared -- three values differ):")
    for off, n in mod2g.most_common(8):
        print(f"  0x{off:09x}  x{n}")

    # --- explicit refutation of the withdrawn claim -------------------------
    print("\nrefutation of the withdrawn '<2GiB base> + 0x39af000' claim:")
    for v in uniq[:6]:
        b = v - 0x39AF000
        print(f"  fault 0x{v:012x} -> minus 0x39af000 = 0x{b:012x}  "
              f"2GiB-aligned: {b % BAND_2G == 0}")

    # --- what a match should test instead -----------------------------------
    print("\nany future allocation match must use a *measured* invariant.")
    print("The only one with support is the 2 MiB-page offset above; a candidate")
    print("allocation A explains a fault at V if (V - A.base) is consistent with a")
    print("stride product the code can actually compute, which is what the GDN")
    print("state-index audit (handover 45) tests directly.")


if __name__ == "__main__":
    main()
