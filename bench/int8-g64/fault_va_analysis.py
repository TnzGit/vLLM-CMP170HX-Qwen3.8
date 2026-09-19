#!/usr/bin/env python3
"""Analyse Xid 31 fault addresses for alignment / repetition structure.

Absolute CUDA virtual addresses are not actionable on their own, but their
*alignment* and their *relative* repetition across engine processes can be. Each
engine process gets a similar allocator layout, so a faulting VA that recurs at
the same low bits across different processes is more likely a fixed tensor base
plus a deterministic bad offset than a random wild pointer.

Print, for each distinct faulting VA: how often it occurred, its low bits at
2 MiB / 64 KiB granularity (the sizes CUDA allocations are actually rounded to),
and whether the set clusters.
"""
import re
import subprocess
import sys
from collections import Counter

raw = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True,
                     errors="ignore").stdout

vas = [int(m.group(1).replace("_", ""), 16)
       for m in re.finditer(r"faulted @ 0x([0-9a-f_]+)", raw)]
if not vas:
    print("no Xid 31 fault addresses found")
    sys.exit(0)

counts = Counter(vas)
total = len(vas)
print(f"Xid 31 events: {total}   distinct VAs: {len(counts)}")
print()


def align_report(va: int) -> str:
    return (f"2MiB=0x{va & ((1 << 21) - 1):07x}  "
            f"64KiB=0x{va & 0xFFFF:04x}  "
            f"4KiB=0x{va & 0xFFF:03x}")


print(f"{'n':>4}  {'VA':>18}  {'low bits (2MiB/64KiB/4KiB)':<40}")
for va, n in counts.most_common(14):
    print(f"{n:>4}  0x{va:016x}  {align_report(va)}")

print()
# Do the faults cluster in a narrow address band? A tight band is consistent with
# one allocation (or one arena) being addressed out of range.
lo, hi = min(vas), max(vas)
print(f"address range: 0x{lo:016x} .. 0x{hi:016x}  "
      f"(span {(hi - lo) / 2**30:.2f} GiB)")

# If faults were uniformly spread over the 64 GiB framebuffer we would expect a
# wide span; a narrow span suggests a specific allocation is implicated.
band = 2 ** 31  # 2 GiB
clusters = Counter(v // band for v in vas)
print(f"\n2 GiB bands touched: {len(clusters)}")
for b, n in clusters.most_common(6):
    print(f"  0x{b * band:011x}-0x{(b + 1) * band - 1:011x}  {n} events")

# Repeating low bits across DIFFERENT high bits would indicate a fixed offset
# within different allocations.
low16 = Counter(v & 0xFFFF for v in vas)
repeated = [(k, n) for k, n in low16.most_common(6) if n > 1]
print(f"\nrepeated low-16 bits (would indicate a fixed in-allocation offset):")
if repeated:
    for k, n in repeated:
        print(f"  0x{k:04x}  x{n}")
else:
    print("  none")
