#!/usr/bin/env python3
"""Offsets of each Xid 31 fault VA within its 2 GiB band.

The band-level clustering is only informative if the *offsets inside a band*
repeat. A repeated offset means a fixed bad address relative to whatever
allocation occupies that band; scattered offsets mean the whole arena is suspect.

Also reports, for each band, the offset histogram at 2 MiB granularity, since
CUDA rounds large allocations to 2 MiB -- so an offset that lands on a clean 2 MiB
boundary is itself a hint about which kind of allocation is being addressed.
"""
import re
import subprocess
from collections import Counter, defaultdict

raw = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True,
                     errors="ignore").stdout
vas = [int(m.group(1).replace("_", ""), 16)
       for m in re.finditer(r"faulted @ 0x([0-9a-f_]+)", raw)]

BAND = 1 << 31  # 2 GiB
by_band = defaultdict(list)
for v in vas:
    by_band[v // BAND].append(v)

print(f"{len(vas)} Xid 31 events across {len(by_band)} 2 GiB bands\n")
for band, vs in sorted(by_band.items(), key=lambda kv: -len(kv[1])):
    base = band * BAND
    offs = [v - base for v in vs]
    uniq = Counter(offs)
    print(f"band 0x{base:011x} .. 0x{base + BAND - 1:011x}: {len(vs)} events, "
          f"{len(uniq)} distinct offsets")
    for off, n in uniq.most_common(6):
        print(f"    n={n:<3} offset 0x{off:09x} = {off / 2**20:9.3f} MiB "
              f"(mod 2MiB = 0x{off % (2 * 1024 * 1024):06x})")
    print()

# The actionable question: does any offset repeat ACROSS different bands? That
# would mean a fixed relative address regardless of which allocation is there.
all_offs = Counter()
for band, vs in by_band.items():
    all_offs.update(v - band * BAND for v in vs)
cross = [(o, n) for o, n in all_offs.most_common(10) if n > 1]
print("offsets repeating across bands (fixed relative address):")
if cross:
    for o, n in cross:
        print(f"  0x{o:09x} = {o / 2**20:.3f} MiB  x{n}")
else:
    print("  none -- offsets do not repeat across bands")
