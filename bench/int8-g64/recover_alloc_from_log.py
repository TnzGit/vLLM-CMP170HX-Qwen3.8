#!/usr/bin/env python3
"""Recover the segment map from the engine log and match the fault displacement.

alloc_probe.py printed the segment table to the engine log, but both the APIServer
(which has no GPU work) and the EngineCore wrote their JSON to the same path, so
the APIServer's empty dump clobbered the real one. The log still has the EngineCore
table, so parse it from there.

Then answer the question directly: is there a segment base such that
`base + 0x39af000` equals an address at which a fault was actually observed?
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter

FAULT_OFFSET = 0x39AF000


def segments_from_log(path: str) -> list[dict]:
    text = open(path, errors="ignore").read()
    pat = re.compile(
        r"base (0x[0-9a-f]{16})\s+size\s+([0-9.]+) MiB\s+"
        r"active\s+([0-9.]+) MiB\s+blocks (\d+)")
    seen: dict[int, dict] = {}
    for m in pat.finditer(text):
        addr = int(m.group(1), 16)
        seen[addr] = {
            "address": addr,
            "address_hex": m.group(1),
            "size": int(float(m.group(2)) * 2**20),
            "size_mib": float(m.group(2)),
            "active_mib": float(m.group(3)),
            "n_blocks": int(m.group(4)),
        }
    return sorted(seen.values(), key=lambda r: -r["size"])


def fault_addresses() -> list[int]:
    raw = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True,
                         errors="ignore").stdout
    return sorted({int(m.group(1).replace("_", ""), 16)
                   for m in re.finditer(r"faulted @ 0x([0-9a-f_]+)", raw)})


def main() -> None:
    log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/graph_engine_NONE.log"
    segs = segments_from_log(log)
    faults = fault_addresses()
    print(f"segments recovered from log: {len(segs)}")
    print(f"distinct observed fault VAs (all generations): {len(faults)}")

    out = "/tmp/alloc_from_log.json"
    json.dump(segs, open(out, "w", encoding="utf-8"), indent=2)
    print(f"wrote {out}\n")

    print("largest segments:")
    for s in segs[:14]:
        print("  {}  {:10.3f} MiB  active {:10.3f} MiB".format(
            s["address_hex"], s["size_mib"], s["active_mib"]))

    bases = {s["address"] for s in segs}
    print("\nexact match: observed fault VA == segment_base + 0x39af000")
    hits = 0
    for v in faults:
        b = v - FAULT_OFFSET
        if b in bases:
            hits += 1
            s = next(x for x in segs if x["address"] == b)
            print("  fault 0x{:016x} == base 0x{:016x} + 0x39af000   "
                  "size {:.3f} MiB".format(v, b, s["size_mib"]))
    if not hits:
        print("  no exact match against THIS generation's bases")
        print("  (expected: the probe dumped a process that had not faulted, and")
        print("   dmesg retains many earlier generations with different bases)")

    # Generation-independent statement: what sizes do 2 GiB-aligned segments have?
    two_gib = Counter()
    for s in segs:
        if s["address"] % (1 << 31) == 0:
            two_gib[round(s["size_mib"], 3)] += 1
    print("\n2 GiB-aligned segments by size (candidate pool/major buffers):")
    for size_mib, n in two_gib.most_common(12):
        print("  {:10.3f} MiB  x{}".format(size_mib, n))

    # Does any non-2GiB-aligned segment have a size that could host 0x39af000?
    print("\nnon-2GiB-aligned segments larger than 0x39af000:")
    for s in segs:
        if s["address"] % (1 << 31) and s["size"] > FAULT_OFFSET:
            print("  0x{:016x}  {:10.3f} MiB  active {:10.3f} MiB".format(
                s["address"], s["size_mib"], s["active_mib"]))


if __name__ == "__main__":
    main()
