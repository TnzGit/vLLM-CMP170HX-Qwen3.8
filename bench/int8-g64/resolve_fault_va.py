#!/usr/bin/env python3
"""Resolve Xid 31 fault addresses against the engine's live KV allocation bases.

The KV backing allocations are 2 GiB-aligned and 5168 MiB each (handover 38.5),
so a fault address that lands inside one of them can be converted to an offset
that is meaningful: "the faulting read was N MiB into the KV pool for layers X..Y".

Also checks the reverse: for each fault VA, the distance to the nearest
allocation base and to its end, which distinguishes "inside a live pool", "just
past the end of a pool" (a classic overrun) and "in an unmapped gap".
"""
import argparse
import re
import subprocess
import sys

TWO_GIB = 1 << 31


def fault_vas() -> list[int]:
    raw = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True,
                         errors="ignore").stdout
    return [int(m.group(1).replace("_", ""), 16)
            for m in re.finditer(r"faulted @ 0x([0-9a-f_]+)", raw)]


def parse_alloc_bases(log_path: str) -> dict[int, int]:
    """{base: size} from the ALLOC kv_cache lines."""
    out: dict[int, int] = {}
    pat = re.compile(r"ALLOC kv_cache base=0x([0-9a-f]+) size=(\d+) bytes")
    try:
        text = open(log_path, errors="ignore").read()
    except OSError as exc:
        print(f"cannot read {log_path}: {exc}", file=sys.stderr)
        return out
    for m in pat.finditer(text):
        base = int(m.group(1), 16)
        out.setdefault(base, int(m.group(2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="engine log with ALLOC lines")
    args = ap.parse_args()

    bases = parse_alloc_bases(args.log)
    vas = fault_vas()
    if not bases:
        print("no ALLOC lines found -- is patches/log-alloc-bases.patch applied "
              "to v1/worker/gpu/attn_utils.py?", file=sys.stderr)
        sys.exit(2)

    print("live KV backing allocations:")
    for b, s in sorted(bases.items()):
        print(f"  base 0x{b:016x}  size {s / 2**20:9.2f} MiB  "
              f"end 0x{b + s:016x}  (2GiB-aligned: {b % TWO_GIB == 0})")

    print(f"\n{len(vas)} Xid 31 fault addresses\n")
    print(f"{'fault VA':>18}  {'verdict':<26} {'offset in pool MiB':>18}  "
          f"{'dist to nearest base MiB':>24}")
    inside = past_end = gap = 0
    for v in sorted(set(vas)):
        hit = None
        for b, s in bases.items():
            if b <= v < b + s:
                hit = (b, s)
                break
        if hit:
            inside += 1
            off = (v - hit[0]) / 2**20
            print(f"0x{v:016x}  {'INSIDE KV POOL':<26} {off:>18.3f}  "
                  f"{0.0:>24.3f}")
        else:
            # nearest base below and nearest end above
            below = [(v - b, b, s) for b, s in bases.items() if b <= v]
            if below:
                d, b, s = min(below)
                past = (v - (b + s)) / 2**20
                past_end += 1
                print(f"0x{v:016x}  {'PAST END of pool':<26} {'-':>18}  "
                      f"base+{d / 2**20:>18.3f} (over end by {past:.3f})")
            else:
                gap += 1
                print(f"0x{v:016x}  {'BELOW all pools':<26} {'-':>18}  {'-':>24}")

    print(f"\nsummary: inside={inside} past_end={past_end} below_all={gap}")
    print("\nNote: dmesg is a ring buffer containing faults from EARLIER engine "
          "processes too. Bases differ per process, so only faults whose band "
          "matches the current bases are resolved here; compare bands, not exact "
          "addresses.")


if __name__ == "__main__":
    main()
