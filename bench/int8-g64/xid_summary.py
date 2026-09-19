#!/usr/bin/env python3
"""Summarise NVRM Xid events on the lab host, separated by time window.

Run on the host that owns the GPU (needs `sudo -n dmesg`). The point is to
separate *this* investigation's faults from historical ones, because dmesg is a
ring buffer that keeps events from earlier unrelated sessions: an Xid seen 3 days
ago must not be attributed to today's runs.
"""
import re
import subprocess
import sys

now = float(open("/proc/uptime").read().split()[0])
raw = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True,
                     errors="ignore").stdout

events = []
for line in raw.splitlines():
    m = re.match(r"\[\s*([0-9.]+)\]", line)
    if not m:
        continue
    t = float(m.group(1))
    x = re.search(r"Xid \(PCI:[0-9a-f:]+\): (\d+)", line)
    if not x:
        continue
    addr = re.search(r"faulted @ 0x([0-9a-f_]+)", line)
    events.append({
        "t": t,
        "xid": int(x.group(1)),
        "addr": addr.group(1) if addr else None,
        "line": line,
    })

if not events:
    print("no Xid events found")
    sys.exit(0)

print(f"uptime now: {now:.0f}s ({now/86400:.2f} days)")
print()
for win, name in ((3600, "last 1h"), (21600, "last 6h"),
                  (86400, "last 24h"), (1e12, "all")):
    counts = {}
    for e in events:
        if now - e["t"] <= win:
            counts[e["xid"]] = counts.get(e["xid"], 0) + 1
    summary = ", ".join(f"Xid{k}={v}" for k, v in sorted(counts.items())) or "(none)"
    print(f"{name:>9}: {summary}")

print()
x31 = [e for e in events if e["xid"] == 31]
if x31:
    first, last = min(e["t"] for e in x31), max(e["t"] for e in x31)
    print(f"Xid 31: {len(x31)} events, first at uptime {first:.0f}s "
          f"(={(now-first)/3600:.1f}h ago), last at {last:.0f}s "
          f"(={(now-last)/60:.0f}min ago)")
    addrs = {}
    for e in x31:
        if e["addr"]:
            addrs[e["addr"]] = addrs.get(e["addr"], 0) + 1
    print(f"  distinct MMU fault addresses: {len(addrs)}")
    for a, c in sorted(addrs.items(), key=lambda x: -x[1])[:8]:
        print(f"    {c:>3}x  0x{a}")

x13 = [e for e in events if e["xid"] == 13]
if x13:
    first, last = min(e["t"] for e in x13), max(e["t"] for e in x13)
    print()
    print(f"Xid 13: {len(x13)} events, first at uptime {first:.0f}s "
          f"(={(now-first)/3600:.1f}h ago), last at {last:.0f}s "
          f"(={(now-last)/60:.0f}min ago)")
    recent = [e for e in x13 if now - e["t"] <= 21600]
    print(f"  in the last 6h: {len(recent)}")
