"""Summarize sparse-read e2e JSONs without reinterpretation.

Tag convention used by the execution prompt:
  dense-250000-r1
  s32-250000-r1
  s48-250000-r1
  s65-250000-r1

Rows are grouped by exact prompt contract SHA/token count and arm.  Medians are
reported only when at least one valid result exists.  Delta is always relative
to the dense arm for the same prompt contract.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path


def median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def fmt(v, digits=2):
    return "null" if v is None else f"{v:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    paths: list[Path] = []
    for item in args.files:
        matches = glob.glob(item)
        paths.extend(Path(x) for x in (matches or [item]))

    groups = defaultdict(lambda: defaultdict(list))
    for path in paths:
        row = json.loads(path.read_text())
        tag = row["tag"]
        arm = tag.split("-", 1)[0]
        contract = row["prompt"]
        key = (contract["prompt_sha256"], int(contract["prompt_tokens"]))
        groups[key][arm].append(row)

    print(
        "| prompt_tokens | arm | n | decode tok/s | delta vs dense | "
        "counter ms/step | tok/step | preemptions |"
    )
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    order = {"dense": 0, "s32": 1, "s48": 2, "s65": 3}
    for (_, prompt_tokens), arms in sorted(groups.items(), key=lambda kv: kv[0][1]):
        dense_tps = median(
            x["decode_interval"]["tok_s"] for x in arms.get("dense", [])
        )
        for arm, rows in sorted(
            arms.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])
        ):
            tps = median(x["decode_interval"]["tok_s"] for x in rows)
            ms = median(
                x["counter_interval"]["ms_per_spec_iteration"] for x in rows
            )
            tstep = median(x["counter_interval"]["tok_per_step"] for x in rows)
            pre = median(x["counter_interval"]["preemptions_delta"] for x in rows)
            delta = None
            if tps is not None and dense_tps:
                delta = (tps / dense_tps - 1.0) * 100.0
            print(
                f"| {prompt_tokens} | {arm} | {len(rows)} | {fmt(tps)} | "
                f"{fmt(delta, 1)}% | {fmt(ms, 3)} | {fmt(tstep, 3)} | {fmt(pre, 0)} |"
            )

    # Fail loudly if an experiment set has sparse results but no same-contract dense arm.
    bad = []
    for key, arms in groups.items():
        if any(a != "dense" for a in arms) and "dense" not in arms:
            bad.append(key)
    if bad:
        raise SystemExit(f"sparse results without same-contract dense baseline: {bad}")


if __name__ == "__main__":
    main()
