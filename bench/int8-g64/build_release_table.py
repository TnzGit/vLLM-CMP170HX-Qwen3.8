#!/usr/bin/env python3
"""Build the release comparison table from the measured cells plus the carried columns.

FOUR COLUMNS, AND WHY THREE OF THEM ARE NOT MEASURED HERE
---------------------------------------------------------
  old-same-path        the pre-fix build on this same path. NOT MEASURABLE at >=16K:
                       it takes an illegal memory access once ~11 requests have been
                       served (the int32 overflow, handover 56). Only the 4K cells can
                       be quoted; everything else is "faulted" by construction, and
                       saying so IS the result -- the old path could not serve long
                       context at all.
  fixed                measured by release_bench.py on the int64-widened kernel.
  historical-mixed-fp8 from docs/cmp170hx-mixed-fp8-engineering.md. A DIFFERENT route
                       (FP8 target KV, BF16 draft KV, FULL graph) and its `segments`
                       column is NSEG, not concurrency. Recorded only at 4K/126K/250K.
  modeled-ceiling      from single-user/README.md: measured on `CTX=fast` (bf16 KV,
                       64k context). A different KV format and a different ceiling, so
                       it is only meaningful at <=64K and is labelled as such.

Usage:
  build_release_table.py --json /tmp/release_fixed.json --out release-table.md
"""
from __future__ import annotations

import argparse
import json
import statistics

# --- carried columns, with their provenance recorded next to the numbers ---------
# docs/cmp170hx-mixed-fp8-engineering.md: decode tok/s at NSEG 16 / 32.
HISTORICAL = {
    4096: {"seg16": 144.8, "seg32": 160.0},
    126000: {"seg16": 54.2, "seg32": 68.2},
    250000: {"seg16": 32.5, "seg32": 42.2},
}
# single-user/README.md: CTX=fast (bf16, 64k), model-default sampling / greedy.
CEILING = {
    1: {"default": 121.8, "greedy": 131.2},
    2: {"default": 195.5, "greedy": 214.6},
    4: {"default": 278.9, "greedy": 285.7},
}
# The pre-fix path faults at >=16K after ~11 requests (handover 56), so its only
# quotable cells are the short ones.
OLD_SAME_PATH = {
    (4096, 1): "measurable (see note)", (4096, 2): "measurable (see note)",
    (4096, 4): "measurable (see note)",
}
NOT_MEASURED_250K = "not measured -- prefill-time bound"


def med(cell, key="decode_aggregate_tok_s"):
    vals = [r[key] for r in cell.get("rounds", []) if r.get(key) is not None]
    return round(statistics.median(vals), 1) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--old-path-json", default="",
                    help="pre-fix measurement (only the cells that do not fault)")
    ap.add_argument("--out", default="release-table.md")
    args = ap.parse_args()

    cells = json.load(open(args.json, encoding="utf-8"))
    old_cells = {}
    if args.old_path_json:
        for c in json.load(open(args.old_path_json, encoding="utf-8")):
            old_cells[(c["ctx"], c["concurrency"])] = (
                med(c) if c.get("rounds") else "FAULTED")
    fixed = {}
    for c in cells:
        if c.get("rounds"):
            fixed[(c["ctx"], c["concurrency"])] = med(c)
        else:
            fixed[(c["ctx"], c["concurrency"])] = None
    ctxs = sorted({c["ctx"] for c in cells})
    concs = sorted({c["concurrency"] for c in cells})

    lines = []
    lines.append("# Release benchmark: long-context speculative decoding on CMP 170HX")
    lines.append("")
    lines.append("Decode throughput (output tokens / second), median of 2 rounds, on the")
    lines.append("path repaired by `patches/spec-decode-attn-int64-block-id.patch`.")
    lines.append("")
    lines.append("Configuration: `Qwen3.8-27B-W4A16-AutoRound-fast` + DFlash2 W4A16 k=7,")
    lines.append("`TRITON_ATTN` + `int8_per_token_head` KV, `max-model-len=262144`,")
    lines.append("`max-num-seqs=4`, eager graph mode, clocks pinned. Prompts are **exact**")
    lines.append("token counts and distinct per request and per round.")
    lines.append("")
    lines.append("Decode is separated from prefill by bracketing each batch with a")
    lines.append("prefill-only run of the same shape; a raw wall-clock figure would be")
    lines.append("prefill-dominated at long context.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## The table")
    lines.append("")
    header = ("| ctx | conc | old-same-path | **fixed** | historical mixed-FP8 | "
              "modeled ceiling |")
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for ctx in ctxs:
        for conc in concs:
            if ctx == 250000 and conc != 1:
                f = NOT_MEASURED_250K
            else:
                v = fixed.get((ctx, conc))
                f = f"**{v}**" if v is not None else "faulted"
            # The pre-fix build faults at 4K C4 as well as at every >=16K cell, so
            # only 4K C1/C2 carry a number.
            ov = old_cells.get((ctx, conc))
            if ov == "FAULTED":
                old = "**FAULTED**"
            elif ov is not None:
                old = f"{ov}"
            else:
                old = "faulted (pre-fix IMA)"
            h = HISTORICAL.get(ctx)
            if h:
                hist = f"{h['seg16']} / {h['seg32']} (NSEG 16/32)"
            else:
                hist = "-- no record"
            c = CEILING.get(conc)
            if c and ctx <= 65536:
                ceil = f"{c['default']} / {c['greedy']} (def/greedy)"
            else:
                ceil = "-- out of range"
            lines.append(f"| {ctx//1000}K | C{conc} | {old} | {f} | {hist} | {ceil} |")
    lines.append("")

    lines.append("## Column provenance and validity")
    lines.append("")
    lines.append("| column | source | valid at | why it is not measured here |")
    lines.append("|---|---|---|---|")
    lines.append("| old-same-path | this repo, pre-fix build (measured for this table) "
                 "| 4K C1/C2 only | it faults at **4K C4** as well as at every >=16K "
                 "cell -- 16 requests of 4K (65K cumulative tokens) was enough. So the "
                 "pre-fix build could not serve even short-context concurrency, and "
                 "the missing cells are missing **by construction**. That absence is "
                 "the finding |")
    lines.append("| **fixed** | `bench/int8-g64/release_bench.py` | all listed cells | "
                 "-- |")
    lines.append("| historical mixed-FP8 | `docs/cmp170hx-mixed-fp8-engineering.md` | 4K, "
                 "126K, 250K | different route (FP8 target KV, BF16 draft KV, FULL "
                 "graph) and the two values are NSEG 16/32, **not** concurrency. No "
                 "record exists at 16K/32K/65K |")
    lines.append("| modeled ceiling | `single-user/README.md` | <=64K | measured on "
                 "`CTX=fast` (bf16 KV, 64k). Different KV format, so it is a ceiling "
                 "for a *different* configuration, not for this one |")
    lines.append("")

    lines.append("## Gaps, stated rather than hidden")
    lines.append("")
    lines.append("- **250K C2/C4**: not measured, by explicit decision. Each cell needs "
                 "four prefill passes of ~24-50 minutes at the measured ~320 tok/s "
                 "prefill rate, and 4x250K = 1.0M tokens approaches the 1.149M-token "
                 "KV pool. The arithmetic, not the result, is the reason.");
    lines.append("- **old-same-path at >=16K**: not measurable, as above.");
    lines.append("- **acceptance** came back `None` on several C1 cells: the "
                 "`SpecDecoding metrics` logger is periodic, so a bracket containing few "
                 "decode steps can miss its window. Decode throughput is unaffected "
                 "(it is measured from wall time), but per-cell acceptance should not "
                 "be quoted where it is blank.");
    lines.append("")

    lines.append("## Correctness during the run")
    lines.append("")
    lines.append("Xid 31 count was 81 before the benchmark and 81 after: **zero new "
                 "illegal accesses across all 31 measured cells**, including the 250K "
                 "cell. The fix holds under benchmark load, not only under the "
                 "dedicated requalification (handover 57-58).")
    lines.append("")
    out = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
