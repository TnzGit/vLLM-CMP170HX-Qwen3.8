#!/usr/bin/env python3
"""Diff the two ab_quality.py arms: token agreement and logprob drift.

Reports, over aligned tokens of the same prompt:
  * exact token-prefix agreement (how far the two arms decode identically)
  * mean |delta logprob| where both arms emitted the same token
  * mean logprob of each arm (a lower value means the model found the text less
    likely, which is what a lossy KV cache shows up as)

Usage: ab_quality_compare.py quality-baseline.json quality-g64.json
"""
from __future__ import annotations

import json
import sys


def load(path: str) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {r["prompt_index"]: r for r in d["results"]}


def main() -> None:
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)

    common_prefix_total = 0
    tokens_total = 0
    deltas = []
    lp_a, lp_b = [], []
    identical_prompts = 0
    per_prompt = []

    for idx in sorted(set(a) & set(b)):
        ta, tb = a[idx]["tokens"], b[idx]["tokens"]
        la, lb = a[idx]["token_logprobs"], b[idx]["token_logprobs"]
        n = min(len(ta), len(tb))
        same = 0
        for i in range(n):
            if ta[i] == tb[i]:
                same += 1
                if la[i] is not None and lb[i] is not None:
                    deltas.append(abs(la[i] - lb[i]))
            else:
                break
        common_prefix_total += same
        tokens_total += max(len(ta), len(tb))
        lp_a.extend(v for v in la if v is not None)
        lp_b.extend(v for v in lb if v is not None)
        if ta == tb:
            identical_prompts += 1
        per_prompt.append((idx, len(ta), len(tb), same))

    print(f"A = {a_path}")
    print(f"B = {b_path}")
    print(f"prompts compared        : {len(per_prompt)}")
    print(f"byte-identical outputs  : {identical_prompts}/{len(per_prompt)}")
    print(f"shared token prefix     : {common_prefix_total}/{tokens_total} "
          f"({100.0 * common_prefix_total / max(1, tokens_total):.2f}%)")
    if deltas:
        print(f"mean |dlogprob| on agreement: {sum(deltas) / len(deltas):.5f} "
              f"(n={len(deltas)}, max={max(deltas):.5f})")
    print(f"mean logprob A          : {sum(lp_a) / max(1, len(lp_a)):.4f} (n={len(lp_a)})")
    print(f"mean logprob B          : {sum(lp_b) / max(1, len(lp_b)):.4f} (n={len(lp_b)})")
    print("\nper-prompt (idx, tokens_A, tokens_B, identical_prefix):")
    for idx, na, nb, same in per_prompt:
        flag = "" if na == nb and same == na else "   <-- diverged"
        print(f"  {idx:2d}  {na:4d} {nb:4d} {same:4d}{flag}")


if __name__ == "__main__":
    main()
