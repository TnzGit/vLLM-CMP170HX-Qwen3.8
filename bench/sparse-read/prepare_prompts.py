"""Prepare deterministic prompt files reused byte-for-byte across A/B arms.

These labels are approximate context targets.  The formal token count is the
server-reported usage.prompt_tokens recorded by e2e_decode_gate.py.  The first
dense run creates a contract containing that exact count; every sparse run must
match it.

The default source is the varied long corpus created by bench/make_long_corpus.py.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--contexts",
        type=csv_ints,
        default=[4096, 32000, 65000, 126000, 250000],
    )
    ap.add_argument(
        "--corpus",
        type=Path,
        default=Path.home() / "bench/labd_corpus_long.txt",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "bench/sparse-read",
    )
    ap.add_argument(
        "--chars-per-token",
        type=float,
        default=2.90,
        help="only chooses a deterministic byte cut; actual tokens come from API usage",
    )
    args = ap.parse_args()

    corpus = args.corpus.read_text()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ctx in args.contexts:
        # Leave room for chat framing and the fixed task suffix.  This need not
        # hit ctx exactly; equality across arms is enforced by the contract.
        want_chars = max(1024, int((ctx - 256) * args.chars_per_token))
        if want_chars > len(corpus):
            raise SystemExit(
                f"corpus too short for ctx={ctx}: need {want_chars} chars, "
                f"have {len(corpus)}"
            )
        marker = (
            f"\n\n[SPARSE_GATE_CONTEXT_LABEL={ctx}; "
            "THIS MARKER MAKES CONTEXT CELLS UNIQUE]\n"
        )
        text = corpus[:want_chars] + marker
        path = args.out_dir / f"ctx{ctx}.txt"
        path.write_text(text)
        print(
            f"{ctx:7d} {path} bytes={path.stat().st_size} "
            f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
        )


if __name__ == "__main__":
    main()
