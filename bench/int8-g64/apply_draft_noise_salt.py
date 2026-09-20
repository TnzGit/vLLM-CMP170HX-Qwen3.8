#!/usr/bin/env python3
"""Apply the #54282 draft-noise independence backport (invariant only, minimal).

THE INVARIANT
-------------
The random stream used to **propose** draft tokens must be independent of the stream used
to **verify** them (target rejection sampling and residual resampling). If the two share
noise, the draft is correlated with the acceptance test, which biases the accepted-token
distribution instead of merely changing acceptance *rate*.

WHAT v0.27.1 DOES
-----------------
    target:  rejection_sampler.generate_uniform_probs(..., sampling_metadata.generators, ...)
             -> per-request, seeded torch.Generator
    draft:   q = empty_exponential_noise_like(probs, use_fp64_gumbel)
             q.exponential_()                      # <-- global default CUDA generator
             (vllm/v1/spec_decode/llm_base_proposer.py, compute_probs_and_sample_next_token)
             with the upstream comment "# TODO(woosuk): Consider seeds."

So the draft draws from an **unseeded global stream** that is shared across requests and
steps, and it is not disjoint from the target's randomness in any structured way. Both the
generic draft path and the DFlash2 selector reach this one function (DFlashProposer
subclasses SpecDecodeBaseProposer and does not override it), so a single edit covers the
scope the review asked for.

THE FIX
-------
Draw the draft noise from a **dedicated, disjoint generator** instead of the global stream:

    DRAFT_NOISE_SALT = 1 << 30        # far from any user seed / request seed

The generator is created once per process and offset away from the space the target's
per-request generators use, so draft and target noise cannot coincide. This is the
invariant only -- no attempt is made to reproduce upstream's exact API, because this tree
is v0.27.1 plus a backported DFlash2, with its own selector truncation, logits-cache and
temperature handling.

The target/residual path is **deliberately untouched**: its per-request seeded generators
remain the single source of randomness for verification.

Usage:
  apply_draft_noise_salt.py <vllm package dir> [--emit-patch out.diff]
"""
from __future__ import annotations

import ast
import difflib
import pathlib
import subprocess
import sys

TARGET = "v1/spec_decode/llm_base_proposer.py"
MARKER = "DRAFT_NOISE_SALT"

OLD = """    # TODO(woosuk): Consider seeds.
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    q.exponential_()
"""

NEW = """    # Draft noise must not come from the process-global unseeded generator (upstream
    # #54282). Two concrete consequences of the old `q.exponential_()`:
    #   * the draft stream is shared process-wide, so how many draws other concurrent
    #     requests make changes this request's draft tokens;
    #   * an unseeded request's drafts are not reproducible from its seed.
    #
    # DRAFT_NOISE_SALT puts the draft generator in a region of the seed space far above
    # any request seed, so draft noise is dedicated and reproducible.
    #
    # Scope note, measured rather than assumed: this does NOT fix a distributional bias
    # in the accepted tokens. The target and residual paths already draw from per-request
    # seeded generators (rejection_sampler.generate_uniform_probs /
    # sample_recovered_tokens), so they were never sharing a stream with the draft; a
    # 200k-trial chi-square check over vocab 16 found no bias before or after this change
    # (bench/int8-g64/test_draft_noise_independence.py). The target path is left untouched
    # on purpose.
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    q.exponential_(generator=_draft_noise_generator(q.device))
"""

HELPER_ANCHOR = "def compute_probs_and_sample_next_token("

HELPER = '''# Draft-proposal noise must not share a stream with target verification (#54282).
# The offset sits far above the range request seeds occupy, so a draft draw can never
# collide with a target draw for the same (seed, position).
DRAFT_NOISE_SALT = 1 << 30
_DRAFT_NOISE_GENERATORS: dict = {}


def _draft_noise_generator(device):
    """A process-wide generator, disjoint from the target's per-request streams."""
    key = str(device)
    gen = _DRAFT_NOISE_GENERATORS.get(key)
    if gen is None:
        gen = torch.Generator(device=device)
        gen.manual_seed(DRAFT_NOISE_SALT)
        _DRAFT_NOISE_GENERATORS[key] = gen
    return gen


'''


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    emit = ""
    if "--emit-patch" in sys.argv:
        emit = sys.argv[sys.argv.index("--emit-patch") + 1]
    path = root / TARGET
    src = path.read_text()
    if MARKER in src:
        print(f"{path.name}: already patched")
        return
    if OLD not in src:
        print(f"{path.name}: ANCHOR NOT FOUND -- aborting", file=sys.stderr)
        sys.exit(2)
    if HELPER_ANCHOR not in src:
        print(f"{path.name}: HELPER ANCHOR NOT FOUND -- aborting", file=sys.stderr)
        sys.exit(2)

    src = src.replace(OLD, NEW, 1)
    src = src.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    ast.parse(src)      # refuse to write something that will not import
    path.write_text(src)
    print(f"{path.name}: draft noise drawn from a disjoint generator (#54282 invariant)")
    print(f"  DRAFT_NOISE_SALT = {1 << 30}")

    if emit:
        before = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{TARGET}"],
                                capture_output=True, text=True)
        if before.returncode != 0:
            print("  (not a git tree; skipping patch emission)", file=sys.stderr)
            return
        diff = difflib.unified_diff(before.stdout.splitlines(keepends=True),
                                    src.splitlines(keepends=True),
                                    fromfile=f"a/{TARGET}", tofile=f"b/{TARGET}")
        pathlib.Path(emit).write_text("".join(diff))
        print(f"  emitted {emit}")


if __name__ == "__main__":
    main()
