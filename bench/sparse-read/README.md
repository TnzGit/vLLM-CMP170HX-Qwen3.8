# Sparse-read execution runbook

This directory is intentionally prescriptive.  The GPU operator executes these
steps and returns the artifacts; it does not redesign the experiment.

Design and stop criteria: [../../docs/sparse-read-oracle-design.md](../../docs/sparse-read-oracle-design.md)  
Implementation contract: [../../experimental/cmp170hx-sparse-read/README.md](../../experimental/cmp170hx-sparse-read/README.md)

## 0. Safety

- production port **8000 is never touched**;
- research server is **8002 only**;
- do not use `pkill -f`;
- after any CUDA failure, treat that engine as poisoned and restart it;
- record Xid **delta**, not the machine's lifetime count;
- actual EngineCore `/proc/<pid>/environ` and startup logs override launcher intent.

## 1. Install into both vLLM sites

The host used for M7 has a runtime vLLM and a shadow mixed-FP8 site.  Patch both.

```bash
RUNTIME=/home/base-node/.codex_tasks/pixelml-cmp170hx/runtime-v0271/venv/lib/python3.12/site-packages/vllm
SHADOW=/home/base-node/.codex_tasks/pixelml-cmp170hx/mixed-fp8-test-site/vllm

experimental/cmp170hx-sparse-read/install.sh --dry-run --site "$RUNTIME"
experimental/cmp170hx-sparse-read/install.sh --dry-run --site "$SHADOW"

experimental/cmp170hx-sparse-read/install.sh --apply --site "$RUNTIME"
experimental/cmp170hx-sparse-read/install.sh --apply --site "$SHADOW"

experimental/cmp170hx-sparse-read/install.sh --check --site "$RUNTIME"
experimental/cmp170hx-sparse-read/install.sh --check --site "$SHADOW"

"$VENV/bin/python" -m py_compile \
  "$RUNTIME/v1/attention/ops/spec_decode_attn.py" \
  "$SHADOW/v1/attention/ops/spec_decode_attn.py" \
  experimental/cmp170hx-sparse-read/apply_sparse_read.py \
  bench/sparse-read/*.py
```

If any dry-run/check/compile fails: **stop and return the exact error**.  Do not
edit the installed source by hand.

## 2. Kernel gate first

With no API server using the research GPU:

```bash
mkdir -p ~/bench/results/sparse-read
"$VENV/bin/python" bench/sparse-read/spec_attn_sparse_ctx_scan.py \
  | tee ~/bench/results/sparse-read/microbench.txt
```

Required before e2e:

- every `table_ok=True`;
- every below-budget identity row remains bit-identical;
- no CUDA/Xid error;
- 126K/250K sparse latency clearly follows the chosen active budget.

If the 250K / 32K-budget kernel path does not show a meaningful structural
reduction (roughly <1.2x is an immediate warning), stop and report instead of
building more features.

## 3. Freeze A/B inputs

```bash
# If the varied long corpus does not exist, build it using the repository tool.
"$VENV/bin/python" bench/make_long_corpus.py

"$VENV/bin/python" bench/sparse-read/prepare_prompts.py
```

Never regenerate these files between dense and sparse arms.

## 4. E2E staging

Formal long profile:

- k=3;
- MAX_SEQS=1;
- NSEG=35;
- FULL CUDA Graph;
- 1350 MHz locked;
- 180 W;
- port 8002.

Launch one arm at a time:

```bash
bench/sparse-read/start_gate_server.sh dense
bench/sparse-read/start_gate_server.sh 32000
bench/sparse-read/start_gate_server.sh 48000
bench/sparse-read/start_gate_server.sh 65000
```

The launcher stays in the foreground by design.  The operator may put it in a
dedicated terminal or a controlled background process group.  Record the exact
PID(s).  Never reuse an engine for a different arm or formal context cell.

For each running arm/context/round:

```bash
"$VENV/bin/python" bench/sparse-read/e2e_decode_gate.py \
  --tag ARM-CONTEXT-rROUND \
  --prompt-file ~/bench/sparse-read/ctxCONTEXT.txt \
  --contract ~/bench/sparse-read/ctxCONTEXT.contract.json \
  --port 8002
```

Use arm tags exactly:

- `dense`
- `s32`
- `s48`
- `s65`

Example:

```bash
"$VENV/bin/python" bench/sparse-read/e2e_decode_gate.py \
  --tag dense-250000-r1 \
  --prompt-file ~/bench/sparse-read/ctx250000.txt \
  --contract ~/bench/sparse-read/ctx250000.contract.json \
  --port 8002
```

### Cost-saving order

Do not run the whole matrix blindly.

1. 4K dense vs s32: below-budget identity/control.
2. 250K dense vs s32: primary structural gate, preferably 3 rounds.
3. If 250K wins materially, run 126K dense vs s32.
4. Then run 250K s48 and s65.
5. Only after those succeed fill the remaining 32K/65K/126K control cells.

A practical high-value 250K region is ~90+ tok/s, but this is not a correctness
threshold.  Primary evidence is a repeatable reduction in ms/spec-iteration plus
e2e decode improvement outside spread.

## 5. Runtime evidence for every fresh engine

Save:

- startup log;
- `nvidia-smi` clock/power snapshot;
- Xid count before/after;
- parsed effective block/page line;
- first `[cmp-sparse-read]` line for sparse arms;
- EngineCore environment from `/proc/<pid>/environ`.

The sparse startup evidence must show the intended values, especially:

```text
VLLM_SPARSE_KV_READ=1
VLLM_SPARSE_KV_READ_TOKENS=<arm>
VLLM_SPEC_DECODE_ATTN_SEGMENTS=35
VLLM_FP8_SPEC_VERIFY=1
VLLM_FP8_SPEC_FULL_CG=1
DFLASH_TOKENS=3
MAX_SEQS=1
```

Do not infer these from the wrapper.

## 6. Summarize

```bash
"$VENV/bin/python" bench/sparse-read/analyze_results.py \
  ~/bench/results/sparse_read_*.json \
  | tee ~/bench/results/sparse-read/summary.md
```

Return the JSONs and raw logs with the table.  Do not summarize a failed cell
into a plausible number.

## 7. Forbidden follow-up during this campaign

Without a new design review, do not implement or tune:

- Mean-K / semantic retrieval;
- CPU or NVMe KV;
- page geometry;
- scheduler/admission;
- sparse draft KV;
- dynamic per-request sparse budgets;
- NSEG / k / MAX_SEQS;
- alternative sparse policies.

A failed gate is a result, not permission to move the goalposts.
