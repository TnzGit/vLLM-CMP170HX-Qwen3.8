# Gotchas

Things that each cost us hours, in rough order of pain. Worth skimming before you debug something that looks like a vLLM bug.

[← back to the main README](../README.md)

1. **A benchmark cannot tell you the output is garbage.** The int8-activation
   path served nonsense for an hour of beautiful throughput numbers before a
   perplexity check caught it. Whatever you change, run
   `bench/quality_battery.py` (perplexity + GSM8K against the live server)
   before you believe a tok/s number.
2. **Restart onto a dirty GPU and you silently lose 25%.** vLLM profiles free
   memory once at startup. If the previous process is still releasing VRAM at
   that moment, the cache pool comes out ~40% smaller and stays that way. No
   warning, the server runs fine, throughput is just quietly bad. The systemd
   units in both mode dirs carry an `ExecStartPre` gate that waits for the GPU
   to be actually free.
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not optional.** The
   DeltaNet prefill kernels allocate transient workspace; without it the
   allocator fragments and the engine OOMs at runtime once
   `gpu-memory-utilization` goes past ~0.975.
4. **With MTP enabled, even that isn't enough — single-user mode runs
   `gpu-memory-utilization 0.93`.** The speculative decode path's DeltaNet
   workspace grows beyond what vLLM's startup memory profiling measures, and
   the engine dies mid-request on long generations at 0.95+. It survives short
   benchmarks, which is exactly how it fools you. We soak-tested 0.93 with a
   100k-token prompt plus 6k-token generations at 4 concurrent.
5. **The torch.compile cache does not know about your env vars.** Switching
   `INT8_LAYERS` between runs replays a compiled graph that expects the other
   layer set and dies with `KeyError: 'input_global_scale'`. Our patch
   registers the selection env vars with vLLM so they become part of the cache
   key; if you invent your own, `VLLM_DISABLE_COMPILE_CACHE=1`.
6. **Random-token benchmarks are meaningless for speculative decoding.** The same
   server does 35, 83 or 151 tok/s on `--dataset-name random` depending on what
   the noise turns into, because acceptance depends entirely on whether the
   drafter can guess it. Use real prompts (`--dataset-name custom`). (This is our
   own measurement, not a description of anyone else's harness — ninfer-3090's
   published cohorts use short real prompts, not random tokens.)
7. **Bigger prefill chunks make things worse.** `--max-num-batched-tokens
   8192` inflates the profiled activation peak, which shrinks the cache pool,
   which caps concurrency. 2048 wins on this card.
8. **Benchmark twice.** The first run after any restart includes JIT warmup
   and reads 30-50% low.
9. **`--language-model-only` drops the vision tower cleanly** (no weights
   loaded), and it is the default in both start scripts. `VISION=1` keeps the
   tower for a client that sends images. The tower is **0.858 GiB**, not the
   2.7 GB this entry used to give: `model.visual.*` sums to 0.858 GiB of BF16 in both
   `Qwen3.8-27B-W4A16-AutoRound` and the `-fast` variant, and a runtime A/B
   agrees — model loading reads 15.13 GiB against 14.26 with
   `--language-model-only`, same server, same config, with non-weight overhead
   0.42 against 0.41 GiB either way. Quantization is not the explanation: the
   tower is BF16 in both dirs. Where 2.7 GB came from I could not work out.
   The KV pool came out within 0.4% across that pair (183,673 against 184,438
   tokens at 150k/fp8) — but the profiled activation peak differed by 0.87 GiB
   between those two starts, which is the same run-to-run swing the V2 runner
   shows, so read the pool difference as noise rather than as a measurement of
   what the tower costs.

   On `SPEC=dflash2` that 0.858 GiB is not optional headroom, it is the difference
   between booting and not. Measured here on a 3090 at 250 W, `SPEC=dflash2 VISION=1
   VISION_OFFLOAD=0`, `CTX=fast`: the engine dies in graph capture with
   `torch.OutOfMemoryError: Tried to allocate 960.00 MiB ... 787.50 MiB is free`
   (`spec_decode_attn.py:184`, the split-KV verify `part_o` buffer). The pool there is
   pinned by bytes (`KV_MEM`), so the tower cannot come out of the KV cache — it comes
   out of the ~1.1 GiB transient margin, and it is 0.85 of it. `VISION_OFFLOAD=1` (the
   default) makes the same config come up at the full 69,758-token pool and read images.
   The `SPEC=mtp` path has no such problem: it boots either way, and there the pool is
   profiling-sized, so what the tower costs is buried in the ±0.87 GiB profiling swing
   above (79,271 against 80,055 tokens across the pair — 1%, i.e. noise).

   What `VISION_OFFLOAD=1` does: the tower's weights live in pinned host RAM and each
   module is copied to the GPU for the duration of its own forward
   (`patches/vision-tower-cpu-offload.patch`). Isolated-tower measurement, RTX 3090 at
   PCIe 4.0 x16, one 8192-patch image, median of 10 forwards: resident weights 891.3 ->
   9.0 MiB, peak allocation 1160.5 -> 308.2 MiB, encode 296 -> 333 ms, output bit-exact
   against the resident tower. Note which offload path that is: vLLM's UVA *zero-copy*
   mode saves the same memory and costs 3327 ms, because the GEMMs then re-read operand
   tiles over PCIe inside the inner loop. The patch forces the bulk-copy path and does
   not touch what `--cpu-offload-gb` does elsewhere -- which could not reach the tower
   anyway, since the offloader is only installed in `make_layers()`.

   One precision from re-verifying the premise on a headless 3090, same 250 W:
   with nothing else on the card, `VISION_OFFLOAD=0` *does* boot -- and lands at
   440 MiB free after boot, inside gotcha 39's kill zone (396 MiB free died on a
   concurrent burst where 436 survived). The hard no-boot above needs something
   else holding a share of the card -- the measuring box also ran a desktop
   compositor and a browser, which is the normal state of a 3090 in a
   workstation. Same conclusion from both geometries: a resident tower puts the
   engine at the headroom cliff, the offload puts it at the full margin, and
   that is why the default is on. On the current tree the pool prints 68,605
   tokens (`KV_MEM` has moved since the 69,758 above was measured), identical
   between `VISION=0` and `VISION=1`, and the image round-trip reads a marker
   that exists only in the pixels either way.
10. **`prompt_logprobs` on long prompts OOMs the engine at 0.972 utilization**
    (a 300-token prompt needs ~300 MB of fp32 logits and there is no headroom).
    Run quality checks at 0.93.
11. **Don't chase the DeltaNet kernels.** `bench/tune_gdn.py` microbenchmarks
    the decode kernel across block/warp configs: it already runs at ~85% of the
    3090's memory bandwidth and every variant lands within 3%. The state dtype
    (point 3 above) is the lever, not the kernel.

12. **The draft vocabulary is the single-user ceiling.** A draft head can only
    propose tokens in its id list; a miss is a certain rejection that also ends
    the chain. Count the list over the model's *own* outputs (`drafter/gen_data.py`,
    then the frequency step in `prepare/build_draft_vocab.py`), not over web text —
    92% vs 97.5% coverage was the difference between 98 and 109 tok/s greedy.
    Coverage saturates around 40k rows; the model only ever emits ~54k distinct
    tokens.
13. **FlashAttention-2 does not split KV for multi-query decode.** With k
    speculative tokens the verify step has k+1 queries per request and FA2's
    varlen path then runs one thread block per (request, head): 24 blocks on 82
    SMs, 57 µs per layer at 1.5k context and 1.3 ms at 16k. vLLM's Triton
    unified attention has the same restriction (`max_seqlen_q > 1` → 2-D
    kernel). `patches/spec-decode-attn.patch` (`VLLM_SPEC_DECODE_ATTN=1`, bf16
    KV only) is a 180-line Triton fix. Watch its query cap: the kernel used to
    handle at most `BLOCK_M / (heads per kv head)` = 10 query tokens and fall
    back silently past that, which doubled the step at 25k context the moment
    the verify block grew to 16. It now tiles the query rows instead.
14. **Greedy is not deterministic across drafter configs.** The target rounds
    differently when it verifies 5 tokens vs 1, so a different drafter changes
    the generated text at near-ties and the 8-prompt acceptance numbers move
    ±3%. Repeat before trusting a small difference; `drafter/README.md` has an
    offline chain simulator that removes the noise.
15. **A stale torch.compile cache bites anything that changes tensor shapes
    behind vLLM's back.** The compiled graph bakes in e.g. the Marlin workspace
    size; a new env knob that changes it must be registered in `envs.py`
    (`patches/speed-knobs-envs.patch`) or you get `assert_size_stride ...
    expected size 328==82` from a cached artifact.
16. **The very first start gets a smaller KV pool.** vLLM sizes the pool from
    the peak memory of a profiling forward pass, and on a cold torch.compile
    cache that pass also runs inductor's autotuning: batch mode profiles a
    1.96 GiB activation peak instead of 1.09 GiB and comes up with 196k KV
    tokens instead of 224k (`Maximum concurrency ... 1.31x` in the log instead
    of 1.49x). Restart once after the cache is warm (venv: `~/.cache/vllm`,
    Docker: the `qwen-cache` volume) and the pool is back to the README numbers.
    (The WSL2 notes above pin it the other way round — record the cold-start
    `--kv-cache-memory` recommendation and pass it via `EXTRA_ARGS` — if you
    prefer the extra transient headroom to the extra KV pages.)
17. **vLLM picks the speculative method from the model *path*.** `"dflash" in
    model_path` switches `method` to dflash — for the *target* too, since MTP
    uses the target path as its draft model. A checkout under a directory with
    "dflash" in its name turns `SPEC=mtp` into a crash in `EAGLEConfig`
    (`'Qwen3_5Config' object has no attribute 'vocab_size'`). Name your
    directories accordingly.
18. **The V2 model runner (`SPEC=dflash2`) does not count its CUDA graphs when
    sizing the KV pool** (~1.2 GiB on top of whatever `--gpu-memory-utilization`
    you asked for), the hybrid allocator sizes KV groups by the smallest layer
    bucket, and the profiled activation peak varies by ~1 GiB between starts of
    the *same* config — three ways to get a server that either wastes a quarter
    of its pool or dies mid-request. `patches/hybrid-kv-groups-v2-cudagraph.patch`
    fixes the first two; for the third, pin the pool in bytes
    (`--kv-cache-memory`, what `KV_MEM` does) instead of tuning utilization. That
    runner also answers `thinking_token_budget` with 400, and the first request
    after a cold start JIT-compiles four Triton kernels (~5 s once; cached in
    `~/.triton`).
19. **`INT8_LAYERS=.` needs `GPU_UTIL=0.95`.** Quantizing the activations of every linear
    layer (rather than just the MLP) is worth ~11% throughput — 1,042 vs 942 tok/s at 64
    concurrent — but the extra per-layer scratch no longer fits batch mode's 0.972: the
    engine dies with `torch.OutOfMemoryError` inside `chunk_fwd_o` once ~17 requests are
    resident, which reads as every request returning 500 while `/health` still answers.
20. **A Triton kernel's scratch buffers may not grow after CUDA graph capture.** The
    split-KV verify attention sizes its partial buffers from the longest query block it has
    been asked for. Once the block got longer than the drafter's — and once a small prefill
    chunk could land on the same kernel — that "longest so far" changed mid-run, the buffers
    were reallocated, and the captured decode graph went on reading the freed ones:
    `CUDA error: an illegal memory access was encountered`, a few hundred tokens into the
    first request. `VLLM_SPEC_DECODE_ATTN_QMAX` fixes the size at startup instead. For
    DFlash parallel drafting, `single-user/start_qwen.sh` sets it to `1 + 2 *
    DFLASH_TOKENS`, matching vLLM's scheduler reorder threshold; `1 + DFLASH_TOKENS`
    is insufficient for valid uneven paths and for the scheduler-realistic warmup.
21. **Async scheduling pins the number of speculative tokens.** vLLM only feeds draft token
    ids — and therefore the *count* the worker wants verified — back to the scheduler on the
    synchronous path (`EngineCore.post_step`). With async scheduling on, every decode step is
    padded to `num_speculative_tokens` and a worker asking for fewer is ignored, silently.
    Adaptive block length (`LOOKUP=1` with `DFLASH_TOKENS > 7`) needs `ASYNC_SCHED=0`; at
    batch 1 that costs under 1%.
22. **`--async-scheduling` is already the default in 0.27.1.** The flag exists and passing it
    changes nothing; `--no-async-scheduling` is what turns it off. Two hours of "the adaptive
    block isn't working" was this.
23. **A longer verify block costs KV pool per request slot, not per token.**
    `--mamba-cache-mode align` reserves `2 + num_speculative_blocks` recurrent-state pages
    per slot, so `DFLASH_TOKENS=31` with 8 slots wants 5.3 GiB before a single token of
    context and refuses to start. Single-user mode drops to 4 slots when the block is long,
    which is what makes the long block affordable at all.
24. **The DFlash draft pass is a captured CUDA graph, so its Python runs once.**
    `DFlashSpeculator._generate_draft` — everything the speculator does per step, including
    the lookup — is replayed from a graph. The Triton kernels inside it do run every step and
    do read live buffers, so the lookup itself works; but host-side Python in there executes
    at *capture* time only. A counter, a pinned copy of a flag, a decision computed there is
    frozen at whatever the warm-up produced, silently. Anything the host must see per step
    belongs in a method the model runner calls per step (`next_num_draft_tokens`), reading
    device tensors the replayed kernels wrote. Three separate "the trigger doesn't fire"
    debugging rounds were this.
25. **`torch.cuda.is_current_stream_capturing()` is not a usable guard on this path.** It
    reads True inside the captured draft pass — which is correct, and exactly why a guard
    written as `if not is_current_stream_capturing():` silently disables the code it guards
    for the entire run, not just during warm-up.
26. **rsync preserves mtimes, and Python trusts mtimes.** Copying a source file into
    `site-packages` with `rsync -a` can leave the `.pyc` newer than the `.py`, in which case
    the interpreter keeps running the old bytecode and every measurement lands on the
    previous revision. Delete `__pycache__` after installing patched files.
27. **A shorter draft block than `num_speculative_tokens` loses the decode CUDA graphs.**
    The V2 runner captures uniform-decode graphs at `decode_query_len = num_speculative_tokens
    + 1` and dispatch requires an exact match, so scheduling the drafter's 8-token block on a
    16-token server matches nothing and the step runs piecewise: 27.9 ms against 25.9 ms for
    the same work, on every short step. `cudagraph_utils.py` already knows how to capture
    several decode lengths (it does it for dynamic speculative decoding); the lookup patch
    adds the drafter's block to that list. Costs 1.8 GiB of graphs instead of 1.45.
28. **A verify block costs step time in steps, not smoothly, and the two stairs are at 16
    and 21 query tokens.** Measured on a copy at 25k context: 39.5 ms per step at 16 query
    tokens (`DFLASH_TOKENS=15`), 47.8 at 19, 47.2 at 21 — a jump between 16 and 19 and then
    flat. The first stair is the target's W4A16 GEMMs: GPTQ-Marlin tiles the M dimension in
    16 rows (`m_block_size = 16 * thread_m_blocks`, `thread_m_blocks = div_ceil(prob_m,
    16)`), so a 17th query token buys a second M block in all 64 layers and the tokens up to
    32 are then free. The second is the verify attention: `SpecDecodeAttention._plan`
    (patches/spec-decode-attn.patch) puts `q_len * G` rows in a 128-row tile, so with this
    model's `G = 24/4 = 6` one tile holds `128 // 6 = 21` query tokens and a 22nd re-reads
    the request's whole KV segment (250/583/1132 us per layer at 8/16/32).
    So there are exactly two sensible block lengths — 16 query tokens, the last one on the
    bottom stair, and 21, the most tokens obtainable for the price of the second. 31 pays
    both stairs and was never worth measuring; two attempts to start it died on memory
    first.
29. **A verify block that outgrows its CUDA-graph reservation OOMs at run time, not at
    startup.** `--kv-cache-memory` pins the pool, so `VLLM_V2_CUDAGRAPH_MEM_MIB` no longer
    sizes it — it only reserves headroom, and if it under-reserves, the server starts, logs a
    healthy pool, and then dies on the first prefill with 50 MiB left. Graph memory grows
    with the block: measured 1.82 GiB at `DFLASH_TOKENS=15`, 2.12 at 18, 2.27 at 20 (the
    capture list length barely matters — 2.21 GiB at 20 with `CG` cut from 63 to 42). Budget
    a request as `64 KiB * context + 102 MiB * (DFLASH_TOKENS + 2)`, the second term being
    the aligned recurrent-state pages, and take the extra graph memory out of the pool.
30. **The draft model is not redundant during a copy, even when the lookup overwrites every
    token it proposed.** It looks like free money: on a step the lookup controller selected,
    a qualifying match is long enough to take the head of the block too, so all seven of the
    drafter's tokens are replaced before anything is verified — skip its forward and save
    ~3 ms of a 39 ms step. Measured, that trade loses: 15.21 tokens per step becomes 13.79
    for a 5% cheaper step, a net 6% down. The drafter is covering the positions *past the
    end of the match*, which is exactly where a copy lands when the text it is reproducing
    diverges. Restricting the skip to steps where the match reaches the end of the block
    recovers the acceptance but only two runs in three — the flag it keys on is one step
    stale, and a stricter condition is more sensitive to that. Both variants are gone; this
    entry is here so the idea does not look untried.
31. **Any controller state that outlives one step has to be per-request, or batch > 1 stops
    being reproducible.** The lookup's block-length decision is batch-wide by design — a long
    block costs step time on every request in the batch — and taking it from the current
    step's flags is fine, because those are a function of the requests present. Holding it
    across steps is not: `VLLM_DFLASH2_LOOKUP_STICKY` keeps the long block on through steps
    where the flags say no, so with several requests in flight the block length a copying
    request gets depends on when the *others* arrived, and the block is one chunk through the
    recurrent layers, so that changes its greedy text. `bench/labd_soak.py` caught a verbatim
    copy coming out differently in two rounds of an identical four-way batch, and OK in three
    of three with the hold off. It is now applied only with one request in flight. The proper
    fix is per-request draft counts, which `get_uniform_token_count` in
    `gpu/cudagraph_utils.py` will not dispatch a graph for — a ragged batch runs piecewise
    and costs 8%, more than the hold is worth.
32. **Halving the KV element size can *cost* memory on a hybrid model with a draft model.**
    `unify_kv_cache_spec_page_size` equalizes page sizes by scaling a layer's block size up by
    the integer ratio `max_page / own_page`, and pads the *page* instead when that ratio is not
    an integer. Sliding-window layers are born at the backend's smallest kernel block — 16 —
    precisely because the code picking it assumes unify will scale it up
    (`_largest_kernel_block_within` in `model_executor/layers/attention/attention.py`: "the
    smallest block is fine — `unify` scales it up by an integer ratio"). When the ratio is not
    an integer that assumption fails silently and every block of that layer pays a whole
    primary page. Divisibility here holds at bf16 only by coincidence — the target's 4 KV heads
    × 256 and the DFlash2 drafter's 8 × 128 both come to 4096 B per token per layer — and
    `int8_per_token_head` breaks it by adding one fp32 scale *per head* (2080 vs 2112 B/token;
    2112 = 2⁶·3·11 shares no factor with the primary page). The drafter's 5 layers then took
    `cdiv(2047 + 4096, 16) + 1 = 385` blocks of 1.71 MiB at 1.88% utilisation — a constant
    5.155 GiB, 75.6% of the per-request budget. Measured: int8 needed **6.82 GiB to serve
    32,768 tokens** where bf16 serves 69,758 in 5.2 GiB, i.e. 2.4× worse from halving the
    dtype. `patches/hybrid-sw-block-promote.patch` rounds such a layer's block *up* instead
    (16 → 864), which turns that into 138,696 tokens. The tell in a log is an "estimated
    maximum model length" that is a small multiple of 16.
33. **The aligned recurrent-state pages scale with the verify block, not with the slot count.**
    Gotcha 23 says "per request slot"; that is wrong. Measured by asking for an impossible
    `max_model_len` and fitting the two numbers vLLM prints: the fixed term is 0.88 GiB at
    `DFLASH_TOKENS=7` and 1.66 GiB at 15 — the ratio 0.53 is exactly 9/17, i.e. `(k+2)` — while
    `MAX_SEQS` 1 against 8 moves it by about **8 MiB in total**. So dropping to one slot for a
    genuinely single-user server buys no context at all, and `MAX_SEQS=4` at a long block is
    about CUDA graph memory, not state pages.

    That is about the SIZE of the pool. It says nothing about how much of the pool a
    *running* request takes, and there the per-request model is right — it is the same
    page, and the two arrive at it independently (0.88 GiB fitted here, ~0.82 GiB from
    the live occupancy below). Measured live at `CTX=fast` (`bench/conc_ladder.py`, and the ramp in the
    issue-25 notes): one resident `dflash2` request with an empty context occupies
    **15.8%** of the 69,758-token pool, so six or seven fit and the next one is
    **preempted**; with 4k-token prompts it is 19.8% and five fit, with 16k-token
    prompts two. One MTP request takes 8.2% of its
    86,727, so eight fit. Both numbers are the k+1 recurrent-state slots, which is why
    they are in the ratio 8:5. The two facts together are the whole of the concurrency
    story for this mode: extra seats do not cost you pool, and they do not buy you
    residents either.
34. **Asking for an impossible `max_model_len` is the cheapest way to read the memory model.**
    vLLM prints "X GiB KV cache is needed ... available Y GiB ... estimated maximum model
    length is Z" and dies in ~90 s, before torch.compile finishes and long before graph
    capture. Two such points give slope and intercept for `needed(context)`, and the slope
    comes out at exactly `16 × 4 × 256 × 2 × 2 = 65,536` B/token for bf16 — so the fit can be
    checked against arithmetic rather than trusted. Beware that `estimate_max_model_len` is a
    binary search over `max_memory_usage_bytes`, which rounds up to whole blocks, so the
    estimate is quantised by the block size: at an 864-token block the granularity is coarse
    and a two-point inversion at small lengths is unreliable.
35. **`KV_MEM` assumes the card is headless, and the failure lands long after
    startup looks fine.** The single-user pool is pinned in bytes rather than sized
    from `GPU_UTIL` (gotcha 33 and the comment in `single-user/start_qwen.sh` say
    why), and 5.2 GiB is what fits when nothing else is on the GPU. With a desktop
    session on the same card — Xorg plus a compositor plus a browser is easily
    ~1.3 GiB — the server still starts, still captures its graphs, still reports a
    pool, and then dies later on a real request when the spec-decode `part_o`
    buffer cannot get its ~1.5 GiB (`spec_decode_attn.py`). Nothing at startup
    warns you. On a card you also render on, drop `KV_MEM` by at least what the
    desktop is holding (`nvidia-smi` before you start the server): `KV_MEM=4000000000`
    was enough for the reporter of
    [#12](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/12). Setting `KV_MEM=`
    empty falls back to `GPU_UTIL`, which profiles the actual free memory instead.
36. **A model dir with no `tokenizer.json` is not an error to transformers — it is an
    empty vocabulary, and vLLM reports it as a reasoning-parser problem.**
    `AutoTokenizer.from_pretrained` on a dir that has `config.json` but no tokenizer
    files returns a `Qwen2Tokenizer` with `vocab_size == 1` that encodes *everything*
    to `[]` — `tok.encode("hello world")` is `[]`, not an exception. Nothing complains
    until `VllmConfig.__post_init__` asks the qwen3 reasoning parser for `<think>`,
    gets `[]` back, and raises

    ```
    ReasoningConfig: failed to tokenize reasoning strings:
    reasoning_start_str='', reasoning_end_str=''.
    ```

    which names neither the tokenizer nor the directory, and prints the strings as
    empty because they are the *unset* config fields, not the ones the parser supplied.
    Reported as [#15](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/15), where it
    looked like a `SPEC=dflash2` bug: it reproduces with no speculative config at all,
    and the reason only the single-user modes failed is that they serve
    `models/Qwen3.8-27B-W4A16-AutoRound-fast` while batch mode serves the base dir.
    `verify.sh` now encodes `<think>` against every dir we pass to `--model` instead of
    only checking that the dir exists, and `docker/prepare.sh` counts `tokenizer.json`
    as part of a complete download.
37. **Bug B needs a prefix-cache HIT, and then fires at one prompt length in every
    128. It is not dflash2-only.**
    Under `CTX=huge` with a CAPTURED (FULL) verify step, a request that hits the
    prefix cache and whose prompt length lands on one particular residue mod 128
    collapses: `SPEC=dflash2 DFLASH_TOKENS=7` gives 1.97 tok/step and degenerate
    repetition (`4/3595` characters verbatim, one 40-char block ×79), `SPEC=mtp`
    stops dead and returns `""` or `"#"` with `finish_reason=stop`. Every other
    residue is 794/794 verbatim.

    **The location is deterministic; the damage is not.** Repeats are bit-identical
    on one server, but the same `mtp` residue has now produced three different
    outputs on three geometries: an empty answer, a one-character answer, and — on a
    box running `MAX_LEN=240000` with a tool parser attached — 400 tokens of fluent
    Danish that open with a malformed `<think>` under `enable_thinking=false` and
    invent a translation task, `2/1146` verbatim at 3.38 tok/step
    ([#25](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25), mjungnickel18).
    So do not test for a symptom. The only property all three share is that the copy
    did not come back, which is what `bench/verbatim.py` scores and what both sweeps
    now judge on.

    Two conditions, and it took two people to see both. The **hit** is necessary:
    a fresh server, one request, no warm-up, never collapses at any length
    ([#13](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/13), mjungnickel18) —
    which is also why `PREFIX_CACHE=0` always looked clean. The **residue** decides
    whether a hit corrupts, and it is a clean function of the draft count:

    | config | k | verify block L=k+1 | attention block | broken R | free slots 128-R |
    |---|---|---|---|---|---|
    | `dflash2`, `DFLASH_TOKENS=7` | 7 | 8 | 2176 (=17x128) | 124 | 4 |
    | `dflash2`, `DFLASH_TOKENS=5` | 5 | 6 | **2176** | **122** | 6 |
    | `dflash2`, `DFLASH_TOKENS=3` | 3 | 4 | 2048 (=16x128) | 120 | 8 |
    | `mtp`, `DRAFT_TOKENS=3` | 3 | 4 | 2048 | 120 | 8 |

    **R = 117 + k**, equivalently the final 128-token tile has exactly `11 - k`
    free slots, equivalently `L + free = 12` in every configuration measured.
    Fitted on three values of k across two speculators, so treat the constant as a
    fit rather than a derivation — but note what it implies: the step reserves or
    touches a **fixed 12 slots regardless of the verify block length**, which is
    the most specific lead this bug has produced.

    The `DFLASH_TOKENS=5` row is the one that matters for method. Same attention
    block as `DFLASH_TOKENS=7` (2176), different broken residue (122 against 124),
    which rules out the attention block size. An earlier version of this entry
    claimed R tracked the verify block on three points where the two co-varied;
    that was retracted as unevidenced, and then confirmed by running the
    configuration that separates them.

    Confirmed periodic in every case: 24,956 / 25,084 / 25,212 / 25,340 at k=7,
    25,082 / 25,210 / 25,338 at k=5, 25,080 / 25,208 / 25,336 at k=3. Hold the document
    byte-identical and pad the *instruction* by one token and a broken length goes
    clean, so it is the token count rather than the corpus. What is established: `mtp` and
    `dflash2` at the same draft count break at the same residue, so the drafter is
    not implicated and the shared multi-query verify against a partially-hit
    prefix is.

    Mitigation, as it stands at HEAD: `CTX=huge` forces `cudagraph_mode=PIECEWISE`
    for `SPEC=mtp`, which is clean at every residue and costs nothing measurable —
    `SPEC=mtp` over 8k/16k/32k/50k is 87.8/86.1/70.4/63.5 tok/s captured against
    93.5/83.8/70.3/59.6 piecewise. This repo previously scoped that workaround to
    `dflash2` on the theory that MTP's short verify step captures correctly; it
    does not, and `SPEC=mtp CTX=huge` shipped with the bug.

    `dflash2` has since got FULL capture back (`a75ee4b` fixed its residue, and
    `b356e31` swept **all 128** residues under FULL with 0 broken), so the two
    speculators are no longer on the same default. `mtp` keeps PIECEWISE as a
    correctness constraint until residue 4 comes back verbatim under a full sweep,
    not until a particular symptom stops appearing.

    A third trap, learned the hard way on `DFLASH_TOKENS=15`: `bench/bugb_sweep.py`
    used to report the RAW prompt length, not the chat-templated one the engine
    actually sees (+12 tokens for the Qwen3 wrapper). That offset is why the rule
    first read as `R = 117 + k` and then as a mysterious constant 12; both were the
    same relation seen through a harness bug. It also made a k=15 sweep look
    structureless until the offset was applied, at which point the lowest-acceptance
    row sat exactly on `== L`. The script now templates before counting.

    Do not judge a row by its failure signature, and that includes `repeats`. An
    earlier version of this entry said "only `repeats` tells you whether it actually
    collapsed"; two of the three shapes above repeat nothing, and a rule that
    demanded repetition is what filed the `mtp` break as "diverged, probably fine"
    through several full sweeps. Both sweeps now score **coverage** — the fraction of
    the answer's 40-character windows that occur in the source — against the median
    of the other lengths in the same run (`bench/verbatim.py`, which self-tests
    against all three shapes: `venv/bin/python bench/verbatim.py`). Coverage rather
    than the old longest-prefix column because a prefix match reports `38/791` for a
    single wrong character at offset 38 no matter how good the rest is; the prefix
    and repeat counts are still printed, but nothing is decided on them.

    Two traps for anyone measuring this. Sweep prompt length in steps of **1
    token** — at a coarse grid one broken sample below and one above reads as a
    cliff, which is how it was first diagnosed. And send each length to a **fresh
    server**, or request N inherits request N-1's blocks and you measure history
    instead of length; `bench/labd_bench.py` sends two warm-ups on `doc[:4000]`,
    which arms the trigger for everything after it. `bench/bugb_sweep.py` prints
    the `mod 128` column for this.

    And do not sample residues. With one broken length in 128, five distinct samples
    miss it `C(127,5)/C(128,5)` = 123/128 = **96%** of the time — this repo once
    wrote 82% there, which is the figure for six broken residues, and hung a
    "5 of 5 clean" claim on it. `bench/residue_sweep.py` walks all 128 by stepping
    the pad one token at a time, which covers each residue exactly once.
38. **The decode-graph budget is sized for 64 query tokens, and `MAX_SEQS` multiplies
    into it.** `CG = MAX_SEQS x (k+1)` is what the V2 runner captures, and
    `VLLM_V2_CUDAGRAPH_MEM_MIB` is reserved for what the shipped defaults produce —
    8x8 at `DFLASH_TOKENS=7`, 4x16 at 15, i.e. 64 either way. Ask for
    `DFLASH_TOKENS=15 MAX_SEQS=8` and it becomes 128: the server boots, captures its
    graphs, answers `/health`, and then dies on the first concurrent batch with
    `torch.OutOfMemoryError` inside the engine — `EngineDeadError`, every request 500,
    `/health` still 200. Same shape as gotcha 18 and as
    [#18](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/18): a memory bill that
    the startup profile does not see. `single-user/start_qwen.sh` now caps the derived
    `CG` at 64, which leaves every shipped default untouched and makes the oversized
    batches run piecewise instead of not at all. Set `CG` explicitly to override, and
    raise `VLLM_V2_CUDAGRAPH_MEM_MIB` with it.

39. **The engine needs non-KV headroom for its first real batch, `MAX_SEQS` and
    `KV_MEM` are two doors into the same shortfall — and on WSL2 the failure is
    silent.** First seen as a seat-count death: `MAX_SEQS > 12` at `CTX=huge` kills
    the engine and the graphs are innocent — same visible failure as gotcha 38
    (boots, captures, `/health` 200, dies on the first prompt with
    `torch.OutOfMemoryError`), different bill. `CG` is pinned at its 64 cap in every
    one of the runs below, so the memory is going to allocations that scale with
    `max_num_seqs` itself, not with the captured batch. Free VRAM after boot on one
    24 GiB 3090, `SPEC=dflash2 CTX=huge PREFIX_CACHE=1` k=7, then a single ~3.7k-token
    prompt:

    | `MAX_SEQS` | free after boot | one ~3.7k prompt |
    |---|---|---|
    | 8 | 596 MiB | ok |
    | 10 | 456 MiB | ok |
    | 12 | 416 MiB | ok |
    | **16** | **356 MiB** | **dead** |

    The allocator says it plainly: `expandable_segments: memory mapping failed with OOM
    on device 0 while trying to map 20971520 bytes (free: 20578304, total:
    25272516608)` — 20 MB wanted against 20 MB left, on a card with 24 GB. It needs no
    concurrency at all: `num_running_reqs=1`, `step_counter=0`, `kv_cache_usage=0.18`.
    Reproduced twice with byte-identical counters.

    The seat count is only one door into that shortfall.
    [@mjungnickel18 named the real subject](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25#issuecomment-5392694387)
    — *how much non-KV headroom does the engine need*, with `MAX_SEQS` and `KV_MEM` as
    two doors into the same room — after a `KV_MEM` pin on his box produced a failure
    this table does not contain (below). The same room walked through the `KV_MEM`
    door, seats pinned at 8, salted prompts, best of 3, prefill tok/s from TTFT:

    | `KV_MEM` | free after boot | 4k / 16k prefill | 8×16k concurrent | outcome |
    |---|---|---|---|---|
    | 5,261,334,938 (stock) | 576 MiB | 1,156 / 1,107 | ok — free bottoms at 110 MiB | ok |
    | 5,414,427,034 | 436 MiB | 1,159 / 1,100 | ok — free bottoms at **8 MiB** | ok |
    | 5,466,855,834 | **396 MiB** | 1,155 / 1,099 — full speed | **dead in 34 s, every request 500** | dead, twice |

    Same fingerprint at the bottom: the identical four failed 20,971,520-byte mappings
    with byte-identical free counters across both repeats, ending in
    `torch.OutOfMemoryError: Tried to allocate 24.00 MiB ... 37.62 MiB is free`. Two
    refinements the second ladder forces. The transient working set is *elastic*: it
    takes ~380 MiB when the room exists (watch `memory.free` during a prefill: 576 →
    194 MiB at stock) but squeezes without measurable cost — at 436 MiB free the 8-way
    cell ran at full throughput with 8 MiB left. What is rigid is the first real
    batch's allocation bill, sized by `max_num_seqs` and by how many requests actually
    run: 16 seats died on a single prompt, while 8 seats at 396 MiB prefilled 16k
    single-stream at full speed and died the moment eight ran at once. No
    configuration anywhere on either ladder was ever merely *slow*.

    That last sentence is the platform note, and it is the part that cost a week of
    cross-box debugging in [#25](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25):
    on bare-metal Linux this failure has exactly two states, full speed or a loud named
    `torch.OutOfMemoryError`. Under WSL2 the WDDM driver backs the failed mapping with
    host memory instead, so the same exhaustion produces **no error at all** — just
    prefill at a fifth of the rate (232 tok/s at 4k against ~1,000 healthy, measured by
    @mjungnickel18 under a `KV_MEM` pin that left ~630 MiB free). A WSL user who raises
    `KV_MEM` gets the context they asked for, no warning, and 5–10× the TTFT, with
    nothing in the logs and no `nvidia-smi` number that flags it. The two boxes in
    [#25](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25) make it concrete: the
    same pin (`KV_MEM=6871947673` at `CTX=fast`, `MAX_SEQS=2`; the boxes produce
    byte-identical pool geometry, 81,368 tokens at the fixed sibling pin) read
    ~630 MiB free after boot on WSL and served — slowly — for four days, while on bare
    metal it boots with 98 MiB free and the first prompt kills the engine. WSL's free-after-boot overstates the Linux number by roughly whatever WDDM
    is host-backing, so a headroom rule of thumb tuned on one platform does not
    transfer to the other in either direction. The detector is the one that found it: a
    prompt-length ladder against a known-good rate. On WSL, ladder any `KV_MEM` above
    stock before trusting it; the launcher now prints a warning when the pin exceeds
    the profile default.

    The launcher warns above 12 seats rather than clamping, because unlike `CG` this is
    a VRAM budget rather than a shape: a card bigger than 24 GiB has room where this
    one does not. Seats and pool trade against each other — if you want seats, buy them
    with a lower `KV_MEM`; if you want context, the ladder above is the price list, and
    on this card the floor at 8 seats sits between 396 and 436 MiB of free headroom.
    The shipped `CTX=huge` default is `MAX_SEQS=2`, so nothing here is reachable
    without an override.

    Worth reading next to the concurrency section of the README: seats above the
    residency were already useless (they queue, then preempt). Past 12 at `CTX=huge`
    they stop being useless and become fatal.

40. **Tool calling / structured output under a speculator killed requests at the
    grammar's end** ([#31](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/31),
    fixed by `patches/xgrammar-spec-terminated.patch`). A speculative verify window
    can legally accept tokens past the point where the xgrammar matcher terminates —
    the newline after a closing `</tool_call>` tag, the stop token itself, anything
    after it under `ignore_eos`. 0.27.1 treats both arrivals as failure, the
    scheduler logs `Unexpected: grammar rejected tokens ... Terminating request`,
    and the client gets an HTTP error for a request whose output was completely
    valid. The longer the verify block, the more reliably the window covers the
    tokens around the stop, which is why `DFLASH_TOKENS=15` + `--tool-call-parser`
    surfaced it first. Reproduced on the shipped config with a `json_schema` +
    `ignore_eos` request — `grammar rejected tokens [16, 22, 198, 92, 248046, 198]`,
    where 92 is the brace that completes the JSON, 248046 the stop token that
    terminates the matcher, and the trailing newline killed the request. The patch
    backports upstream's current semantics: tokens after termination are ignored,
    real mid-grammar rejections still fail loudly.

    Two log signatures to keep apart, because they look alike. The fatal one is the
    `grammar rejected tokens` line above — gone with the patch. The non-fatal one is
    a burst of `Failed to advance FSM for request ... Please file an issue.` with
    **no** `Terminating request` after it: that is the bitmask builder advancing
    draft tokens past a reasoning end that landed mid-window, a rejection the code
    explicitly tolerates. It is noise, the request completes normally, and it
    predates (and survives) this fix.

41. **sm80 (GA100) Marlin repack can Xid-31 the whole card under memory
    pressure — and the kernel in the traceback is innocent.** Community
    finding, [@ahnguyen17 in #27](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/27#issuecomment-5397500895),
    on a CMP 170HX 40 GB: with ~27 GB resident, `gptq_marlin_repack`'s GB-scale
    int64 intermediates (k×n int64 ≈ 1.4 GB per 27B layer, several live at
    once) churn sm80 VMM mappings until an unrelated, trivially correct
    elementwise kernel takes an async write fault — the faulting frame drifts
    between runs, the Xid 31 wedges the card until reboot, and
    `compute-sanitizer` is clean on sm86 with identical inputs. Their
    workaround, serving in production since: compute the repack on CPU
    (bit-exact, ~3 min extra boot) —
    [`sm80-int8-repack-cpu-fallback.patch`](https://github.com/ahnguyen17/cmp-170hx-vllm)
    — with `expandable_segments` kept **off**, which on that card is an
    independent Xid-31 trigger. Not shipped here (no sm80 to regression-test
    against); recorded so the next GA100/A100 report starts from the answer
    instead of from five reboots.


42. **The OffloadingConnector's CPU tier can be silently useless: uniform
    blocks meet asymmetric chunk sizes, and one request evicts everything
    (issue #33).** The tier allocates equal-size blocks sized for the LARGEST
    group's offload chunk. Under KVarN the drafter's sliding-window group
    carries 128-token chunks against the 2,176-token maximum, so every SW
    crumb occupies a full ~14.6 MiB block — a single 23k-token request eats
    ~264 of a 4 GiB tier's 293 blocks and LRU-evicts every previous
    document. Stores succeed, `complete_store` succeeds, and every
    cross-request lookup is a MISS: 41 GB written, 0 bytes ever read back,
    with nothing in the logs. On bf16 KV the SW group happens to share the
    large per-token size (gotcha 25's 4096-B coincidence), the geometry
    stays uniform, and the same connector uplifts at PCIe speed — the KV
    dtype was never the mechanism, the chunk geometry it induces was. Since
    `offload-dflash-eagle-groups.patch` the config builder warns at boot
    with the waste factor and the `cpu_bytes_to_use` multiplier that would
    compensate (~17x under KVarN). Same patch fixes an adjacent quiet bug:
    upstream only ever sets `is_eagle_group` for DeepSeek V4, so the
    connector's fallback marked EVERY group as draft attention under
    `method=dflash`/`mtp` and silently excluded each group's trailing chunk
    from store while decoding; with dflash the flag now lands on the
    drafter's sliding-window group alone. Also: on bare-metal Linux the
    connector refuses this stack's default allocator
    (`expandable_segments:True`) at config validation — run it with
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (or the cumem
    allocator), which the WSL2 branch of the launchers already defaults to.
    And when eviction probing, keep the resend prompt BYTE-identical: a
    two-token label difference shifts every block hash and manufactures a
    convincing, fake "per-request hash instability" (ask how we know).

43. **The split-KV segment count is a graph-time tuning parameter, not a live
    knob.** The verifier's partial-output workspace is captured by FULL CUDA
    Graph, so `VLLM_SPEC_DECODE_ATTN_SEGMENTS` is read once and must not change
    until process restart. The generic default stays 16. On CMP 170HX with the
    896-token static-FP8 target geometry, 32 segments reduced verifier pass time
    from 62.8 to 50.0 ms at 126K and from 103.3 to 77.5 ms at 250K. 64 segments
    bought less than 2% more in the isolated long-context kernel scan and hurt
    short-context latency. The FULL mixed-FP8 service profiles therefore pin 32.

44. **SM80 FP8 verifier decoding belongs in a BF16 LUT, not in per-element
    arithmetic.** Triton cannot lower native E4M3FN loads on SM80, but every
    finite E4M3FN value is exactly representable in BF16. A 256-entry BF16 LUT
    removes masks, `tl.exp2`, and FP32-to-BF16 conversion from the hottest
    split-KV loop while preserving the previous NaN-to-zero behavior. On the
    CMP 170HX this raised C1 decode from 68.2 to 81.8 tok/s at 126K and from
    42.2 to 53.8 tok/s at 250K with unchanged acceptance and zero preemptions.

45. **A 255-register verifier is not automatically fixed by a lower register cap.**
    The production q=8 static-FP8 partial kernel reported 255 registers/thread,
    96 bytes of local memory/thread and 12.5% theoretical occupancy.  Compiling it
    with `maxnreg` 192, 168 or 160 increased local memory to 320, 512 and 632 bytes
    and slowed it by 37-50%.  The useful fix was to shorten live ranges: keep scores,
    maxima and normalizers in FP32, round only the per-tile running output to FP16,
    and scalarize the block ID for the integral 896/32 page/tile geometry.  This cut
    the local frame to 32 bytes and reduced FULL-graph step time 8-11% at 126K/250K.
    Treat occupancy as a diagnostic, not a target; forcing occupancy by spilling is
    worse than the original kernel.

46. **Power-of-two tiles can hide useful masked work even after spills are gone.**
    DFlash2 q=8 with six GQA heads has 48 valid rows, while the generic verifier uses
    `BLOCK_M=64`.  Keeping one CTA per request/KV-head/segment but evaluating 32+16
    rows removes the 16 padded accumulators without rereading K/V.  On CMP 170HX this
    left occupancy unchanged and improved FULL-graph step time another 3% at 126K and
    5% at 250K.  Splitting into separate CTAs per query head is not equivalent: that
    would reread every K/V tile six times and should remain a rejected design.

47. **Page-table metadata should follow page lifetime, not tile lifetime.**  The
    q8 verifier uses 32-token tiles inside 896-token pages, so one physical block
    ID is valid for 28 consecutive loop iterations.  Carrying that scalar and
    reloading it only at a page boundary was bit-identical and saved 0.4-0.5% at
    the FULL-model step boundary from 4K through 250K.  This is a small but robust
    win; unlike increasing split count or duplicating query-row kernels, it adds
    no KV scan, reduction work, or CUDA Graph node.

48. **A tiny immutable LUT can still create enormous global-transaction waste.**
    The SM80 FP8 verifier's 512-byte BF16 decode table was cache-resident, but
    data-dependent scalar indices made ordinary Triton global loads account for
    85,998,528 excessive sectors (64% of the total).  In the exact NVIDIA-only
    q8/GQA6 path, `ld.global.nc.u16` routes those immutable reads through the
    read-only cache: NCU reported only 76,288 excessive sectors, kernel duration
    fell from about 2.13 to 1.82 ms, and FULL-graph decode improved 2.3% at 126K
    and 3.2% at 250K.  Keep the portable load in generic paths; inline PTX is a
    measured architecture specialization, not a replacement for Triton's normal
    lowering on other GPUs.

49. **Match a one-wave split grid to the GPU; more splits are not monotonic.**
    This verifier launches one CTA per KV head and segment.  With four KV heads,
    NSEG32 gives 128 CTAs on a 70-SM CMP 170HX.  At two resident CTAs per SM the
    one-wave capacity is 140 CTAs, so NCU reports 0.91 waves; NSEG35 fills all 140
    resident slots.  NSEG40/48/64 create a second-wave tail and were slower.
    NSEG35 reduced FULL-graph pass time 3-5% at 126K/250K without moving 4K.
    Because Triton `arange` requires a power of two, keep logical NSEG=35 for
    partials but pad/mask only the final combine reduction.  This is
    hardware-shape tuning: do not copy 35 to a GPU with a different SM count,
    resident-CTA limit or KV-head count without redoing the wave calculation and
    A/B.

50. **The next integer after a wave-aligned split is not a useful compromise.**
    NSEG36 launches 144 CTAs: four CTAs spill into a second resident wave on the
    70-SM CMP 170HX.  Against NSEG35 it was 45-54% slower at 70K, 126K, 200K and
    250K in the isolated q=8 verifier scan.  This is much worse than the four-CTA
    tail suggests because the non-matching segmentation also changes generated
    loop and memory geometry.  Keep NSEG35; do not sweep adjacent integers as if
    segment count were a smooth tuning knob.

51. **Cache hints after `ld.global.nc` did not improve the long-context LUT path.**
    Four interleaved runs of `ld.global.nc.L2::64B.u16` were statistically neutral
    at 250K, and three runs of `ld.global.nc.L1::evict_last.u16` were neutral to
    slower.  Both showed high variance at 126K and retained exact output, but
    neither passed the requirement to improve 126K and 250K together.  The useful
    change is the read-only cache route itself; extra prefetch/eviction decoration
    is rejected unless a future architecture-specific NCU trace establishes a new
    bottleneck.

52. **Use exact-token, counter-aware A/B requests for speculative tuning.**
    `bench/context_ab.py` sends token IDs rather than approximate repeated text,
    changes a salt to defeat accidental prefix-cache reuse, streams the response,
    and records TTFT, decode tok/s, speculative steps, accepted tokens, tokens per
    step, milliseconds per step and preemptions.  This separates kernel latency
    changes from DFlash acceptance drift.  The prompt corpus is tokenized with an
    explicit limit so a 250K benchmark does not first allocate a multi-million-token
    host sequence.

53. **Do not transplant the verifier's 140-CTA reasoning into Marlin.**  The
    verifier can keep two CTAs resident per SM; the profiled Marlin target GEMMs
    use about 163 KiB dynamic shared memory and launch 70 CTAs, one per SM.  An
    SM80-only source build that prioritized `(128,64,128)` over the stock
    `(128,128,256)` tile regressed step latency 12-18%.  Prioritizing
    `(64,128,128)` regressed it 9-12%.  Both candidates passed load and execution
    smoke tests, so these are real performance results rather than build
    failures.  Changing `blocks_per_sm` or grid size would also change Marlin's
    workspace/lock/reduction protocol and is not a safe follow-up to these
    negative tile tests.

54. **`inline_asm_elementwise(pack=2)` is not a vectorized random LUT load.**
    The FP8 decode table uses one data-dependent scalar address per lane.  Merely
    changing the Triton inline-assembly call from `pack=1` to `pack=2` aborts
    compilation with `number of input constraints does not match number of
    parameters`; it does not lower to a useful `ld.v2.u16`.  A genuine packed
    load would require paired contiguous addresses, which this lookup does not
    have.  Keep the scalar read-only-cache load rather than trying to force
    vectorization through the pack metadata.

55. **More Triton pipeline stages are not free latency hiding for the q8
    verifier.**  Changing only the production q8/GQA6 launch from
    `num_stages=1` to 2 really generated 22 `cp.async` instructions, but dynamic
    shared memory rose from 43,008 to 73,728 bytes.  It was 4.2% slower at 4K,
    only 0.7% faster at 126K, and 2.1% slower at 250K.  Stage 3 generated 32
    `cp.async` instructions, used 90,112 bytes and regressed long contexts by
    36-40%, consistent with losing the two-resident-CTA geometry behind NSEG35.
    Keep stage 1.  Future latency hiding must control shared-memory lifetime and
    accumulator liveness explicitly rather than relying on a launch hint.

56. **Lower registers/thread can still produce a worse verifier CTA.**  An
    eight-warp q8/GQA6 build used 167 rather than 250 registers/thread and did
    not spill, but its 256 threads consume about 42.8K registers per CTA.  That
    permits only one CTA/SM instead of the baseline's two 4-warp CTAs, leaving
    both layouts at roughly eight resident warps/SM while removing CTA-level
    independence.  Pairing it with NSEG17 (68 CTAs for 70 SMs) did not rescue
    it: isolated latency was 75% slower at 126K and 79% slower at 250K.  NSEG18
    was worse because two CTAs formed a long second-wave tail.  Keep 4 warps and
    NSEG35; optimize live ranges without changing this resident-grid geometry.

57. **A smaller partial workspace can trigger a larger verifier kernel.**  The
    q8 static-FP8 running accumulator is FP16, so changing `part_o` from FP32 to
    FP16 looked like a free 50% workspace reduction.  On Triton 3.7.1 it changed
    lowering enough to raise dynamic shared memory from 43,008 to about 57,344
    bytes and retained 252-255 registers/thread.  Three interleaved scans showed
    6-15% regression at 126K and 5-8% at 250K.  Keep the FP32 external partial
    buffer; workspace byte count is not a proxy for generated-kernel cost.

58. **Disabling LICM lowers a register count without fixing verifier liveness.**
    Replacing the q8 KV loop with `tl.range(..., disable_licm=True)` reduced the
    compiled count from 250 to 239 registers/thread but raised dynamic shared
    memory from 43,008 to 57,344 bytes.  Three interleaved scans averaged 4.2%
    slower at 126K and 1.3% slower at 250K.  The persistent 48x256 accumulator
    still spans the whole loop; compiler scheduling hints cannot make that state
    disappear.  Keep normal LICM and move structural experiments to explicit
    row partitioning or a custom CUDA producer/consumer kernel.

59. **Three smaller source groups are still one long-lived 48-row state.**  A
    q8/GQA6 rewrite from 32+16 rows to 16+16+16 passed every strengthened
    correctness case, but Triton kept all three 16x256 accumulators live across
    the KV loop.  It compiled to 248 registers/thread and 57,344 bytes shared,
    versus 250 and 43,008 for the qualified kernel, then regressed interleaved
    126K/250K latency by about 16.9%/14.8%.  Source grouping is not lifetime
    control; further work needs an explicit storage/synchronization design.

60. **Tensor-core instruction count does not prove tensor-core utilization.**
    The standalone V7-E2 CUDA verifier lowered both QK and PV to BF16 WMMA and
    executed exactly the same 6,048,768 tensor-pipe instructions as the
    qualified Triton kernel at 126K.  It was still about 19.5x slower.  NCU
    showed why: an unswizzled 256-column BF16 shared layout caused 190.54 M
    shared-load bank conflicts versus Triton's 2.02 M, long-scoreboard stalls
    rose from 12.09% to 51.20%, and tensor-pipe active time fell from 17.21%
    to 0.82%.  Its scalar decode/feed path also executed 1.079 B instructions
    versus 126.0 M.  Validate shared layout, feed efficiency and stalls with
    counters; `mma_sync` in source or SASS is not a performance result.

61. **Legal per-CTA shared memory can still cross a residency cliff.**  Padding
    the V7 WMMA Q/K/V leading dimensions from 256 to 264 reduced shared-load
    bank conflicts from 190.54 M to 63.51 M and passed full correctness, but
    grew dynamic shared from 81,920 to 83,200 bytes.  The latter is legal for
    one CTA yet reduced measured residency from two CTAs/SM to one on GA100.
    Long-context latency regressed about 55%.  Always query active CTAs after
    a shared-layout change; fitting below the per-block opt-in limit is not
    evidence that the intended multi-CTA occupancy survives.

62. **A small serialized tail can be cheaper than losing a resident CTA.**
    Keeping padded Q/K/V but shrinking the FP32 PV scratch from 16x256 to
    16x224 restored two CTAs/SM.  A second 16x32 tail phase added barriers and
    made 4K about 3% slower, yet improved 70K-250K roughly 5% over the
    unpadded E2 because the bank-conflict reduction finally survived at two
    CTA residency.  This remains far from production-fast, but it is a useful
    design rule: explicitly serialize a small tail when doing so preserves a
    major occupancy tier, and verify the trade with both short and long tiers.

63. **Moving a tiny decode LUT to shared memory does not cure a scalar feed
    pipeline.** V7-E4a copied the 256-entry BF16 FP8 table into 512 B of shared
    memory and preserved two-CTA residency. It passed the full correctness
    gate and improved E2 by 6.1%/5.6% at 126K/250K, but long-scoreboard stalls
    remained 51.97%, tensor-pipe activity only reached 0.88%, and shared-load
    conflicts rose to 72.22 M. The dominant cost is still scalar raw-byte
    loading/addressing and staging, not the LUT's global-memory residence.
    Vectorize raw loads and hoist address bases before attempting more lookup
    caching.

64. **A vector global load can remain a scalar decode pipeline.** V7-E4b
    changed raw K/V reads to aligned 16-byte `uint4` loads and hoisted tile
    address bases without changing occupancy or correctness. It nevertheless
    regressed 4K by 13% and was flat from 70K through 250K. NCU still showed
    about 1.09 B instructions, 52% long-scoreboard stalls and 0.88% tensor
    activity because every vector still had to be unpacked byte by byte and
    indexed through the shared LUT before WMMA. Distinguish vector transport
    from vector conversion; stage compact raw bytes separately if the goal is
    to break the global-load/decode dependency chain.

65. **Separating transport from decode does not remove decode cost.** V7-E5
    staged each compact 8-KiB raw K/V matrix in an otherwise idle shared Q/P
    buffer before decoding it to padded BF16. It passed all gates and improved
    4K/70K by 1.2%/2.9%, but 126K-250K latency and every important NCU counter
    were flat. The remaining per-byte random shared-LUT read still dominates
    the feed path. After validating all encodings, direct E4M3FN bit conversion
    is a better next test than adding more transport stages.

66. **For E4M3FN on SM80, exact bit synthesis can beat a decode LUT by a wide
    margin.** V7-E6 exhaustively matched all 256 raw codes against PyTorch and
    then removed hot-loop shared LUT reads. Long-context latency fell 43-44%,
    instructions dropped from 1.093 B to 711.7 M, long-scoreboard stalls from
    51.99% to 28.48%, and tensor activity rose to 1.51%, with no occupancy or
    correctness regression. A tiny table is not automatically cheap when each
    divergent byte creates a shared lookup and conflict; prove the finite/
    subnormal/NaN bit mapping and prefer arithmetic when its ISA cost is lower.

67. **Staging can be worth its barriers when it breaks a global scoreboard
    chain.** V7-E7 removed E6's compact raw staging and cut barrier stalls from
    19.23% to 11.19%, yet long-scoreboard stalls jumped from 28.48% to 54.34%
    and 126K/250K latency regressed about 68%. Direct byte loads serialize the
    decode feed despite fewer synchronizations. Preserve staged transport and
    reduce its phase count or overlap it; do not optimize barrier percentage in
    isolation.

68. **Preserve the application's invalid-code contract when replacing a
    numeric LUT.** PyTorch converts E4M3FN `0x7f/0xff` to NaN, but this verifier
    deliberately fail-closes those two encodings to zero. E6's first exhaustive
    test accidentally validated PyTorch NaNs; cross-review caught it before
    integration, and E8 corrected the device helper/test. Exhaustive encoding
    tests must compare against the runtime contract, not merely a library cast.

69. **Fewer source barriers need not reduce measured barrier stalls.** E8
    staged K and V together and cut four explicit phases to two, reducing
    instructions about 5%, yet barrier stalls stayed near 19.3%. Long latency
    improved only 1-2% and 4K regressed 7.7%. Arrival imbalance and the work
    between barriers matter more than the source-level barrier count.

70. **A public FP8x2 conversion can still be slower than exact integer decode
    on SM80.** CUDA 13.0 on the qualified host exposes
    `__nv_cvt_fp8x2_to_halfraw2`, but not a direct FP8x2-to-BF16 intrinsic.
    E9b therefore required half-to-float-to-BF16 conversion after the paired
    decode. It was 8.3% faster at 4K, but 2.8-3.0% slower from 126K to 250K and
    executed 783.4 M rather than E6's 711.7 M instructions. Check the target
    toolkit's actual header and end type; vector width alone is not a useful
    optimization if a scalar type bridge is inserted afterward.

71. **Segment-local block metadata can be correct and measurable yet too small
    to matter.** V7-E10a prefetched up to sixteen K/V physical page bases into
    256 bytes of already-reserved shared memory and preserved a safe fallback
    for wider segments. It passed every correctness/high-ID gate and reduced
    126K instructions from about 711.7 M to 710.9 M, but improved stable
    126K-250K latency by only 0.7-0.9%. The block-table/address work was not the
    dominant feed cost; repeated Q preparation and FP8 decode remain much
    larger. Do not infer a large benefit merely because an independent kernel
    stages page IDs. Measure the fraction of instructions and stalls removed.

72. **For small-query long-context verify, Q preparation must be outside the
    KV-tile loop.** V7-E11 assigned one 16-row Q group to each of three warps
    and retained sixteen BF16 WMMA A fragments per owner warp across the scan.
    Despite rising from 79 to 166 registers/thread, shared memory remained the
    occupancy limiter, so the kernel kept two CTAs/SM with zero spill. Stable
    126K-250K latency improved about 38-39%, instructions fell 37%, and
    long-scoreboard stalls fell 28.48% -> 9.28%. Register count by itself is
    not a rejection criterion; calculate the actual occupancy limiter and
    verify local bytes/spills. The next bottleneck is now barrier/short
    dependency cost, not Q reload or global scoreboard.

73. **Independent GQA groups should not serialize CTA-wide softmax/PV.**
    V7-E12 kept E11's three persistent-Q owner warps but gave each a disjoint
    BF16 P pack and FP32 PV scratch slice. Replacing three group-level CTA
    phases with warp-local work plus one tile-tail barrier improved stable
    126K-250K latency by about 19-20%. NCU barrier stalls fell 29.52% -> 15.47%
    and tensor activity rose 2.41% -> 3.01%. Keep score and compact-P storage
    disjoint: in-place FP32-to-BF16 compaction creates a cross-lane overwrite
    race even when every warp owns a separate group.

74. **Fewer WMMA scratch-store bank conflicts need not improve the complete
    store-to-merge chain.** V7-E13a padded each FP32 `16x16` PV scratch from
    `ld=16` to `ld=20`. It preserved resources and correctness but slowed the
    stable 126K-250K tiers by 0.2-0.5%. The padded row phase helps the opaque
    WMMA store pattern while making the following scalar merge loads less
    favorable. Measure the whole producer/consumer chain; do not optimize one
    side's bank map in isolation.

75. **Score-row padding is only measurable under controlled clocks.** V7-E13b
    padded FP32 score rows from 32 to 36 columns while restoring E12's dense PV
    scratch. With the CMP170HX locked at 1350MHz, latency improved 3.7-6.8%
    from 4K through 250K; at 126K shared-load conflicts fell about 61% and
    short-scoreboard stalls fell 21.01% -> 15.85%. An earlier unlocked run
    falsely suggested short-tier regressions because Boost state differed. Lock
    or interleave clocks for kernel A/B; never choose dispatch from sequential
    auto-Boost samples.

76. **P-row padding can remove shared conflicts without changing KV traffic.**
    V7-E14 kept score `ld=36` and padded the BF16 P operand from `ld=32` to
    `ld=40`. At 126K, shared-load conflicts fell 67% and store conflicts 39%,
    while latency improved a further 3.2% with identical resources and exact
    outputs. The smaller wall-time gain shows that conflict counters are a
    bottleneck diagnostic, not a direct speedup forecast.

77. **Accumulator row padding alone does not move the remaining conflict wall.**
    V7-E15 changed only the persistent FP16 accumulator from physical `ld=256`
    to `ld=258`, keeping logical D=256 and preserving two-CTA residency after
    allocator rounding. Correctness/high-block-ID passed, but three locked-clock
    scans stayed within about ±0.1% of E14. NCU at 126K remained at roughly
    9.08M shared loads and 14.49M stores with 14.6% barrier and 14.5% short
    scoreboard stalls. Attribute the specific shared-store instructions before
    trying another accumulator layout; retain E15 only as a negative control.

78. **Removing a per-tile barrier requires moving every live consumer off the
    raw staging alias, and still needs one publication barrier.** V7-E17 moved
    the FP32 score packs from `q_shared` into the tail of `tmp_shared`, then
    removed the tile-tail CTA barrier. At 1350MHz this produced a stable
    1.0-1.5% gain over E16 from 20K through 250K, with barrier stalls falling
    12.86% -> 12.50% and no resource change. Do not remove the final
    end-of-loop barrier: warp 3 can otherwise publish/read `acc_shared` while
    an owner warp is still finishing its last PV merge. Verify the entire
    producer/consumer lifetime before deleting a synchronization point.

79. **A resident shared LUT can beat “zero extra shared loads” when decode
    arithmetic dominates.** V7-E18 reused the already-copied 256-entry BF16
    E4M3FN LUT for K/V conversion instead of recomputing exponent/mantissa and
    `clz` per byte. Shared-bank conflicts and scoreboard stalls increased, but
    fixed-clock wall latency fell 11.6% at 4K and 17-22% from 20K to 250K with
    unchanged DRAM traffic, registers and occupancy. Keep exact LUT semantics
    (including fail-closed NaN codes) and measure end-to-end latency; do not
    reject a lookup path solely because shared transaction counters rise.

80. **Pairing LUT decode accesses is a separate win from choosing the LUT.**
    V7-E19 loads two adjacent raw FP8 bytes as `uint16_t`, performs two exact
    shared-LUT reads and stores two BF16 values as one `uint32_t`. The mapping
    is even-aligned for the 256-wide tile, so no byte-order or tail special
    case is needed. At locked 1350MHz it cut E18 latency another 4.4-9.4%
    (up to 29% versus E17) with unchanged 132 registers, 81,856B shared and
    two-CTA occupancy. Keep alignment assertions; do not generalize the pair
    mapping to odd-width or unaligned cache geometries without a new gate.

81. **Random `__ldg` LUT reads are slower than the resident shared table on
    this SM80 path.** V7-E20 kept E19's paired decode and changed only the
address space; correctness and occupancy were identical, but locked-clock
latency regressed 1.7-6.1% (about 6% at long context). The 256-entry table
is small enough that shared-bank conflicts are cheaper than per-lane
read-only-cache misses. Retain E19 unless a different broadcast/constant
access pattern is measured.

82. **An idle warp can hide the next K-tile load when the raw staging alias is
already free.** V7-E21 uses warp 3 to fetch the next block id and raw K bytes
while the three owner warps perform QK/softmax/PV for the current tile. The
next iteration's existing CTA barrier publishes the prefetch before decode;
V still uses the normal all-thread stage because the alias holds only one raw
matrix. On locked clocks this reduced query-8 verifier latency 2.2-11.1% over
E19, with the largest gains at 126K-250K, and NCU long-scoreboard stalls fell
from 17.70% to about 7.6%. Keep the final publication barrier and verify
request-local block ids: a persistent next-tile pointer would create stale
state across requests. The extra register (133 vs 132) did not reduce the
two-CTA occupancy, but this is still an isolated prototype until multi-request,
CUDA Graph and vLLM A/B gates pass.

83. **CUDA Graph compatibility is shape- and-address-specific.** V7-E22
captured the E21 `partial`+`combine` sequence for two requests
(`lengths=[895,896]`, `q_lens=[5,8]`) and replayed it successfully with
`max_abs=0.001953`. This proves the kernel's synchronization is capture-safe,
not that one graph can serve arbitrary scheduler batches. Integration must
maintain a graph pool keyed by capture shape (and refresh request-local block
tables before replay), with eager fallback for misses.

84. **Aligning LUT entries is not automatically worth a shared-memory budget
trade.** V7-E23 repacked the 256 BF16 entries into aligned 32-bit slots and
reclaimed the reserved temporary tail, preserving 81,856 B and two-CTA
occupancy. Correctness passed, but fixed-clock scans regressed 4K by about
3.3% and improved 20K-250K by less than 1%, below the acceptance gate. Keep
the compact E21 LUT representation unless a new access pattern changes this
balance.

85. **Overlapping V staging with owner compute is not useful if the copy loses
parallelism.** V7-E24 used idle warp 3 to stage V while the owner warps ran
QK/softmax, but that made the copy four times less parallel than the original
all-thread stage. Correctness remained exact, yet fixed-clock latency regressed
about 25% at 20K-250K. Keep E21's K-only prefetch and all-thread V stage unless
a true multi-warp/double-buffer design preserves copy bandwidth without
losing two-CTA occupancy.

86. **`cp.async` does not guarantee a win for a small warp-prefetch group.**
V7-E25 kept a synchronous fallback but still added commit/wait-group overhead
and raised registers from 133 to 134. Fixed-clock scans regressed 1.2-1.3% at
20K-250K and about 4.6% at 4K. Retain E21's synchronous warp-3 K prefetch
unless a design can pipeline multiple groups without an immediate wait.

87. **Standalone prototype timing is not a vLLM baseline.** V7-E26 routed
the real SpecDecodeAttention API to E21 for one exact static-FP8 shape and
compared identical inputs against the existing Triton q8 specialization.
E21 was 1.52-1.83x slower at 4K/126K/250K despite max_abs 0.000008-0.000015.
Do not integrate a standalone kernel without an apples-to-apples API test;
profile the Triton path and E21 under the same wrapper before changing
dispatch.

88. **A streaming cache modifier is not automatically better for q8 KV.**
V7-E27 applied `.cg` only to the q8 global K/V loads. Locked-clock results
changed by -0.2%/-1.2%/+0.5% at 4K/126K/250K, with identical output. Keep the
existing cache policy unless a workload-specific trace shows L1 pollution.

89. **More warps can be a severe regression for the q8 specialization.**
V7-E28 changed only the launch from four to eight warps; locked-clock latency
was 22-78% worse at 4K/126K/250K with identical output. Keep four warps and
measure resource/launch effects before increasing parallelism.

90. **Wider KV tiles can hurt the q8 specialization.** V7-E29 changed only
TILE 32→64; latency regressed 11% at 4K and roughly 63-65% at 126K/250K.
Keep TILE=32 unless register/live-state and long-context measurements show a
real benefit.

91. **Replacing a hot 256-entry FP8 LUT with arithmetic can be worse.**
V7-E30's integer/exp2 decoder regressed 12% at 4K and 73-75% at long
contexts. On SM80, retain the LUT unless instruction-level profiling proves
the table lookup is the dominant cost.

92. **Tiny address-conversion wins must be repeated before keeping them.**
V7-E31's q8 int32 block-ID fast path showed sub-0.5% long-context change and
unstable 4K deltas across 500-iteration repeats. Do not keep a short-tier
fast-path on a single noisy scan.

93. **One-tile page-table prefetch is not automatically useful.** V7-E32
prefetched the next q8 page block ID before the boundary, but locked-clock
latency moved only -0.8%/-0.7%/+0.6% at 4K/126K/250K with identical output.
Reject below the measurement gate unless a trace shows address lookup is a
real bottleneck.

94. **A full register accumulator can be numerically right but resource-wrong.**
V7-E33 removed the shared accumulator and reached 255 registers/thread with
124--172B spills. Even with roughly 4% lower isolated latency, spilled
register state invalidates the candidate; require zero spills before judging
any registerization result.

95. **Partial registerization may be safe but too small to matter.** V7-E34
kept four D16 accumulator tiles in registers (168 registers/thread, zero
spills) but improved locked-clock latency by only 1.8%/1.1%/1.2% at
4K/126K/250K. Apply the measurement gate before retaining a hybrid path.

96. **Softmax lane pairing must follow the half-warp row layout.** V7-E35
    correctly maps `local_row = lane & 15` and `half = lane >> 4`, then
    exchanges peers with XOR-16; adjacent-lane pairing (`lane >> 1`,
    `lane & 1`) is wrong for this WMMA layout. E35 used 164 registers/thread
    with zero spills and improved locked-clock verifier latency
    3.2%/6.6%/6.4% at 4K/126K/250K, so it is an accepted isolated candidate.
    Its reduction order differs slightly from E21; validate against the
    reference and still require API/multi-request/graph gates before dispatch.

97. **Legacy Nsight Compute parses application arguments positionally.** On
    2022.4, putting a standalone `--` before the Python command caused the
    benchmark's `--segments`/similar flags to be parsed as profiler options
    and reported a misleading ambiguity. Put the application executable
    first and pass its arguments directly. Matched E21/E35 NCU samples then
    worked and showed the E35 gain without an occupancy or cache-policy
    change.
