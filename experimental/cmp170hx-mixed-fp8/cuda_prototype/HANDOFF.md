# V7 prototype handoff

Status: V7-E5 BF16 WMMA shared-raw-staging experiment passed correctness and
resource gates but failed the long-context performance admission gate. It is
not connected to production dispatch and is retained as measured evidence.

Files:

- `v7_verifier.cu` — standalone PyTorch C++ extension with fixed SM80 partial
  and combine kernels.
- `../../../bench/test_v7_cuda_prototype.py` — JIT build, reference check,
  boundary cases, mixed lengths, and int32/int64 index variants.
- `build_and_smoke.sh` — convenience wrapper for the bench.
- `README.md` — geometry, interface, build command, and limitations.

V7-E5 candidate layout in `v7_verifier.cu`:

- persistent all-row FP16 shared accumulator `[48, 256]`, 24,576 B;
- one decoded BF16 K/V tile physically `[2, 32, 264]`, 33,792 B; only the
  first 256 elements of each row are logical K/V data;
- one BF16 Q/P pack physically `[16, 264]`, 8,448 B; logical Q is 256-wide
  and P is repacked densely as `[16, 32]` with `ld=32`;
- one FP32 score/PV temporary `[16, 224]`, 14,336 B;
- one shared BF16 FP8 decode LUT `[256]`, 512 B;
- total dynamic shared: 81,664 B;
- the first 8,192 B of the Q/P allocation is aliased as compact raw staging
  before Q is loaded; K/V each use 512 aligned 16-byte chunks, four per thread;
- the shared 256-entry BF16 LUT is populated once at kernel entry;
- `m/l` retained in registers, with warp-0 lanes 0..15 owning three row-pack
  states each;
- `part_o/m/l` published once per segment after all tiles, preserving the ABI.

E5 WMMA/decode mapping:

- QK uses two N16 tiles per 16-row pack: row-major BF16 Q with logical
  `[16,256]` and physical `ld=264`, and col-major BF16 K with logical
  `[256,32]` viewed from physical token-major rows with `ld=264`;
- warp 0 lanes 0..15 perform the 32-score causal online softmax, write BF16
  P using compact `ld=32`, and save exact FP32 alpha in the Q/P buffer tail;
- PV main phase covers d=0..223 with fourteen N16 tiles: warps 0..2 own four
  each and warp 3 owns two; stores and merge use the FP32 temporary `ld=224`.
  After a barrier, warp 3 computes tail tiles d=224,240, stores them into the
  temporary prefix as dense 16x32 with `ld=32`, and after another barrier the
  whole CTA merges d=224..255.  Both phases apply `previous * alpha + tmp`.
- each tile stages raw K then barriers, decodes from shared raw bytes through
  the shared LUT then barriers, and repeats for V with a final decode barrier
  before Q loading.  E5 hoists physical block/head bases once per tile, loads
  valid rows through aligned `uint4` chunks, zero-fills incomplete token tails,
  and uses scalar byte fallback for an externally unaligned base rather than
  issuing an unsafe vector read.  The raw alias never overlaps decoded K/V or
  the LUT, and there is no K/V double buffer.

Historical E2 baseline: the unpadded candidate compiled to 88 registers/thread,
zero local spill, 81,920 B dynamic shared memory and two CTAs/SM.  Its complete
correctness gate passed after fixing P's store stride from Q's 256 columns to
compact 32 columns.  The largest ordinary error was 0.003906 and the 4-GiB
high-block-ID case was 0.031250.

It nevertheless failed performance admission: 4K/70K/126K/200K/250K were
488.2/8,546.7/15,269.0/24,167.6/30,208.5 us per layer.  At 126K, NCU measured
the same 6,048,768 tensor instructions as Triton but only 0.82% tensor-pipe
activity, 190.54 M shared-load bank conflicts, 51.20% long-scoreboard stalls
and 1.079 B executed instructions.  Triton measured 17.21%, 2.02 M, 12.09%
and 126.0 M respectively.  This is a shared-layout/scalar-feed failure, not
missing HMMA lowering.

Measured E1 resources were 126 registers/thread, zero stack/spill, 81,920 bytes
dynamic shared and two active CTAs/SM.  The full gate passed with maximum error
0.0625, but latency was 19-29x worse than Triton because QK/PV were scalar.

Historical E3a baseline compiled to 79 registers/thread and zero spill, and
its complete correctness/high-block-ID gate passed.  Padding reduced shared-load bank
conflicts from 190.54 M to 63.51 M, but 83,200 B crossed the residency cliff:
only one CTA/SM remained.  Latency regressed to
882.3/13,220.8/23,600.3/37,307.3/46,681.1 us at
4K/70K/126K/200K/250K.  Tensor-pipe activity fell further to 0.53%.

Historical E3b baseline compiled to 79 registers/thread, zero spill, 81,152 B
dynamic shared and two CTAs/SM.  Full correctness/high-block-ID passed.
Five-tier latency was 502.7/8,093.1/14,481.3/23,061.9/28,746.7 us, about 5%
faster than E2 at long contexts but 3% slower at 4K.  NCU still showed 52.26%
long-scoreboard stalls, 63.51 M shared-load conflicts and only 0.87%
tensor-pipe activity.

Historical E4a measured result:

- resources: 79 registers/thread, zero local spill, 81,664 B dynamic shared,
  two active CTAs/SM;
- complete correctness/high-block-ID gate passed; largest high-block error was
  0.0625 (<0.08);
- 4K/70K/126K/200K/250K latency was
  479.3/8,222.1/14,338.5/22,858.5/28,522.1 us;
- versus E3b this was -4.7%/+1.6%/-1.0%/-0.9%/-0.8%; versus unpadded E2 it
  improved 126K/250K by 6.1%/5.6%;
- NCU still reported 51.97% long-scoreboard stalls and only 0.88% tensor-pipe
  activity. Shared-load conflicts increased from E3b's 63.51 M to 72.22 M.

Historical E4b measured result:

- resources remained 79 registers/thread, zero local spill, 81,664 B shared,
  and two active CTAs/SM;
- full correctness/high-block-ID passed; high-block max error was 0.0625;
- 4K/70K/126K/200K/250K latency was
  541.5/8,240.1/14,349.7/22,862.3/28,517.7 us;
- versus E4a: +13.0%/+0.2%/+0.1%/+0.02%/-0.02%, so it failed admission;
- NCU remained effectively unchanged: 1.093 B executed instructions, 52.00%
  long-scoreboard stalls, 0.88% tensor activity, 72.22 M shared-load and
  31.15 M shared-store conflicts.

E5 measured result:

- resources remained 79 registers/thread, zero local spill, 81,664 B shared
  and two active CTAs/SM;
- full correctness/high-block-ID passed; high-block max error was 0.0625;
- 4K/70K/126K/200K/250K latency was
  473.4/7,984.3/14,355.3/22,863.6/28,509.2 us;
- versus E4a: -1.2%/-2.9%/+0.1%/+0.02%/-0.05%; short/mid improved but long
  contexts did not meet the >=5% admission target;
- NCU remained 1.093 B instructions, 51.99% long scoreboard, 0.88% tensor
  activity and 72.22 M shared-load conflicts.

Next handoff checklist:

1. Replace per-byte shared LUT lookup with exact E4M3FN-to-BF16 bit
   conversion; first exhaustively compare all 256 raw codes against PyTorch.
2. Preserve shared geometry/occupancy and the complete correctness gate.
3. Measure instruction count and shared-load conflicts before judging latency.
4. Keep page carry and P-fragment reuse as later isolated changes.

Known risks:

- WMMA BF16 accumulation, online softmax ordering, two-phase scratch reuse and
  E5's aliasing/staged decode all passed the complete correctness gate.
- E3a proved that 83,200 B leaves one CTA/SM; E3b proved 81,152 B restores two.
  E4a raises the allocation to 81,664 B, leaving only 256 B of headroom; its
  intended two-CTA residency was measured successfully.
- E5 keeps E4b's safe aligned-vector fallback policy but adds K/V staging and
  four explicit phase barriers per tile (three beyond E4b's final barrier).
  The raw alias must not overlap Q writes, decoded K/V or the LUT; the added
  shared-memory traffic and synchronization may erase the intended feed-path
  gain.
- The launcher rejects non-contiguous caches and requests the required
  dynamic shared-memory carveout; WMMA is compiled only for SM80.
- `part_o/m/l` are the established flattened workspace contract, but the
  prototype does not allocate or own that workspace and does not hook the
  active vLLM series.
- A real high physical block-ID test requires a sparse/large KV allocation;
  the bench only exercises the int64 code path with ordinary physical IDs.

Measured E0 resources were 48 registers/thread, 81,920 bytes dynamic shared,
zero local bytes and two active CTAs/SM.  The 895/896/897/4097 correctness
smoke passed at max absolute error 0.000977.  These are historical E0/E1
results and do not qualify the E5 WMMA scaffold by themselves.

`bench/test_v7_cuda_prototype.py --full --high-block-id` subsequently passed
q=6/7, mixed queries, 8K/32K/65K KV and physical block ID 2341.  The synthetic
4.00-GiB two-cache case had max absolute error 0.031250 (<0.08).

No production files, active-series entries or qualified test-site files were
changed.
