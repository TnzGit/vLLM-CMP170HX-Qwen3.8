# V7 prototype handoff

Status: V7-E21 warp-3 K prefetch on top of E19 is the current accepted isolated
scaffold on top of E18/E17/E16. E22 has now qualified a two-request
CUDA-Graph capture/replay on this scaffold. E21 preserves the exact shared LUT
and paired decode, and uses the otherwise idle fourth warp to stage the next
tile's raw K bytes while owner warps compute the current tile. A locked-clock
A/B improved query-8 latency 2.2-11.1% over E19 from 4K through 250K;
correctness, zero-spill, two-CTA and graph-capture gates pass. E21/E22 are not
connected to production dispatch.

Files:

- `v7_verifier.cu` — standalone PyTorch C++ extension with fixed SM80 partial
  and combine kernels.
- `../../../bench/test_v7_cuda_prototype.py` — JIT build, exhaustive device
  decoder check, reference check, boundary cases, mixed lengths, and
  int32/int64 index variants.
- `build_and_smoke.sh` — convenience wrapper for the bench.
- `README.md` — geometry, interface, build command, and limitations.

## V7-E19 paired shared-LUT decode (accepted isolated scaffold)

Two adjacent raw FP8 bytes are loaded as one aligned `uint16_t`; their two
BF16 LUT values are packed into one aligned `uint32_t` shared store. The even
mapping is compile-time aligned and preserves cache bytes, LUT semantics,
scales, geometry, synchronization and ABI. Resources remain 132
registers/thread, zero local bytes/spills, 81,856B dynamic shared and two
active CTAs/SM; exhaustive, mixed-boundary and high-block-ID correctness pass.

Locked-1350MHz query-8 medians (us/layer) are 190.2/516.5/1,260.3/2,464.7/
3,822.9/4,745.7 at 4K/20K/60K/126K/200K/250K, 4.4-9.4% below E18. NCU at
126K reports 12.50% barrier, 17.70% long-scoreboard and 18.75%
short-scoreboard stalls, about 38.40M aggregate shared conflicts and
258.21MB DRAM reads. It remains an isolated scaffold pending multi-request
and end-to-end vLLM A/B.

## V7-E20 global read-only LUT (rejected negative control)

E20 changed only E19's LUT address space to `__ldg` device reads. Full
correctness and resources were unchanged, but three locked-1350MHz scans were
1.7-6.1% slower (long-context medians 2,615.0/4,015.0/5,031.7 us/layer at
126K/200K/250K versus E19's 2,464.7/3,822.9/4,745.7). Random per-lane
read-only-cache latency loses to the shared LUT, so E20 is rejected and the
source remains E19.

## V7-E21 warp-3 next-K prefetch (accepted isolated scaffold)

E21 keeps E19's cache, shared-memory and ABI geometry unchanged. During each
tile's owner-local QK/softmax/PV work, the otherwise idle warp 3 loads the next
tile's block id and raw K bytes into the existing `q_shared` staging alias. The
next iteration begins with the retained CTA barrier, decodes the prefetched K,
then stages/decodes V through the normal all-thread path. The final publication
barrier remains mandatory; no request or block-table state is persisted across
launches.

The candidate passed exhaustive E4M3FN decoding, mixed-boundary lengths,
int32/int64 indices, 4-GiB high-block-ID checks and the existing numerical
tolerance. Resources are 133 registers/thread, zero local bytes/spills,
81,856 B dynamic shared and two active CTAs/SM (one extra register versus E19).

Three locked-1350MHz scans (query length 8, 300 iterations) produced these
medians in us/layer:

| context | E19 | E21 | change |
|---:|---:|---:|---:|
| 4K | 190.2 | 186.1 | -2.2% |
| 20K | 516.5 | 479.0 | -7.3% |
| 60K | 1,260.3 | 1,136.3 | -9.8% |
| 126K | 2,464.7 | 2,205.6 | -10.5% |
| 200K | 3,822.9 | 3,403.6 | -11.0% |
| 250K | 4,745.7 | 4,218.6 | -11.1% |

At 126K, NCU reports 12.50% barrier, about 7.6% long-scoreboard and 19.55%
short-scoreboard stalls, about 38.45M aggregate shared-bank conflicts and
258.21 MB DRAM reads. Relative to E19's 17.70%/18.75% long/short scoreboard,
the prefetch removes most of the exposed K-load wait while slightly shifting
the remaining dependency pressure to short scoreboard; traffic and occupancy
are unchanged. This is a strong isolated result, but the current source still
requires multi-request stress, CUDA Graph capture and end-to-end vLLM A/B
before any integration decision.

## V7-E22 two-request CUDA Graph capture (accepted validation gate)

The final E21 extension was warmed up and captured with a fixed two-request
shape (`lengths=[895,896]`, `q_lens=[5,8]`) using the existing `partial` and
`combine` bindings, then replayed on the same static tensor addresses. Capture
and replay both completed without CUDA errors; the replayed output had
`max_abs=0.001953` against the reference. This validates that the E21 warp-3
prefetch and its retained barriers are compatible with CUDA Graph execution.

This is a fixed-shape/address gate, not proof that arbitrary scheduler shapes
are graph-safe. A vLLM integration must use one graph per supported capture
shape (or eager fallback for shape misses) and must rebind request-local block
tables before replay.

## V7-E18 shared-LUT FP8 decode (accepted isolated scaffold)

The K/V tile decoder indexes the CTA-local 256-entry BF16 LUT copied at kernel
entry. It retains the exact E4M3FN mapping and fail-closed `0x7f/0xff` zeros,
with no change to cache bytes, scales, block geometry or ABI. Resources are
132 registers/thread, zero local bytes/spills, 81,856B dynamic shared and two
active CTAs/SM; exhaustive, mixed-boundary and high-block-ID correctness pass.

Locked-1350MHz query-8 medians (us/layer) are 198.9/558.1/1,378.2/2,709.1/
4,202.8/5,239.6 at 4K/20K/60K/126K/200K/250K, 11.6-21.8% below E17.
NCU at 126K reports 12.50% barrier, 15.61% long-scoreboard and 16.74%
short-scoreboard stalls, about 38.42M aggregate shared conflicts and
258.21MB DRAM reads. The extra shared LUT reads are outweighed by removing
the integer decode instruction path. Run multi-request correctness and
end-to-end vLLM A/B before any integration.

## V7-E16 half2 accumulator merge (accepted isolated scaffold)

The merge loop maps 128 logical FP16 pairs across all 32 lanes, converts each
pair to FP32, applies the FP32 row alpha with FMA, and stores it back as
`half2`. Logical output indexing remains D=256. Resources are 134
registers/thread, zero local bytes/spills, 81,856B dynamic shared (81,920B
allocator round) and two active CTAs/SM. Exhaustive decoder, mixed boundary,
int32/int64 and 4-GiB high-block-ID tests pass.

Locked-1350MHz query-8 medians (us/layer) are 225.3/684.5/1,751.1/3,485.7/
5,439.8/6,757.3 at 4K/20K/60K/126K/200K/250K, or 6.2-12.1% below E14.
NCU at 126K shows 6.049M tensor instructions, 12.098M/17.556M shared
load/store conflicts and 12.86%/13.11%/14.78% barrier/long/short stalls.
The higher conflict counters do not negate the wall-time gain; scalar merge
transactions and dependency length are reduced. Before any production use,
run multi-request correctness and end-to-end vLLM A/B with E14 as baseline.

## V7-E17 disjoint score tail and tile-barrier reduction

The three 2,304-byte FP32 score packs now live at offset 6,912 in the existing
14,336-byte `tmp_shared` allocation. `q_shared` is raw-only after the fixed
load/decode barriers, so the former per-tile CTA barrier is gone. A single
end-of-loop CTA barrier remains before publication; it is required to prevent
warp 3 from reading `acc_shared` while owner warps finish the last merge.

Resources: 132 registers/thread, zero local bytes/spills, 81,856B dynamic
shared (81,920B allocator round), two active CTAs/SM. Full and high-block-ID
correctness passed. Locked-1350MHz query-8 medians in us/layer are
225.0/674.2/1,726.0/3,442.7/5,372.7/6,689.1 at
4K/20K/60K/126K/200K/250K. NCU at 126K reports 12.50% barrier, 12.54%
long-scoreboard and 13.08% short-scoreboard stalls, with about 29.65M
aggregate shared conflicts and 258.21MB DRAM reads. This is accepted as a
low-risk secondary scaffold; it is not wired into vLLM or production.

## V7-E15 negative control (rejected)

E15 changed the persistent FP16 accumulator from physical `ld=256` to `ld=258`
while retaining logical `D=256`. It compiled with 81,856B dynamic shared
(81,920B allocator round), 164 registers/thread, zero local bytes/spills and
two active CTAs/SM. Exhaustive correctness and high-block-ID gates passed.
Three locked-1350MHz 300-iteration scans at 4K/20K/60K/126K/200K/250K showed
no repeatable difference from E14 (within roughly ±0.1% at query=8). NCU at
126K showed 9.079M shared-load and 14.481M shared-store conflicts, effectively
the same as E14, so this factor did not alter the dominant dependency path.

V7-E12 candidate layout in `v7_verifier.cu`:

- persistent all-row FP16 shared accumulator `[48, 256]`, 24,576 B;
- one decoded BF16 K/V tile physically `[2, 32, 264]`, 33,792 B; only the
  first 256 elements of each row are logical K/V data;
- one BF16 Q/scores/raw alias physically `[16, 264]`, 8,448 B; its first 8,192 B
  stage raw K or V before the decode barrier, then carries three disjoint
  FP32 score packs `[3, 16, 32]` (6,144 B);
- one retained FP32 temporary allocation, 14,336 B; its first 3,072 B hold
  three disjoint dense BF16 P packs and its next three 1,024 B slices are
  owner-warp `[16,16]` FP32 PV scratch;
- one reused E6 LUT allocation, 512 B; it is copied once per CTA and remains
  outside the hot pure-bit decode path;
- total dynamic shared: 81,664 B;
- the first 8,192 B of the Q/P allocation aliases one raw K/V matrix at a
  time during each tile load; the persistent Q fragments make a separate Q
  shared copy unnecessary after the initial preparation;
- `fp8_lut` remains a public ABI argument and is copied as in E6, while the
  exact bit decoder does not read it;
- every tile performs E6's one current-page block-table load; no segment-local
  page-base prefetch is present;
- warps 0..2 each own a 16-row Q group, convert/load Q once, and retain sixteen
  K16 row-major WMMA A fragments in registers across every KV tile;
- E12 measured 164 registers/thread with zero local bytes/spills. The
  allocation remains 81,664 B and the driver reports two active CTAs/SM;
- `decode_e4m3fn_bf16` is an exported CUDA helper used by the bench's exhaustive
  256-code device check; it requires 254 finite bit-exact results plus zero for
  `0x7f/0xff`;
- `m/l` and alpha are retained in owner-lane registers, with lanes 0..15
  holding one row-pack state across the segment and warp shuffle supplying
  alpha while merging each scratch tile;
- `part_o/m/l` published once per segment after all tiles, preserving the ABI.

E12 WMMA/decode and synchronization mapping:

- QK uses two N16 tiles per 16-row pack: row-major BF16 Q with logical
  `[16,256]` and physical `ld=264`, and col-major BF16 K with logical
  `[256,32]` viewed from physical token-major rows with `ld=264`;
- owners 0..2 each write its own `[16,32]` FP32 score pack, perform causal
  online softmax without a group CTA barrier, write BF16 P to its disjoint tmp
  pack with compact `ld=32`, and complete all sixteen D16 PV tiles. Each WMMA
  tile stores to that owner's private `[16,16]` FP32 scratch, executes
  `__syncwarp()`, and the same owner merges only its 16 accumulator rows using
  `previous * alpha + scratch`. Warp 3 is idle in this phase.
- each tile uses E6's four phases: coalesced-stage raw K into Q/P's first
  8,192 B as aligned 16-byte `uint4` chunks, barrier, decode K, barrier, then
  repeat for V using the same alias.  An externally unaligned source uses the
  scalar fallback and invalid token tails are zero-filled.  The pure integer
  decoder maps `0x7f/0xff` to explicit BF16 zero.
- after all owners finish a tile, one CTA barrier is retained before the next
  tile's raw stage can reuse `q_shared`; this prevents a fast owner from
  overwriting a slower owner's score/P region. There are no inter-group CTA
  barriers in softmax/PV. Each tile still loads its current page's int32/int64
  block id through the old E6 block-table path, promotes it to int64, and then
  applies cache strides; there is no segment-page table or fallback path.
  Finite normal codes map with `(8+m)*2^(e-10)` to BF16 exponent `e+120`/
  fraction `m<<4`; E4M3FN subnormals normalize from their highest mantissa bit.

E11 baseline qualification result:

- resources: 166 registers/thread, zero local bytes/spills, 81,664 B dynamic
  shared and two active CTAs/SM;
- exhaustive decoder, complete correctness and 4-GiB high-block-ID passed;
- 4K/70K/126K/200K/250K first-run latency was
  346.5/3,397.3/5,020.4/7,841.9/9,791.5 us per layer; repeated long tiers were
  stable at about 5.01/7.85/9.79 ms;
- versus E6, stable 126K/200K/250K improved about 38.3%/38.9%/38.9%;
- NCU at 126K: 447.44 M instructions, 6.049 M tensor instructions, 2.41%
  tensor active, 29.52% barrier, 9.28% long scoreboard, 13.97% short
  scoreboard, 63.514 M/31.097 M shared load/store conflicts and 258.24 MB
  DRAM read.

E12 qualification result:

- resources: 164 registers/thread, zero local bytes/spills, 81,664 B dynamic
  shared and two active CTAs/SM;
- exhaustive decoder, complete correctness and 4-GiB high-block-ID passed;
- 4K/70K/126K/200K/250K first-run latency was
  239.4/2,325.9/4,040.9/6,319.0/7,852.2 us per layer; two repeats were
  241.7/2,382.8/4,056.4/6,304.6/7,885.7 us and
  260.1/2,548.8/4,046.5/6,303.0/7,876.7 us;
- versus E11, stable 126K/200K/250K latency improved about
  19.2%/19.7%/19.5%;
- NCU at 126K: 436.54 M instructions, 6.049 M tensor instructions, 3.01%
  tensor active, 15.47% barrier, 10.74% long scoreboard, 21.01% short
  scoreboard, 0.25% MIO throttle, 69.558 M/25.074 M shared load/store
  conflicts and 258.22 MB DRAM read;
- relative to E11, barrier stalls fell 29.52% -> 15.47% and tensor activity
  rose 2.41% -> 3.01%. Short-scoreboard stalls and shared-load conflicts rose,
  defining the next measurement-led optimization target;
- no production files, active-series entries, or qualified test-site files
  were changed.

Decision: accept E12 as the new hardware-qualified isolated V7 scaffold. Any
follow-up should preserve E12 correctness and target its 21.01%
short-scoreboard/shared-load boundary before production integration.

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

Historical E5 measured result:

- resources remained 79 registers/thread, zero local spill, 81,664 B shared
  and two active CTAs/SM;
- full correctness/high-block-ID passed; high-block max error was 0.0625;
- 4K/70K/126K/200K/250K latency was
  473.4/7,984.3/14,355.3/22,863.6/28,509.2 us;
- versus E4a: -1.2%/-2.9%/+0.1%/+0.02%/-0.05%; short/mid improved but long
  contexts did not meet the >=5% admission target;
- NCU remained 1.093 B instructions, 51.99% long scoreboard, 0.88% tensor
  activity and 72.22 M shared-load conflicts.

Historical E6 measured result:

- the exported CUDA helper passed all 256 encodings: 254 finite BF16 bits were
  exact against PyTorch, and both NaN encodings followed the qualified LUT's
  fail-closed BF16-zero policy;
- resources remained 79 registers/thread, zero spill, 81,664 B shared and two
  active CTAs/SM; full correctness/high-block-ID passed at 0.0625 max error;
- 4K/70K/126K/200K/250K latency was
  378.5/4,957.3/8,130.3/12,845.7/16,041.2 us;
- versus E5 this is -20.0%/-37.9%/-43.4%/-43.8%/-43.7%;
- NCU: executed instructions 1.093 B -> 711.7 M, shared-load conflicts
  72.22 M -> 63.52 M, long scoreboard 51.99% -> 28.48%, tensor activity
  0.88% -> 1.51%; barrier stalls rose to 19.23%.

Historical E7 measured result:

- resources/correctness/exhaustive decoder remained unchanged and passed;
- 4K/70K/126K/200K/250K latency was
  484.6/7,899.3/13,663.6/21,651.6/27,041.6 us, 28-69% slower than E6;
- NCU barrier stalls fell 19.23% -> 11.19%, but long scoreboard rose
  28.48% -> 54.34%, tensor activity fell 1.51% -> 0.92%, and instructions rose
  711.7 M -> 845.6 M.

Historical E8 measured result:

- all 254 finite codes were bit-exact and `0x7f/0xff` fail-closed to zero;
  resources/correctness/high-block-ID remained green;
- 4K/70K/126K/200K/250K latency was
  407.8/4,924.6/7,979.2/12,692.6/15,781.1 us;
- versus E6: +7.7%/-0.7%/-1.9%/-1.2%/-1.6%, failing both the >=5% long gain
  and <=2% 4K-regression admission rules;
- NCU instructions improved 711.7 M -> 675.1 M, but barrier stalls stayed
  19.31%, long scoreboard was 29.11%, and tensor activity 1.53%.

E10a qualification result:

1. Resources: 86 registers/thread, zero local bytes, 81,664 B dynamic shared,
   two active CTAs/SM.
2. Exhaustive decode, complete correctness and 4-GiB high-block-ID passed.
3. Three-run long tiers were stable at about 8.06/12.76/15.91 ms per layer for
   126K/200K/250K, only 0.7-0.9% faster than E6.
4. NCU at 126K: 710.90 M instructions, 19.02% barrier, 28.08% long
   scoreboard, 1.52% tensor active and 258.26 MB DRAM read.

Decision: reject E10a for production admission; preserve it as a correct
measured factorial. Keep production disconnected and start the next structural
experiment from E6 semantics, targeting repeated Q preparation/fragments.

Historical measured E9b result:

- resources: 79 registers/thread, zero local bytes, 81,664 B dynamic shared,
  two active CTAs/SM;
- all 256 decode codes, complete correctness and 4-GiB high-block-ID passed;
- 4K/70K/126K/200K/250K latency:
  346.9/4,829.9/8,354.1/13,228.3/16,511.7 us;
- versus E6: -8.3%/-2.6%/+2.8%/+3.0%/+2.9%;
- NCU at 126K: 783.4 M instructions, 63.51 M shared-load conflicts, 31.21 M
  shared-store conflicts, 18.73% barrier, 27.36% long scoreboard, 1.46%
  tensor active and 258.26 MB DRAM read.

Historical decision: reject E9b for the long-context objective. The CUDA 13.0 public API
ends at half2, and its half-to-float-to-BF16 bridge costs more instructions
than E6 exact bit synthesis over long scans. Preserve this source as a measured
factorial; E10a starts split-local page/base staging from E6, not E9b.

Known risks:

- WMMA, staged decode and E6 bit synthesis all passed exhaustive/complete
  correctness gates; remaining risks are performance and integration scope.
- E3a proved that 83,200 B leaves one CTA/SM; E3b proved 81,152 B restores two.
  E4a raises the allocation to 81,664 B, leaving only 256 B of headroom; its
  intended two-CTA residency was measured successfully.
- E8 reduced instructions but not barrier stalls and missed admission; phase
  count alone is not the relevant synchronization cost.
- E10a's page-base table is deliberately capped at sixteen pages (256 B inside
  the retained 512-B allocation); oversized segments take the old per-tile
  block-table/barrier fallback rather than truncating or indexing past the
  prefetched entries.
- The launcher rejects non-contiguous caches and requests the required
  dynamic shared-memory carveout; WMMA is compiled only for SM80.
- `part_o/m/l` are the established flattened workspace contract, but the
  prototype does not allocate or own that workspace and does not hook the
  active vLLM series.
- A real high physical block-ID test requires a sparse/large KV allocation;
  the bench exercises the int64 code path with a finite-only raw-code payload
  so NaN propagation does not obscure address validation.

Measured E0 resources were 48 registers/thread, 81,920 bytes dynamic shared,
zero local bytes and two active CTAs/SM.  The 895/896/897/4097 correctness
smoke passed at max absolute error 0.000977.  These are historical E0/E1
results and do not qualify the E10a WMMA scaffold by themselves.

`bench/test_v7_cuda_prototype.py --full --high-block-id` subsequently passed
q=6/7, mixed queries, 8K/32K/65K KV and physical block ID 2341.  The synthetic
4.00-GiB two-cache case had max absolute error 0.031250 (<0.08).

No production files, active-series entries or qualified test-site files were
changed.

## V7-E23 aligned 32-bit LUT container (rejected negative control)

E23 repacked each BF16 LUT value into an aligned 32-bit shared slot, reclaiming
the unused 512-byte temporary tail so total dynamic shared memory and
occupancy stayed unchanged. Full correctness, mixed requests and the high
block-ID gate passed (maximum error 0.0625, below the 0.08 limit), with 133
registers/thread, 81,856 B shared and two CTAs/SM.

Three locked-1350MHz scans were not a meaningful improvement over E21. Median
us/layer at 4K/20K/60K/126K/200K/250K was 192.2/475.0/1,128.5/2,188.2/
3,375.9/4,182.8 versus E21's 186.1/479.0/1,136.3/2,205.6/3,403.6/4,218.6:
long-context change was only 0.7-0.9% while 4K regressed 3.3%. The candidate
is rejected as below the single-factor gate; source remains E21.

## V7-E24 warp-3 V prefetch (rejected negative control)

E24 moved V staging after K decode and let warp 3 copy the current tile's V
bytes while owner warps computed QK/softmax, then decoded V before PV. Full
correctness and high-ID checks passed with unchanged 133-register, 81,856-B,
two-CTA resources. However, a single warp's V copy is four times less
parallel than the original all-thread stage; two locked-clock scans regressed
about 25% at 20K-250K (250K about 5.29 ms/layer versus E21's 4.22 ms).
The candidate is rejected and the isolated source has been restored to E21.

## V7-E26 vLLM API dispatch wrapper (rejected)

A disposable wrapper tested routing the real SpecDecodeAttention API to E21
for the exact qualified static-FP8 shape (Hq/Hkv/D=24/4/256, qmax=8,
NSEG=35, block=896), with Triton fallback for all other requests. Numerical
error stayed at 0.000008-0.000015, but E21 was 2.52x/2.83x/2.75x the Triton
latency at 4K/126K/250K (180.8/2109.1/4111.0 us versus 71.8/746.7/1495.6).
This rejects direct dispatch integration; the isolated source remains E21.
The result also shows that the earlier standalone E21 scan was not an
apples-to-apples vLLM baseline.

## V7-E27 q8 global K/V `.cg` cache policy (rejected)

Changing only the q8 kernel's raw K/V loads to `cache_modifier=".cg"` passed
the same numerical output. Locked-1350MHz q=8/NSEG=35 results (us/layer)
were 53.7/774.6/1529.6 current versus 53.6/765.2/1537.2 with `.cg` at
4K/126K/250K. The mixed ±1.2% result is below the gate; source remains E21.

## V7-E28 q8 warp count 4→8 (rejected)

Changing only the q8 launch to eight warps produced 66.4/1379.2/2708.1
us/layer versus 54.6/776.1/1548.1 at 4K/126K/250K (locked 1350MHz,
q=8/NSEG=35). Output was identical, but the candidate regressed 22-78% and
was rejected.

NCU on the same partial launch measured Triton q8 at 895 us, 73.53% memory
throughput and 16.30% DRAM throughput, versus E21 at 2.52 ms, 54.65% and
5.79%. Both were 128-thread/grid-140 launches with 12.5% theoretical
occupancy; E21 used 81.86 KiB shared and 133 registers/thread, while Triton
used 57.34 KiB and 252 registers/thread. This points to E21's shared FP8/LUT
staging overhead and weaker effective cache traffic, not a missing occupancy
opportunity.

## V7-E29 q8 TILE 32→64 (rejected)

Only the q8 K/V tile width changed to 64. Locked-1350MHz q=8/NSEG=35
latency was 59.5/1278.5/2502.7 us/layer versus 53.6/775.7/1533.3 at
4K/126K/250K, with identical output. The wider tile is rejected.

## V7-E30 q8 arithmetic E4M3 decode (rejected)

Replacing the q8 256-entry LUT with arithmetic sign/exponent/mantissa decode
passed finite output checks but regressed locked-clock latency by 12% at 4K
and 73-75% at 126K/250K (64.2/1349.6/2658.2 versus 57.1/772.2/1532.7
us/layer). No source change kept.

## V7-E31 q8 block-ID int32 fast path (rejected)

Removing only the two explicit q8 block-ID int64 conversions changed long
latency by under 0.5%; repeated 500-iteration 4K pairs varied from -0.9% to
+4.6% to +2.0%, so the short result was noise. No source change kept.

## V7-E32 q8 next-page block-table prefetch (rejected)

The candidate loaded the next page block ID one tile before the page
boundary and consumed it at the boundary. Locked-1350MHz q=8/NSEG=35 scans
were 55.9/55.3 us at 4K, 775.8/770.4 us at 126K and 1533.7/1543.2 us at
250K (baseline/candidate). Output was identical; deltas (-0.8%, -0.7%,
+0.6%) are below the 2% gate, so no source change was kept.

## V7-E25 cp.async K prefetch (rejected negative control)

E25 replaced E21's warp-3 synchronous vector loads for the next K tile with
SM80 `cp.async` groups, retaining a conservative synchronous fallback for
unaligned/tail chunks. Full correctness and high-ID checks passed, but the
candidate rose to 134 registers/thread and was consistently slower: three
locked-clock scans were about 1.2-1.3% slower at 20K-250K and about 4.6%
slower at 4K. Commit/wait-group overhead outweighed any additional overlap;
the isolated source has been restored to E21.

## V7-E33 full register-resident accumulator (rejected)

Moving the entire 24,768-byte persistent FP16 accumulator into per-lane
`__half2` registers was numerically exact and gave about 3.7%/4.2%/4.5%
lower latency at 4K/126K/250K under locked 1350MHz. It is not resource-safe:
ptxas reached 255 registers/thread and emitted 124--172B spill stores/loads.
The candidate was discarded; the qualified E21 source remains unchanged.

## V7-E34 partial register accumulator (rejected)

Registerizing only four of the sixteen D16 output tiles passed correctness
and used 168 registers/thread with zero spills. Locked-1350MHz latency was
only 1.8%/1.1%/1.2% lower at 4K/126K/250K, below the 2% gate; no source
change was kept.

## V7-E35 cooperative half-warp softmax (accepted isolated candidate)

E35 splits each owner row's 32 score columns into two 16-column halves. Use
`local_row = lane & 15`, `half = lane >> 4`, and exchange peers with
`__shfl_xor_sync(..., 16)`; do not use adjacent lane pairs. The lower half
publishes the combined alpha/m/l state. Standard, mixed and high-ID
reference checks passed (max error 0.000977 standard, 0.062500 high-ID).
ptxas reported 164 registers/thread, zero spills, and unchanged shared/
two-CTA geometry. Locked-1350MHz q=8/NSEG=35 E21/E35 medians were
185.8/179.8 us at 4K, 2209.5/2064.0 us at 126K and 4220.9/3952.6 us at
250K (3.2%/6.6%/6.4% faster). Retain this as an isolated candidate; before
vLLM dispatch, run matched API, two-request and CUDA-Graph A/B checks.

### E35 NCU attribution

Matched Nsight Compute 2022.4 sampling of the 4K/q=8 partial launch measured
202.912 us for E21 and 195.136 us for E35. Compute-memory throughput was
23.98%/24.85%, DRAM 2.38%/2.47%, L1/TEX 24.62%/25.50%, L1 hit
87.86%/87.93% and L2 hit 55.19%/54.96% (E21/E35). Launch geometry and
shared memory were unchanged (128 threads, grid 140, 81.856 KiB, two CTA
limit); registers were 133/164. The gain therefore tracks reduced softmax
serialization rather than an occupancy or cache-policy change.
