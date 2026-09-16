# V7 prototype handoff

Status: V7-E16 half2 accumulator merge is the current accepted isolated
scaffold. It preserves E13b score `ld=36`, E14 P `ld=40`, and E15's safe
accumulator `ld=258`, then changes only scalar accumulator merging to aligned
half2/float2 FMA pairs. A locked-clock A/B improved query-8 latency 6.2-12.1%
over E14 from 4K through 250K; correctness, zero-spill and two-CTA gates pass.
E15's stride-only factor was rejected and remains documented as a negative
control. E16 is not connected to production dispatch.

Files:

- `v7_verifier.cu` — standalone PyTorch C++ extension with fixed SM80 partial
  and combine kernels.
- `../../../bench/test_v7_cuda_prototype.py` — JIT build, exhaustive device
  decoder check, reference check, boundary cases, mixed lengths, and
  int32/int64 index variants.
- `build_and_smoke.sh` — convenience wrapper for the bench.
- `README.md` — geometry, interface, build command, and limitations.

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
