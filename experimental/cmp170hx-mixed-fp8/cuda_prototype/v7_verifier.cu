// Standalone V7 fixed-geometry verifier prototype.
//
// This file intentionally has no vLLM/FlashInfer dependency and is not wired
// into any production dispatch.  It mirrors the split-KV partial/combine
// workspace contract so that the kernel can be qualified in isolation.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>

#include <cstdint>
#include <limits>

namespace py = pybind11;

namespace {

constexpr int kHq = 24;
constexpr int kHkv = 4;
constexpr int kGroup = kHq / kHkv;
constexpr int kD = 256;
constexpr int kQmax = 8;
constexpr int kBlockSize = 896;
constexpr int kTile = 32;
constexpr int kRawChunkBytes = 16;
constexpr int kNseg = 35;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kRows = kGroup * kQmax;
constexpr int kRowsPerGroup = 16;
constexpr int kRowGroups = kRows / kRowsPerGroup;
// Pad the physical BF16 WMMA rows by eight elements.  The logical head
// dimension remains 256; P is a separate dense [16,32] operand below.
constexpr int kWmmaLd = 264;
constexpr int kKvElementsPerTile = kTile * kD;
constexpr int kKvMatrixElements = kTile * kWmmaLd;
constexpr int kRawChunksPerMatrix = kKvElementsPerTile / kRawChunkBytes;
constexpr int kRawChunksPerThread = kRawChunksPerMatrix / kThreads;
constexpr int kRawStageBytes = kKvElementsPerTile * sizeof(unsigned char);
// E6 retains E5's padded physical BF16 WMMA rows and logical 256-wide
// matrices.  The 16-row Q/P buffer is reused for each of the three row groups;
// its tail also carries the FP32 alpha values between softmax and PV fusion.
constexpr int kKvSharedBytes = 2 * kKvMatrixElements *
                               static_cast<int>(sizeof(uint16_t));
constexpr int kAccBytes = kRows * kD * static_cast<int>(sizeof(uint16_t));
constexpr int kQBytes = kRowsPerGroup * kWmmaLd *
                        static_cast<int>(sizeof(uint16_t));
// E6 keeps E5's FP32 PV scratch wide enough for fourteen N16 tiles.  The
// final two N16 tiles are reused through the first 16x32 entries after the
// main phase has merged, so the logical output/workspace width stays 256.
constexpr int kPvMainD = 224;
constexpr int kPvTailD = kD - kPvMainD;
constexpr int kPvMainTiles = kPvMainD / 16;
constexpr int kPvTailTiles = kPvTailD / 16;
constexpr int kTmpBytes = kRowsPerGroup * kPvMainD *
                          static_cast<int>(sizeof(float));
constexpr int kKvSharedOffset = kAccBytes;
constexpr int kQSharedOffset = kKvSharedOffset + kKvSharedBytes;
constexpr int kTmpSharedOffset = kQSharedOffset + kQBytes;
constexpr int kFp8LutEntries = 256;
constexpr int kFp8LutBytes = kFp8LutEntries * static_cast<int>(sizeof(uint16_t));
constexpr int kFp8LutSharedOffset = kTmpSharedOffset + kTmpBytes;
constexpr int kSharedBytes = kFp8LutSharedOffset + kFp8LutBytes;

static_assert(kGroup == 6, "V7 geometry requires GQA group size six");
static_assert(kBlockSize % kTile == 0, "V7 page must contain whole tiles");
static_assert(kThreads == 128, "V7 geometry requires four warps");
static_assert(sizeof(uint4) == kRawChunkBytes,
              "E6 raw K/V vector chunk must be 16 bytes");
static_assert(alignof(uint4) == kRawChunkBytes,
              "E6 raw K/V vector chunk must be 16-byte aligned");
static_assert(kD % kRawChunkBytes == 0,
              "E6 head dimension must contain whole vector chunks");
static_assert(kRawChunksPerMatrix == 512,
              "E6 one raw K/V matrix must contain 512 vector chunks");
static_assert(kRawChunksPerThread == 4,
              "E6 each thread must load four chunks per K/V matrix");
static_assert(kWmmaLd == 264, "E6 WMMA rows must use leading dimension 264");
static_assert(kRowsPerGroup == 16, "E6 WMMA tiles require sixteen rows");
static_assert(kRowGroups == 3, "E6 WMMA layout requires three row groups");
static_assert(kPvMainD == 224, "E6 PV main phase must cover D=224");
static_assert(kPvTailD == 32, "E6 PV tail phase must cover D=32");
static_assert(kPvMainTiles == 14, "E6 PV main phase requires fourteen tiles");
static_assert(kPvTailTiles == 2, "E6 PV tail phase requires two tiles");
static_assert(kKvSharedBytes == 33792, "E6 K/V tile must be 33,792 bytes");
static_assert(kAccBytes == 24576, "E6 accumulator must be 24,576 bytes");
static_assert(kQBytes == 8448, "E6 Q/P buffer must be 8,448 bytes");
static_assert(kRawStageBytes == 8192,
              "E6 compact raw staging buffer must be 8,192 bytes");
static_assert(kRawStageBytes <= kQBytes,
              "E6 raw staging must fit within the Q/P shared buffer");
static_assert(kQSharedOffset + kRawStageBytes <= kTmpSharedOffset,
              "E6 raw staging must not overlap temporary storage");
static_assert(kQSharedOffset % kRawChunkBytes == 0,
              "E6 raw staging base must be 16-byte aligned");
static_assert(kQSharedOffset >= kKvSharedOffset + kKvSharedBytes,
              "E6 raw staging must not overlap decoded K/V output");
static_assert(kTmpSharedOffset == 66816,
              "E6 temporary tile offset must be 66,816 bytes");
static_assert(kTmpBytes == 14336, "E6 temporary tile must be 14,336 bytes");
static_assert(kFp8LutEntries == 256, "E6 LUT must have 256 entries");
static_assert(kFp8LutBytes == 512, "E6 shared LUT must be 512 bytes");
static_assert(kFp8LutSharedOffset == 81152,
              "E6 shared LUT offset must be 81,152 bytes");
static_assert(kFp8LutSharedOffset >= kTmpSharedOffset + kTmpBytes,
              "E6 shared LUT must not overlap temporary storage");
static_assert(kSharedBytes == 81664, "E6 shared layout must be 81,664 bytes");
static_assert(kSharedBytes <= 98304,
              "E6 shared layout must fit SM80 per-CTA dynamic shared limit");

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ float fp16_bits_to_float(uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return __half2float(raw);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ uint16_t float_to_fp16_bits(float value) {
  return __half_as_ushort(__float2half_rn(value));
}

// Decode an E4M3FN byte without a table or floating-point instructions.  For
// finite normal values, (8+m) * 2^(e-10) maps directly to BF16's exponent and
// seven-bit fraction.  E4M3FN subnormals are still normal in BF16; normalize
// the three-bit mantissa with its highest set bit.  PyTorch uses one canonical
// quiet BF16 NaN for both signed E4M3FN NaN encodings, so follow that explicit
// policy rather than attempting to preserve a NaN sign/payload.
__device__ __forceinline__ uint16_t fp8_e4m3fn_to_bf16_bits(uint8_t code) {
  const uint16_t sign = static_cast<uint16_t>(code & 0x80u) << 8;
  const int exponent = static_cast<int>((code >> 3) & 0xFu);
  const int mantissa = static_cast<int>(code & 0x7u);
  if (exponent == 0xF && mantissa == 0x7) {
    return 0x7FC0u;
  }
  if (exponent == 0) {
    if (mantissa == 0) {
      return sign;
    }
    const int top = 31 - __clz(static_cast<unsigned int>(mantissa));
    const int bf16_exponent = top - 9 + 127;
    const int bf16_fraction =
        (mantissa - (1 << top)) << (7 - top);
    return sign | static_cast<uint16_t>(bf16_exponent << 7) |
           static_cast<uint16_t>(bf16_fraction);
  }
  const int bf16_exponent = exponent - 7 + 127;
  const int bf16_fraction = mantissa << 4;
  return sign | static_cast<uint16_t>(bf16_exponent << 7) |
         static_cast<uint16_t>(bf16_fraction);
}

template <bool IsI64>
__device__ __forceinline__ int64_t load_index(const void* ptr, int64_t index) {
  if constexpr (IsI64) {
    return reinterpret_cast<const int64_t*>(ptr)[index];
  } else {
    return static_cast<int64_t>(reinterpret_cast<const int32_t*>(ptr)[index]);
  }
}

template <bool BlockI64>
__device__ __forceinline__ int64_t load_block_id(
    const void* block_table, int64_t index) {
  return load_index<BlockI64>(block_table, index);
}

// Stage one raw FP8 K or V tile in the compact shared_q alias.  K and V are
// physical [32, 264] token-major BF16 matrices, but their raw source rows are
// only 256 bytes wide.  This stage writes one logical [32,256] uint8 matrix;
// the alias is not reused for Q until load_kv_bf16's final decode barrier.
__device__ __forceinline__ void stage_raw_kv(
    unsigned char* raw_stage,
    const unsigned char* cache,
    int64_t tile_base,
    int64_t stride_s,
    int64_t tile_token,
    int64_t kv_len) {
  const int tid = threadIdx.x;
  constexpr int kChunksPerMatrix = kD / kRawChunkBytes;
  for (int chunk = tid; chunk < kRawChunksPerMatrix; chunk += blockDim.x) {
    const int token_in_tile = chunk / kChunksPerMatrix;
    const int d0 = (chunk % kChunksPerMatrix) * kRawChunkBytes;
    const int raw_offset = token_in_tile * kD + d0;
    unsigned char* dst = raw_stage + raw_offset;
    const int64_t token = tile_token + token_in_tile;
    if (token < 0 || token >= kv_len) {
      const uint4 zero = {0, 0, 0, 0};
      *reinterpret_cast<uint4*>(dst) = zero;
      continue;
    }

    const int64_t raw_base =
        tile_base + static_cast<int64_t>(token_in_tile) * stride_s + d0;
    const uintptr_t raw_address = reinterpret_cast<uintptr_t>(cache) +
                                  static_cast<uintptr_t>(raw_base);
    // The chunk mapping gives d0 in {0,...,240}; keep the bound explicit so a
    // future geometry change cannot turn the vector load into an over-read.
    const bool row_in_bounds = d0 >= 0 && d0 + kRawChunkBytes <= kD;
    if (!row_in_bounds) {
      const uint4 zero = {0, 0, 0, 0};
      *reinterpret_cast<uint4*>(dst) = zero;
      continue;
    }
    const bool aligned = (raw_address & (kRawChunkBytes - 1)) == 0;
    if (aligned) {
      // Both source and destination are 16-byte aligned for the normal
      // contiguous NHD path.  No register-side byte unpack occurs here.
      *reinterpret_cast<uint4*>(dst) =
          *reinterpret_cast<const uint4*>(cache + raw_base);
    } else {
      // Preserve correctness for an external unaligned base without issuing
      // an unaligned uint4 load.  The normal allocator/cache geometry takes
      // the vector path above.
      #pragma unroll
      for (int i = 0; i < kRawChunkBytes; ++i) {
        dst[i] = cache[raw_base + i];
      }
    }
  }
}

__device__ __forceinline__ void decode_raw_kv(
    uint16_t* shared_kv,
    const unsigned char* raw_stage,
    bool is_v) {
  const int tid = threadIdx.x;
  for (int element = tid; element < kKvElementsPerTile;
       element += blockDim.x) {
    const int token_in_tile = element / kD;
    const int d = element % kD;
    const int physical = token_in_tile * kWmmaLd + d;
    shared_kv[(is_v ? kKvMatrixElements : 0) + physical] =
        fp8_e4m3fn_to_bf16_bits(raw_stage[element]);
  }
}

// Decode one raw FP8 tile through compact shared staging.  The four barriers
// make each alias lifetime explicit: stage K, decode K, stage V, decode V.
// The caller may load Q only after this function returns.
__device__ __forceinline__ void load_kv_bf16(
    uint16_t* shared_kv,
    unsigned char* raw_stage,
    const unsigned char* k_cache,
    const unsigned char* v_cache,
    int64_t tile_token,
    int64_t block_size,
    int64_t kvh,
    int64_t block_id,
    int64_t k_stride_b,
    int64_t k_stride_s,
    int64_t k_stride_h,
    int64_t v_stride_b,
    int64_t v_stride_s,
    int64_t v_stride_h,
    int64_t kv_len) {
  const int64_t tile_slot = tile_token % block_size;
  const int64_t k_tile_base =
      block_id * k_stride_b + tile_slot * k_stride_s + kvh * k_stride_h;
  const int64_t v_tile_base =
      block_id * v_stride_b + tile_slot * v_stride_s + kvh * v_stride_h;

  stage_raw_kv(raw_stage, k_cache, k_tile_base, k_stride_s, tile_token, kv_len);
  __syncthreads();
  decode_raw_kv(shared_kv, raw_stage, false);
  __syncthreads();

  stage_raw_kv(raw_stage, v_cache, v_tile_base, v_stride_s, tile_token, kv_len);
  __syncthreads();
  decode_raw_kv(shared_kv, raw_stage, true);
  __syncthreads();
}

template <bool QIsBF16, bool IndexI64, bool BlockI64>
__global__ void v7_partial_kernel(
    const uint16_t* q,
    const unsigned char* k_cache,
    const unsigned char* v_cache,
    const void* block_table,
    const void* seqused_k,
    const void* cu_q,
    float* part_o,
    float* part_m,
    float* part_l,
    const uint16_t* fp8_lut,
    float static_k_scale,
    float static_v_scale,
    float scale,
    int64_t stride_qt,
    int64_t stride_qh,
    int64_t stride_kb,
    int64_t stride_ks,
    int64_t stride_kh,
    int64_t stride_vb,
    int64_t stride_vs,
    int64_t stride_vh,
    int64_t stride_bt) {
  extern __shared__ unsigned char shared[];
  uint16_t* acc_shared = reinterpret_cast<uint16_t*>(shared);
  uint16_t* kv_shared =
      reinterpret_cast<uint16_t*>(shared + kKvSharedOffset);
  uint16_t* q_shared =
      reinterpret_cast<uint16_t*>(shared + kQSharedOffset);
  // Before Q is loaded, the compact first 8,192 B of q_shared is aliased as
  // raw_stage for one K or V matrix at a time.  It never overlaps KV output,
  // the Q/P tail, or the shared LUT; load_kv_bf16's final barrier ends its
  // lifetime before any Q store begins.
  unsigned char* raw_stage = reinterpret_cast<unsigned char*>(q_shared);
  float* tmp_shared = reinterpret_cast<float*>(shared + kTmpSharedOffset);
  uint16_t* fp8_lut_shared = reinterpret_cast<uint16_t*>(
      shared + kFp8LutSharedOffset);
  // The P matrix occupies the first 16x32 BF16 entries.  The remaining
  // buffer space carries alpha as exact FP32 values until the PV fusion.
  float* alpha_shared =
      reinterpret_cast<float*>(q_shared + kRowsPerGroup * kTile);
  __nv_bfloat16* q_bf16 = reinterpret_cast<__nv_bfloat16*>(q_shared);
  __nv_bfloat16* k_bf16 = reinterpret_cast<__nv_bfloat16*>(kv_shared);
  __nv_bfloat16* v_bf16 =
      reinterpret_cast<__nv_bfloat16*>(kv_shared + kKvMatrixElements);

  const int req = static_cast<int>(blockIdx.x);
  const int kvh = static_cast<int>(blockIdx.y);
  const int seg = static_cast<int>(blockIdx.z);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Retain E5's 512-B shared LUT allocation and load it once per CTA so the
  // shared-memory geometry stays comparable.  E6's hot decode path is pure
  // bit arithmetic and deliberately does not read this table; the existing
  // accumulator-init barrier still publishes the copied entries.
  const int lut_index = tid * 2;
  fp8_lut_shared[lut_index] = fp8_lut[lut_index];
  fp8_lut_shared[lut_index + 1] = fp8_lut[lut_index + 1];

  const int64_t q_start = load_index<IndexI64>(cu_q, req);
  const int64_t q_len = load_index<IndexI64>(cu_q, req + 1) - q_start;
  const int64_t kv_len = load_index<IndexI64>(seqused_k, req);
  const int64_t tiles_total = (kv_len + kTile - 1) / kTile;
  const int64_t tiles_per_seg = (tiles_total + kNseg - 1) / kNseg;
  const int64_t t0 = static_cast<int64_t>(seg) * tiles_per_seg;
  const int64_t t1 = min(t0 + tiles_per_seg, tiles_total);

  // Persistent FP16 running value state is shared by all three row groups.
  // Warp 0 lanes 0..15 each own the three m/l states for one local row.
  float m_state[kRowGroups];
  float l_state[kRowGroups];
  #pragma unroll
  for (int group = 0; group < kRowGroups; ++group) {
    m_state[group] = -CUDART_INF_F;
    l_state[group] = 0.0f;
  }
  for (int idx = tid; idx < kRows * kD; idx += blockDim.x) {
    acc_shared[idx] = float_to_fp16_bits(0.0f);
  }
  __syncthreads();

  // Empty segments are common when context is short relative to NSEG.  Write
  // a canonical neutral partial so stale workspace values cannot leak into a
  // later standalone combine call.
  if (t0 >= t1 || q_len <= 0) {
    for (int idx = tid; idx < kRows * kD; idx += blockDim.x) {
      const int row = idx / kD;
      const int d = idx % kD;
      const int qi = row / kGroup;
      const int group = row % kGroup;
      const int h = kvh * kGroup + group;
      const int64_t pi =
          (static_cast<int64_t>(req) * kHq + h) * kQmax * kNseg +
          static_cast<int64_t>(qi) * kNseg + seg;
      (void)d;
      part_o[pi * kD + d] = 0.0f;
    }
    if (warp == 0 && lane < kRowsPerGroup) {
      #pragma unroll
      for (int group = 0; group < kRowGroups; ++group) {
        const int row = group * kRowsPerGroup + lane;
        const int qi = row / kGroup;
        const int qgroup = row % kGroup;
        const int h = kvh * kGroup + qgroup;
        const int64_t pi =
            (static_cast<int64_t>(req) * kHq + h) * kQmax * kNseg +
            static_cast<int64_t>(qi) * kNseg + seg;
        part_m[pi] = -CUDART_INF_F;
        part_l[pi] = 0.0f;
      }
    }
    return;
  }

  __shared__ int64_t block_id_shared;
  for (int64_t tile = t0; tile < t1; ++tile) {
    if (tid == 0) {
      const int64_t page = (tile * kTile) / kBlockSize;
      block_id_shared = load_block_id<BlockI64>(
          block_table, static_cast<int64_t>(req) * stride_bt + page);
    }
    __syncthreads();
    load_kv_bf16(
        kv_shared, raw_stage, k_cache, v_cache, tile * kTile, kBlockSize, kvh,
        block_id_shared, stride_kb, stride_ks, stride_kh, stride_vb, stride_vs,
        stride_vh, kv_len);

    // The Q/P buffer is reused for each 16-row pack.  Loading Q once per pack
    // per tile is required because P occupies its first 32 columns.  This
    // begins only after load_kv_bf16's final V-decode barrier, when raw_stage
    // is no longer live.
    #pragma unroll
    for (int row_group = 0; row_group < kRowGroups; ++row_group) {
      for (int idx = tid; idx < kRowsPerGroup * kD; idx += blockDim.x) {
        const int local_row = idx / kD;
        const int d = idx % kD;
        const int row = row_group * kRowsPerGroup + local_row;
        const int qi = row / kGroup;
        const int qgroup = row % kGroup;
        float q_value = 0.0f;
        if (qi < q_len) {
          const int64_t q_offset =
              (q_start + qi) * stride_qt +
              static_cast<int64_t>(kvh * kGroup + qgroup) * stride_qh + d;
          q_value = QIsBF16 ? bf16_bits_to_float(q[q_offset])
                            : fp16_bits_to_float(q[q_offset]);
        }
        q_shared[local_row * kWmmaLd + d] =
            float_to_bf16_bits(q_value * scale);
      }
      __syncthreads();

      // QK: warps 0 and 1 each produce one 16-token N tile.  K is naturally
      // token-major logical [32,256] in physical rows of ld=264; WMMA reads
      // it as logical col-major [256,16] with ld=264.
      if (warp < 2) {
        const int n_base = warp * 16;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::col_major>
            b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            c_frag;
        nvcuda::wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for (int k0 = 0; k0 < kD; k0 += 16) {
          nvcuda::wmma::load_matrix_sync(a_frag, q_bf16 + k0, kWmmaLd);
          nvcuda::wmma::load_matrix_sync(
              b_frag, k_bf16 + n_base * kWmmaLd + k0, kWmmaLd);
          nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        nvcuda::wmma::store_matrix_sync(
            tmp_shared + n_base, c_frag, kTile, nvcuda::wmma::mem_row_major);
      }
      __syncthreads();

      // Warp 0 lanes 0..15 each own one row and evaluate all 32 scores.  P
      // is rounded to BF16 in the Q/P buffer; alpha remains FP32 in its tail.
      if (warp == 0 && lane < kRowsPerGroup) {
        const int local_row = lane;
        const int row = row_group * kRowsPerGroup + local_row;
        const int qi = row / kGroup;
        const bool row_ok = qi < q_len;
        const int64_t q_pos = kv_len - q_len + qi;
        float old_m = m_state[row_group];
        float old_l = l_state[row_group];
        float tile_max = -CUDART_INF_F;
        #pragma unroll
        for (int token_in_tile = 0; token_in_tile < kTile;
             ++token_in_tile) {
          const int64_t token = tile * kTile + token_in_tile;
          const bool valid = row_ok && token < kv_len && token <= q_pos;
          const float score = valid
              ? tmp_shared[local_row * kTile + token_in_tile] * static_k_scale
              : -CUDART_INF_F;
          tile_max = fmaxf(tile_max, score);
        }
        const float new_m = fmaxf(old_m, tile_max);
        const float max_base = isinf(new_m) ? 0.0f : new_m;
        const float alpha = isinf(old_m) ? 0.0f : __expf(old_m - max_base);
        float p_sum = 0.0f;
        #pragma unroll
        for (int token_in_tile = 0; token_in_tile < kTile;
             ++token_in_tile) {
          const int64_t token = tile * kTile + token_in_tile;
          const bool valid = row_ok && token < kv_len && token <= q_pos;
          const float score = valid
              ? tmp_shared[local_row * kTile + token_in_tile] * static_k_scale
              : -CUDART_INF_F;
          const float p = valid ? __expf(score - max_base) : 0.0f;
          p_sum += p;
          // Q occupied this buffer as a physical [16,264] matrix for QK.  Q
          // is dead after that WMMA pass, so repack P densely as [16,32] for
          // the PV operand.  Using the old 256-column stride here leaves
          // every row except row zero outside the matrix consumed by WMMA.
          q_shared[local_row * kTile + token_in_tile] =
              float_to_bf16_bits(p * static_v_scale);
        }
        const float new_l = old_l * alpha + p_sum;
        m_state[row_group] = new_m;
        l_state[row_group] = new_l;
        alpha_shared[local_row] = alpha;
      }
      __syncthreads();

      // PV main phase: cover d=0..223 with fourteen N16 tiles.  Warps 0..2
      // own four tiles each and warp 3 owns two.  A is P [16,32] row-major;
      // B is the decoded V logical [32,256] row-major tile with physical
      // row stride ld=264.  P remains dense [16,32] with ld=32.
      {
        const int d_base = warp * 4 * 16;
        #pragma unroll
        for (int output_tile = 0; output_tile < 4; ++output_tile) {
          // Warp 3 has only two valid main-phase tiles (d=192,208).
          if (warp < 3 || output_tile < 2) {
            const int d0 = d_base + output_tile * 16;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                   __nv_bfloat16, nvcuda::wmma::row_major>
                a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                   __nv_bfloat16, nvcuda::wmma::row_major>
                b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                                   float>
                c_frag;
            nvcuda::wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for (int k0 = 0; k0 < 32; k0 += 16) {
              nvcuda::wmma::load_matrix_sync(a_frag, q_bf16 + k0, kTile);
              nvcuda::wmma::load_matrix_sync(
                  b_frag, v_bf16 + k0 * kWmmaLd + d0, kWmmaLd);
              nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            nvcuda::wmma::store_matrix_sync(
                tmp_shared + d0, c_frag, kPvMainD,
                nvcuda::wmma::mem_row_major);
          }
        }
      }
      __syncthreads();

      // Merge the main phase's FP32 PV result into d=0..223.  Keeping alpha
      // in FP32 avoids introducing an extra rounding step in the online
      // softmax state while preserving E1's FP16 accumulator ABI.
      for (int idx = tid; idx < kRowsPerGroup * kPvMainD; idx += blockDim.x) {
        const int local_row = idx / kPvMainD;
        const int d = idx % kPvMainD;
        const int row = row_group * kRowsPerGroup + local_row;
        const float alpha = alpha_shared[local_row];
        const float previous = fp16_bits_to_float(acc_shared[row * kD + d]);
        acc_shared[row * kD + d] =
            float_to_fp16_bits(previous * alpha + tmp_shared[idx]);
      }
      __syncthreads();

      // PV tail phase: warp 3 computes d=224,240 only.  The main scratch has
      // already been merged, so reuse its first 16x32 entries as a dense tail
      // tile with ld=32.
      if (warp == 3) {
        #pragma unroll
        for (int output_tile = 0; output_tile < kPvTailTiles; ++output_tile) {
          const int d0 = kPvMainD + output_tile * 16;
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                 __nv_bfloat16, nvcuda::wmma::row_major>
              a_frag;
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                 __nv_bfloat16, nvcuda::wmma::row_major>
              b_frag;
          nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
              c_frag;
          nvcuda::wmma::fill_fragment(c_frag, 0.0f);
          #pragma unroll
          for (int k0 = 0; k0 < 32; k0 += 16) {
            nvcuda::wmma::load_matrix_sync(a_frag, q_bf16 + k0, kTile);
            nvcuda::wmma::load_matrix_sync(
                b_frag, v_bf16 + k0 * kWmmaLd + d0, kWmmaLd);
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
          }
          nvcuda::wmma::store_matrix_sync(
              tmp_shared + (d0 - kPvMainD), c_frag, kTile,
              nvcuda::wmma::mem_row_major);
        }
      }
      __syncthreads();

      // Merge the dense tail [16,32] into d=224..255 across the whole CTA.
      for (int idx = tid; idx < kRowsPerGroup * kPvTailD; idx += blockDim.x) {
        const int local_row = idx / kPvTailD;
        const int d = kPvMainD + idx % kPvTailD;
        const int row = row_group * kRowsPerGroup + local_row;
        const float alpha = alpha_shared[local_row];
        const float previous = fp16_bits_to_float(acc_shared[row * kD + d]);
        acc_shared[row * kD + d] =
            float_to_fp16_bits(previous * alpha + tmp_shared[idx]);
      }
      __syncthreads();
    }
  }

  // Publish the established FP32 workspace ABI once per segment; no part_o/m/l
  // traffic occurs inside the tile loop.
  for (int idx = tid; idx < kRows * kD; idx += blockDim.x) {
    const int row = idx / kD;
    const int d = idx % kD;
    const int qi = row / kGroup;
    const int group = row % kGroup;
    const int h = kvh * kGroup + group;
    const int64_t pi =
        (static_cast<int64_t>(req) * kHq + h) * kQmax * kNseg +
        static_cast<int64_t>(qi) * kNseg + seg;
    part_o[pi * kD + d] = fp16_bits_to_float(acc_shared[idx]);
  }
  if (warp == 0 && lane < kRowsPerGroup) {
    #pragma unroll
    for (int row_group = 0; row_group < kRowGroups; ++row_group) {
      const int row = row_group * kRowsPerGroup + lane;
      const int qi = row / kGroup;
      const int group = row % kGroup;
      const int h = kvh * kGroup + group;
      const int64_t pi =
          (static_cast<int64_t>(req) * kHq + h) * kQmax * kNseg +
          static_cast<int64_t>(qi) * kNseg + seg;
      part_m[pi] = m_state[row_group];
      part_l[pi] = l_state[row_group];
    }
  }
}

template <bool QIsBF16, bool IndexI64>
__global__ void v7_combine_kernel(
    const float* part_o,
    const float* part_m,
    const float* part_l,
    uint16_t* out,
    const void* cu_q,
    int64_t stride_ot,
    int64_t stride_oh) {
  const int req = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int qi = static_cast<int>(blockIdx.z);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t q_start = load_index<IndexI64>(cu_q, req);
  const int64_t q_len = load_index<IndexI64>(cu_q, req + 1) - q_start;
  if (qi >= q_len) {
    return;
  }
  // One warp covers all 256 output columns (eight columns per lane).  The
  // partial kernel above intentionally uses all four warps; combine is kept
  // single-warp to avoid four identical stores to the same output row.
  if (warp != 0) {
    return;
  }
  const int64_t base =
      (static_cast<int64_t>(req) * kHq + h) * kQmax * kNseg +
      static_cast<int64_t>(qi) * kNseg;
  float m_max = -CUDART_INF_F;
  for (int seg = 0; seg < kNseg; ++seg) {
    m_max = fmaxf(m_max, part_m[base + seg]);
  }
  const float max_base = isinf(m_max) ? 0.0f : m_max;
  float l_total = 0.0f;
  for (int seg = 0; seg < kNseg; ++seg) {
    const float m = part_m[base + seg];
    const float weight = isinf(m) ? 0.0f : __expf(m - max_base);
    l_total += part_l[base + seg] * weight;
  }
  const float inv_l = 1.0f / fmaxf(l_total, 1.0e-30f);
  for (int d = lane; d < kD; d += 32) {
    float value = 0.0f;
    for (int seg = 0; seg < kNseg; ++seg) {
      const float m = part_m[base + seg];
      const float weight = isinf(m) ? 0.0f : __expf(m - max_base);
      value += part_o[(base + seg) * kD + d] * weight;
    }
    const float result = value * inv_l;
    const int64_t out_offset =
        static_cast<int64_t>(q_start + qi) * stride_ot +
        static_cast<int64_t>(h) * stride_oh + d;
    out[out_offset] = QIsBF16 ? float_to_bf16_bits(result)
                              : float_to_fp16_bits(result);
  }
}

__global__ void v7_decode_e4m3fn_kernel(
    const uint8_t* codes, uint16_t* out, int64_t count) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[index] = fp8_e4m3fn_to_bf16_bits(codes[index]);
  }
}

void check_common(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_sm80(const torch::Tensor& tensor) {
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 8 && props->minor == 0,
              "V7 prototype is fixed to SM80; got compute capability ",
              props->major, ".", props->minor);
  TORCH_CHECK(tensor.get_device() == c10::cuda::current_device(),
              "all V7 tensors must be on the current CUDA device");
}

torch::Tensor decode_e4m3fn_bf16(const torch::Tensor& codes) {
  check_common(codes, "codes");
  check_sm80(codes);
  TORCH_CHECK(codes.scalar_type() == at::kByte,
              "codes must be uint8 E4M3FN encodings");
  TORCH_CHECK(codes.dim() == 1 && codes.is_contiguous(),
              "codes must be a contiguous 1-D tensor");
  TORCH_CHECK(codes.numel() <= std::numeric_limits<int32_t>::max(),
              "codes tensor is too large for the decoder helper");
  auto out = at::empty(codes.sizes(), codes.options().dtype(at::kBFloat16));
  const int64_t count = codes.numel();
  if (count != 0) {
    const int64_t blocks64 = (count + kThreads - 1) / kThreads;
    TORCH_CHECK(blocks64 <= std::numeric_limits<unsigned int>::max(),
                "codes tensor requires too many decoder blocks");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    v7_decode_e4m3fn_kernel<<<static_cast<unsigned int>(blocks64), kThreads,
                              0, stream>>>(
        codes.data_ptr<uint8_t>(), reinterpret_cast<uint16_t*>(out.data_ptr()),
        count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

void check_partial_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& seqused_k,
    const torch::Tensor& cu_q,
    const torch::Tensor& part_o,
    const torch::Tensor& part_m,
    const torch::Tensor& part_l,
    const torch::Tensor& lut) {
  check_common(q, "q");
  check_common(k_cache, "k_cache");
  check_common(v_cache, "v_cache");
  check_common(block_table, "block_table");
  check_common(seqused_k, "seqused_k");
  check_common(cu_q, "cu_q");
  check_common(part_o, "part_o");
  check_common(part_m, "part_m");
  check_common(part_l, "part_l");
  check_common(lut, "fp8_lut");
  check_sm80(q);
  TORCH_CHECK(k_cache.device() == q.device() && v_cache.device() == q.device() &&
                  block_table.device() == q.device() &&
                  seqused_k.device() == q.device() && cu_q.device() == q.device() &&
                  part_o.device() == q.device() && part_m.device() == q.device() &&
                  part_l.device() == q.device() && lut.device() == q.device(),
              "all V7 partial tensors must share q's CUDA device");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 ||
                  q.scalar_type() == at::kHalf,
              "q must be BF16 or FP16");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHq && q.size(2) == kD,
              "q must have shape [tokens, 24, 256]");
  TORCH_CHECK(q.stride(2) == 1,
              "q must have contiguous D dimension");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte &&
                  v_cache.scalar_type() == at::kByte,
              "k_cache/v_cache must be raw uint8 E4M3 bytes");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 &&
                  k_cache.size(1) == kBlockSize && k_cache.size(2) == kHkv &&
                  k_cache.size(3) == kD &&
                  v_cache.size(0) == k_cache.size(0) &&
                  v_cache.size(1) == k_cache.size(1) &&
                  v_cache.size(2) == k_cache.size(2) &&
                  v_cache.size(3) == k_cache.size(3),
              "caches must have shape [blocks, 896, 4, 256]");
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(),
              "V7 prototype requires contiguous NHD cache tensors");
  TORCH_CHECK(block_table.dim() == 2 &&
                  (block_table.scalar_type() == at::kInt ||
                   block_table.scalar_type() == at::kLong),
              "block_table must be int32 or int64 [requests, pages]");
  TORCH_CHECK(cu_q.dim() == 1 && cu_q.numel() >= 1 &&
                  seqused_k.dim() == 1,
              "cu_q must be [requests + 1] and seqused_k must be [requests]");
  TORCH_CHECK((seqused_k.scalar_type() == at::kInt ||
               seqused_k.scalar_type() == at::kLong) &&
                  (cu_q.scalar_type() == at::kInt ||
                   cu_q.scalar_type() == at::kLong),
              "seqused_k/cu_q must be int32 or int64");
  TORCH_CHECK(seqused_k.scalar_type() == cu_q.scalar_type(),
              "seqused_k and cu_q must use the same integer dtype");
  TORCH_CHECK(part_o.scalar_type() == at::kFloat &&
                  part_m.scalar_type() == at::kFloat &&
                  part_l.scalar_type() == at::kFloat && lut.scalar_type() ==
                      at::kBFloat16,
              "part_o/m/l must be FP32 and fp8_lut must be BF16");
  TORCH_CHECK(part_o.is_contiguous() && part_m.is_contiguous() &&
                  part_l.is_contiguous() && lut.is_contiguous() &&
                  lut.numel() == 256,
              "workspace/LUT must be contiguous; LUT must contain 256 entries");
  TORCH_CHECK(part_o.dim() == 2 && part_o.size(1) == kD &&
                  part_m.dim() == 1 && part_l.dim() == 1,
              "part_o must be [workspace_rows, 256], part_m/l must be 1-D");
}

template <bool QIsBF16, bool IndexI64, bool BlockI64>
void launch_partial(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& seqused_k,
    const torch::Tensor& cu_q,
    const torch::Tensor& part_o,
    const torch::Tensor& part_m,
    const torch::Tensor& part_l,
    const torch::Tensor& lut,
    float static_k_scale,
    float static_v_scale,
    float scale) {
  const int64_t num_reqs = cu_q.size(0) - 1;
  TORCH_CHECK(num_reqs >= 0 && num_reqs <= 65535,
              "invalid request count");
  TORCH_CHECK(
      part_o.numel() >= num_reqs * kHq * kQmax * kNseg * kD &&
          part_m.numel() >= num_reqs * kHq * kQmax * kNseg &&
          part_l.numel() >= num_reqs * kHq * kQmax * kNseg,
      "part_o/m/l workspace is smaller than the fixed V7 geometry");
  TORCH_CHECK(block_table.size(0) >= num_reqs,
              "block_table has fewer rows than cu_q requests");
  const dim3 grid(static_cast<unsigned int>(num_reqs), kHkv, kNseg);
  const dim3 block(kThreads);
  auto kernel = v7_partial_kernel<QIsBF16, IndexI64, BlockI64>;
  AT_CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  kernel<<<grid, block, kSharedBytes, stream>>>(
      reinterpret_cast<const uint16_t*>(q.data_ptr()),
      k_cache.data_ptr<uint8_t>(), v_cache.data_ptr<uint8_t>(),
      block_table.data_ptr(), seqused_k.data_ptr(), cu_q.data_ptr(),
      part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
      reinterpret_cast<const uint16_t*>(lut.data_ptr()), static_k_scale,
      static_v_scale, scale, q.stride(0), q.stride(1), k_cache.stride(0),
      k_cache.stride(1), k_cache.stride(2), v_cache.stride(0),
      v_cache.stride(1), v_cache.stride(2), block_table.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void partial(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& seqused_k,
    const torch::Tensor& cu_q,
    const torch::Tensor& part_o,
    const torch::Tensor& part_m,
    const torch::Tensor& part_l,
    const torch::Tensor& fp8_lut,
    double static_k_scale,
    double static_v_scale,
    double scale) {
  check_partial_inputs(q, k_cache, v_cache, block_table, seqused_k, cu_q,
                       part_o, part_m, part_l, fp8_lut);
  TORCH_CHECK(q.size(0) <= std::numeric_limits<int32_t>::max(),
              "q token count is too large for this prototype");
  const bool q_bf16 = q.scalar_type() == at::kBFloat16;
  const bool index_i64 = cu_q.scalar_type() == at::kLong;
  const bool block_i64 = block_table.scalar_type() == at::kLong;
  if (q_bf16 && index_i64 && block_i64) {
    launch_partial<true, true, true>(q, k_cache, v_cache, block_table, seqused_k,
                                     cu_q, part_o, part_m, part_l, fp8_lut,
                                     static_cast<float>(static_k_scale),
                                     static_cast<float>(static_v_scale),
                                     static_cast<float>(scale));
  } else if (q_bf16 && index_i64) {
    launch_partial<true, true, false>(q, k_cache, v_cache, block_table, seqused_k,
                                      cu_q, part_o, part_m, part_l, fp8_lut,
                                      static_cast<float>(static_k_scale),
                                      static_cast<float>(static_v_scale),
                                      static_cast<float>(scale));
  } else if (q_bf16 && block_i64) {
    launch_partial<true, false, true>(q, k_cache, v_cache, block_table, seqused_k,
                                      cu_q, part_o, part_m, part_l, fp8_lut,
                                      static_cast<float>(static_k_scale),
                                      static_cast<float>(static_v_scale),
                                      static_cast<float>(scale));
  } else if (q_bf16) {
    launch_partial<true, false, false>(q, k_cache, v_cache, block_table, seqused_k,
                                       cu_q, part_o, part_m, part_l, fp8_lut,
                                       static_cast<float>(static_k_scale),
                                       static_cast<float>(static_v_scale),
                                       static_cast<float>(scale));
  } else if (index_i64 && block_i64) {
    launch_partial<false, true, true>(q, k_cache, v_cache, block_table, seqused_k,
                                      cu_q, part_o, part_m, part_l, fp8_lut,
                                      static_cast<float>(static_k_scale),
                                      static_cast<float>(static_v_scale),
                                      static_cast<float>(scale));
  } else if (index_i64) {
    launch_partial<false, true, false>(q, k_cache, v_cache, block_table, seqused_k,
                                       cu_q, part_o, part_m, part_l, fp8_lut,
                                       static_cast<float>(static_k_scale),
                                       static_cast<float>(static_v_scale),
                                       static_cast<float>(scale));
  } else if (block_i64) {
    launch_partial<false, false, true>(q, k_cache, v_cache, block_table, seqused_k,
                                       cu_q, part_o, part_m, part_l, fp8_lut,
                                       static_cast<float>(static_k_scale),
                                       static_cast<float>(static_v_scale),
                                       static_cast<float>(scale));
  } else {
    launch_partial<false, false, false>(q, k_cache, v_cache, block_table, seqused_k,
                                        cu_q, part_o, part_m, part_l, fp8_lut,
                                        static_cast<float>(static_k_scale),
                                        static_cast<float>(static_v_scale),
                                        static_cast<float>(scale));
  }
}

template <bool QIsBF16, bool IndexI64>
void launch_combine(
    const torch::Tensor& part_o,
    const torch::Tensor& part_m,
    const torch::Tensor& part_l,
    const torch::Tensor& out,
    const torch::Tensor& cu_q) {
  const int64_t num_reqs = cu_q.size(0) - 1;
  const dim3 grid(static_cast<unsigned int>(num_reqs), kHq, kQmax);
  const dim3 block(kThreads);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  v7_combine_kernel<QIsBF16, IndexI64><<<grid, block, 0, stream>>>(
      part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(out.data_ptr()), cu_q.data_ptr(), out.stride(0),
      out.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void combine(
    const torch::Tensor& part_o,
    const torch::Tensor& part_m,
    const torch::Tensor& part_l,
    const torch::Tensor& out,
    const torch::Tensor& cu_q) {
  check_common(part_o, "part_o");
  check_common(part_m, "part_m");
  check_common(part_l, "part_l");
  check_common(out, "out");
  check_common(cu_q, "cu_q");
  check_sm80(out);
  TORCH_CHECK(part_o.device() == out.device() &&
                  part_m.device() == out.device() &&
                  part_l.device() == out.device() &&
                  cu_q.device() == out.device(),
              "all V7 combine tensors must share out's CUDA device");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 ||
                  out.scalar_type() == at::kHalf,
              "out must be BF16 or FP16");
  TORCH_CHECK(out.dim() == 3 && out.size(1) == kHq && out.size(2) == kD,
              "out must have shape [tokens, 24, 256]");
  TORCH_CHECK(out.stride(2) == 1,
              "out must have contiguous D dimension");
  TORCH_CHECK(part_o.scalar_type() == at::kFloat &&
                  part_m.scalar_type() == at::kFloat &&
                  part_l.scalar_type() == at::kFloat,
              "part_o/m/l must be FP32");
  TORCH_CHECK(part_o.is_contiguous() && part_m.is_contiguous() &&
                  part_l.is_contiguous() && cu_q.is_cuda(),
              "workspace must be contiguous CUDA tensors");
  TORCH_CHECK(part_o.dim() == 2 && part_o.size(1) == kD &&
                  part_m.dim() == 1 && part_l.dim() == 1,
              "part_o must be [workspace_rows, 256], part_m/l must be 1-D");
  TORCH_CHECK(cu_q.dim() == 1 && cu_q.numel() >= 1,
              "cu_q must be [requests + 1]");
  const int64_t num_reqs = cu_q.size(0) - 1;
  TORCH_CHECK(num_reqs >= 0 &&
                  part_o.numel() >= num_reqs * kHq * kQmax * kNseg * kD &&
                  part_m.numel() >= num_reqs * kHq * kQmax * kNseg &&
                  part_l.numel() >= num_reqs * kHq * kQmax * kNseg,
              "part_o/m/l workspace is smaller than the fixed V7 geometry");
  TORCH_CHECK(cu_q.scalar_type() == at::kInt ||
                  cu_q.scalar_type() == at::kLong,
              "cu_q must be int32 or int64");
  const bool q_bf16 = out.scalar_type() == at::kBFloat16;
  const bool index_i64 = cu_q.scalar_type() == at::kLong;
  if (q_bf16 && index_i64) {
    launch_combine<true, true>(part_o, part_m, part_l, out, cu_q);
  } else if (q_bf16) {
    launch_combine<true, false>(part_o, part_m, part_l, out, cu_q);
  } else if (index_i64) {
    launch_combine<false, true>(part_o, part_m, part_l, out, cu_q);
  } else {
    launch_combine<false, false>(part_o, part_m, part_l, out, cu_q);
  }
}

py::dict resources() {
  auto kernel = v7_partial_kernel<true, false, false>;
  AT_CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
  cudaFuncAttributes attributes{};
  AT_CUDA_CHECK(cudaFuncGetAttributes(
      &attributes, reinterpret_cast<const void*>(kernel)));
  int active_blocks = 0;
  AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, reinterpret_cast<const void*>(kernel), kThreads,
      kSharedBytes));
  py::dict result;
  result["threads_per_cta"] = kThreads;
  result["dynamic_shared_bytes"] = kSharedBytes;
  result["registers_per_thread"] = attributes.numRegs;
  result["static_shared_bytes"] = attributes.sharedSizeBytes;
  result["local_bytes"] = attributes.localSizeBytes;
  result["max_threads_per_block"] = attributes.maxThreadsPerBlock;
  result["active_ctas_per_sm"] = active_blocks;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partial", &partial,
        "V7 fixed SM80 FP8 split-KV partial (standalone prototype)");
  m.def("combine", &combine,
        "V7 fixed SM80 FP8 split-KV combine (standalone prototype)");
  m.def("resources", &resources,
        "V7 partial kernel attributes and occupancy (standalone prototype)");
  m.def("decode_e4m3fn_bf16", &decode_e4m3fn_bf16,
        "V7 device E4M3FN-to-BF16 bit decoder exhaustive helper");
}
