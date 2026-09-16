// Standalone V7 fixed-geometry verifier prototype.
//
// This file intentionally has no vLLM/FlashInfer dependency and is not wired
// into any production dispatch.  It mirrors the split-KV partial/combine
// workspace contract so that the kernel can be qualified in isolation.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
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
constexpr int kNseg = 35;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kRows = kGroup * kQmax;
constexpr int kKvBytesPerTile = kTile * kD;
constexpr int kKvStageBytes = 2 * kKvBytesPerTile;
constexpr int kQBytes = kRows * kD * static_cast<int>(sizeof(float));
constexpr int kSharedBytes = kQBytes + 2 * kKvStageBytes;

static_assert(kGroup == 6, "V7 geometry requires GQA group size six");
static_assert(kBlockSize % kTile == 0, "V7 page must contain whole tiles");
static_assert(kThreads == 128, "V7 geometry requires four warps");

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

template <bool IsI64>
__device__ __forceinline__ int64_t load_index(const void* ptr, int64_t index) {
  if constexpr (IsI64) {
    return reinterpret_cast<const int64_t*>(ptr)[index];
  } else {
    return static_cast<int64_t>(reinterpret_cast<const int32_t*>(ptr)[index]);
  }
}

// The existing LUT is a 256-entry BF16 tensor.  Keeping the bit conversion
// explicit makes the E4M3FN NaN-to-zero and BF16 rounding semantics visible.
__device__ __forceinline__ float lut_decode(const uint16_t* lut, uint8_t raw) {
  return bf16_bits_to_float(lut[static_cast<int>(raw)]);
}

__device__ __forceinline__ float warp_max(float value) {
  for (int delta = 16; delta > 0; delta >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int delta = 16; delta > 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

// SM80 cp.async is used only for valid 16-byte chunks.  Invalid tail rows are
// written synchronously as zero, which gives the same masked-load semantics as
// the Triton kernel.  The two shared-memory stages are independent, so the
// next tile can be copied while the current tile is being computed.
__device__ __forceinline__ void cp_async_16(
    unsigned char* shared_ptr, const unsigned char* global_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_addr =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(shared_addr), "l"(global_ptr));
#else
  *reinterpret_cast<uint4*>(shared_ptr) =
      *reinterpret_cast<const uint4*>(global_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : :);
#endif
}

__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" : :);
#endif
}

template <bool BlockI64>
__device__ __forceinline__ int64_t load_block_id(
    const void* block_table, int64_t index) {
  return load_index<BlockI64>(block_table, index);
}

__device__ __forceinline__ void load_kv_stage(
    unsigned char* shared_kv,
    const unsigned char* k_cache,
    const unsigned char* v_cache,
    int64_t tile_token,
    int64_t block_size,
    int64_t kvh,
    int stage,
    int64_t block_id,
    int64_t k_stride_b,
    int64_t k_stride_s,
    int64_t k_stride_h,
    int64_t v_stride_b,
    int64_t v_stride_s,
    int64_t v_stride_h,
    int64_t kv_len) {
  const int tid = threadIdx.x;
  const int stage_offset = stage * kKvStageBytes;
  // 1024 x 16-byte chunks cover one K and one V tile.  Eight chunks per
  // thread keep the producer work balanced across all four warps.
  for (int chunk = tid; chunk < 2 * kKvBytesPerTile / 16;
       chunk += blockDim.x) {
    const bool is_v = chunk >= kKvBytesPerTile / 16;
    const int local_chunk = chunk % (kKvBytesPerTile / 16);
    const int row = local_chunk / (kD / 16);
    const int col = (local_chunk % (kD / 16)) * 16;
    const int64_t token = tile_token + row;
    const int64_t slot = token % block_size;
    // Promote the physical block ID before multiplying by the cache stride.
    // This is the same high-block-ID safety rule as the existing verifier.
    const int64_t base = is_v
        ? block_id * v_stride_b + slot * v_stride_s + kvh * v_stride_h + col
        : block_id * k_stride_b + slot * k_stride_s + kvh * k_stride_h + col;
    const unsigned char* global_ptr = is_v ? v_cache + base : k_cache + base;
    unsigned char* shared_ptr =
        shared_kv + stage_offset + (is_v ? kKvBytesPerTile : 0) +
        local_chunk * 16;
    if (token < kv_len) {
      cp_async_16(shared_ptr, global_ptr);
    } else {
      for (int byte = 0; byte < 16; ++byte) {
        shared_ptr[byte] = 0;
      }
    }
  }
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
  float* q_shared = reinterpret_cast<float*>(shared);
  unsigned char* kv_shared = shared + kQBytes;

  const int req = static_cast<int>(blockIdx.x);
  const int kvh = static_cast<int>(blockIdx.y);
  const int seg = static_cast<int>(blockIdx.z);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int64_t q_start = load_index<IndexI64>(cu_q, req);
  const int64_t q_len = load_index<IndexI64>(cu_q, req + 1) - q_start;
  const int64_t kv_len = load_index<IndexI64>(seqused_k, req);
  const int64_t tiles_total = (kv_len + kTile - 1) / kTile;
  const int64_t tiles_per_seg = (tiles_total + kNseg - 1) / kNseg;
  const int64_t t0 = static_cast<int64_t>(seg) * tiles_per_seg;
  const int64_t t1 = min(t0 + tiles_per_seg, tiles_total);

  // Load q into a compact FP32 shared tile.  The actual query tensor may be
  // BF16 or FP16; V7's output dtype follows the input query dtype.
  for (int idx = tid; idx < kRows * kD; idx += blockDim.x) {
    const int row = idx / kD;
    const int d = idx % kD;
    const int qi = row / kGroup;
    const int group = row % kGroup;
    float q_value = 0.0f;
    if (qi < q_len) {
      const int64_t q_offset = (q_start + qi) * stride_qt +
                               static_cast<int64_t>(kvh * kGroup + group) *
                                   stride_qh + d;
      q_value = QIsBF16 ? bf16_bits_to_float(q[q_offset])
                        : fp16_bits_to_float(q[q_offset]);
    }
    // Match the existing q8 path's `(q * scale).to(bfloat16)` before the
    // BF16 dot, including FP16-query callers.
    q_shared[idx] = bf16_bits_to_float(float_to_bf16_bits(q_value * scale));
  }
  __syncthreads();

  const int row_begin = warp * (kRows / kWarps);
  const int row_end = row_begin + (kRows / kWarps);
  // Empty segments are common when context is short relative to NSEG.  Write
  // a canonical neutral partial so stale workspace values cannot leak into a
  // later standalone combine call.
  if (t0 >= t1 || q_len <= 0) {
    for (int row = row_begin; row < row_end; ++row) {
      const int qi = row / kGroup;
      const int group = row % kGroup;
      const int h = kvh * kGroup + group;
      const int64_t pi =
          (static_cast<int64_t>(req) * kHq + h) * kQmax * kNseg +
          static_cast<int64_t>(qi) * kNseg + seg;
      for (int d = lane; d < kD; d += 32) {
        part_o[pi * kD + d] = 0.0f;
      }
      if (lane == 0) {
        part_m[pi] = -CUDART_INF_F;
        part_l[pi] = 0.0f;
      }
    }
    return;
  }

  __shared__ int64_t current_block_shared;
  __shared__ int64_t next_block_id;
  int stage = 0;
  const int64_t first_block_index = (t0 * kTile) / kBlockSize;
  if (tid == 0) {
    current_block_shared = load_block_id<BlockI64>(
        block_table, static_cast<int64_t>(req) * stride_bt + first_block_index);
  }
  __syncthreads();
  int64_t current_block_id = current_block_shared;

  load_kv_stage(
      kv_shared, k_cache, v_cache, t0 * kTile, kBlockSize, kvh, stage,
      current_block_id, stride_kb, stride_ks, stride_kh, stride_vb, stride_vs,
      stride_vh, kv_len);
  cp_async_commit();
  cp_async_wait();
  __syncthreads();

  for (int64_t tile = t0; tile < t1; ++tile) {
    const bool has_next = tile + 1 < t1;
    const int next_stage = stage ^ 1;
    if (has_next) {
      const int64_t next_token = (tile + 1) * kTile;
      const int64_t next_page = next_token / kBlockSize;
      if (tid == 0) {
        next_block_id = load_block_id<BlockI64>(
            block_table, static_cast<int64_t>(req) * stride_bt + next_page);
      }
      __syncthreads();
      load_kv_stage(
          kv_shared, k_cache, v_cache, next_token, kBlockSize, kvh,
          next_stage, next_block_id, stride_kb, stride_ks, stride_kh,
          stride_vb, stride_vs, stride_vh, kv_len);
      cp_async_commit();
    }

    unsigned char* k_shared = kv_shared + stage * kKvStageBytes;
    unsigned char* v_shared = k_shared + kKvBytesPerTile;
    // Correctness scaffold: retain the established ABI by checkpointing the
    // running state through part_o/m/l after each tile.  E1 should move this
    // state on-chip; the extra global traffic is intentional in this version.
    for (int row = row_begin; row < row_end; ++row) {
      const int qi = row / kGroup;
      const bool row_ok = qi < q_len;
      const int64_t q_pos = kv_len - q_len + qi;
      const int group = row % kGroup;
      const int h = kvh * kGroup + group;
      const int64_t pi =
          (static_cast<int64_t>(req) * kHq + h) * kQmax * kNseg +
          static_cast<int64_t>(qi) * kNseg + seg;
      float m = lane == 0
          ? (tile == t0 ? -CUDART_INF_F : part_m[pi])
          : 0.0f;
      float l = lane == 0 ? (tile == t0 ? 0.0f : part_l[pi]) : 0.0f;
      m = __shfl_sync(0xffffffffu, m, 0);
      l = __shfl_sync(0xffffffffu, l, 0);
      float acc[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int d = lane + 32 * j;
        acc[j] = tile == t0 ? 0.0f : part_o[pi * kD + d];
      }

      // The four warps each own 12 query/group rows.  Within a row, every
      // lane owns one score token and eight output dimensions, so all four
      // warps perform useful math on every tile.
      const int token_lane = lane;
      const int64_t token = tile * kTile + token_lane;
      float score = -CUDART_INF_F;
      if (row_ok && token < kv_len && token <= q_pos) {
        float dot = 0.0f;
        for (int d = 0; d < kD; ++d) {
          dot += q_shared[row * kD + d] *
                 lut_decode(fp8_lut, k_shared[token_lane * kD + d]);
        }
        score = dot * static_k_scale;
      }
      const float tile_max = warp_max(score);
      const float m_new = fmaxf(m, tile_max);
      const float max_base = isinf(m_new) ? 0.0f : m_new;
      const float alpha = isinf(m) ? 0.0f : __expf(m - max_base);
      const float p = isinf(score) ? 0.0f : __expf(score - max_base);
      const float p_sum = warp_sum(p);
      l = l * alpha + p_sum;
      for (int j = 0; j < 8; ++j) {
        const int d = lane + 32 * j;
        float value_sum = 0.0f;
        for (int source_lane = 0; source_lane < 32; ++source_lane) {
          const float source_p = __shfl_sync(0xffffffffu, p, source_lane);
          const float weight = bf16_bits_to_float(
              float_to_bf16_bits(source_p * static_v_scale));
          const float value = lut_decode(
              fp8_lut, v_shared[source_lane * kD + d]);
          value_sum += weight * value;
        }
        acc[j] = fp16_bits_to_float(float_to_fp16_bits(
            fp16_bits_to_float(float_to_fp16_bits(acc[j])) * alpha +
            value_sum));
      }
      m = m_new;

      // Each lane writes eight of the 256 output columns.  m/l are identical
      // across the warp after the reductions, so lane zero writes the scalar
      // partial metadata once.
      for (int j = 0; j < 8; ++j) {
        part_o[pi * kD + lane + 32 * j] = acc[j];
      }
      if (lane == 0) {
        part_m[pi] = m;
        part_l[pi] = l;
      }
    }

    if (has_next) {
      cp_async_wait();
      __syncthreads();
      stage = next_stage;
      current_block_id = next_block_id;
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
}
