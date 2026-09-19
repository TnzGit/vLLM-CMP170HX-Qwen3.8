// Standalone E38 INT8-G64 probe using NInfer's qualified SM80 tiled kernel.
// This is an isolated ABI adapter only; it does not modify vLLM dispatch.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "page_stride_kernel.cuh"
#include "paged_kv_address_strided.cuh"

namespace py = pybind11;
using Geometry = ninfer::ops::Gqa27Geometry;

#ifndef E38_KEYBLOCK
#define E38_KEYBLOCK 32
#endif
#ifndef E38_WARPS
#define E38_WARPS 6
#endif
#ifndef E38_MINBLOCKS
#define E38_MINBLOCKS 2
#endif
#ifndef E38_DYNAMIC
#define E38_DYNAMIC 0
#endif

namespace {

void check_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// Padded page-local G64 views keep an HND tail but stride between pages, so
// only their leading stride may differ from the contiguous layout.
void check_g64_cache(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  const auto shape = t.sizes();
  const auto stride = t.strides();
  TORCH_CHECK(stride[3] == 1 && stride[2] == shape[3] &&
                  stride[1] == shape[2] * shape[3],
              name, " must keep an HND tail stride, got ", stride);
}

int64_t g64_code_page_stride(const torch::Tensor& cache_k,
                             const torch::Tensor& cache_v) {
  TORCH_CHECK(cache_k.stride(0) == cache_v.stride(0),
              "K/V code page strides disagree");
  return cache_k.stride(0);
}

// Each plane carries its own page stride: interleaved page-local storage jumps
// the whole physical page, the compact plane-packed ABI uses the scale plane's
// own tight stride. Both are valid; the kernel just needs the right value.
int64_t g64_scale_page_stride(const torch::Tensor& cache_k_scale) {
  return cache_k_scale.stride(0);
}

void partial(const torch::Tensor& q,
             const torch::Tensor& pos,
             const torch::Tensor& cache_k,
             const torch::Tensor& cache_v,
             const torch::Tensor& cache_k_scale,
             const torch::Tensor& cache_v_scale,
             const torch::Tensor& block_table,
             const torch::Tensor& partial_acc,
             const torch::Tensor& partial_m,
             const torch::Tensor& partial_l,
             float scale,
             int split_count,
             int logical_capacity) {
  for (const auto& item : {std::pair<const torch::Tensor&, const char*>{q, "q"},
                           {pos, "pos"},
                           {block_table, "block_table"}, {partial_acc, "partial_acc"},
                           {partial_m, "partial_m"}, {partial_l, "partial_l"}}) {
    check_cuda(item.first, item.second);
  }
  check_g64_cache(cache_k, "cache_k");
  check_g64_cache(cache_v, "cache_v");
  check_g64_cache(cache_k_scale, "cache_k_scale");
  check_g64_cache(cache_v_scale, "cache_v_scale");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.numel() == 8 * 24 * 256,
              "q must be contiguous BF16 [8,24,256]");
  TORCH_CHECK(pos.scalar_type() == torch::kInt32 && pos.numel() == 8,
              "pos must be int32 [8]");
  TORCH_CHECK(cache_k.scalar_type() == torch::kChar && cache_v.scalar_type() == torch::kChar,
              "cache K/V must be int8");
  TORCH_CHECK(cache_k_scale.scalar_type() == torch::kHalf &&
                  cache_v_scale.scalar_type() == torch::kHalf,
              "cache scales must be FP16");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 && block_table.dim() == 1,
              "block_table must be int32 [pages]");
  TORCH_CHECK(split_count > 0 && split_count <= Geometry::DecodeSplits,
              "invalid split_count");
  TORCH_CHECK(partial_acc.scalar_type() == torch::kBFloat16 && partial_acc.is_contiguous(),
              "partial_acc must be contiguous BF16");
  TORCH_CHECK(partial_m.scalar_type() == torch::kFloat && partial_l.scalar_type() == torch::kFloat,
              "partial stats must be FP32");

  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(Geometry::KVHeads, split_count, 1);
  constexpr int Warps = E38_WARPS;
  constexpr int KeyBlock = E38_KEYBLOCK;
  using CacheInput = ninfer::ops::GqaCachedInput;
  constexpr std::size_t DynamicBytes = E38_DYNAMIC ? static_cast<std::size_t>(4 * KeyBlock * ninfer::ops::kGqaHeadDim) : 0u;
  if constexpr (E38_DYNAMIC) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    ninfer::ops::gqa_attention_decode_i8_tiled_kernel<
                        Geometry, 8, Warps, E38_MINBLOCKS, KeyBlock, E38_DYNAMIC,
                        false, false, false, false, false, CacheInput>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(DynamicBytes)) == cudaSuccess,
                "failed to opt in E38 dynamic shared memory");
  }
  ninfer::ops::gqa_attention_decode_i8_tiled_kernel<
      Geometry, 8, Warps, E38_MINBLOCKS, KeyBlock, E38_DYNAMIC, false, false, false, false, false, CacheInput>
      <<<grid, Warps * 32, DynamicBytes, stream>>>(
          static_cast<const __nv_bfloat16*>(q.data_ptr()), CacheInput{},
          static_cast<const int32_t*>(pos.data_ptr()),
          static_cast<int8_t*>(cache_k.data_ptr()),
          reinterpret_cast<uint8_t*>(cache_v.data_ptr()),
          reinterpret_cast<__half*>(cache_k_scale.data_ptr()),
          reinterpret_cast<__half*>(cache_v_scale.data_ptr()),
          static_cast<const int32_t*>(block_table.data_ptr()), nullptr, nullptr,
          static_cast<int32_t>(block_table.numel()), 8, 0, logical_capacity, scale,
          g64_code_page_stride(cache_k, cache_v),
          g64_scale_page_stride(cache_k_scale),
          reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr()),
          static_cast<float*>(partial_m.data_ptr()), static_cast<float*>(partial_l.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce(const torch::Tensor& partial_acc,
            const torch::Tensor& partial_m,
            const torch::Tensor& partial_l,
            const torch::Tensor& pos,
            const torch::Tensor& out,
            int split_count) {
  check_cuda(partial_acc, "partial_acc");
  check_cuda(partial_m, "partial_m");
  check_cuda(partial_l, "partial_l");
  check_cuda(pos, "pos");
  check_cuda(out, "out");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.numel() == 8 * 24 * 256,
              "out must be contiguous BF16 [8,24,256]");
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(Geometry::QHeads, 1, 8);
  ninfer::ops::gqa_attention_small_t_reduce_output_kernel<
      Geometry, 256, true, false, false, false><<<grid, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(partial_acc.data_ptr()),
      static_cast<const float*>(partial_m.data_ptr()),
      static_cast<const float*>(partial_l.data_ptr()),
      static_cast<const int32_t*>(pos.data_ptr()), nullptr, 8, 8, 0, 1, split_count,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void append(const torch::Tensor& q,
            const torch::Tensor& new_k,
            const torch::Tensor& new_v,
            const torch::Tensor& pos,
            const torch::Tensor& cache_k,
            const torch::Tensor& cache_v,
            const torch::Tensor& cache_k_scale,
            const torch::Tensor& cache_v_scale,
            const torch::Tensor& block_table,
            const torch::Tensor& partial_acc,
            const torch::Tensor& partial_m,
            const torch::Tensor& partial_l,
            float scale,
            int split_count,
            int logical_capacity) {
  for (const auto& item : {std::pair<const torch::Tensor&, const char*>{q, "q"},
                           {new_k, "new_k"}, {new_v, "new_v"}, {pos, "pos"},
                           {block_table, "block_table"}, {partial_acc, "partial_acc"},
                           {partial_m, "partial_m"}, {partial_l, "partial_l"}}) {
    check_cuda(item.first, item.second);
  }
  check_g64_cache(cache_k, "cache_k");
  check_g64_cache(cache_v, "cache_v");
  check_g64_cache(cache_k_scale, "cache_k_scale");
  check_g64_cache(cache_v_scale, "cache_v_scale");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.numel() == 8 * 24 * 256,
              "q must be contiguous BF16 [8,24,256]");
  TORCH_CHECK(new_k.scalar_type() == torch::kBFloat16 && new_k.numel() == 8 * 4 * 256 &&
                  new_v.scalar_type() == torch::kBFloat16 && new_v.numel() == 8 * 4 * 256,
              "new K/V must be contiguous BF16 [8,4,256]");
  TORCH_CHECK(pos.scalar_type() == torch::kInt32 && pos.numel() == 8,
              "pos must be int32 [8]");
  TORCH_CHECK(cache_k.scalar_type() == torch::kChar && cache_v.scalar_type() == torch::kChar,
              "cache K/V must be int8");
  TORCH_CHECK(cache_k_scale.scalar_type() == torch::kHalf &&
                  cache_v_scale.scalar_type() == torch::kHalf,
              "cache scales must be FP16");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 && block_table.dim() == 1,
              "block_table must be int32 [pages]");
  TORCH_CHECK(split_count > 0 && split_count <= Geometry::DecodeSplits,
              "invalid split_count");
  TORCH_CHECK(partial_acc.scalar_type() == torch::kBFloat16 && partial_acc.is_contiguous(),
              "partial_acc must be contiguous BF16");
  TORCH_CHECK(partial_m.scalar_type() == torch::kFloat && partial_l.scalar_type() == torch::kFloat,
              "partial stats must be FP32");
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(Geometry::KVHeads, split_count, 1);
  constexpr int Warps = E38_WARPS;
  constexpr int KeyBlock = E38_KEYBLOCK;
  constexpr std::size_t DynamicBytes = E38_DYNAMIC ? static_cast<std::size_t>(4 * KeyBlock * ninfer::ops::kGqaHeadDim) : 0u;
  using CacheInput = ninfer::ops::GqaAppendInput;
  if constexpr (E38_DYNAMIC) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    ninfer::ops::gqa_attention_decode_i8_tiled_kernel<
                        Geometry, 8, Warps, E38_MINBLOCKS, KeyBlock, E38_DYNAMIC,
                        false, false, false, false, false, CacheInput>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(DynamicBytes)) == cudaSuccess,
                "failed to opt in E38 dynamic shared memory");
  }
  ninfer::ops::gqa_attention_decode_i8_tiled_kernel<
      Geometry, 8, Warps, E38_MINBLOCKS, KeyBlock, E38_DYNAMIC, false, false, false, false, false, CacheInput>
      <<<grid, Warps * 32, DynamicBytes, stream>>>(
          static_cast<const __nv_bfloat16*>(q.data_ptr()),
          CacheInput{static_cast<const __nv_bfloat16*>(new_k.data_ptr()),
                     static_cast<const __nv_bfloat16*>(new_v.data_ptr())},
          static_cast<const int32_t*>(pos.data_ptr()),
          static_cast<int8_t*>(cache_k.data_ptr()),
          reinterpret_cast<uint8_t*>(cache_v.data_ptr()),
          reinterpret_cast<__half*>(cache_k_scale.data_ptr()),
          reinterpret_cast<__half*>(cache_v_scale.data_ptr()),
          static_cast<const int32_t*>(block_table.data_ptr()), nullptr, nullptr,
          static_cast<int32_t>(block_table.numel()), 8, 0, logical_capacity, scale,
          g64_code_page_stride(cache_k, cache_v),
          g64_scale_page_stride(cache_k_scale),
          reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr()),
          static_cast<float*>(partial_m.data_ptr()), static_cast<float*>(partial_l.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void partial_batch(const torch::Tensor& q,
                   const torch::Tensor& pos,
                   const torch::Tensor& cache_k,
                   const torch::Tensor& cache_v,
                   const torch::Tensor& cache_k_scale,
                   const torch::Tensor& cache_v_scale,
                   const torch::Tensor& block_tables,
                   const torch::Tensor& valid_columns,
                   const torch::Tensor& table_rows,
                   const torch::Tensor& partial_acc,
                   const torch::Tensor& partial_m,
                   const torch::Tensor& partial_l,
                   float scale,
                   int split_count,
                   int logical_capacity) {
  for (const auto& item : {std::pair<const torch::Tensor&, const char*>{q, "q"},
                           {pos, "pos"},
                           {block_tables, "block_tables"}, {valid_columns, "valid_columns"},
                           {table_rows, "table_rows"}, {partial_acc, "partial_acc"},
                           {partial_m, "partial_m"}, {partial_l, "partial_l"}}) {
    check_cuda(item.first, item.second);
  }
  check_g64_cache(cache_k, "cache_k");
  check_g64_cache(cache_v, "cache_v");
  check_g64_cache(cache_k_scale, "cache_k_scale");
  check_g64_cache(cache_v_scale, "cache_v_scale");
  const int Batch = static_cast<int>(block_tables.size(0));
  constexpr int Tokens = 8;
  constexpr int Warps = E38_WARPS;
  constexpr int KeyBlock = E38_KEYBLOCK;
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.numel() == Batch * Tokens * 24 * 256,
              "q must be contiguous BF16 [batch,8,24,256]");
  TORCH_CHECK(Batch >= 1 && Batch <= 4, "batch must be in [1,4]");
  TORCH_CHECK(pos.scalar_type() == torch::kInt32 && pos.numel() == Batch * Tokens,
              "pos must be int32 [batch,8]");
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt32 && block_tables.dim() == 2 &&
                  block_tables.size(0) == Batch,
              "block_tables must be int32 [batch,pages]");
  TORCH_CHECK(valid_columns.scalar_type() == torch::kInt32 && valid_columns.numel() == Batch &&
                  table_rows.scalar_type() == torch::kInt32 && table_rows.numel() == Batch,
              "valid_columns/table_rows must be int32 [batch]");
  TORCH_CHECK(cache_k.scalar_type() == torch::kChar && cache_v.scalar_type() == torch::kChar,
              "cache K/V must be int8");
  TORCH_CHECK(cache_k_scale.scalar_type() == torch::kHalf &&
                  cache_v_scale.scalar_type() == torch::kHalf,
              "cache scales must be FP16");
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  using CacheInput = ninfer::ops::GqaCachedInput;
  const dim3 grid(Geometry::KVHeads, split_count, Batch);
  constexpr std::size_t DynamicBytes = E38_DYNAMIC ? static_cast<std::size_t>(4 * KeyBlock * ninfer::ops::kGqaHeadDim) : 0u;
  if constexpr (E38_DYNAMIC) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    ninfer::ops::gqa_attention_decode_i8_tiled_kernel<
                        Geometry, Tokens, Warps, E38_MINBLOCKS, KeyBlock, E38_DYNAMIC,
                        false, false, false, true, false, CacheInput>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(DynamicBytes)) == cudaSuccess,
                "failed to opt in E38 dynamic shared memory");
  }
  ninfer::ops::gqa_attention_decode_i8_tiled_kernel<
      Geometry, Tokens, Warps, E38_MINBLOCKS, KeyBlock, E38_DYNAMIC, false, false, false, true, false, CacheInput>
      <<<grid, Warps * 32, DynamicBytes, stream>>>(
          static_cast<const __nv_bfloat16*>(q.data_ptr()), CacheInput{},
          static_cast<const int32_t*>(pos.data_ptr()),
          static_cast<int8_t*>(cache_k.data_ptr()),
          reinterpret_cast<uint8_t*>(cache_v.data_ptr()),
          reinterpret_cast<__half*>(cache_k_scale.data_ptr()),
          reinterpret_cast<__half*>(cache_v_scale.data_ptr()),
          static_cast<const int32_t*>(block_tables.data_ptr()),
          static_cast<const int32_t*>(valid_columns.data_ptr()),
          static_cast<const int32_t*>(table_rows.data_ptr()),
          static_cast<int32_t>(block_tables.size(1)), Tokens, 0, logical_capacity, scale,
          g64_code_page_stride(cache_k, cache_v),
          g64_scale_page_stride(cache_k_scale),
          reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr()),
          static_cast<float*>(partial_m.data_ptr()), static_cast<float*>(partial_l.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce_batch(const torch::Tensor& partial_acc,
                  const torch::Tensor& partial_m,
                  const torch::Tensor& partial_l,
                  const torch::Tensor& pos,
                  const torch::Tensor& out,
                  int split_count) {
  check_cuda(partial_acc, "partial_acc");
  check_cuda(partial_m, "partial_m");
  check_cuda(partial_l, "partial_l");
  check_cuda(pos, "pos");
  check_cuda(out, "out");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.numel() % (8 * 24 * 256) == 0,
              "out must be contiguous BF16 [batch,8,24,256]");
  const int Batch = static_cast<int>(out.numel() / (8 * 24 * 256));
  TORCH_CHECK(Batch >= 1 && Batch <= 4, "batch must be in [1,4]");
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(Geometry::QHeads, 1, Batch * 8);
  ninfer::ops::gqa_attention_small_t_reduce_output_kernel<
      Geometry, 256, true, true, false, false><<<grid, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(partial_acc.data_ptr()),
      static_cast<const float*>(partial_m.data_ptr()),
      static_cast<const float*>(partial_l.data_ptr()),
      static_cast<const int32_t*>(pos.data_ptr()), nullptr, 8, 8, 0, Batch, split_count,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partial", &partial, "E38 INT8-G64 partial");
  m.def("append", &append, "E38 INT8-G64 append and partial");
  m.def("reduce", &reduce, "E38 INT8-G64 reduce");
  m.def("partial_batch", &partial_batch, "E38 INT8-G64 batch partial");
  m.def("reduce_batch", &reduce_batch, "E38 INT8-G64 batch reduce");
}
