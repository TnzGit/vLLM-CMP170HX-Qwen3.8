#pragma once

// Isolated candidate: stride-aware paged KV addressing for the E38 adapter.
// Adds page-stride parameters without changing the e40-ninfer installed tree.
// The historical contiguous layout is recovered when page_stride equals the
// natural LeadingExtent * page_size * HeadExtent layout, so a caller that keeps
// computing the historical stride keeps the old byte addresses.

#include "ops/kernel/paged_kv_address.cuh"

#include <cstdint>

namespace ninfer::ops {

// Stride-aware element offsets: physical_page contributes page_stride, and the
// head/leading terms keep the historical within-page geometry (head plane =
// LeadingExtent * page_size elements), so a contiguous caller passing the
// natural stride reproduces the original byte addresses exactly.
template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t paged_kv_page_head_offset_strided(
    std::int64_t page_stride, std::int32_t physical_page, std::int32_t head) {
    return page_stride * physical_page +
           static_cast<std::int64_t>(LeadingExtent) * kPagedKVPageSize * head;
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t paged_kv_element_offset_strided(
    std::int64_t page_stride, std::int32_t physical_page, std::int32_t head,
    std::int32_t page_offset, std::int32_t leading) {
    return paged_kv_page_head_offset_strided<LeadingExtent, HeadExtent>(
               page_stride, physical_page, head) +
           static_cast<std::int64_t>(LeadingExtent) * page_offset + leading;
}

} // namespace ninfer::ops
