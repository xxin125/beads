#pragma once

#include <beads/core/cuda_macros.cuh>
#include <beads/core/device_buffer.cuh>
#include <beads/core/types.hpp>
#include <system/state/device_particles.cuh>
#include <system/state/tag_to_slot_map.cuh>
#include <system/topology/topology.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace beads {
namespace system::topology {

struct ExcludedTagPair {
  index_t tag_i = 0;
  index_t tag_j = 0;
};

enum class ExclusionSlotRuntimeMode {
  None,
  InlineOnly,
  InlinePlusOverflow,
};

const char* exclusion_slot_runtime_mode_name(
    ExclusionSlotRuntimeMode mode) noexcept;

struct ExclusionSourceConstView {
  index_t tag_count = 0;
  const index_t* offsets = nullptr;
  const index_t* partners = nullptr;
  const index_t* degree_by_tag = nullptr;
  const index_t* overflow_count_by_tag = nullptr;
};

class ExclusionSourceRuntime {
 public:
  ExclusionSourceRuntime() = default;
  ExclusionSourceRuntime(
      index_t tag_count,
      const std::vector<ExcludedTagPair>& pairs);

  void assign(
      index_t tag_count,
      const std::vector<ExcludedTagPair>& pairs);
  bool empty() const noexcept { return unique_pair_count_ == 0; }
  index_t tag_count() const noexcept { return tag_count_; }
  index_t unique_pair_count() const noexcept { return unique_pair_count_; }
  index_t max_degree() const noexcept { return max_degree_; }
  index_t overflow_slot_count() const noexcept { return overflow_slot_count_; }
  index_t total_overflow_partners() const noexcept {
    return total_overflow_partners_;
  }
  ExclusionSourceConstView view() const noexcept;

  void download_offsets(std::vector<index_t>& host) const;
  void download_partners(std::vector<index_t>& host) const;

 private:
  index_t tag_count_ = 0;
  index_t unique_pair_count_ = 0;
  index_t max_degree_ = 0;
  index_t overflow_slot_count_ = 0;
  index_t total_overflow_partners_ = 0;
  DeviceBuffer<index_t> offsets_;
  DeviceBuffer<index_t> partners_;
  DeviceBuffer<index_t> degree_by_tag_;
  DeviceBuffer<index_t> overflow_count_by_tag_;
};

struct ExclusionSlotInfo {
  index_t degree = 0;
  index_t overflow_offset = 0;
  index_t overflow_count = 0;
};

struct SlotExclusionConstView {
  ExclusionSlotRuntimeMode mode = ExclusionSlotRuntimeMode::None;
  index_t slot_count = 0;
  const ExclusionSlotInfo* slot_info = nullptr;
  const index_t* inline_partners = nullptr;
  const index_t* overflow_partners = nullptr;
};

class SlotExclusionRuntime {
 public:
  static constexpr index_t inline_capacity = 6;

  // Rebuild enqueues the current-slot projection on stream; same-stream
  // neighbor builders may consume the returned view without host synchronization.
  void reset_none(index_t slot_count);
  void rebuild_from_source(
      const ExclusionSourceRuntime& source,
      const state::DeviceParticles& particles,
      const state::TagToSlotMap& tag_to_slot_map,
      cudaStream_t stream,
      std::uint64_t slot_generation);

  index_t slot_count() const noexcept { return slot_count_; }
  ExclusionSlotRuntimeMode mode() const noexcept { return mode_; }
  index_t max_degree() const noexcept { return max_degree_; }
  index_t total_overflow_partners() const noexcept {
    return total_overflow_partners_;
  }
  // Matches the TagToSlotMap generation whose projection has been enqueued.
  std::uint64_t slot_generation() const noexcept { return slot_generation_; }
  SlotExclusionConstView view() const noexcept;

  void download_slot_info(std::vector<ExclusionSlotInfo>& host) const;
  void download_inline_partners(std::vector<index_t>& host) const;
  void download_overflow_partners(std::vector<index_t>& host) const;

 private:
  void ensure_sort_workspace(
      index_t total_overflow_partners,
      index_t slot_count,
      cudaStream_t stream);
  void ensure_scan_workspace(index_t scan_count, cudaStream_t stream);

  index_t slot_count_ = 0;
  ExclusionSlotRuntimeMode mode_ = ExclusionSlotRuntimeMode::None;
  index_t max_degree_ = 0;
  index_t total_overflow_partners_ = 0;
  std::uint64_t slot_generation_ = 0;

  DeviceBuffer<ExclusionSlotInfo> slot_info_;
  DeviceBuffer<index_t> inline_partners_;
  DeviceBuffer<index_t> overflow_offsets_;
  DeviceBuffer<index_t> overflow_counts_;
  DeviceBuffer<index_t> overflow_partners_;
  DeviceBuffer<index_t> overflow_sorted_partners_;
  DeviceBuffer<std::byte> scan_workspace_;
  std::size_t scan_workspace_bytes_ = 0;
  DeviceBuffer<std::byte> sort_workspace_;
  std::size_t sort_workspace_bytes_ = 0;
};

std::vector<ExcludedTagPair> compile_bond_graph_excluded_pairs(
    const HostTopology& topology,
    index_t n_particles,
    index_t distance);

ExclusionSourceRuntime compile_bond_graph_exclusion_source(
    const HostTopology& topology,
    index_t n_particles,
    index_t distance);

template <ExclusionSlotRuntimeMode Mode>
BEADS_HOST_DEVICE inline bool slot_is_excluded(
    SlotExclusionConstView exclusions,
    index_t slot_i,
    index_t slot_j) noexcept {
  if constexpr (Mode == ExclusionSlotRuntimeMode::None) {
    return false;
  } else {
    const ExclusionSlotInfo info = exclusions.slot_info[slot_i];
    const index_t* inline_base =
        exclusions.inline_partners +
        slot_i * SlotExclusionRuntime::inline_capacity;
    const index_t inline_count =
        info.degree < SlotExclusionRuntime::inline_capacity
            ? info.degree
            : SlotExclusionRuntime::inline_capacity;
    for (index_t i = 0; i < inline_count; ++i) {
      if (inline_base[i] == slot_j) {
        return true;
      }
    }
    if constexpr (Mode == ExclusionSlotRuntimeMode::InlineOnly) {
      return false;
    } else {
      index_t begin = info.overflow_offset;
      index_t end = begin + info.overflow_count;
      while (begin < end) {
        const index_t mid = begin + (end - begin) / 2;
        const index_t candidate = exclusions.overflow_partners[mid];
        if (candidate == slot_j) {
          return true;
        }
        if (candidate < slot_j) {
          begin = mid + 1;
        } else {
          end = mid;
        }
      }
      return false;
    }
  }
}

}  // namespace system::topology
}  // namespace beads
