#pragma once

#include <beads/core/device_buffer.cuh>
#include <beads/core/types.hpp>
#include <system/state/device_particles.cuh>

#include <cstdint>
#include <vector>

namespace beads {
namespace system::state {

class TagToSlotMap {
 public:
  TagToSlotMap() = default;

  static constexpr index_t missing_slot() noexcept {
    return static_cast<index_t>(-1);
  }

  // Prepares a fixed-N one-based tag map. System tags are validated as a
  // 1..N permutation, so a prepared map is never silently resized.
  void prepare(index_t n_particles);
  void mark_dirty() noexcept { dirty_ = true; }
  bool is_dirty() const noexcept { return dirty_; }
  // Current means rebuild work has been enqueued in stream order; same-stream
  // consumers may use the map without a host synchronization.
  bool is_current() const noexcept;
  index_t particle_count() const noexcept { return particle_count_; }
  // Generation advances when rebuild work is enqueued, not when the host has
  // synchronized that work.
  std::uint64_t generation() const noexcept { return generation_; }

  bool rebuild_if_dirty(
      const DeviceParticles& particles,
      cudaStream_t stream = nullptr);
  void rebuild(
      const DeviceParticles& particles,
      cudaStream_t stream = nullptr);

  const DeviceBuffer<index_t>& slots_by_tag() const noexcept {
    return slots_by_tag_;
  }
  DeviceBuffer<index_t>& slots_by_tag() noexcept { return slots_by_tag_; }

  void download_slots_by_tag(std::vector<index_t>& host) const;

 private:
  bool is_prepared() const noexcept;
  void require_prepared() const;

  DeviceBuffer<index_t> slots_by_tag_;
  index_t particle_count_ = 0;
  bool dirty_ = true;
  std::uint64_t generation_ = 0;
};

}  // namespace system::state
}  // namespace beads
