#pragma once

#include <beads/core/cuda_macros.cuh>
#include <beads/core/device_buffer.cuh>
#include <beads/core/types.hpp>

#include <cuda_runtime.h>

namespace beads {
namespace simulation::neighbor {

BEADS_HOST_DEVICE BEADS_FORCE_INLINE index_t neighbor_index_slot(
    index_t slot,
    index_t n_particles,
    index_t particle
) noexcept
{
  return slot * n_particles + particle;
}

struct NeighborListView {
  index_t n_particles = 0;
  index_t max_neighbors = 0;
  index_t* neighbor_count = nullptr;
  index_t* neighbor_index = nullptr;
  int* overflow_flag = nullptr;
};

struct NeighborListConstView {
  index_t n_particles = 0;
  index_t max_neighbors = 0;
  const index_t* neighbor_count = nullptr;
  const index_t* neighbor_index = nullptr;
  const int* overflow_flag = nullptr;
};

class NeighborList {
 public:
  NeighborList() = default;
  NeighborList(index_t n_particles, index_t max_neighbors);
  ~NeighborList();

  NeighborList(const NeighborList&) = delete;
  NeighborList& operator=(const NeighborList&) = delete;
  NeighborList(NeighborList&&) = delete;
  NeighborList& operator=(NeighborList&&) = delete;

  void resize(index_t n_particles, index_t max_neighbors);

  index_t n_particles() const noexcept { return n_particles_; }
  index_t max_neighbors() const noexcept { return max_neighbors_; }

  void clear_counts(cudaStream_t stream = nullptr);
  void clear_build_state(cudaStream_t stream = nullptr);

  int download_overflow_flag(cudaStream_t stream = nullptr) const;

  NeighborListView view() noexcept;
  NeighborListConstView view() const noexcept;

 private:
  void ensure_overflow_host_storage() const;
  void release_overflow_host_storage_noexcept() noexcept;

  index_t n_particles_ = 0;
  index_t max_neighbors_ = 0;
  DeviceBuffer<index_t> neighbor_count_;
  DeviceBuffer<index_t> neighbor_index_;
  DeviceBuffer<int> overflow_flag_;
  mutable int* overflow_flag_host_ = nullptr;
};

}  // namespace simulation::neighbor
}  // namespace beads
