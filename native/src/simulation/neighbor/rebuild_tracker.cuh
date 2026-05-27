#pragma once

#include <beads/core/device_buffer.cuh>
#include <beads/core/types.hpp>
#include <system/geometry/box_geometry.hpp>
#include <system/state/device_particles.cuh>

#include <cuda_runtime.h>

#include <cstddef>

namespace beads {
namespace simulation::neighbor {

class NeighborRebuildTracker {
 public:
  NeighborRebuildTracker() = default;
  ~NeighborRebuildTracker();

  NeighborRebuildTracker(const NeighborRebuildTracker&) = delete;
  NeighborRebuildTracker& operator=(const NeighborRebuildTracker&) = delete;
  NeighborRebuildTracker(NeighborRebuildTracker&&) = delete;
  NeighborRebuildTracker& operator=(NeighborRebuildTracker&&) = delete;

  void resize(index_t n_particles, cudaStream_t stream = nullptr);

  index_t n_particles() const noexcept { return n_particles_; }

  void capture_reference(
      const system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream = nullptr);

  bool exceeds_displacement_threshold(
      const system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box,
      real_t threshold_sq,
      cudaStream_t stream = nullptr);

 private:
  void ensure_threshold_storage();
  void release_threshold_storage_noexcept() noexcept;

  index_t n_particles_ = 0;
  DeviceBuffer<real_t> reference_position_x_;
  DeviceBuffer<real_t> reference_position_y_;
  DeviceBuffer<real_t> reference_position_z_;
  DeviceBuffer<image_t> reference_image_x_;
  DeviceBuffer<image_t> reference_image_y_;
  DeviceBuffer<image_t> reference_image_z_;
  DeviceBuffer<int> threshold_exceeded_;
  int* threshold_exceeded_host_ = nullptr;
  bool has_reference_ = false;
};

}  // namespace simulation::neighbor
}  // namespace beads
