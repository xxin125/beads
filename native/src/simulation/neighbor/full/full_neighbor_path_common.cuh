#pragma once

#include <beads/core/cuda_macros.cuh>
#include <beads/core/types.hpp>
#include <system/geometry/box_geometry_view.cuh>
#include <simulation/neighbor/neighbor_list.cuh>
#include <system/state/device_particles.cuh>
#include <system/topology/exclusions.cuh>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace beads {
namespace simulation::neighbor {
namespace full {

inline constexpr int kFullNeighborBlockSize = 256;

struct ParticlePositionConstView {
  index_t n_particles = 0;
  const real_t* position_x = nullptr;
  const real_t* position_y = nullptr;
  const real_t* position_z = nullptr;
};

inline ParticlePositionConstView position_view(
    const system::state::DeviceParticles& particles) noexcept {
  return ParticlePositionConstView{
      particles.n_particles(),
      particles.position_x().data(),
      particles.position_y().data(),
      particles.position_z().data()};
}

inline int full_neighbor_grid_size(index_t n_particles, int block_size) {
  if (block_size <= 0) {
    throw std::invalid_argument("CUDA block size must be positive.");
  }
  const auto items = static_cast<std::size_t>(n_particles);
  const auto block = static_cast<std::size_t>(block_size);
  const std::size_t block_count = (items + block - 1) / block;
  if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("CUDA grid size exceeds launch capacity.");
  }
  return static_cast<int>(block_count);
}

template <system::topology::ExclusionSlotRuntimeMode ExclusionMode>
__device__ inline void append_neighbor_or_mark_overflow(
    index_t particle_i,
    index_t particle_j,
    real_t xi,
    real_t yi,
    real_t zi,
    ParticlePositionConstView particles,
    system::geometry::BoxGeometryView box,
    real_t cutoff_sq,
    system::topology::SlotExclusionConstView exclusions,
    index_t& count,
    bool& overflowed,
    NeighborListView neighbors
)
{
  if (particle_i == particle_j) {
    return;
  }
  if constexpr (ExclusionMode != system::topology::ExclusionSlotRuntimeMode::None) {
    if (system::topology::slot_is_excluded<ExclusionMode>(
            exclusions, particle_i, particle_j)) {
      return;
    }
  }

  const real_t dx = system::geometry::minimum_image_delta_x_for_wrapped_positions(
      box, xi - particles.position_x[particle_j]);
  const real_t dy = system::geometry::minimum_image_delta_y_for_wrapped_positions(
      box, yi - particles.position_y[particle_j]);
  const real_t dz = system::geometry::minimum_image_delta_z_for_wrapped_positions(
      box, zi - particles.position_z[particle_j]);
  const real_t r2 = dx * dx + dy * dy + dz * dz;
  if (r2 > cutoff_sq) {
    return;
  }

  if (count < neighbors.max_neighbors) {
    neighbors.neighbor_index[neighbor_index_slot(
        count, neighbors.n_particles, particle_i)] = particle_j;
    ++count;
    return;
  }

  overflowed = true;
}

}  // namespace full
}  // namespace simulation::neighbor
}  // namespace beads
