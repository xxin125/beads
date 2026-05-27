#include "direct_neighbor_builder.cuh"

#include <beads/core/cuda_check.cuh>
#include <system/geometry/box_geometry_view.cuh>
#include <simulation/neighbor/full/full_neighbor_path_common.cuh>

namespace beads {
namespace simulation::neighbor {
namespace full {
namespace {

template <system::topology::ExclusionSlotRuntimeMode ExclusionMode>
__global__ void build_direct_neighbor_list_kernel(
    ParticlePositionConstView particles,
    system::geometry::BoxGeometryView box,
    real_t search_cutoff_sq,
    system::topology::SlotExclusionConstView exclusions,
    NeighborListView neighbor_list
)
{
  const index_t i = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= particles.n_particles) {
    return;
  }

  const real_t xi = particles.position_x[i];
  const real_t yi = particles.position_y[i];
  const real_t zi = particles.position_z[i];

  index_t count = 0;
  bool overflowed = false;
  for (index_t j = 0; j < particles.n_particles; ++j) {
    append_neighbor_or_mark_overflow<ExclusionMode>(
        i,
        j,
        xi,
        yi,
        zi,
        particles,
        box,
        search_cutoff_sq,
        exclusions,
        count,
        overflowed,
        neighbor_list);
  }
  if (overflowed && neighbor_list.overflow_flag != nullptr) {
    atomicExch(neighbor_list.overflow_flag, 1);
  }

  neighbor_list.neighbor_count[i] = count;
}

}  // namespace

void build_direct_neighbor_list(
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    const NeighborPlan& plan,
    NeighborList& neighbor_list,
    system::topology::SlotExclusionConstView exclusions,
    cudaStream_t stream) {
  neighbor_list.clear_build_state(stream);

  const int grid = full_neighbor_grid_size(
      plan.n_particles, kFullNeighborBlockSize);
  switch (exclusions.mode) {
    case system::topology::ExclusionSlotRuntimeMode::None:
      build_direct_neighbor_list_kernel<
          system::topology::ExclusionSlotRuntimeMode::None><<<
          grid, kFullNeighborBlockSize, 0, stream>>>(
              position_view(particles),
              system::geometry::make_box_geometry_view(box),
              plan.search_cutoff_sq,
              exclusions,
              neighbor_list.view());
      break;
    case system::topology::ExclusionSlotRuntimeMode::InlineOnly:
      build_direct_neighbor_list_kernel<
          system::topology::ExclusionSlotRuntimeMode::InlineOnly><<<
          grid, kFullNeighborBlockSize, 0, stream>>>(
              position_view(particles),
              system::geometry::make_box_geometry_view(box),
              plan.search_cutoff_sq,
              exclusions,
              neighbor_list.view());
      break;
    case system::topology::ExclusionSlotRuntimeMode::InlinePlusOverflow:
      build_direct_neighbor_list_kernel<
          system::topology::ExclusionSlotRuntimeMode::InlinePlusOverflow><<<
          grid, kFullNeighborBlockSize, 0, stream>>>(
              position_view(particles),
              system::geometry::make_box_geometry_view(box),
              plan.search_cutoff_sq,
              exclusions,
              neighbor_list.view());
      break;
  }
  BEADS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace full
}  // namespace simulation::neighbor
}  // namespace beads
