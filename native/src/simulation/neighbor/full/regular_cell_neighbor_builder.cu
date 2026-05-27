#include "regular_cell_neighbor_builder.cuh"

#include <beads/core/cuda_check.cuh>
#include <system/geometry/box_geometry_view.cuh>
#include <simulation/neighbor/full/full_neighbor_path_common.cuh>

namespace beads {
namespace simulation::neighbor {
namespace full {
namespace {

struct CellCoord {
  index_t x = 0;
  index_t y = 0;
  index_t z = 0;
};

__device__ index_t flatten_cell(
    index_t ix,
    index_t iy,
    index_t iz,
    const NeighborCellGeometry& geometry
) noexcept
{
  return ix + geometry.nx * (iy + geometry.ny * iz);
}

__device__ CellCoord unflatten_cell(
    index_t cell,
    const NeighborCellGeometry& geometry
) noexcept
{
  const index_t ix = cell % geometry.nx;
  const index_t yz = cell / geometry.nx;
  const index_t iy = yz % geometry.ny;
  const index_t iz = yz / geometry.ny;
  return CellCoord{ix, iy, iz};
}

__device__ index_t wrapped_axis_cell(
    index_t center,
    int delta,
    index_t count
) noexcept
{
  if (delta < 0) {
    return center == 0 ? count - 1 : center - 1;
  }
  if (delta > 0) {
    const index_t next = center + 1;
    return next == count ? 0 : next;
  }
  return center;
}

template <system::topology::ExclusionSlotRuntimeMode ExclusionMode>
__global__ void build_regular_cell_neighbor_list_kernel(
    ParticlePositionConstView particles,
    system::geometry::BoxGeometryView box,
    NeighborCellGeometry cell_geometry,
    real_t search_cutoff_sq,
    CellListBuildDataConstView cell_list_data,
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
  const CellCoord cell_coord = unflatten_cell(cell_list_data.particle_cell[i], cell_geometry);

  index_t count = 0;
  bool overflowed = false;
  for (int dz = -1; dz <= 1; ++dz) {
    const index_t iz = wrapped_axis_cell(cell_coord.z, dz, cell_geometry.nz);
    for (int dy = -1; dy <= 1; ++dy) {
      const index_t iy = wrapped_axis_cell(cell_coord.y, dy, cell_geometry.ny);
      for (int dx = -1; dx <= 1; ++dx) {
        const index_t ix = wrapped_axis_cell(cell_coord.x, dx, cell_geometry.nx);
        const index_t neighbor_cell = flatten_cell(ix, iy, iz, cell_geometry);
        const index_t begin = cell_list_data.cell_offset[neighbor_cell];
        const index_t count_in_cell = cell_list_data.cell_count[neighbor_cell];
        for (index_t slot = 0; slot < count_in_cell; ++slot) {
          const index_t j = cell_list_data.cell_particle[begin + slot];
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
      }
    }
  }
  if (overflowed && neighbor_list.overflow_flag != nullptr) {
    atomicExch(neighbor_list.overflow_flag, 1);
  }

  neighbor_list.neighbor_count[i] = count;
}

}  // namespace

void build_regular_cell_neighbor_list_from_cell_list(
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    const NeighborPlan& plan,
    const CellListBuildData& cell_list_data,
    NeighborList& neighbor_list,
    system::topology::SlotExclusionConstView exclusions,
    cudaStream_t stream) {
  neighbor_list.clear_build_state(stream);

  const int grid = full_neighbor_grid_size(
      plan.n_particles, kFullNeighborBlockSize);
  switch (exclusions.mode) {
    case system::topology::ExclusionSlotRuntimeMode::None:
      build_regular_cell_neighbor_list_kernel<
          system::topology::ExclusionSlotRuntimeMode::None><<<
          grid, kFullNeighborBlockSize, 0, stream>>>(
              position_view(particles),
              system::geometry::make_box_geometry_view(box),
              plan.cell_geometry,
              plan.search_cutoff_sq,
              cell_list_data.view(),
              exclusions,
              neighbor_list.view());
      break;
    case system::topology::ExclusionSlotRuntimeMode::InlineOnly:
      build_regular_cell_neighbor_list_kernel<
          system::topology::ExclusionSlotRuntimeMode::InlineOnly><<<
          grid, kFullNeighborBlockSize, 0, stream>>>(
              position_view(particles),
              system::geometry::make_box_geometry_view(box),
              plan.cell_geometry,
              plan.search_cutoff_sq,
              cell_list_data.view(),
              exclusions,
              neighbor_list.view());
      break;
    case system::topology::ExclusionSlotRuntimeMode::InlinePlusOverflow:
      build_regular_cell_neighbor_list_kernel<
          system::topology::ExclusionSlotRuntimeMode::InlinePlusOverflow><<<
          grid, kFullNeighborBlockSize, 0, stream>>>(
              position_view(particles),
              system::geometry::make_box_geometry_view(box),
              plan.cell_geometry,
              plan.search_cutoff_sq,
              cell_list_data.view(),
              exclusions,
              neighbor_list.view());
      break;
  }
  BEADS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace full
}  // namespace simulation::neighbor
}  // namespace beads
