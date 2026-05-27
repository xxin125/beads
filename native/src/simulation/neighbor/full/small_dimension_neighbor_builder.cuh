#pragma once

#include <cuda_runtime.h>

#include <system/geometry/box_geometry.hpp>
#include <simulation/neighbor/neighbor_list.cuh>
#include <simulation/neighbor/neighbor_plan.hpp>
#include <simulation/neighbor/cell_list_build.cuh>
#include <system/state/device_particles.cuh>
#include <system/topology/exclusions.cuh>

namespace beads {
namespace simulation::neighbor {
namespace full {

void build_small_dimension_neighbor_list_from_cell_list(
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    const NeighborPlan& plan,
    const CellListBuildData& cell_list_data,
    NeighborList& neighbor_list,
    system::topology::SlotExclusionConstView exclusions = {},
    cudaStream_t stream = nullptr);

}  // namespace full
}  // namespace simulation::neighbor
}  // namespace beads
