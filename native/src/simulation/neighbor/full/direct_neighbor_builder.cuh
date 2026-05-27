#pragma once

#include <beads/core/types.hpp>
#include <system/geometry/box_geometry.hpp>
#include <simulation/neighbor/neighbor_list.cuh>
#include <simulation/neighbor/neighbor_plan.hpp>
#include <system/state/device_particles.cuh>
#include <system/topology/exclusions.cuh>

#include <cuda_runtime.h>

namespace beads {
namespace simulation::neighbor {
namespace full {

void build_direct_neighbor_list(
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    const NeighborPlan& plan,
    NeighborList& neighbor_list,
    system::topology::SlotExclusionConstView exclusions = {},
    cudaStream_t stream = nullptr);

}  // namespace full
}  // namespace simulation::neighbor
}  // namespace beads
