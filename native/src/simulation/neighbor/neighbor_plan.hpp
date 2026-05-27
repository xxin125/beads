#pragma once

#include <beads/core/types.hpp>
#include <system/geometry/box_geometry.hpp>
#include <input/native_spec.hpp>

namespace beads {
namespace simulation::neighbor {

enum class NeighborBuildPathKind {
  Direct,
  RegularCell,
  SmallDimension,
};

const char* neighbor_build_path_name(NeighborBuildPathKind path_kind) noexcept;

struct NeighborCellGeometry {
  index_t nx = 0;
  index_t ny = 0;
  index_t nz = 0;
  index_t n_cells = 0;
  real_t cell_size_x = real_t{0};
  real_t cell_size_y = real_t{0};
  real_t cell_size_z = real_t{0};
  real_t inv_cell_size_x = real_t{0};
  real_t inv_cell_size_y = real_t{0};
  real_t inv_cell_size_z = real_t{0};
};

struct NeighborPlan {
  NeighborPlan(
      const input::NeighborSpec& neighbor,
      real_t force_cutoff,
      index_t n_particles,
      const system::geometry::BoxGeometry& box);

  index_t n_particles = 0;
  real_t force_cutoff = real_t{0};
  real_t cutoff_buffer = real_t{0};
  real_t search_cutoff = real_t{0};
  real_t search_cutoff_sq = real_t{0};
  index_t max_neighbors = 0;
  runstep_t rebuild_check_every = 0;
  runstep_t reorder_every_rebuild = 0;
  real_t half_shortest_box_length = real_t{0};
  NeighborBuildPathKind path_kind = NeighborBuildPathKind::Direct;
  bool uses_cell_list = false;
  NeighborCellGeometry cell_geometry;
};

}  // namespace simulation::neighbor
}  // namespace beads
