#include "neighbor_plan.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace beads {
namespace simulation::neighbor {
namespace {

constexpr real_t kCellSizeRelativeTolerance = real_t{1.0e-5};

void require_finite_positive(real_t value, const char* context) {
  if (!std::isfinite(static_cast<double>(value)) || value <= real_t{0}) {
    throw std::invalid_argument(std::string(context) + " must be finite and positive.");
  }
}

void require_finite_nonnegative(real_t value, const char* context) {
  if (!std::isfinite(static_cast<double>(value)) || value < real_t{0}) {
    throw std::invalid_argument(
        std::string(context) + " must be finite and non-negative.");
  }
}

index_t checked_axis_cell_count(
    real_t length,
    real_t search_cutoff,
    const char* context) {
  require_finite_positive(length, context);

  const double raw_count = std::floor(
      static_cast<double>(length) / static_cast<double>(search_cutoff));
  if (!std::isfinite(raw_count)) {
    throw std::invalid_argument(
        std::string(context) + " cell count must be finite.");
  }
  if (raw_count < 1.0) {
    return 1;
  }
  if (raw_count > static_cast<double>(std::numeric_limits<index_t>::max())) {
    throw std::invalid_argument(
        std::string(context) + " cell count exceeds index_t range.");
  }
  return static_cast<index_t>(raw_count);
}

index_t checked_cell_product(index_t nx, index_t ny, index_t nz) {
  const auto sx = static_cast<std::size_t>(nx);
  const auto sy = static_cast<std::size_t>(ny);
  const auto sz = static_cast<std::size_t>(nz);
  const auto index_max = static_cast<std::size_t>(
      std::numeric_limits<index_t>::max());

  if (sx == 0 || sy == 0 || sz == 0) {
    throw std::invalid_argument("neighbor n_cells must be positive.");
  }
  if (sy != 0 && sx > std::numeric_limits<std::size_t>::max() / sy) {
    throw std::invalid_argument("neighbor n_cells exceeds size_t range.");
  }
  const std::size_t xy = sx * sy;
  if (sz != 0 && xy > std::numeric_limits<std::size_t>::max() / sz) {
    throw std::invalid_argument("neighbor n_cells exceeds size_t range.");
  }
  const std::size_t xyz = xy * sz;
  if (xyz > index_max) {
    throw std::invalid_argument("neighbor n_cells exceeds index_t range.");
  }
  return static_cast<index_t>(xyz);
}

real_t cell_size_for_axis(real_t length, index_t cell_count, const char* context) {
  if (cell_count == 0) {
    throw std::invalid_argument(std::string(context) + " cell count must be positive.");
  }
  const real_t cell_size = length / static_cast<real_t>(cell_count);
  require_finite_positive(cell_size, context);
  return cell_size;
}

real_t inverse_cell_size(real_t cell_size, const char* context) {
  const real_t inverse = real_t{1} / cell_size;
  require_finite_positive(inverse, context);
  return inverse;
}

void require_cell_size_supports_search_cutoff(
    index_t cell_count,
    real_t cell_size,
    real_t search_cutoff,
    const char* context) {
  if (cell_count < 3) {
    return;
  }
  const real_t tolerance =
      std::max(real_t{1.0e-6}, search_cutoff * kCellSizeRelativeTolerance);
  if (cell_size + tolerance < search_cutoff) {
    throw std::invalid_argument(
        std::string(context) +
        " cell size must be at least neighbor search_cutoff.");
  }
}

NeighborBuildPathKind choose_path_kind(index_t nx, index_t ny, index_t nz) noexcept {
  if (nx < 3 && ny < 3 && nz < 3) {
    return NeighborBuildPathKind::Direct;
  }
  if (nx >= 3 && ny >= 3 && nz >= 3) {
    return NeighborBuildPathKind::RegularCell;
  }
  return NeighborBuildPathKind::SmallDimension;
}

void validate_path_kind(
    NeighborBuildPathKind path_kind,
    bool uses_cell_list,
    index_t nx,
    index_t ny,
    index_t nz) {
  const NeighborBuildPathKind expected = choose_path_kind(nx, ny, nz);
  if (path_kind != expected) {
    throw std::logic_error("neighbor build path kind does not match cell counts.");
  }
  if (uses_cell_list != (path_kind != NeighborBuildPathKind::Direct)) {
    throw std::logic_error(
        "neighbor uses_cell_list does not match build path kind.");
  }
}

NeighborCellGeometry make_cell_geometry(
    const system::geometry::BoxGeometry& box,
    real_t search_cutoff) {
  NeighborCellGeometry geometry;
  geometry.nx = checked_axis_cell_count(
      box.lengths[0], search_cutoff, "neighbor x box length");
  geometry.ny = checked_axis_cell_count(
      box.lengths[1], search_cutoff, "neighbor y box length");
  geometry.nz = checked_axis_cell_count(
      box.lengths[2], search_cutoff, "neighbor z box length");
  geometry.n_cells = checked_cell_product(geometry.nx, geometry.ny, geometry.nz);

  geometry.cell_size_x = cell_size_for_axis(
      box.lengths[0], geometry.nx, "neighbor x cell size");
  geometry.cell_size_y = cell_size_for_axis(
      box.lengths[1], geometry.ny, "neighbor y cell size");
  geometry.cell_size_z = cell_size_for_axis(
      box.lengths[2], geometry.nz, "neighbor z cell size");
  geometry.inv_cell_size_x =
      inverse_cell_size(geometry.cell_size_x, "neighbor x inverse cell size");
  geometry.inv_cell_size_y =
      inverse_cell_size(geometry.cell_size_y, "neighbor y inverse cell size");
  geometry.inv_cell_size_z =
      inverse_cell_size(geometry.cell_size_z, "neighbor z inverse cell size");

  require_cell_size_supports_search_cutoff(
      geometry.nx, geometry.cell_size_x, search_cutoff, "neighbor x");
  require_cell_size_supports_search_cutoff(
      geometry.ny, geometry.cell_size_y, search_cutoff, "neighbor y");
  require_cell_size_supports_search_cutoff(
      geometry.nz, geometry.cell_size_z, search_cutoff, "neighbor z");

  return geometry;
}

}  // namespace

const char* neighbor_build_path_name(NeighborBuildPathKind path_kind) noexcept {
  switch (path_kind) {
    case NeighborBuildPathKind::Direct:
      return "direct";
    case NeighborBuildPathKind::RegularCell:
      return "regular_cell";
    case NeighborBuildPathKind::SmallDimension:
      return "small_dimension";
  }
  return "unknown";
}

NeighborPlan::NeighborPlan(
    const input::NeighborSpec& neighbor,
    real_t force_cutoff,
    index_t n_particles,
    const system::geometry::BoxGeometry& box)
    : n_particles(n_particles),
      force_cutoff(force_cutoff),
      cutoff_buffer(neighbor.cutoff_buffer),
      search_cutoff(force_cutoff + neighbor.cutoff_buffer),
      search_cutoff_sq(search_cutoff * search_cutoff),
      max_neighbors(neighbor.max_neighbors),
      rebuild_check_every(neighbor.rebuild_check_every),
      reorder_every_rebuild(neighbor.sort_every_rebuild),
      half_shortest_box_length(system::geometry::half_shortest_length(box)) {
  if (this->n_particles == 0) {
    throw std::invalid_argument("neighbor n_particles must be positive.");
  }
  require_finite_positive(this->force_cutoff, "neighbor force_cutoff");
  require_finite_nonnegative(this->cutoff_buffer, "neighbor cutoff_buffer");
  require_finite_positive(this->search_cutoff, "neighbor search_cutoff");

  if (!std::isfinite(static_cast<double>(search_cutoff_sq)) ||
      search_cutoff_sq <= real_t{0}) {
    throw std::invalid_argument(
        "neighbor search_cutoff_sq must be finite and positive.");
  }
  if (max_neighbors == 0) {
    throw std::invalid_argument("neighbor max_neighbors must be positive.");
  }
  if (rebuild_check_every == 0) {
    throw std::invalid_argument("neighbor rebuild_check_every must be positive.");
  }
  if (reorder_every_rebuild == 0) {
    throw std::invalid_argument("neighbor.sort_every_rebuild must be positive.");
  }
  require_finite_positive(
      half_shortest_box_length,
      "neighbor half shortest box length");

  if (search_cutoff > half_shortest_box_length) {
    throw std::invalid_argument(
        "neighbor search_cutoff must not exceed half shortest box length.");
  }

  cell_geometry = make_cell_geometry(box, search_cutoff);
  path_kind = choose_path_kind(
      cell_geometry.nx, cell_geometry.ny, cell_geometry.nz);
  uses_cell_list = path_kind != NeighborBuildPathKind::Direct;
  validate_path_kind(
      path_kind,
      uses_cell_list,
      cell_geometry.nx,
      cell_geometry.ny,
      cell_geometry.nz);
}

}  // namespace simulation::neighbor
}  // namespace beads
