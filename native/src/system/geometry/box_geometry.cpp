#include "box_geometry.hpp"

#include <algorithm>
#include <cstddef>

namespace beads {
namespace system::geometry {

void BoxGeometry::set_bounds(const real_t* box_bound) noexcept {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    lower[axis] = box_bound[axis];
    upper[axis] = box_bound[3 + axis];
    lengths[axis] = upper[axis] - lower[axis];
    half_lengths[axis] = real_t{0.5} * lengths[axis];
    inv_lengths[axis] = real_t{1} / lengths[axis];
  }
}

real_t shortest_length(const BoxGeometry& box) noexcept {
  return std::min(box.lengths[0], std::min(box.lengths[1], box.lengths[2]));
}

real_t half_shortest_length(const BoxGeometry& box) noexcept {
  return real_t{0.5} * shortest_length(box);
}

real_t volume(const BoxGeometry& box) noexcept {
  return box.lengths[0] * box.lengths[1] * box.lengths[2];
}

}  // namespace system::geometry
}  // namespace beads
