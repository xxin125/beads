#pragma once

#include <beads/core/types.hpp>

#include <array>

namespace beads {
namespace system::geometry {

struct BoxGeometry {
  BoxGeometry() = default;

  void set_bounds(const real_t* box_bound) noexcept;

  std::array<real_t, 3> lower = {real_t{0}, real_t{0}, real_t{0}};
  std::array<real_t, 3> upper = {real_t{0}, real_t{0}, real_t{0}};
  std::array<real_t, 3> lengths = {real_t{0}, real_t{0}, real_t{0}};
  std::array<real_t, 3> half_lengths = {real_t{0}, real_t{0}, real_t{0}};
  std::array<real_t, 3> inv_lengths = {real_t{0}, real_t{0}, real_t{0}};
};

real_t shortest_length(const BoxGeometry& box) noexcept;
real_t half_shortest_length(const BoxGeometry& box) noexcept;
real_t volume(const BoxGeometry& box) noexcept;

}  // namespace system::geometry
}  // namespace beads
