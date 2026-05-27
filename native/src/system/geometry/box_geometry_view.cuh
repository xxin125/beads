#pragma once

#include <beads/core/cuda_macros.cuh>
#include <beads/core/types.hpp>
#include <system/geometry/box_geometry.hpp>

#include <cmath>

namespace beads {
namespace system::geometry {

struct BoxGeometryView {
  real_t lower_x = real_t{0};
  real_t lower_y = real_t{0};
  real_t lower_z = real_t{0};

  real_t length_x = real_t{0};
  real_t length_y = real_t{0};
  real_t length_z = real_t{0};

  real_t half_length_x = real_t{0};
  real_t half_length_y = real_t{0};
  real_t half_length_z = real_t{0};

  real_t inv_length_x = real_t{0};
  real_t inv_length_y = real_t{0};
  real_t inv_length_z = real_t{0};
};

inline BoxGeometryView make_box_geometry_view(const BoxGeometry& box) noexcept {
  return BoxGeometryView{
      box.lower[0],
      box.lower[1],
      box.lower[2],
      box.lengths[0],
      box.lengths[1],
      box.lengths[2],
      box.half_lengths[0],
      box.half_lengths[1],
      box.half_lengths[2],
      box.inv_lengths[0],
      box.inv_lengths[1],
      box.inv_lengths[2]};
}

// Requires both coordinates to be wrapped into the same orthorhombic box.
// This is not a general minimum-image helper for arbitrary unwrapped
// displacement; use image-aware displacement logic or a future general MIC
// helper for large/unwrapped displacements.
BEADS_HOST_DEVICE BEADS_FORCE_INLINE real_t minimum_image_delta_for_wrapped_positions(
    real_t delta,
    real_t length,
    real_t half_length
) noexcept
{
  if (delta > half_length) {
    delta -= length;
  } else if (delta < -half_length) {
    delta += length;
  }
  return delta;
}

BEADS_HOST_DEVICE BEADS_FORCE_INLINE real_t minimum_image_delta_x_for_wrapped_positions(
    const BoxGeometryView& box,
    real_t delta
) noexcept
{
  return minimum_image_delta_for_wrapped_positions(delta, box.length_x, box.half_length_x);
}

BEADS_HOST_DEVICE BEADS_FORCE_INLINE real_t minimum_image_delta_y_for_wrapped_positions(
    const BoxGeometryView& box,
    real_t delta
) noexcept
{
  return minimum_image_delta_for_wrapped_positions(delta, box.length_y, box.half_length_y);
}

BEADS_HOST_DEVICE BEADS_FORCE_INLINE real_t minimum_image_delta_z_for_wrapped_positions(
    const BoxGeometryView& box,
    real_t delta
) noexcept
{
  return minimum_image_delta_for_wrapped_positions(delta, box.length_z, box.half_length_z);
}

// Uses floor-based wrapping and can return multi-box crossing counts.
// Position must be finite, the crossing count must fit in image_t, and callers
// accumulating into particle image fields assume no image_t overflow. The
// minimal core does not check image overflow on this hot path.
BEADS_HOST_DEVICE BEADS_FORCE_INLINE image_t wrap_position_axis_with_delta(
    real_t& position,
    real_t lower,
    real_t length,
    real_t inv_length
) noexcept
{
  const real_t crossings = floor((position - lower) * inv_length);
  position -= crossings * length;
  return static_cast<image_t>(crossings);
}

BEADS_HOST_DEVICE BEADS_FORCE_INLINE image_t wrap_position_x_with_delta(
    const BoxGeometryView& box,
    real_t& position
) noexcept
{
  return wrap_position_axis_with_delta(
      position, box.lower_x, box.length_x, box.inv_length_x);
}

BEADS_HOST_DEVICE BEADS_FORCE_INLINE image_t wrap_position_y_with_delta(
    const BoxGeometryView& box,
    real_t& position
) noexcept
{
  return wrap_position_axis_with_delta(
      position, box.lower_y, box.length_y, box.inv_length_y);
}

BEADS_HOST_DEVICE BEADS_FORCE_INLINE image_t wrap_position_z_with_delta(
    const BoxGeometryView& box,
    real_t& position
) noexcept
{
  return wrap_position_axis_with_delta(
      position, box.lower_z, box.length_z, box.inv_length_z);
}

}  // namespace system::geometry
}  // namespace beads
