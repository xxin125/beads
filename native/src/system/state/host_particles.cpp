#include "host_particles.hpp"

#include <cstddef>
#include <vector>

namespace beads {
namespace system::state {
namespace {

template <typename T>
std::size_t element_count(const input::ArrayViewSpec<T>& view) {
  std::size_t count = 1;
  for (const std::size_t extent : view.shape) {
    count *= extent;
  }
  return count;
}

template <typename T>
std::vector<T> copy_array_view(const input::ArrayViewSpec<T>& view) {
  const std::size_t count = element_count(view);
  return std::vector<T>(view.data, view.data + count);
}

template <typename T>
std::vector<T> copy_interleaved_axis(
    const input::ArrayViewSpec<T>& view,
    index_t n_particles,
    std::size_t axis) {
  const std::size_t count = static_cast<std::size_t>(n_particles);
  std::vector<T> values(count);
  for (std::size_t particle = 0; particle < count; ++particle) {
    values[particle] = view.data[3 * particle + axis];
  }
  return values;
}

}  // namespace

HostParticles::HostParticles(const input::SystemSpec& system)
    : n_particles(system.n_particles),
      position_x(copy_interleaved_axis(system.positions, system.n_particles, 0)),
      position_y(copy_interleaved_axis(system.positions, system.n_particles, 1)),
      position_z(copy_interleaved_axis(system.positions, system.n_particles, 2)),
      velocity_x(copy_interleaved_axis(system.velocities, system.n_particles, 0)),
      velocity_y(copy_interleaved_axis(system.velocities, system.n_particles, 1)),
      velocity_z(copy_interleaved_axis(system.velocities, system.n_particles, 2)),
      masses(copy_array_view(system.masses)),
      types(copy_array_view(system.types)),
      tags(copy_array_view(system.tags)),
      molecule_ids(copy_array_view(system.molecule_ids)),
      image_x(copy_interleaved_axis(system.images, system.n_particles, 0)),
      image_y(copy_interleaved_axis(system.images, system.n_particles, 1)),
      image_z(copy_interleaved_axis(system.images, system.n_particles, 2)) {}

}  // namespace system::state
}  // namespace beads
