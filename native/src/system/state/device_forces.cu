#include "device_forces.cuh"

#include <beads/core/cuda_check.cuh>

#include <cstddef>

namespace beads {
namespace system::state {

DeviceForces::DeviceForces(index_t n_particles)
    : n_particles_(n_particles),
      force_x_(static_cast<std::size_t>(n_particles)),
      force_y_(static_cast<std::size_t>(n_particles)),
      force_z_(static_cast<std::size_t>(n_particles)) {}

void DeviceForces::clear() {
  if (n_particles_ == 0) {
    return;
  }

  const std::size_t byte_count =
      static_cast<std::size_t>(n_particles_) * sizeof(real_t);
  BEADS_CUDA_CHECK(cudaMemset(force_x_.data(), 0, byte_count));
  BEADS_CUDA_CHECK(cudaMemset(force_y_.data(), 0, byte_count));
  BEADS_CUDA_CHECK(cudaMemset(force_z_.data(), 0, byte_count));
}

}  // namespace system::state
}  // namespace beads
