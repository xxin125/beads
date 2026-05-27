#include "device_particles.cuh"

#include <beads/core/cuda_check.cuh>

#include <cstddef>
#include <vector>

namespace beads {
namespace system::state {
namespace {

template <typename T>
void upload_vector(
    DeviceBuffer<T>& device_buffer,
    const std::vector<T>& host_values) {
  if (host_values.empty()) {
    return;
  }
  BEADS_CUDA_CHECK(cudaMemcpy(
      device_buffer.data(),
      host_values.data(),
      host_values.size() * sizeof(T),
      cudaMemcpyHostToDevice));
}

}  // namespace

void DeviceParticleReorderScratch::resize(index_t n_particles) {
  const auto count = static_cast<std::size_t>(n_particles);
  position_x_.resize(count);
  position_y_.resize(count);
  position_z_.resize(count);

  velocity_x_.resize(count);
  velocity_y_.resize(count);
  velocity_z_.resize(count);

  mass_.resize(count);
  type_.resize(count);
  tag_.resize(count);
  molecule_id_.resize(count);

  image_x_.resize(count);
  image_y_.resize(count);
  image_z_.resize(count);
  n_particles_ = n_particles;
}

DeviceParticles::DeviceParticles(const HostParticles& host_particles)
    : n_particles_(host_particles.n_particles),
      position_x_(static_cast<std::size_t>(host_particles.n_particles)),
      position_y_(static_cast<std::size_t>(host_particles.n_particles)),
      position_z_(static_cast<std::size_t>(host_particles.n_particles)),
      velocity_x_(static_cast<std::size_t>(host_particles.n_particles)),
      velocity_y_(static_cast<std::size_t>(host_particles.n_particles)),
      velocity_z_(static_cast<std::size_t>(host_particles.n_particles)),
      mass_(static_cast<std::size_t>(host_particles.n_particles)),
      type_(static_cast<std::size_t>(host_particles.n_particles)),
      tag_(static_cast<std::size_t>(host_particles.n_particles)),
      molecule_id_(static_cast<std::size_t>(host_particles.n_particles)),
      image_x_(static_cast<std::size_t>(host_particles.n_particles)),
      image_y_(static_cast<std::size_t>(host_particles.n_particles)),
      image_z_(static_cast<std::size_t>(host_particles.n_particles)) {
  upload_vector(position_x_, host_particles.position_x);
  upload_vector(position_y_, host_particles.position_y);
  upload_vector(position_z_, host_particles.position_z);

  upload_vector(velocity_x_, host_particles.velocity_x);
  upload_vector(velocity_y_, host_particles.velocity_y);
  upload_vector(velocity_z_, host_particles.velocity_z);

  upload_vector(mass_, host_particles.masses);
  upload_vector(type_, host_particles.types);
  upload_vector(tag_, host_particles.tags);
  upload_vector(molecule_id_, host_particles.molecule_ids);

  upload_vector(image_x_, host_particles.image_x);
  upload_vector(image_y_, host_particles.image_y);
  upload_vector(image_z_, host_particles.image_z);
}

void DeviceParticles::swap_reorderable_fields(
    DeviceParticleReorderScratch& scratch) noexcept {
  position_x_.swap(scratch.position_x_);
  position_y_.swap(scratch.position_y_);
  position_z_.swap(scratch.position_z_);

  velocity_x_.swap(scratch.velocity_x_);
  velocity_y_.swap(scratch.velocity_y_);
  velocity_z_.swap(scratch.velocity_z_);

  mass_.swap(scratch.mass_);
  type_.swap(scratch.type_);
  tag_.swap(scratch.tag_);
  molecule_id_.swap(scratch.molecule_id_);

  image_x_.swap(scratch.image_x_);
  image_y_.swap(scratch.image_y_);
  image_z_.swap(scratch.image_z_);
  scratch.n_particles_ = n_particles_;
}

}  // namespace system::state
}  // namespace beads
