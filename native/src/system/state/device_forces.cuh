#pragma once

#include <beads/core/device_buffer.cuh>
#include <beads/core/types.hpp>

namespace beads {
namespace system::state {

struct DeviceForcesView {
  index_t n_particles = 0;
  real_t* force_x = nullptr;
  real_t* force_y = nullptr;
  real_t* force_z = nullptr;
};

struct DeviceForcesConstView {
  index_t n_particles = 0;
  const real_t* force_x = nullptr;
  const real_t* force_y = nullptr;
  const real_t* force_z = nullptr;
};

class DeviceForces {
 public:
  DeviceForces() = default;
  explicit DeviceForces(index_t n_particles);

  // Slot-ordered force scratch. Physical particle reorder does not reorder this
  // buffer; consumers must only read it after a force evaluation has refreshed
  // forces for the current particle slot order.
  index_t n_particles() const noexcept { return n_particles_; }

  DeviceForcesView view() noexcept {
    return DeviceForcesView{
        n_particles_,
        force_x_.data(),
        force_y_.data(),
        force_z_.data()};
  }

  DeviceForcesConstView view() const noexcept {
    return DeviceForcesConstView{
        n_particles_,
        force_x_.data(),
        force_y_.data(),
        force_z_.data()};
  }

  DeviceBuffer<real_t>& force_x() noexcept { return force_x_; }
  DeviceBuffer<real_t>& force_y() noexcept { return force_y_; }
  DeviceBuffer<real_t>& force_z() noexcept { return force_z_; }

  const DeviceBuffer<real_t>& force_x() const noexcept { return force_x_; }
  const DeviceBuffer<real_t>& force_y() const noexcept { return force_y_; }
  const DeviceBuffer<real_t>& force_z() const noexcept { return force_z_; }

  void clear();

 private:
  index_t n_particles_ = 0;
  DeviceBuffer<real_t> force_x_;
  DeviceBuffer<real_t> force_y_;
  DeviceBuffer<real_t> force_z_;
};

}  // namespace system::state
}  // namespace beads
