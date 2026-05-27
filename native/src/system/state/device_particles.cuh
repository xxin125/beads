#pragma once

#include <beads/core/device_buffer.cuh>
#include <beads/core/types.hpp>
#include <system/state/host_particles.hpp>

namespace beads {
namespace system::state {

struct DeviceParticlesConstView {
  index_t n_particles = 0;
  const real_t* position_x = nullptr;
  const real_t* position_y = nullptr;
  const real_t* position_z = nullptr;
  const type_id_t* type = nullptr;
  const index_t* tag = nullptr;
};

struct DeviceParticlesView {
  index_t n_particles = 0;
  real_t* position_x = nullptr;
  real_t* position_y = nullptr;
  real_t* position_z = nullptr;
  real_t* velocity_x = nullptr;
  real_t* velocity_y = nullptr;
  real_t* velocity_z = nullptr;
  const real_t* mass = nullptr;
  image_t* image_x = nullptr;
  image_t* image_y = nullptr;
  image_t* image_z = nullptr;
};

struct DeviceParticleReorderScratchView {
  index_t n_particles = 0;

  real_t* position_x = nullptr;
  real_t* position_y = nullptr;
  real_t* position_z = nullptr;

  real_t* velocity_x = nullptr;
  real_t* velocity_y = nullptr;
  real_t* velocity_z = nullptr;

  real_t* mass = nullptr;
  type_id_t* type = nullptr;
  index_t* tag = nullptr;
  index_t* molecule_id = nullptr;

  image_t* image_x = nullptr;
  image_t* image_y = nullptr;
  image_t* image_z = nullptr;
};

class DeviceParticleReorderScratch {
 public:
  DeviceParticleReorderScratch() = default;
  explicit DeviceParticleReorderScratch(index_t n_particles) {
    resize(n_particles);
  }

  index_t n_particles() const noexcept { return n_particles_; }
  void resize(index_t n_particles);

  DeviceParticleReorderScratchView view() noexcept {
    return DeviceParticleReorderScratchView{
        n_particles_,
        position_x_.data(),
        position_y_.data(),
        position_z_.data(),
        velocity_x_.data(),
        velocity_y_.data(),
        velocity_z_.data(),
        mass_.data(),
        type_.data(),
        tag_.data(),
        molecule_id_.data(),
        image_x_.data(),
        image_y_.data(),
        image_z_.data()};
  }

 private:
  friend class DeviceParticles;

  index_t n_particles_ = 0;

  DeviceBuffer<real_t> position_x_;
  DeviceBuffer<real_t> position_y_;
  DeviceBuffer<real_t> position_z_;

  DeviceBuffer<real_t> velocity_x_;
  DeviceBuffer<real_t> velocity_y_;
  DeviceBuffer<real_t> velocity_z_;

  DeviceBuffer<real_t> mass_;
  DeviceBuffer<type_id_t> type_;
  DeviceBuffer<index_t> tag_;
  DeviceBuffer<index_t> molecule_id_;

  DeviceBuffer<image_t> image_x_;
  DeviceBuffer<image_t> image_y_;
  DeviceBuffer<image_t> image_z_;
};

class DeviceParticles {
 public:
  DeviceParticles() = default;
  explicit DeviceParticles(const HostParticles& host_particles);

  index_t n_particles() const noexcept { return n_particles_; }
  void swap_reorderable_fields(DeviceParticleReorderScratch& scratch) noexcept;

  DeviceParticlesView view() noexcept {
    return DeviceParticlesView{
        n_particles_,
        position_x_.data(),
        position_y_.data(),
        position_z_.data(),
        velocity_x_.data(),
        velocity_y_.data(),
        velocity_z_.data(),
        mass_.data(),
        image_x_.data(),
        image_y_.data(),
        image_z_.data()};
  }

  DeviceParticlesConstView view() const noexcept {
    return DeviceParticlesConstView{
        n_particles_,
        position_x_.data(),
        position_y_.data(),
        position_z_.data(),
        type_.data(),
        tag_.data()};
  }

  DeviceBuffer<real_t>& position_x() noexcept { return position_x_; }
  DeviceBuffer<real_t>& position_y() noexcept { return position_y_; }
  DeviceBuffer<real_t>& position_z() noexcept { return position_z_; }

  DeviceBuffer<real_t>& velocity_x() noexcept { return velocity_x_; }
  DeviceBuffer<real_t>& velocity_y() noexcept { return velocity_y_; }
  DeviceBuffer<real_t>& velocity_z() noexcept { return velocity_z_; }

  DeviceBuffer<real_t>& mass() noexcept { return mass_; }
  DeviceBuffer<type_id_t>& type() noexcept { return type_; }
  DeviceBuffer<index_t>& tag() noexcept { return tag_; }
  DeviceBuffer<index_t>& molecule_id() noexcept { return molecule_id_; }

  DeviceBuffer<image_t>& image_x() noexcept { return image_x_; }
  DeviceBuffer<image_t>& image_y() noexcept { return image_y_; }
  DeviceBuffer<image_t>& image_z() noexcept { return image_z_; }

  const DeviceBuffer<real_t>& position_x() const noexcept {
    return position_x_;
  }
  const DeviceBuffer<real_t>& position_y() const noexcept {
    return position_y_;
  }
  const DeviceBuffer<real_t>& position_z() const noexcept {
    return position_z_;
  }

  const DeviceBuffer<real_t>& velocity_x() const noexcept {
    return velocity_x_;
  }
  const DeviceBuffer<real_t>& velocity_y() const noexcept {
    return velocity_y_;
  }
  const DeviceBuffer<real_t>& velocity_z() const noexcept {
    return velocity_z_;
  }

  const DeviceBuffer<real_t>& mass() const noexcept { return mass_; }
  const DeviceBuffer<type_id_t>& type() const noexcept { return type_; }
  const DeviceBuffer<index_t>& tag() const noexcept { return tag_; }
  const DeviceBuffer<index_t>& molecule_id() const noexcept {
    return molecule_id_;
  }

  const DeviceBuffer<image_t>& image_x() const noexcept { return image_x_; }
  const DeviceBuffer<image_t>& image_y() const noexcept { return image_y_; }
  const DeviceBuffer<image_t>& image_z() const noexcept { return image_z_; }

 private:
  index_t n_particles_ = 0;

  DeviceBuffer<real_t> position_x_;
  DeviceBuffer<real_t> position_y_;
  DeviceBuffer<real_t> position_z_;

  DeviceBuffer<real_t> velocity_x_;
  DeviceBuffer<real_t> velocity_y_;
  DeviceBuffer<real_t> velocity_z_;

  DeviceBuffer<real_t> mass_;
  DeviceBuffer<type_id_t> type_;
  DeviceBuffer<index_t> tag_;
  DeviceBuffer<index_t> molecule_id_;

  DeviceBuffer<image_t> image_x_;
  DeviceBuffer<image_t> image_y_;
  DeviceBuffer<image_t> image_z_;
};

}  // namespace system::state
}  // namespace beads
