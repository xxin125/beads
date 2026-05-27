#pragma once

#include <beads/core/types.hpp>
#include <dynamics/integrator/integrator.cuh>
#include <system/geometry/box_geometry.hpp>
#include <system/state/device_forces.cuh>
#include <system/state/device_particles.cuh>

#include <cuda_runtime.h>

#include <memory>

namespace beads {
namespace input {
struct DynamicsSpec;
}  // namespace input

namespace dynamics {
namespace integrator {

class VelocityVerletIntegrator final : public Integrator {
 public:
  explicit VelocityVerletIntegrator(real_t dt);

  real_t dt() const noexcept override { return dt_; }
  const char* style() const noexcept override { return "velocity_verlet"; }

  void pre_force(
      system::state::DeviceParticles& particles,
      const system::state::DeviceForces& forces,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream = nullptr) const override;
  void post_force(
      system::state::DeviceParticles& particles,
      const system::state::DeviceForces& forces,
      cudaStream_t stream = nullptr) const override;

 private:
  real_t dt_ = real_t{0};
};

std::unique_ptr<Integrator> make_velocity_verlet_integrator(
    const input::DynamicsSpec& dynamics);

}  // namespace integrator
}  // namespace dynamics
}  // namespace beads
