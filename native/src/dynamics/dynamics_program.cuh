#pragma once

#include <beads/core/types.hpp>
#include <dynamics/integrator/integrator.cuh>
#include <dynamics/thermostat/thermostat.cuh>
#include <system/geometry/box_geometry.hpp>
#include <system/state/device_forces.cuh>
#include <system/state/device_particles.cuh>

#include <cuda_runtime.h>

#include <memory>
#include <optional>

namespace beads {
namespace input {
struct DynamicsSpec;
}  // namespace input
namespace system::units {
struct UnitSystem;
}  // namespace system::units

namespace dynamics {

class DynamicsProgram {
 public:
  explicit DynamicsProgram(
      std::unique_ptr<integrator::Integrator> integrator,
      std::unique_ptr<thermostat::Thermostat> thermostat = nullptr);

  DynamicsProgram(const DynamicsProgram&) = delete;
  DynamicsProgram& operator=(const DynamicsProgram&) = delete;
  DynamicsProgram(DynamicsProgram&&) noexcept = default;
  DynamicsProgram& operator=(DynamicsProgram&&) noexcept = default;

  real_t dt() const;
  const char* style() const noexcept;
  const char* thermostat_style() const noexcept;

  void pre_force(
      system::state::DeviceParticles& particles,
      const system::state::DeviceForces& forces,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream = nullptr) const;
  void post_force(
      system::state::DeviceParticles& particles,
      const system::state::DeviceForces& forces,
      cudaStream_t stream = nullptr) const;
  void apply_post_velocity_controls(
      system::state::DeviceParticles& particles,
      cudaStream_t stream = nullptr) const;

 private:
  std::unique_ptr<integrator::Integrator> integrator_;
  std::unique_ptr<thermostat::Thermostat> thermostat_;
};

std::optional<DynamicsProgram> make_dynamics_program(
    const input::DynamicsSpec& dynamics,
    const system::units::UnitSystem& units);

}  // namespace dynamics
}  // namespace beads
