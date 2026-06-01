#include "dynamics_program.cuh"

#include <dynamics/integrator/velocity_verlet.cuh>
#include <dynamics/thermostat/berendsen_thermostat.cuh>
#include <input/native_spec.hpp>
#include <system/units/unit_system.hpp>

#include <stdexcept>
#include <utility>

namespace beads {
namespace dynamics {
namespace {

std::unique_ptr<thermostat::Thermostat> make_thermostat(
    const std::optional<input::ThermostatSpec>& thermostat_spec,
    const system::units::UnitSystem& units) {
  if (!thermostat_spec) {
    return nullptr;
  }
  if (thermostat_spec->style == "berendsen") {
    return thermostat::make_berendsen_thermostat(*thermostat_spec, units);
  }
  throw std::invalid_argument(
      "dynamics.thermostat.style supports only berendsen.");
}

}  // namespace

DynamicsProgram::DynamicsProgram(
    std::unique_ptr<integrator::Integrator> integrator,
    std::unique_ptr<thermostat::Thermostat> thermostat)
    : integrator_(std::move(integrator)),
      thermostat_(std::move(thermostat)) {
  if (!integrator_) {
    throw std::invalid_argument("DynamicsProgram requires an integrator.");
  }
}

DynamicsProgram make_dynamics_program(
    const input::DynamicsSpec& dynamics,
    const system::units::UnitSystem& units) {
  if (dynamics.style == "velocity_verlet") {
    return DynamicsProgram(
        integrator::make_velocity_verlet_integrator(dynamics),
        make_thermostat(dynamics.thermostat, units));
  }

  throw std::invalid_argument(
      "dynamics.style must be \"velocity_verlet\".");
}

real_t DynamicsProgram::dt() const {
  return integrator_->dt();
}

const char* DynamicsProgram::style() const noexcept {
  return integrator_->style();
}

const char* DynamicsProgram::thermostat_style() const noexcept {
  return thermostat_ ? thermostat_->style() : "none";
}

void DynamicsProgram::pre_force(
    system::state::DeviceParticles& particles,
    const system::state::DeviceForces& forces,
    const system::geometry::BoxGeometry& box,
    cudaStream_t stream) const {
  integrator_->pre_force(particles, forces, box, stream);
}

void DynamicsProgram::post_force(
    system::state::DeviceParticles& particles,
    const system::state::DeviceForces& forces,
    cudaStream_t stream) const {
  integrator_->post_force(particles, forces, stream);
}

void DynamicsProgram::apply_post_velocity_controls(
    system::state::DeviceParticles& particles,
    cudaStream_t stream) const {
  if (thermostat_) {
    thermostat_->apply(particles, integrator_->dt(), stream);
  }
}

}  // namespace dynamics
}  // namespace beads
