#pragma once

#include <beads/core/device_buffer.cuh>
#include <dynamics/thermostat/thermostat.cuh>
#include <input/native_spec.hpp>
#include <system/units/unit_system.hpp>

#include <cstddef>
#include <memory>

namespace beads {
namespace dynamics::thermostat {

class BerendsenThermostat final : public Thermostat {
 public:
  BerendsenThermostat(
      const input::ThermostatSpec& thermostat,
      const system::units::UnitSystem& units);

  const char* style() const noexcept override { return "berendsen"; }

  void apply(
      system::state::DeviceParticles& particles,
      real_t dt,
      cudaStream_t stream = nullptr) const override;

 private:
  void ensure_workspace(index_t n_particles) const;

  real_t target_thermal_energy_ = real_t{0};
  real_t tau_ = real_t{0};

  mutable index_t prepared_particle_count_ = 0;
  mutable index_t kinetic_partial_count_ = 0;
  mutable int cub_partial_count_ = 0;
  mutable std::size_t sum_workspace_bytes_ = 0;
  mutable DeviceBuffer<real_t> kinetic_partials_;
  mutable DeviceBuffer<real_t> kinetic_total_;
  mutable DeviceBuffer<std::byte> sum_workspace_;
};

std::unique_ptr<Thermostat> make_berendsen_thermostat(
    const input::ThermostatSpec& thermostat,
    const system::units::UnitSystem& units);

}  // namespace dynamics::thermostat
}  // namespace beads
