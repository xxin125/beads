#pragma once

#include <beads/core/types.hpp>

#include <string_view>

namespace beads {
namespace system::units {

enum class UnitStyle {
  Reduced,
  NmKjMol,
};

struct UnitSystem {
  UnitStyle style = UnitStyle::Reduced;
  const char* public_name = "reduced";
  real_t boltzmann_constant = real_t{1};
  real_t pressure_scale = real_t{1};
  const char* length_label = "reduced";
  const char* energy_label = "reduced";
  const char* mass_label = "reduced";
  const char* time_label = "reduced";
  const char* temperature_label = "reduced";
  const char* pressure_label = "reduced";

  real_t temperature_from_kinetic_energy_dof(
      real_t kinetic_energy,
      real_t dof) const noexcept {
    if (!(dof > real_t{0}) || !(boltzmann_constant > real_t{0})) {
      return real_t{0};
    }
    return (real_t{2} * kinetic_energy) / (dof * boltzmann_constant);
  }

  real_t thermal_energy_from_temperature(real_t temperature) const noexcept {
    return boltzmann_constant * temperature;
  }

  real_t pressure_from_kinetic_energy_and_virial(
      real_t kinetic_energy,
      real_t virial,
      real_t volume) const noexcept {
    if (!(volume > real_t{0})) {
      return real_t{0};
    }
    return pressure_scale *
        ((real_t{2} * kinetic_energy + virial) / (real_t{3} * volume));
  }
};

UnitSystem unit_system_from_public_name(std::string_view name);

}  // namespace system::units
}  // namespace beads
