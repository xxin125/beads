#include "unit_system.hpp"

#include <stdexcept>

namespace beads {
namespace system::units {
namespace {

constexpr real_t kNmKjMolBoltzmannConstant =
    static_cast<real_t>(0.00831446261815324);
constexpr real_t kNmKjMolPressureScale =
    static_cast<real_t>(16.60539067);

UnitSystem reduced_unit_system() noexcept {
  return UnitSystem{
      UnitStyle::Reduced,
      "reduced",
      real_t{1},
      real_t{1},
      "reduced",
      "reduced",
      "reduced",
      "reduced",
      "reduced",
      "reduced"};
}

UnitSystem nm_kjmol_unit_system() noexcept {
  return UnitSystem{
      UnitStyle::NmKjMol,
      "nm_kjmol",
      kNmKjMolBoltzmannConstant,
      kNmKjMolPressureScale,
      "nm",
      "kJ/mol",
      "u",
      "ps",
      "K",
      "bar"};
}

}  // namespace

UnitSystem unit_system_from_public_name(std::string_view name) {
  if (name == "reduced") {
    return reduced_unit_system();
  }
  if (name == "nm_kjmol") {
    return nm_kjmol_unit_system();
  }
  throw std::invalid_argument(
      "system.units must be \"reduced\" or \"nm_kjmol\".");
}

}  // namespace system::units
}  // namespace beads
