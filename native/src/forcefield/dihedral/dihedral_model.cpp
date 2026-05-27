#include "dihedral_model.hpp"

#include <forcefield/style_param_reader.hpp>
#include <system/state/host_state.hpp>

#include <stdexcept>
#include <utility>

namespace beads {
namespace forcefield {
namespace dihedral {
namespace {

std::string owner_label(const std::string& style_name, std::string_view owner) {
  std::string label = "Dihedral(\"";
  label += style_name;
  label += "\") ";
  label += owner;
  return label;
}

}  // namespace

DihedralModel::DihedralModel(std::string style_name)
    : style_name_(std::move(style_name)) {}

void DihedralModel::configure(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state) {
  require_not_configured();
  require_dihedral_style(forcefield);
  read_settings(*forcefield.dihedral_style);

  if (host_state.topology().dihedrals().empty()) {
    throw std::invalid_argument(dihedral_label() + " requires topology dihedrals.");
  }
  if (forcefield.dihedral_coeffs.empty()) {
    throw std::invalid_argument(
        dihedral_label() + " requires at least one dihedral_coeff.");
  }

  begin_topology(host_state);
  for (const input::DihedralCoeffSpec& dihedral_coeff :
       forcefield.dihedral_coeffs) {
    require_dihedral_coeff_style(dihedral_coeff);
    read_coeff(dihedral_coeff);
  }
  finish_configuration();
  configured_ = true;
}

std::string DihedralModel::dihedral_label() const {
  return "Dihedral(\"" + style_name_ + "\")";
}

void DihedralModel::require_exact_parameter_keys(
    const input::StyleParamMap& params,
    std::initializer_list<const char*> required_keys,
    std::string_view owner) const {
  require_exact_style_parameter_keys(
      params,
      required_keys,
      owner_label(style_name_, owner));
}

real_t DihedralModel::require_nonnegative_real_parameter(
    const input::StyleParamMap& params,
    const char* key,
    std::string_view owner) const {
  return require_nonnegative_real_style_parameter(
      params,
      key,
      owner_label(style_name_, owner));
}

std::int64_t DihedralModel::require_integer_parameter(
    const input::StyleParamMap& params,
    const char* key,
    std::string_view owner) const {
  return require_integer_style_parameter(
      params,
      key,
      owner_label(style_name_, owner));
}

void DihedralModel::require_not_configured() const {
  if (configured_) {
    throw std::logic_error(dihedral_label() + " is already configured.");
  }
}

void DihedralModel::require_dihedral_style(
    const input::ForceFieldSpec& forcefield) const {
  if (!forcefield.dihedral_style) {
    throw std::invalid_argument(
        dihedral_label() + " requires active dihedral_style.");
  }
  if (forcefield.dihedral_style->style != style_name_) {
    throw std::invalid_argument(
        "Dihedral(\"" + style_name_ + "\") received dihedral_style \"" +
        forcefield.dihedral_style->style + "\".");
  }
}

void DihedralModel::require_dihedral_coeff_style(
    const input::DihedralCoeffSpec& dihedral_coeff) const {
  if (dihedral_coeff.style != style_name_) {
    throw std::invalid_argument(
        dihedral_label() + " dihedral_coeff style must be \"" +
        style_name_ + "\".");
  }
}

}  // namespace dihedral
}  // namespace forcefield
}  // namespace beads
