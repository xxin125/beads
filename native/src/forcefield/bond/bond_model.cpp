#include "bond_model.hpp"

#include <forcefield/style_param_reader.hpp>
#include <system/state/host_state.hpp>

#include <stdexcept>
#include <utility>

namespace beads {
namespace forcefield {
namespace bond {
namespace {

std::string owner_label(const std::string& style_name, std::string_view owner) {
  std::string label = "Bond(\"";
  label += style_name;
  label += "\") ";
  label += owner;
  return label;
}

}  // namespace

BondModel::BondModel(std::string style_name)
    : style_name_(std::move(style_name)) {}

void BondModel::configure(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state) {
  require_not_configured();
  require_bond_style(forcefield);
  read_settings(*forcefield.bond_style);

  if (host_state.topology().bonds().empty()) {
    throw std::invalid_argument(bond_label() + " requires topology bonds.");
  }
  if (forcefield.bond_coeffs.empty()) {
    throw std::invalid_argument(bond_label() + " requires at least one bond_coeff.");
  }

  begin_topology(host_state);
  for (const input::BondCoeffSpec& bond_coeff : forcefield.bond_coeffs) {
    require_bond_coeff_style(bond_coeff);
    read_coeff(bond_coeff);
  }
  finish_configuration();
  configured_ = true;
}

std::string BondModel::bond_label() const {
  return "Bond(\"" + style_name_ + "\")";
}

void BondModel::require_exact_parameter_keys(
    const input::StyleParamMap& params,
    std::initializer_list<const char*> required_keys,
    std::string_view owner) const {
  require_exact_style_parameter_keys(
      params,
      required_keys,
      owner_label(style_name_, owner));
}

real_t BondModel::require_nonnegative_real_parameter(
    const input::StyleParamMap& params,
    const char* key,
    std::string_view owner) const {
  return require_nonnegative_real_style_parameter(
      params,
      key,
      owner_label(style_name_, owner));
}

void BondModel::require_not_configured() const {
  if (configured_) {
    throw std::logic_error(bond_label() + " is already configured.");
  }
}

void BondModel::require_bond_style(
    const input::ForceFieldSpec& forcefield) const {
  if (!forcefield.bond_style) {
    throw std::invalid_argument(bond_label() + " requires active bond_style.");
  }
  if (forcefield.bond_style->style != style_name_) {
    throw std::invalid_argument(
        "Bond(\"" + style_name_ + "\") received bond_style \"" +
        forcefield.bond_style->style + "\".");
  }
}

void BondModel::require_bond_coeff_style(
    const input::BondCoeffSpec& bond_coeff) const {
  if (bond_coeff.style != style_name_) {
    throw std::invalid_argument(
        bond_label() + " bond_coeff style must be \"" + style_name_ + "\".");
  }
}

}  // namespace bond
}  // namespace forcefield
}  // namespace beads
