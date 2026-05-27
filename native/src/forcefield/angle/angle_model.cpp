#include "angle_model.hpp"

#include <forcefield/style_param_reader.hpp>
#include <system/state/host_state.hpp>

#include <stdexcept>
#include <utility>

namespace beads {
namespace forcefield {
namespace angle {
namespace {

std::string owner_label(const std::string& style_name, std::string_view owner) {
  std::string label = "Angle(\"";
  label += style_name;
  label += "\") ";
  label += owner;
  return label;
}

}  // namespace

AngleModel::AngleModel(std::string style_name)
    : style_name_(std::move(style_name)) {}

void AngleModel::configure(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state) {
  require_not_configured();
  require_angle_style(forcefield);
  read_settings(*forcefield.angle_style);

  if (host_state.topology().angles().empty()) {
    throw std::invalid_argument(angle_label() + " requires topology angles.");
  }
  if (forcefield.angle_coeffs.empty()) {
    throw std::invalid_argument(angle_label() + " requires at least one angle_coeff.");
  }

  begin_topology(host_state);
  for (const input::AngleCoeffSpec& angle_coeff : forcefield.angle_coeffs) {
    require_angle_coeff_style(angle_coeff);
    read_coeff(angle_coeff);
  }
  finish_configuration();
  configured_ = true;
}

std::string AngleModel::angle_label() const {
  return "Angle(\"" + style_name_ + "\")";
}

void AngleModel::require_exact_parameter_keys(
    const input::StyleParamMap& params,
    std::initializer_list<const char*> required_keys,
    std::string_view owner) const {
  require_exact_style_parameter_keys(
      params,
      required_keys,
      owner_label(style_name_, owner));
}

real_t AngleModel::require_nonnegative_real_parameter(
    const input::StyleParamMap& params,
    const char* key,
    std::string_view owner) const {
  return require_nonnegative_real_style_parameter(
      params,
      key,
      owner_label(style_name_, owner));
}

void AngleModel::require_not_configured() const {
  if (configured_) {
    throw std::logic_error(angle_label() + " is already configured.");
  }
}

void AngleModel::require_angle_style(
    const input::ForceFieldSpec& forcefield) const {
  if (!forcefield.angle_style) {
    throw std::invalid_argument(angle_label() + " requires active angle_style.");
  }
  if (forcefield.angle_style->style != style_name_) {
    throw std::invalid_argument(
        "Angle(\"" + style_name_ + "\") received angle_style \"" +
        forcefield.angle_style->style + "\".");
  }
}

void AngleModel::require_angle_coeff_style(
    const input::AngleCoeffSpec& angle_coeff) const {
  if (angle_coeff.style != style_name_) {
    throw std::invalid_argument(
        angle_label() + " angle_coeff style must be \"" + style_name_ + "\".");
  }
}

}  // namespace angle
}  // namespace forcefield
}  // namespace beads
