#include "pair_model.hpp"

#include <forcefield/style_param_reader.hpp>

#include <stdexcept>
#include <utility>

namespace beads {
namespace forcefield {
namespace pair {
namespace {

std::string owner_label(const std::string& style_name, std::string_view owner) {
  std::string label = "Pair(\"";
  label += style_name;
  label += "\") ";
  label += owner;
  return label;
}

}  // namespace

PairModel::PairModel(std::string style_name)
    : style_name_(std::move(style_name)) {}

void PairModel::configure(
    const input::ForceFieldSpec& forcefield,
    type_id_t active_type_count) {
  require_not_configured();
  require_pair_style(forcefield.pair_style);
  read_settings(forcefield.pair_style);

  if (forcefield.pair_coeffs.empty()) {
    throw std::invalid_argument(pair_label() + " requires at least one pair_coeff.");
  }

  begin_coeffs(active_type_count);
  for (const input::PairCoeffSpec& pair_coeff : forcefield.pair_coeffs) {
    require_pair_coeff_style(pair_coeff);
    read_coeff(pair_coeff);
  }
  finish_configuration();
  configured_ = true;
}

std::string PairModel::pair_label() const {
  return "Pair(\"" + style_name_ + "\")";
}

ForceEvalObservableLayout PairModel::observable_layout(
    index_t,
    const ForceEvalRequest& request) const {
  if (request.empty()) {
    return {};
  }
  throw std::invalid_argument(
      pair_label() + " does not support requested force observables.");
}

void PairModel::evaluate_forces(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const simulation::neighbor::NeighborList& neighbor_list,
    const system::geometry::BoxGeometry& box,
    const ForceEvalRequest& request,
    const ForceObservableBuffers&,
    cudaStream_t stream) const {
  if (request.empty()) {
    compute_forces(particles, forces, neighbor_list, box, stream);
    return;
  }
  throw std::invalid_argument(
      pair_label() + " does not support requested force observables.");
}

void PairModel::require_exact_parameter_keys(
    const input::StyleParamMap& params,
    std::initializer_list<const char*> required_keys,
    std::string_view owner) const {
  require_exact_style_parameter_keys(
      params,
      required_keys,
      owner_label(style_name_, owner));
}

real_t PairModel::require_positive_real_parameter(
    const input::StyleParamMap& params,
    const char* key,
    std::string_view owner) const {
  return require_positive_real_style_parameter(
      params,
      key,
      owner_label(style_name_, owner));
}

void PairModel::require_allowed_parameter_keys(
    const input::StyleParamMap& params,
    std::initializer_list<const char*> allowed_keys,
    std::string_view owner) const {
  require_allowed_style_parameter_keys(
      params,
      allowed_keys,
      owner_label(style_name_, owner));
}
bool PairModel::optional_boolean_parameter(
    const input::StyleParamMap& params,
    const char* key,
    bool default_value,
    std::string_view owner) const {
  return optional_boolean_style_parameter(
      params,
      key,
      default_value,
      owner_label(style_name_, owner));
}

void PairModel::require_not_configured() const {
  if (configured_) {
    throw std::logic_error(pair_label() + " is already configured.");
  }
}

void PairModel::require_pair_style(const input::PairStyleSpec& pair_style) const {
  if (pair_style.style != style_name_) {
    throw std::invalid_argument(
        "Pair(\"" + style_name_ + "\") received pair_style \"" +
        pair_style.style + "\".");
  }
}

void PairModel::require_pair_coeff_style(
    const input::PairCoeffSpec& pair_coeff) const {
  if (pair_coeff.style != style_name_) {
    throw std::invalid_argument(
        pair_label() + " pair_coeff style must be \"" + style_name_ + "\".");
  }
}

}  // namespace pair
}  // namespace forcefield
}  // namespace beads
