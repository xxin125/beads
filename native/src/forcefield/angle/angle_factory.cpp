#include "angle_factory.hpp"

#include <forcefield/angle/angle_harmonic.hpp>

#include <stdexcept>

namespace beads {
namespace forcefield {
namespace angle {
namespace {

using CreateAngleModel = std::unique_ptr<AngleModel> (*)(
    const input::ForceFieldSpec&,
    const system::state::HostState&);

struct BuiltinAngleModel {
  const char* style = nullptr;
  CreateAngleModel create = nullptr;
};

template <typename AngleModelT>
std::unique_ptr<AngleModel> make_angle_model(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state) {
  auto model = std::make_unique<AngleModelT>();
  model->configure(forcefield, host_state);
  return model;
}

constexpr BuiltinAngleModel kBuiltinAngleModels[] = {
    {HarmonicAngleModel::kStyleName, &make_angle_model<HarmonicAngleModel>},
};

}  // namespace

std::unique_ptr<AngleModel> create_angle_model(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state) {
  if (!forcefield.angle_style) {
    return nullptr;
  }
  for (const BuiltinAngleModel& builtin : kBuiltinAngleModels) {
    if (forcefield.angle_style->style == builtin.style) {
      return builtin.create(forcefield, host_state);
    }
  }
  throw std::invalid_argument(
      "unsupported angle style \"" + forcefield.angle_style->style + "\".");
}

}  // namespace angle
}  // namespace forcefield
}  // namespace beads
