#include "bond_factory.hpp"

#include <forcefield/bond/bond_harmonic.hpp>

#include <stdexcept>

namespace beads {
namespace forcefield {
namespace bond {
namespace {

using CreateBondModel = std::unique_ptr<BondModel> (*)(
    const input::ForceFieldSpec&,
    const system::state::HostState&);

struct BuiltinBondModel {
  const char* style = nullptr;
  CreateBondModel create = nullptr;
};

template <typename BondModelT>
std::unique_ptr<BondModel> make_bond_model(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state) {
  auto model = std::make_unique<BondModelT>();
  model->configure(forcefield, host_state);
  return model;
}

constexpr BuiltinBondModel kBuiltinBondModels[] = {
    {HarmonicBondModel::kStyleName, &make_bond_model<HarmonicBondModel>},
};

}  // namespace

std::unique_ptr<BondModel> create_bond_model(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state) {
  if (!forcefield.bond_style) {
    return nullptr;
  }
  for (const BuiltinBondModel& builtin : kBuiltinBondModels) {
    if (forcefield.bond_style->style == builtin.style) {
      return builtin.create(forcefield, host_state);
    }
  }
  throw std::invalid_argument(
      "unsupported bond style \"" + forcefield.bond_style->style + "\".");
}

}  // namespace bond
}  // namespace forcefield
}  // namespace beads
