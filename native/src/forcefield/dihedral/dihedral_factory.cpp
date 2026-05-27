#include "dihedral_factory.hpp"

#include <forcefield/dihedral/dihedral_harmonic.hpp>

#include <stdexcept>

namespace beads {
namespace forcefield {
namespace dihedral {
namespace {

using CreateDihedralModel = std::unique_ptr<DihedralModel> (*)(
    const input::ForceFieldSpec&,
    const system::state::HostState&);

struct BuiltinDihedralModel {
  const char* style = nullptr;
  CreateDihedralModel create = nullptr;
};

template <typename DihedralModelT>
std::unique_ptr<DihedralModel> make_dihedral_model(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state) {
  auto model = std::make_unique<DihedralModelT>();
  model->configure(forcefield, host_state);
  return model;
}

constexpr BuiltinDihedralModel kBuiltinDihedralModels[] = {
    {HarmonicDihedralModel::kStyleName,
     &make_dihedral_model<HarmonicDihedralModel>},
};

}  // namespace

std::unique_ptr<DihedralModel> create_dihedral_model(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state) {
  if (!forcefield.dihedral_style) {
    return nullptr;
  }
  for (const BuiltinDihedralModel& builtin : kBuiltinDihedralModels) {
    if (forcefield.dihedral_style->style == builtin.style) {
      return builtin.create(forcefield, host_state);
    }
  }
  throw std::invalid_argument(
      "unsupported dihedral style \"" + forcefield.dihedral_style->style + "\".");
}

}  // namespace dihedral
}  // namespace forcefield
}  // namespace beads
