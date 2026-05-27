#include "pair_factory.hpp"

#include <forcefield/pair/pair_lj.hpp>

#include <stdexcept>

namespace beads {
namespace forcefield {
namespace pair {
namespace {

using CreatePairModel = std::unique_ptr<PairModel> (*)(
    const input::ForceFieldSpec&,
    type_id_t);

struct BuiltinPairModel {
  const char* style = nullptr;
  CreatePairModel create = nullptr;
};

template <typename PairModelT>
std::unique_ptr<PairModel> make_pair_model(
    const input::ForceFieldSpec& forcefield,
    type_id_t active_type_count) {
  auto model = std::make_unique<PairModelT>();
  model->configure(forcefield, active_type_count);
  return model;
}

constexpr BuiltinPairModel kBuiltinPairModels[] = {
    {LjPairModel::kStyleName, &make_pair_model<LjPairModel>},
};

}  // namespace

std::unique_ptr<PairModel> create_pair_model(
    const input::ForceFieldSpec& forcefield,
    type_id_t active_type_count) {
  for (const BuiltinPairModel& builtin : kBuiltinPairModels) {
    if (forcefield.pair_style.style == builtin.style) {
      return builtin.create(forcefield, active_type_count);
    }
  }

  throw std::invalid_argument(
      "unsupported pair style \"" + forcefield.pair_style.style + "\".");
}

}  // namespace pair
}  // namespace forcefield
}  // namespace beads
