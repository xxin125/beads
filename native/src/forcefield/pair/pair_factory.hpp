#pragma once

#include <beads/core/types.hpp>
#include <input/native_spec.hpp>
#include <forcefield/pair/pair_model.hpp>

#include <memory>

namespace beads {
namespace forcefield {
namespace pair {

std::unique_ptr<PairModel> create_pair_model(
    const input::ForceFieldSpec& forcefield,
    type_id_t active_type_count);

}  // namespace pair
}  // namespace forcefield
}  // namespace beads
