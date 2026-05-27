#pragma once

#include <forcefield/bond/bond_model.hpp>
#include <input/native_spec.hpp>

#include <memory>

namespace beads {
namespace system::state {
class HostState;
}
namespace forcefield {
namespace bond {

std::unique_ptr<BondModel> create_bond_model(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state);

}  // namespace bond
}  // namespace forcefield
}  // namespace beads
