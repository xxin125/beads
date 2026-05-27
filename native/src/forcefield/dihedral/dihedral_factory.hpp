#pragma once

#include <input/native_spec.hpp>

#include <memory>

namespace beads {
namespace system::state {
class HostState;
}
namespace forcefield {
namespace dihedral {

class DihedralModel;

std::unique_ptr<DihedralModel> create_dihedral_model(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state);

}  // namespace dihedral
}  // namespace forcefield
}  // namespace beads
