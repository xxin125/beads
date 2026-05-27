#pragma once

#include <input/native_spec.hpp>

#include <memory>

namespace beads {
namespace system::state {
class HostState;
}
namespace forcefield {
namespace angle {

class AngleModel;

std::unique_ptr<AngleModel> create_angle_model(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state);

}  // namespace angle
}  // namespace forcefield
}  // namespace beads
