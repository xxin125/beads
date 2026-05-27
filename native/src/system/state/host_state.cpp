#include "host_state.hpp"

namespace beads {
namespace system::state {

HostState::HostState(const input::SystemSpec& system)
    : units_(beads::system::units::unit_system_from_public_name(system.units)),
      particles_(system),
      topology_(system.topology),
      forces_(system.n_particles) {
  box_.set_bounds(system.box_bound.data);
}

}  // namespace system::state
}  // namespace beads
