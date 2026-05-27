#include "device_state.cuh"

namespace beads {
namespace system::state {

DeviceState::DeviceState(const HostState& host_state)
    : particles_(host_state.particles()),
      forces_(host_state.particles().n_particles) {
  forces_.clear();
}

}  // namespace system::state
}  // namespace beads
