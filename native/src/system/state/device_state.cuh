#pragma once

#include <system/state/device_forces.cuh>
#include <system/state/device_particles.cuh>
#include <system/state/host_state.hpp>

namespace beads {
namespace system::state {

class DeviceState {
 public:
  explicit DeviceState(const HostState& host_state);

  DeviceParticles& particles() noexcept { return particles_; }
  DeviceForces& forces() noexcept { return forces_; }

  const DeviceParticles& particles() const noexcept { return particles_; }
  const DeviceForces& forces() const noexcept { return forces_; }

 private:
  DeviceParticles particles_;
  DeviceForces forces_;
};

}  // namespace system::state
}  // namespace beads
