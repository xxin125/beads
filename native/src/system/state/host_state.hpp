#pragma once

#include <input/native_spec.hpp>
#include <system/geometry/box_geometry.hpp>
#include <system/state/host_forces.hpp>
#include <system/state/host_particles.hpp>
#include <system/topology/topology.hpp>
#include <system/units/unit_system.hpp>

namespace beads {
namespace system::state {

class HostState {
 public:
  explicit HostState(const input::SystemSpec& system);

  const system::geometry::BoxGeometry& box() const noexcept { return box_; }
  const HostParticles& particles() const noexcept { return particles_; }
  const system::topology::HostTopology& topology() const noexcept {
    return topology_;
  }
  const beads::system::units::UnitSystem& units() const noexcept {
    return units_;
  }

 private:
  beads::system::units::UnitSystem units_;
  system::geometry::BoxGeometry box_;
  HostParticles particles_;
  system::topology::HostTopology topology_;
  HostForces forces_;
};

}  // namespace system::state
}  // namespace beads
