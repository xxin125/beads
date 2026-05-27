#pragma once

#include <beads/core/types.hpp>
#include <system/geometry/box_geometry.hpp>
#include <system/state/device_forces.cuh>
#include <system/state/device_particles.cuh>

#include <cuda_runtime.h>

namespace beads {
namespace dynamics {
namespace integrator {

class Integrator {
 public:
  virtual ~Integrator() = default;

  virtual real_t dt() const = 0;
  virtual const char* style() const noexcept = 0;

  virtual void pre_force(
      system::state::DeviceParticles& particles,
      const system::state::DeviceForces& forces,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream = nullptr) const = 0;
  virtual void post_force(
      system::state::DeviceParticles& particles,
      const system::state::DeviceForces& forces,
      cudaStream_t stream = nullptr) const = 0;
};

}  // namespace integrator
}  // namespace dynamics
}  // namespace beads
