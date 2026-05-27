#pragma once

#include <beads/core/types.hpp>
#include <system/state/device_particles.cuh>

#include <cuda_runtime.h>

namespace beads {
namespace dynamics::thermostat {

class Thermostat {
 public:
  virtual ~Thermostat() = default;

  virtual const char* style() const noexcept = 0;

  virtual void apply(
      system::state::DeviceParticles& particles,
      real_t dt,
      cudaStream_t stream = nullptr) const = 0;
};

}  // namespace dynamics::thermostat
}  // namespace beads
