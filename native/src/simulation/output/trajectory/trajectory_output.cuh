#pragma once

#include <beads/core/types.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace beads {
namespace system::geometry {
class BoxGeometry;
}
namespace system::state {
class DeviceParticles;
class DeviceForces;
}
namespace system::units {
struct UnitSystem;
}
namespace simulation::output {
class HostWriter;
}
namespace simulation::output::trajectory {

struct TrajectoryOutputConfig {
  runstep_t every = 0;
  std::string prefix;
  std::vector<std::string> fields;
};

class TrajectoryOutput {
 public:
  TrajectoryOutput(TrajectoryOutputConfig config, HostWriter& writer);
  ~TrajectoryOutput();

  TrajectoryOutput(const TrajectoryOutput&) = delete;
  TrajectoryOutput& operator=(const TrajectoryOutput&) = delete;
  TrajectoryOutput(TrajectoryOutput&&) noexcept;
  TrajectoryOutput& operator=(TrajectoryOutput&&) noexcept;

  bool has_output() const noexcept;
  void prepare(
      index_t n_particles,
      const system::units::UnitSystem& units);
  void stage_if_due(
      runstep_t step,
      const system::state::DeviceParticles& particles,
      const system::state::DeviceForces& forces,
      const system::geometry::BoxGeometry& box,
      cudaStream_t dynamics_stream,
      cudaStream_t transfer_stream);
  bool needs_force_state(runstep_t step) const noexcept;
  void poll_ready();
  void drain_pending();
  void flush_file();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace simulation::output::trajectory
}  // namespace beads
