#pragma once

#include <forcefield/force_eval.cuh>
#include <simulation/output/log/log_output.cuh>
#include <simulation/output/output_demand.cuh>

#include <cuda_runtime.h>

#include <memory>

namespace beads {
namespace input {
struct OutputSpec;
}
namespace system::state {
class DeviceParticles;
class DeviceForces;
}
namespace system::geometry {
class BoxGeometry;
}
namespace system::units {
struct UnitSystem;
}
namespace simulation::output {

class OutputSystem {
 public:
  OutputSystem(const input::OutputSpec& output, OutputDemand demand);
  ~OutputSystem();

  OutputSystem(const OutputSystem&) = delete;
  OutputSystem& operator=(const OutputSystem&) = delete;
  OutputSystem(OutputSystem&&) noexcept;
  OutputSystem& operator=(OutputSystem&&) noexcept;

  bool has_output() const noexcept;
  bool needs_force_state(runstep_t step) const noexcept;
  forcefield::ForceEvalRequest max_force_request() const noexcept;
  forcefield::ForceEvalRequest force_request(runstep_t step) const noexcept;
  void prepare(
      index_t n_particles,
      const system::units::UnitSystem& units,
      const forcefield::ForceEvalObservableLayout& layout);
  void stage_if_due(
      runstep_t step,
      const system::state::DeviceParticles& particles,
      const system::state::DeviceForces& forces,
      const system::geometry::BoxGeometry& box,
      const forcefield::ForceEvalResult& force_result,
      cudaStream_t dynamics_stream,
      cudaStream_t transfer_stream);
  void poll_ready();
  void begin_run(const log::LogRunStartSummary& summary);
  void finish_run(
      const log::LogRunEndSummary& summary,
      runstep_t step,
      const system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box);
  void flush();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace simulation::output
}  // namespace beads
