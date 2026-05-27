#pragma once

#include <beads/core/types.hpp>
#include <dynamics/dynamics_program.cuh>
#include <forcefield/force_eval.cuh>
#include <input/native_spec.hpp>
#include <forcefield/force_field_model.hpp>
#include <simulation/neighbor/neighbor_system.hpp>
#include <simulation/output/output_system.cuh>
#include <simulation/run_streams.cuh>
#include <system/state/device_state.cuh>
#include <system/state/host_state.hpp>

#include <optional>
#include <stdexcept>
#include <chrono>

namespace beads {
namespace simulation {

class NotImplementedFeature : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class SimulationRun {
 public:
  // Direct C++ specs are trusted internal inputs. User-facing paths must cross
  // the Python/native-dict validation boundary before constructing this type.
  explicit SimulationRun(const input::SimulationSpec& spec);

  void execute();

  const system::state::HostState& host_state() const noexcept { return host_state_; }
  const system::state::DeviceState& device_state() const noexcept { return device_state_; }

 private:
  const dynamics::DynamicsProgram& require_dynamics_program() const;
  forcefield::ForceEvalResult compute_forces_for_step(runstep_t step);
  forcefield::ForceEvalResult initialize_force_state();
  void stage_output_for_step(
      runstep_t step,
      const forcefield::ForceEvalResult& force_result);
  void finish_run();
  void execute_zero_step_run();
  void execute_dynamics_run(const dynamics::DynamicsProgram& dynamics);
  void advance_dynamics_step(
      const dynamics::DynamicsProgram& dynamics,
      runstep_t physical_step);

  system::state::HostState host_state_;
  forcefield::ForceFieldModel force_field_;
  RunStreams streams_;
  system::state::DeviceState device_state_;
  simulation::neighbor::NeighborSystem neighbor_system_;
  std::optional<dynamics::DynamicsProgram> dynamics_;
  output::OutputDemand output_demand_;
  forcefield::ForceEvaluator force_evaluator_;
  output::OutputSystem output_;
  output::log::LogRunStartSummary log_start_summary_;
  std::chrono::steady_clock::time_point run_start_time_;
  runstep_t runsteps_ = 0;
};

}  // namespace simulation
}  // namespace beads
