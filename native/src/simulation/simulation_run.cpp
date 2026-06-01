#include "simulation_run.hpp"

#include <beads/core/cuda_check.cuh>

#include <algorithm>
#include <chrono>

namespace beads {
namespace simulation {
namespace {

output::log::LogRunStartSummary make_log_run_start_summary(
    const input::SimulationSpec& spec,
    const system::state::HostState& host_state,
    const neighbor::NeighborSystem& neighbor_system,
    const std::optional<dynamics::DynamicsProgram>& dynamics) {
  output::log::LogRunStartSummary summary;
  summary.units = host_state.units().public_name;
  summary.n_particles = host_state.particles().n_particles;
  summary.runsteps = spec.runsteps;
  summary.box_lower = host_state.box().lower;
  summary.box_upper = host_state.box().upper;
  summary.box_lengths = host_state.box().lengths;
  summary.dynamics_style = spec.dynamics.style;
  if (dynamics) {
    summary.dynamics_dt = dynamics->dt();
    summary.thermostat_style = dynamics->thermostat_style();
  } else if (spec.dynamics.thermostat) {
    summary.thermostat_style = spec.dynamics.thermostat->style;
  }
  summary.cutoff_buffer = spec.neighbor.cutoff_buffer;
  summary.neighbor_path = neighbor_system.path_name();
  summary.rebuild_check_every = spec.neighbor.rebuild_check_every;
  summary.sort_every_rebuild = spec.neighbor.sort_every_rebuild;
  summary.max_neighbors = spec.neighbor.max_neighbors;
  summary.pair_style = spec.forcefield.pair_style.style;
  if (spec.forcefield.bond_style) {
    summary.bond_style = spec.forcefield.bond_style->style;
  }
  if (spec.forcefield.angle_style) {
    summary.angle_style = spec.forcefield.angle_style->style;
  }
  if (spec.forcefield.dihedral_style) {
    summary.dihedral_style = spec.forcefield.dihedral_style->style;
  }
  summary.bonded_exclusion_policy =
      spec.forcefield.bonded_exclusion_policy;
  summary.bonded_exclusion_distance =
      spec.forcefield.bonded_exclusion_distance;
  summary.bonded_excluded_pair_count =
      neighbor_system.excluded_pair_count();
  summary.bond_count = host_state.topology().bond_count();
  summary.angle_count = host_state.topology().angle_count();
  summary.dihedral_count = host_state.topology().dihedral_count();
  summary.thermo_enabled = spec.output.thermo.has_value();
  summary.final_state_enabled = spec.output.final_state.has_value();
  summary.trajectory_enabled = spec.output.trajectory.has_value();
  summary.log_echo = spec.output.log ? spec.output.log->echo : "disabled";
  return summary;
}

}  // namespace

SimulationRun::SimulationRun(
    const input::SimulationSpec& spec,
    InterruptCheckCallback interrupt_check)
    : host_state_(spec.system),
      force_field_(spec.forcefield, host_state_),
      streams_(),
      device_state_(host_state_),
      neighbor_system_(
          spec.neighbor,
          force_field_.max_cutoff(),
          host_state_,
          force_field_.requires_tag_to_slot_map(),
          spec.forcefield.bonded_exclusion_distance),
      dynamics_(dynamics::make_dynamics_program(spec.dynamics, host_state_.units())),
      output_demand_(output::OutputDemand::from_spec(spec.output)),
      force_evaluator_(
          force_field_,
          spec.system.n_particles,
          output_demand_.max_force_request()),
      output_(spec.output, output_demand_),
      log_start_summary_(make_log_run_start_summary(
          spec,
          host_state_,
          neighbor_system_,
          dynamics_)),
      runsteps_(spec.runsteps),
      interrupt_check_(std::move(interrupt_check)) {

  // Dynamic interrupt interval based on particle count:
  // N=1000 -> every 100 steps
  // N=50,000 -> every 2 steps
  // N>100,000 -> every 1 step
  interrupt_check_interval_ = std::max<runstep_t>(1, 100000 / std::max<index_t>(1, spec.system.n_particles));

  output_.prepare(
      spec.system.n_particles,
      host_state_.units(),
      force_evaluator_.observable_layout());
}

const dynamics::DynamicsProgram& SimulationRun::require_dynamics_program() const {
  if (!dynamics_) {
    throw NotImplementedFeature(
        "Dynamics(\"none\") supports only zero-step force evaluation. "
        "Use Dynamics(\"velocity_verlet\", dt=...) for positive runsteps.");
  }
  return *dynamics_;
}

forcefield::ForceEvalResult SimulationRun::compute_forces_for_step(
    runstep_t step) {
  const forcefield::ForceEvalRequest request = output_.force_request(step);
  return force_evaluator_.evaluate(
      device_state_.particles(),
      device_state_.forces(),
      neighbor_system_.neighbor_list(),
      host_state_.box(),
      request,
      neighbor_system_.current_tag_to_slot_map(),
      streams_.dynamics_stream());
}

forcefield::ForceEvalResult SimulationRun::initialize_force_state() {
  neighbor_system_.initialize_for_run(
      device_state_.particles(),
      host_state_.box(),
      streams_.dynamics_stream());
  return compute_forces_for_step(runstep_t{0});
}

void SimulationRun::stage_output_for_step(
    runstep_t step,
    const forcefield::ForceEvalResult& force_result) {
  output_.stage_if_due(
      step,
      device_state_.particles(),
      device_state_.forces(),
      host_state_.box(),
      force_result,
      streams_.dynamics_stream(),
      streams_.transfer_stream());
  output_.poll_ready();
}

void SimulationRun::finish_run() {
  BEADS_CUDA_CHECK(cudaStreamSynchronize(streams_.dynamics_stream()));
  const auto run_end_time = std::chrono::steady_clock::now();
  const double run_wall_s =
      std::chrono::duration<double>(run_end_time - run_start_time_).count();
  output::log::LogRunEndSummary summary;
  summary.runsteps = runsteps_;
  summary.run_wall_s = run_wall_s;
  if (runsteps_ != 0) {
    summary.steps_per_second = static_cast<double>(runsteps_) / run_wall_s;
  }
  summary.neighbor_build_count = neighbor_system_.build_count();
  summary.dangerous_rebuild_count =
      neighbor_system_.dangerous_rebuild_count();
  // Output finish owns any transfer-stream waits needed for staged rows, then
  // appends the run-end log after all pending data rows are submitted.
  output_.finish_run(
      summary,
      runsteps_,
      device_state_.particles(),
      host_state_.box());
}

void SimulationRun::execute_zero_step_run() {
  if (!output_.has_output()) {
    return;
  }

  run_start_time_ = std::chrono::steady_clock::now();
  output_.begin_run(log_start_summary_);
  const forcefield::ForceEvalRequest request = output_.force_request(runstep_t{0});
  forcefield::ForceEvalResult force_result;
  if (!request.empty() || output_.needs_force_state(runstep_t{0})) {
    force_result = initialize_force_state();
  }
  stage_output_for_step(runstep_t{0}, force_result);
  finish_run();
}

void SimulationRun::check_interrupt() {
  if (interrupt_check_) {
    interrupt_check_();
  }
}

void SimulationRun::synchronize_streams_noexcept() noexcept {
  if (streams_.dynamics_stream() != nullptr) {
    cudaStreamSynchronize(streams_.dynamics_stream());
  }
  if (streams_.transfer_stream() != nullptr) {
    cudaStreamSynchronize(streams_.transfer_stream());
  }
}

void SimulationRun::execute_dynamics_run(
    const dynamics::DynamicsProgram& dynamics) {
  run_start_time_ = std::chrono::steady_clock::now();
  output_.begin_run(log_start_summary_);
  const forcefield::ForceEvalResult initial_force_result =
      initialize_force_state();
  stage_output_for_step(runstep_t{0}, initial_force_result);

  try {
    for (runstep_t step = 0; step < runsteps_; ++step) {
      if (step % interrupt_check_interval_ == 0) {
        check_interrupt();
      }
      advance_dynamics_step(dynamics, step + 1);
    }
  } catch (...) {
    synchronize_streams_noexcept();
    throw;
  }

  finish_run();
}

void SimulationRun::advance_dynamics_step(
    const dynamics::DynamicsProgram& dynamics,
    runstep_t physical_step) {
  dynamics.pre_force(
      device_state_.particles(),
      device_state_.forces(),
      host_state_.box(),
      streams_.dynamics_stream());
  neighbor_system_.update_after_position_change(
      device_state_.particles(),
      host_state_.box(),
      physical_step,
      streams_.dynamics_stream());
  const forcefield::ForceEvalResult force_result =
      compute_forces_for_step(physical_step);
  dynamics.post_force(
      device_state_.particles(),
      device_state_.forces(),
      streams_.dynamics_stream());
  dynamics.apply_post_velocity_controls(
      device_state_.particles(),
      streams_.dynamics_stream());
  stage_output_for_step(physical_step, force_result);
}

void SimulationRun::execute() {
  if (runsteps_ == 0) {
    execute_zero_step_run();
    return;
  }

  const dynamics::DynamicsProgram& dynamics = require_dynamics_program();
  execute_dynamics_run(dynamics);
}

}  // namespace simulation
}  // namespace beads
