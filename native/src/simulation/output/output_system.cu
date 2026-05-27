#include "output_system.cuh"

#include <input/native_spec.hpp>
#include <simulation/output/host_writer.hpp>
#include <simulation/output/log/log_output.cuh>
#include <simulation/output/snapshot/final_state_output.cuh>
#include <simulation/output/thermo/thermo_output.cuh>
#include <simulation/output/trajectory/trajectory_output.cuh>
#include <system/geometry/box_geometry.hpp>
#include <system/state/device_forces.cuh>
#include <system/state/device_particles.cuh>
#include <system/units/unit_system.hpp>

#include <algorithm>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace beads {
namespace simulation::output {
namespace {

std::string describe_exception(const std::exception_ptr& error) {
  try {
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
  } catch (const std::exception& exception) {
    return exception.what();
  } catch (...) {
    return "unknown exception";
  }
  return "unknown exception";
}

class OutputErrorCollector {
 public:
  explicit OutputErrorCollector(std::string context)
      : context_(std::move(context)) {}

  template <typename Function>
  void collect(const char* label, Function&& function) {
    try {
      std::forward<Function>(function)();
    } catch (...) {
      add(label, std::current_exception());
    }
  }

  void add(const char* label, const std::exception_ptr& error) {
    std::string message = describe_exception(error);
    if (label != nullptr && label[0] != '\0') {
      message = std::string(label) + ": " + message;
    }
    if (std::find(messages_.begin(), messages_.end(), message) ==
        messages_.end()) {
      messages_.push_back(std::move(message));
    }
  }

  void throw_if_any() const {
    if (messages_.empty()) {
      return;
    }
    std::ostringstream message;
    message << context_ << " failed";
    if (messages_.size() == 1) {
      message << ": " << messages_.front();
    } else {
      message << " with " << messages_.size() << " errors:";
      for (std::size_t index = 0; index < messages_.size(); ++index) {
        message << " [" << (index + 1) << "] " << messages_[index];
      }
    }
    throw std::runtime_error(message.str());
  }

 private:
  std::string context_;
  std::vector<std::string> messages_;
};

}  // namespace

class OutputSystem::Impl {
 public:
  Impl(const input::OutputSpec& output, OutputDemand demand)
      : demand_(demand) {
    if (output.log) {
      log_.emplace(log::LogOutputConfig{
          output.log->echo,
          output.log->prefix},
          writer_);
    }
    if (output.thermo) {
      thermo::ThermoOutputConfig config{
          output.thermo->every,
          output.thermo->prefix};
      if (log_ && log_->active()) {
        config.log_sink = [this](const thermo::ThermoLogRow& row) {
          log_->write_thermo_row_from_writer(row);
        };
      }
      thermo_.emplace(std::move(config), writer_);
    }
    if (output.final_state) {
      final_state_.emplace(snapshot::FinalStateOutputConfig{
          output.final_state->prefix});
    }
    if (output.trajectory) {
      trajectory_.emplace(trajectory::TrajectoryOutputConfig{
          output.trajectory->every,
          output.trajectory->prefix,
          output.trajectory->fields},
          writer_);
    }
  }

  bool has_output() const noexcept {
    return thermo_.has_value() ||
           final_state_.has_value() ||
           trajectory_.has_value() ||
           (log_ && log_->active());
  }

  bool needs_force_state(runstep_t step) const noexcept {
    if (!force_request(step).empty()) {
      return true;
    }
    return trajectory_ && trajectory_->needs_force_state(step);
  }

  forcefield::ForceEvalRequest max_force_request() const noexcept {
    return demand_.max_force_request();
  }

  forcefield::ForceEvalRequest force_request(runstep_t step) const noexcept {
    return demand_.force_request(step);
  }

  void prepare(
      index_t n_particles,
      const system::units::UnitSystem& units,
      const forcefield::ForceEvalObservableLayout& layout) {
    if (thermo_) {
      thermo_->prepare(n_particles, units, layout);
    }
    if (final_state_) {
      final_state_->prepare(n_particles, units);
    }
    if (trajectory_) {
      trajectory_->prepare(n_particles, units);
    }
  }

  void stage_if_due(
      runstep_t step,
      const system::state::DeviceParticles& particles,
      const system::state::DeviceForces& forces,
      const system::geometry::BoxGeometry& box,
      const forcefield::ForceEvalResult& force_result,
      cudaStream_t dynamics_stream,
      cudaStream_t transfer_stream) {
    writer_.rethrow_if_failed();
    if (thermo_) {
      thermo_->stage_if_due(
          step,
          particles,
          box,
          force_result,
          dynamics_stream,
          transfer_stream);
    }
    if (trajectory_) {
      trajectory_->stage_if_due(
          step,
          particles,
          forces,
          box,
          dynamics_stream,
          transfer_stream);
    }
  }

  void poll_ready() {
    writer_.rethrow_if_failed();
    if (thermo_) {
      thermo_->poll_ready();
    }
    if (trajectory_) {
      trajectory_->poll_ready();
    }
  }

  void begin_run(const log::LogRunStartSummary& summary) {
    writer_.rethrow_if_failed();
    if (log_) {
      log_->submit_run_start(summary);
    }
  }

  void flush() {
    OutputErrorCollector errors("output flush");
    collect_pending_output_errors(errors);
    collect_writer_errors(errors);
    collect_file_flush_errors(errors);
    errors.throw_if_any();
  }

  void finish_run(
      const log::LogRunEndSummary& summary,
      runstep_t step,
      const system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box) {
    OutputErrorCollector errors("output finish");
    if (final_state_) {
      errors.collect("final-state output", [&]() {
        final_state_->write_end_state(step, particles, box);
      });
    }
    collect_pending_output_errors(errors);
    if (log_) {
      errors.collect("log run end", [&]() {
        log_->submit_run_end(summary);
      });
    }
    collect_writer_errors(errors);
    collect_file_flush_errors(errors);
    errors.throw_if_any();
  }

 private:
  void collect_pending_output_errors(OutputErrorCollector& errors) {
    if (thermo_) {
      errors.collect("thermo drain", [&]() {
        thermo_->drain_pending();
      });
    }
    if (trajectory_) {
      errors.collect("trajectory drain", [&]() {
        trajectory_->drain_pending();
      });
    }
  }

  void collect_writer_errors(OutputErrorCollector& errors) {
    const std::vector<std::exception_ptr> writer_errors =
        writer_.flush_and_collect_errors();
    for (const std::exception_ptr& error : writer_errors) {
      errors.add("host writer", error);
    }
  }

  void collect_file_flush_errors(OutputErrorCollector& errors) {
    if (thermo_) {
      errors.collect("thermo flush", [&]() {
        thermo_->flush_file();
      });
    }
    if (trajectory_) {
      errors.collect("trajectory flush", [&]() {
        trajectory_->flush_file();
      });
    }
    if (log_) {
      errors.collect("log flush", [&]() {
        log_->flush_file();
      });
    }
  }

  OutputDemand demand_;
  HostWriter writer_;
  std::optional<log::LogOutput> log_;
  std::optional<thermo::ThermoOutput> thermo_;
  std::optional<snapshot::FinalStateOutput> final_state_;
  std::optional<trajectory::TrajectoryOutput> trajectory_;
};

OutputSystem::OutputSystem(const input::OutputSpec& output, OutputDemand demand)
    : impl_(std::make_unique<Impl>(output, demand)) {}

OutputSystem::~OutputSystem() = default;
OutputSystem::OutputSystem(OutputSystem&&) noexcept = default;
OutputSystem& OutputSystem::operator=(OutputSystem&&) noexcept = default;

bool OutputSystem::has_output() const noexcept {
  return impl_->has_output();
}

bool OutputSystem::needs_force_state(runstep_t step) const noexcept {
  return impl_->needs_force_state(step);
}

forcefield::ForceEvalRequest OutputSystem::max_force_request() const noexcept {
  return impl_->max_force_request();
}

forcefield::ForceEvalRequest OutputSystem::force_request(
    runstep_t step) const noexcept {
  return impl_->force_request(step);
}

void OutputSystem::prepare(
    index_t n_particles,
    const system::units::UnitSystem& units,
    const forcefield::ForceEvalObservableLayout& layout) {
  impl_->prepare(n_particles, units, layout);
}

void OutputSystem::stage_if_due(
    runstep_t step,
    const system::state::DeviceParticles& particles,
    const system::state::DeviceForces& forces,
    const system::geometry::BoxGeometry& box,
    const forcefield::ForceEvalResult& force_result,
    cudaStream_t dynamics_stream,
    cudaStream_t transfer_stream) {
  impl_->stage_if_due(
      step,
      particles,
      forces,
      box,
      force_result,
      dynamics_stream,
      transfer_stream);
}

void OutputSystem::poll_ready() {
  impl_->poll_ready();
}

void OutputSystem::begin_run(const log::LogRunStartSummary& summary) {
  impl_->begin_run(summary);
}

void OutputSystem::flush() {
  impl_->flush();
}

void OutputSystem::finish_run(
    const log::LogRunEndSummary& summary,
    runstep_t step,
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box) {
  impl_->finish_run(summary, step, particles, box);
}

}  // namespace simulation::output
}  // namespace beads
