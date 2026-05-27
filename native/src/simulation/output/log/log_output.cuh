#pragma once

#include <beads/core/types.hpp>

#include <array>
#include <memory>
#include <optional>
#include <string>

namespace beads {
namespace simulation::output {
class HostWriter;
namespace thermo {
struct ThermoLogRow;
}
}  // namespace simulation::output

namespace simulation::output::log {

struct LogOutputConfig {
  std::string echo;
  std::optional<std::string> prefix;
};

struct LogRunStartSummary {
  std::string units;
  index_t n_particles = 0;
  runstep_t runsteps = 0;
  std::array<real_t, 3> box_lower = {};
  std::array<real_t, 3> box_upper = {};
  std::array<real_t, 3> box_lengths = {};
  std::string dynamics_style;
  std::string thermostat_style = "none";
  real_t cutoff_buffer = real_t{0};
  std::string neighbor_path;
  runstep_t rebuild_check_every = 0;
  runstep_t sort_every_rebuild = 0;
  index_t max_neighbors = 0;
  std::string pair_style;
  std::optional<std::string> bond_style;
  std::optional<std::string> angle_style;
  std::optional<std::string> dihedral_style;
  std::string bonded_exclusion_policy;
  std::optional<index_t> bonded_exclusion_distance;
  index_t bonded_excluded_pair_count = 0;
  index_t bond_count = 0;
  index_t angle_count = 0;
  index_t dihedral_count = 0;
  bool thermo_enabled = false;
  bool final_state_enabled = false;
  bool trajectory_enabled = false;
  std::string log_echo;
};

struct LogRunEndSummary {
  runstep_t runsteps = 0;
  double run_wall_s = 0.0;
  std::optional<double> steps_per_second;
  runstep_t neighbor_build_count = 0;
  runstep_t dangerous_rebuild_count = 0;
};

class LogOutput {
 public:
  LogOutput(LogOutputConfig config, HostWriter& writer);
  ~LogOutput();

  LogOutput(const LogOutput&) = delete;
  LogOutput& operator=(const LogOutput&) = delete;

  bool active() const noexcept;
  const std::string& echo() const noexcept;

  void submit_run_start(LogRunStartSummary summary);
  void write_thermo_row_from_writer(const thermo::ThermoLogRow& row);
  void submit_run_end(LogRunEndSummary summary);
  void flush_file();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace simulation::output::log
}  // namespace beads
