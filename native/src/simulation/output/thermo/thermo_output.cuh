#pragma once

#include <beads/core/types.hpp>
#include <forcefield/force_eval.cuh>

#include <cuda_runtime.h>

#include <memory>
#include <functional>
#include <string>

namespace beads {
namespace system::state {
class DeviceParticles;
}
namespace system::geometry {
class BoxGeometry;
}
namespace system::units {
struct UnitSystem;
}
namespace simulation::output {
class HostWriter;
}
namespace simulation::output::thermo {

struct ThermoSchema {
  bool include_bond_pe = false;
  bool include_angle_pe = false;
  bool include_dihedral_pe = false;
};

struct ThermoLogRow {
  runstep_t step = 0;
  real_t pe = real_t{0};
  real_t pair_pe = real_t{0};
  real_t bond_pe = real_t{0};
  real_t angle_pe = real_t{0};
  real_t dihedral_pe = real_t{0};
  real_t ke = real_t{0};
  real_t temp = real_t{0};
  real_t press = real_t{0};
  ThermoSchema schema;
};

struct ThermoOutputConfig {
  runstep_t every = 0;
  std::string prefix;
  std::function<void(const ThermoLogRow&)> log_sink;
};

class ThermoOutput {
 public:
  ThermoOutput(ThermoOutputConfig config, HostWriter& writer);
  ~ThermoOutput();

  ThermoOutput(const ThermoOutput&) = delete;
  ThermoOutput& operator=(const ThermoOutput&) = delete;
  ThermoOutput(ThermoOutput&&) noexcept;
  ThermoOutput& operator=(ThermoOutput&&) noexcept;

  bool has_output() const noexcept;
  void prepare(
      index_t n_particles,
      const system::units::UnitSystem& units,
      const forcefield::ForceEvalObservableLayout& layout);
  void stage_if_due(
      runstep_t step,
      const system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box,
      const forcefield::ForceEvalResult& force_result,
      cudaStream_t dynamics_stream,
      cudaStream_t transfer_stream);
  void poll_ready();
  void drain_pending();
  void flush_file();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace simulation::output::thermo
}  // namespace beads
