#pragma once

#include <beads/core/types.hpp>

#include <memory>
#include <string>

namespace beads {
namespace input {
struct FinalStateOutputSpec;
}
namespace system::geometry {
class BoxGeometry;
}
namespace system::state {
class DeviceParticles;
}
namespace system::units {
struct UnitSystem;
}
namespace simulation::output::snapshot {

struct FinalStateOutputConfig {
  std::string prefix;
};

class FinalStateOutput {
 public:
  explicit FinalStateOutput(FinalStateOutputConfig config);
  ~FinalStateOutput();

  FinalStateOutput(const FinalStateOutput&) = delete;
  FinalStateOutput& operator=(const FinalStateOutput&) = delete;
  FinalStateOutput(FinalStateOutput&&) noexcept;
  FinalStateOutput& operator=(FinalStateOutput&&) noexcept;

  void prepare(
      index_t n_particles,
      const system::units::UnitSystem& units);
  void write_end_state(
      runstep_t step,
      const system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace simulation::output::snapshot
}  // namespace beads
