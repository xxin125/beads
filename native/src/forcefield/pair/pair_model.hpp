#pragma once

#include <beads/core/types.hpp>
#include <forcefield/force_eval.cuh>
#include <input/native_spec.hpp>

#include <cuda_runtime.h>

#include <initializer_list>
#include <string>
#include <string_view>

namespace beads {
namespace system::geometry {
class BoxGeometry;
}
namespace simulation::neighbor {
class NeighborList;
}
namespace system::state {
class DeviceForces;
class DeviceParticles;
}
namespace forcefield {
namespace pair {

class PairModel {
 public:
  explicit PairModel(std::string style_name);
  virtual ~PairModel() = default;

  PairModel(const PairModel&) = delete;
  PairModel& operator=(const PairModel&) = delete;
  PairModel(PairModel&&) = delete;
  PairModel& operator=(PairModel&&) = delete;

  void configure(
      const input::ForceFieldSpec& forcefield,
      type_id_t active_type_count);

  const std::string& style_name() const noexcept { return style_name_; }
  virtual real_t max_cutoff() const noexcept = 0;
  virtual ForceEvalObservableLayout observable_layout(
      index_t n_particles,
      const ForceEvalRequest& request) const;
  // Pair models own the base force write for every current particle slot.
  // Listed force models run after this and add to the refreshed pair forces.
  virtual void compute_forces(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const simulation::neighbor::NeighborList& neighbor_list,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream = nullptr) const = 0;
  virtual void evaluate_forces(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const simulation::neighbor::NeighborList& neighbor_list,
      const system::geometry::BoxGeometry& box,
      const ForceEvalRequest& request,
      const ForceObservableBuffers& buffers,
      cudaStream_t stream = nullptr) const;

 protected:
  std::string pair_label() const;
  void require_exact_parameter_keys(
      const input::StyleParamMap& params,
      std::initializer_list<const char*> required_keys,
      std::string_view owner) const;
  real_t require_positive_real_parameter(
      const input::StyleParamMap& params,
      const char* key,
      std::string_view owner) const;

 private:
  void require_not_configured() const;
  void require_pair_style(const input::PairStyleSpec& pair_style) const;
  void require_pair_coeff_style(const input::PairCoeffSpec& pair_coeff) const;

  virtual void read_settings(const input::PairStyleSpec& pair_style) = 0;
  virtual void begin_coeffs(type_id_t active_type_count) = 0;
  virtual void read_coeff(const input::PairCoeffSpec& pair_coeff) = 0;
  virtual void finish_configuration() = 0;

  std::string style_name_;
  bool configured_ = false;
};

}  // namespace pair
}  // namespace forcefield
}  // namespace beads
