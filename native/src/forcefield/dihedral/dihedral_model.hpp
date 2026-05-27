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
namespace system::state {
class DeviceForces;
class DeviceParticles;
class HostState;
class TagToSlotMap;
}
namespace forcefield {
namespace dihedral {

class DihedralModel {
 public:
  explicit DihedralModel(std::string style_name);
  virtual ~DihedralModel() = default;

  DihedralModel(const DihedralModel&) = delete;
  DihedralModel& operator=(const DihedralModel&) = delete;
  DihedralModel(DihedralModel&&) = delete;
  DihedralModel& operator=(DihedralModel&&) = delete;

  void configure(
      const input::ForceFieldSpec& forcefield,
      const system::state::HostState& host_state);

  const std::string& style_name() const noexcept { return style_name_; }
  virtual index_t dihedral_count() const noexcept = 0;
  virtual ForceEvalObservableLayout observable_layout(
      const ForceEvalRequest& request) const = 0;
  virtual void add_forces(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const system::state::TagToSlotMap& tag_to_slot_map,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream = nullptr) const = 0;
  virtual void add_forces_and_observables(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const system::state::TagToSlotMap& tag_to_slot_map,
      const system::geometry::BoxGeometry& box,
      const ForceEvalRequest& request,
      const ForceObservableBuffers& buffers,
      cudaStream_t stream = nullptr) const = 0;

 protected:
  std::string dihedral_label() const;
  void require_exact_parameter_keys(
      const input::StyleParamMap& params,
      std::initializer_list<const char*> required_keys,
      std::string_view owner) const;
  real_t require_nonnegative_real_parameter(
      const input::StyleParamMap& params,
      const char* key,
      std::string_view owner) const;
  std::int64_t require_integer_parameter(
      const input::StyleParamMap& params,
      const char* key,
      std::string_view owner) const;

 private:
  void require_not_configured() const;
  void require_dihedral_style(const input::ForceFieldSpec& forcefield) const;
  void require_dihedral_coeff_style(
      const input::DihedralCoeffSpec& dihedral_coeff) const;

  virtual void read_settings(const input::DihedralStyleSpec& dihedral_style) = 0;
  virtual void begin_topology(
      const system::state::HostState& host_state) = 0;
  virtual void read_coeff(const input::DihedralCoeffSpec& dihedral_coeff) = 0;
  virtual void finish_configuration() = 0;

  std::string style_name_;
  bool configured_ = false;
};

}  // namespace dihedral
}  // namespace forcefield
}  // namespace beads
