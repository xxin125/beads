#pragma once

#include <forcefield/angle/angle_model.hpp>

#include <beads/core/device_buffer.cuh>

#include <vector>

namespace beads {
namespace forcefield {
namespace angle {

class HarmonicAngleModel final : public AngleModel {
 public:
  static constexpr const char* kStyleName = "harmonic";

  struct AngleEntry {
    index_t tag_i = 0;
    index_t tag_j = 0;
    index_t tag_k = 0;
    type_id_t type = 0;
  };

  struct DeviceCoeff {
    real_t k = real_t{0};
    real_t theta0_rad = real_t{0};
  };

  HarmonicAngleModel();

  index_t angle_count() const noexcept override { return angle_count_; }
  ForceEvalObservableLayout observable_layout(
      const ForceEvalRequest& request) const override;
  void add_forces(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const system::state::TagToSlotMap& tag_to_slot_map,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream = nullptr) const override;
  void add_forces_and_observables(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const system::state::TagToSlotMap& tag_to_slot_map,
      const system::geometry::BoxGeometry& box,
      const ForceEvalRequest& request,
      const ForceObservableBuffers& buffers,
      cudaStream_t stream = nullptr) const override;

 private:
  void read_settings(const input::AngleStyleSpec& angle_style) override;
  void begin_topology(const system::state::HostState& host_state) override;
  void read_coeff(const input::AngleCoeffSpec& angle_coeff) override;
  void finish_configuration() override;

  DeviceCoeff make_coeff(const input::AngleCoeffSpec& coeff) const;
  void upload_entries(const std::vector<AngleEntry>& entries);
  void upload_coeffs(const std::vector<DeviceCoeff>& coeffs);

  std::vector<AngleEntry> host_angles_;
  std::vector<DeviceCoeff> host_coeffs_;
  std::vector<bool> coeff_filled_;
  index_t angle_count_ = 0;
  DeviceBuffer<AngleEntry> angles_;
  DeviceBuffer<DeviceCoeff> coeffs_;
};

}  // namespace angle
}  // namespace forcefield
}  // namespace beads
