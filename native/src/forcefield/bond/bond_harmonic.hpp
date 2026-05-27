#pragma once

#include <forcefield/bond/bond_model.hpp>

#include <beads/core/device_buffer.cuh>

#include <vector>

namespace beads {
namespace forcefield {
namespace bond {

class HarmonicBondModel final : public BondModel {
 public:
  static constexpr const char* kStyleName = "harmonic";

  struct BondEntry {
    index_t tag_i = 0;
    index_t tag_j = 0;
    type_id_t type = 0;
  };

  struct DeviceCoeff {
    real_t k = real_t{0};
    real_t r0 = real_t{0};
  };

  HarmonicBondModel();

  index_t bond_count() const noexcept override { return bond_count_; }
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
  void read_settings(const input::BondStyleSpec& bond_style) override;
  void begin_topology(const system::state::HostState& host_state) override;
  void read_coeff(const input::BondCoeffSpec& bond_coeff) override;
  void finish_configuration() override;

  DeviceCoeff make_coeff(const input::BondCoeffSpec& coeff) const;

  void upload_entries(const std::vector<BondEntry>& entries);
  void upload_coeffs(const std::vector<DeviceCoeff>& coeffs);

  std::vector<BondEntry> host_bonds_;
  std::vector<DeviceCoeff> host_coeffs_;
  std::vector<bool> coeff_filled_;
  index_t bond_count_ = 0;
  DeviceBuffer<BondEntry> bonds_;
  DeviceBuffer<DeviceCoeff> coeffs_;
};

}  // namespace bond
}  // namespace forcefield
}  // namespace beads
