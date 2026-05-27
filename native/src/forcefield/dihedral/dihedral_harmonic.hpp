#pragma once

#include <forcefield/dihedral/dihedral_model.hpp>

#include <beads/core/device_buffer.cuh>

#include <vector>

namespace beads {
namespace forcefield {
namespace dihedral {

class HarmonicDihedralModel final : public DihedralModel {
 public:
  static constexpr const char* kStyleName = "harmonic";

  struct DihedralEntry {
    index_t tag_i = 0;
    index_t tag_j = 0;
    index_t tag_k = 0;
    index_t tag_l = 0;
    type_id_t type = 0;
  };

  struct DeviceCoeff {
    real_t k = real_t{0};
    int d = 1;
    int n = 0;
  };

  HarmonicDihedralModel();

  index_t dihedral_count() const noexcept override { return dihedral_count_; }
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
  void read_settings(const input::DihedralStyleSpec& dihedral_style) override;
  void begin_topology(const system::state::HostState& host_state) override;
  void read_coeff(const input::DihedralCoeffSpec& dihedral_coeff) override;
  void finish_configuration() override;

  DeviceCoeff make_coeff(const input::DihedralCoeffSpec& coeff) const;
  void upload_entries(const std::vector<DihedralEntry>& entries);
  void upload_coeffs(const std::vector<DeviceCoeff>& coeffs);

  std::vector<DihedralEntry> host_dihedrals_;
  std::vector<DeviceCoeff> host_coeffs_;
  std::vector<bool> coeff_filled_;
  index_t dihedral_count_ = 0;
  DeviceBuffer<DihedralEntry> dihedrals_;
  DeviceBuffer<DeviceCoeff> coeffs_;
};

}  // namespace dihedral
}  // namespace forcefield
}  // namespace beads
