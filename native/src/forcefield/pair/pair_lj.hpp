#pragma once

#include <forcefield/pair/dense_type_pair_table.cuh>
#include <forcefield/pair/pair_model.hpp>

namespace beads {
namespace forcefield {
namespace pair {

class LjPairModel final : public PairModel {
 public:
  struct DeviceCoeff {
    real_t sigma2 = real_t{0};
    real_t epsilon4 = real_t{0};
    real_t epsilon24 = real_t{0};
    real_t cutoff_sq = real_t{0};
    real_t shift_energy = real_t{0};
  };

  static constexpr const char* kStyleName = "lj";

  LjPairModel();

  real_t max_cutoff() const noexcept override { return cutoff_; }
  auto device_coeffs() const noexcept { return coeffs_.device_view(); }
  ForceEvalObservableLayout observable_layout(
      index_t n_particles,
      const ForceEvalRequest& request) const override;
  void compute_forces(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const simulation::neighbor::NeighborList& neighbor_list,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream = nullptr) const override;
  void evaluate_forces(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const simulation::neighbor::NeighborList& neighbor_list,
      const system::geometry::BoxGeometry& box,
      const ForceEvalRequest& request,
      const ForceObservableBuffers& buffers,
      cudaStream_t stream = nullptr) const override;

 private:
  static DeviceCoeff pack_device_coeff(
      real_t epsilon,
      real_t sigma,
      real_t cutoff,
      bool shift);
  void read_settings(const input::PairStyleSpec& pair_style) override;
  void begin_coeffs(type_id_t active_type_count) override;
  void read_coeff(const input::PairCoeffSpec& pair_coeff) override;
  void finish_configuration() override;

  real_t cutoff_ = real_t{0};
  bool shift_ = false;
  DenseTypePairTable<DeviceCoeff> coeffs_;
};

}  // namespace pair
}  // namespace forcefield
}  // namespace beads
