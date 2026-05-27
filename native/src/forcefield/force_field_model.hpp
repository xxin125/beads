#pragma once

#include <beads/core/types.hpp>
#include <forcefield/angle/angle_model.hpp>
#include <forcefield/bond/bond_model.hpp>
#include <forcefield/dihedral/dihedral_model.hpp>
#include <forcefield/force_eval.cuh>
#include <input/native_spec.hpp>
#include <forcefield/pair/pair_model.hpp>

#include <cuda_runtime.h>

#include <memory>

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
class HostState;
class TagToSlotMap;
}
namespace forcefield {

class ForceFieldModel {
 public:
  ForceFieldModel(
      const input::ForceFieldSpec& forcefield,
      const input::SystemSpec& system);
  ForceFieldModel(
      const input::ForceFieldSpec& forcefield,
      const system::state::HostState& host_state);

  const pair::PairModel& pair() const noexcept { return *pair_; }
  real_t max_cutoff() const noexcept { return pair_->max_cutoff(); }
  bool requires_tag_to_slot_map() const noexcept {
    return bond_ != nullptr || angle_ != nullptr || dihedral_ != nullptr;
  }
  ForceEvalObservableLayout observable_layout(
      index_t n_particles,
      const ForceEvalRequest& request) const;
  // Force evaluation refreshes slot-ordered DeviceForces for the current
  // DeviceParticles order. The required pair model overwrites base forces first;
  // optional listed models then add tag-resolved contributions.
  void compute_forces(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const simulation::neighbor::NeighborList& neighbor_list,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream = nullptr,
      const system::state::TagToSlotMap* tag_to_slot_map = nullptr) const;
  void evaluate_forces(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const simulation::neighbor::NeighborList& neighbor_list,
      const system::geometry::BoxGeometry& box,
      const ForceEvalRequest& request,
      const ForceObservableBuffers& buffers,
      cudaStream_t stream = nullptr,
      const system::state::TagToSlotMap* tag_to_slot_map = nullptr) const;

 private:
  const system::state::TagToSlotMap& require_tag_to_slot_map(
      const system::state::TagToSlotMap* tag_to_slot_map) const;

  type_id_t active_type_count_ = 0;
  std::unique_ptr<pair::PairModel> pair_;
  std::unique_ptr<bond::BondModel> bond_;
  std::unique_ptr<angle::AngleModel> angle_;
  std::unique_ptr<dihedral::DihedralModel> dihedral_;
};

}  // namespace forcefield
}  // namespace beads
