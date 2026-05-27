#pragma once

#include <beads/core/device_buffer.cuh>
#include <beads/core/types.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>

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
class TagToSlotMap;
}
namespace forcefield {

class ForceFieldModel;

enum class ForceObservable : std::uint32_t {
  PairPotentialEnergy = 1u << 0,
  BondPotentialEnergy = 1u << 1,
  AnglePotentialEnergy = 1u << 2,
  DihedralPotentialEnergy = 1u << 3,
  GlobalScalarVirial = 1u << 4,
};

// Optional observable channels requested for one force evaluation.
struct ForceEvalRequest {
  bool empty() const noexcept { return mask_ == 0; }

  bool has(ForceObservable observable) const noexcept {
    return (mask_ & to_mask(observable)) != 0;
  }

  bool is_subset_of(const ForceEvalRequest& supported) const noexcept {
    return (mask_ & ~supported.mask_) == 0;
  }

  void add(ForceObservable observable) noexcept {
    mask_ |= to_mask(observable);
  }

  void merge(const ForceEvalRequest& other) noexcept {
    mask_ |= other.mask_;
  }

 private:
  static constexpr std::uint32_t to_mask(ForceObservable observable) noexcept {
    return static_cast<std::uint32_t>(observable);
  }

  std::uint32_t mask_ = 0;
};

// Writable ForceEvaluator workspace view passed into force models.
struct ForceObservableBuffers {
  real_t* pair_pe_partials = nullptr;
  index_t pair_pe_partial_count = 0;
  real_t* bond_pe_partials = nullptr;
  index_t bond_pe_partial_count = 0;
  real_t* angle_pe_partials = nullptr;
  index_t angle_pe_partial_count = 0;
  real_t* dihedral_pe_partials = nullptr;
  index_t dihedral_pe_partial_count = 0;
  real_t* global_virial_partials = nullptr;
  index_t global_virial_partial_count = 0;
};

struct ForcePartialView {
  const real_t* data = nullptr;
  index_t count = 0;

  bool present() const noexcept {
    return data != nullptr && count != 0;
  }
};

// Borrowed observable views produced by one force evaluation. A view is present
// when it has a readable pointer/count pair, not merely because the observable
// was requested. Pointers refer to ForceEvaluator-owned workspace and remain
// valid only until the next evaluation overwrites that workspace or the
// evaluator is destroyed.
struct ForceEvalResult {
  ForcePartialView pair_pe;
  ForcePartialView bond_pe;
  ForcePartialView angle_pe;
  ForcePartialView dihedral_pe;
  ForcePartialView global_virial;
};

// Prepared observable capacity/layout exposed to output reducers.
struct ForceEvalObservableLayout {
  index_t pair_pe_partial_count = 0;
  index_t bond_pe_partial_count = 0;
  index_t angle_pe_partial_count = 0;
  index_t dihedral_pe_partial_count = 0;
  index_t global_virial_partial_count = 0;
};

ForceEvalRequest make_force_eval_request(
    std::initializer_list<ForceObservable> observables) noexcept;

ForceEvalRequest component_force_request(
    const ForceEvalRequest& request,
    ForceObservable component_pe) noexcept;

ForceEvalObservableLayout make_component_observable_layout(
    const ForceEvalRequest& request,
    ForceObservable component_pe,
    index_t partial_count,
    const char* component_label);

ForceObservableBuffers slice_component_observable_buffers(
    const ForceObservableBuffers& buffers,
    const ForceEvalObservableLayout& layout,
    ForceObservable component_pe,
    index_t global_virial_offset = 0);

void require_observable_buffer_shape(
    const ForceObservableBuffers& buffers,
    ForceObservable observable,
    index_t expected_count,
    const char* error_message);

class ForceEvalWorkspace {
 public:
  void prepare(
      const ForceFieldModel& force_field,
      index_t n_particles,
      const ForceEvalRequest& max_request);

  ForceObservableBuffers buffers_for(const ForceEvalRequest& request);
  const ForceEvalObservableLayout& observable_layout() const noexcept {
    return layout_;
  }

 private:
  ForceEvalObservableLayout layout_;
  DeviceBuffer<real_t> pair_pe_partials_;
  DeviceBuffer<real_t> bond_pe_partials_;
  DeviceBuffer<real_t> angle_pe_partials_;
  DeviceBuffer<real_t> dihedral_pe_partials_;
  DeviceBuffer<real_t> global_virial_partials_;
};

class ForceEvaluator {
 public:
  ForceEvaluator(
      const ForceFieldModel& force_field,
      index_t n_particles,
      const ForceEvalRequest& max_request);

  const ForceEvalObservableLayout& observable_layout() const noexcept {
    return workspace_.observable_layout();
  }

  ForceEvalResult evaluate(
      const system::state::DeviceParticles& particles,
      system::state::DeviceForces& forces,
      const simulation::neighbor::NeighborList& neighbor_list,
      const system::geometry::BoxGeometry& box,
      const ForceEvalRequest& request,
      const system::state::TagToSlotMap* tag_to_slot_map = nullptr,
      cudaStream_t stream = nullptr);

 private:
  const ForceFieldModel* force_field_ = nullptr;
  ForceEvalWorkspace workspace_;
};

}  // namespace forcefield
}  // namespace beads
