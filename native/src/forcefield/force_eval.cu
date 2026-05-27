#include "force_eval.cuh"

#include <forcefield/force_field_model.hpp>

#include <stdexcept>
#include <string>

namespace beads {
namespace forcefield {
namespace {

ForcePartialView present_partials(
    const real_t* partials,
    index_t count) noexcept {
  if (partials == nullptr || count == 0) {
    return {};
  }
  return ForcePartialView{partials, count};
}

index_t& layout_partial_count(
    ForceEvalObservableLayout& layout,
    ForceObservable observable) {
  switch (observable) {
    case ForceObservable::PairPotentialEnergy:
      return layout.pair_pe_partial_count;
    case ForceObservable::BondPotentialEnergy:
      return layout.bond_pe_partial_count;
    case ForceObservable::AnglePotentialEnergy:
      return layout.angle_pe_partial_count;
    case ForceObservable::DihedralPotentialEnergy:
      return layout.dihedral_pe_partial_count;
    case ForceObservable::GlobalScalarVirial:
      return layout.global_virial_partial_count;
  }
  throw std::invalid_argument("unknown force observable.");
}

index_t layout_partial_count(
    const ForceEvalObservableLayout& layout,
    ForceObservable observable) {
  switch (observable) {
    case ForceObservable::PairPotentialEnergy:
      return layout.pair_pe_partial_count;
    case ForceObservable::BondPotentialEnergy:
      return layout.bond_pe_partial_count;
    case ForceObservable::AnglePotentialEnergy:
      return layout.angle_pe_partial_count;
    case ForceObservable::DihedralPotentialEnergy:
      return layout.dihedral_pe_partial_count;
    case ForceObservable::GlobalScalarVirial:
      return layout.global_virial_partial_count;
  }
  throw std::invalid_argument("unknown force observable.");
}

real_t*& buffer_partials(
    ForceObservableBuffers& buffers,
    ForceObservable observable) {
  switch (observable) {
    case ForceObservable::PairPotentialEnergy:
      return buffers.pair_pe_partials;
    case ForceObservable::BondPotentialEnergy:
      return buffers.bond_pe_partials;
    case ForceObservable::AnglePotentialEnergy:
      return buffers.angle_pe_partials;
    case ForceObservable::DihedralPotentialEnergy:
      return buffers.dihedral_pe_partials;
    case ForceObservable::GlobalScalarVirial:
      return buffers.global_virial_partials;
  }
  throw std::invalid_argument("unknown force observable.");
}

real_t* buffer_partials(
    const ForceObservableBuffers& buffers,
    ForceObservable observable) {
  switch (observable) {
    case ForceObservable::PairPotentialEnergy:
      return buffers.pair_pe_partials;
    case ForceObservable::BondPotentialEnergy:
      return buffers.bond_pe_partials;
    case ForceObservable::AnglePotentialEnergy:
      return buffers.angle_pe_partials;
    case ForceObservable::DihedralPotentialEnergy:
      return buffers.dihedral_pe_partials;
    case ForceObservable::GlobalScalarVirial:
      return buffers.global_virial_partials;
  }
  throw std::invalid_argument("unknown force observable.");
}

index_t& buffer_partial_count(
    ForceObservableBuffers& buffers,
    ForceObservable observable) {
  switch (observable) {
    case ForceObservable::PairPotentialEnergy:
      return buffers.pair_pe_partial_count;
    case ForceObservable::BondPotentialEnergy:
      return buffers.bond_pe_partial_count;
    case ForceObservable::AnglePotentialEnergy:
      return buffers.angle_pe_partial_count;
    case ForceObservable::DihedralPotentialEnergy:
      return buffers.dihedral_pe_partial_count;
    case ForceObservable::GlobalScalarVirial:
      return buffers.global_virial_partial_count;
  }
  throw std::invalid_argument("unknown force observable.");
}

index_t buffer_partial_count(
    const ForceObservableBuffers& buffers,
    ForceObservable observable) {
  switch (observable) {
    case ForceObservable::PairPotentialEnergy:
      return buffers.pair_pe_partial_count;
    case ForceObservable::BondPotentialEnergy:
      return buffers.bond_pe_partial_count;
    case ForceObservable::AnglePotentialEnergy:
      return buffers.angle_pe_partial_count;
    case ForceObservable::DihedralPotentialEnergy:
      return buffers.dihedral_pe_partial_count;
    case ForceObservable::GlobalScalarVirial:
      return buffers.global_virial_partial_count;
  }
  throw std::invalid_argument("unknown force observable.");
}

}  // namespace

ForceEvalRequest make_force_eval_request(
    std::initializer_list<ForceObservable> observables) noexcept {
  ForceEvalRequest request;
  for (const ForceObservable observable : observables) {
    request.add(observable);
  }
  return request;
}

ForceEvalRequest component_force_request(
    const ForceEvalRequest& request,
    ForceObservable component_pe) noexcept {
  ForceEvalRequest result;
  if (request.has(component_pe)) {
    result.add(component_pe);
  }
  if (request.has(ForceObservable::GlobalScalarVirial)) {
    result.add(ForceObservable::GlobalScalarVirial);
  }
  return result;
}

ForceEvalObservableLayout make_component_observable_layout(
    const ForceEvalRequest& request,
    ForceObservable component_pe,
    index_t partial_count,
    const char* component_label) {
  const ForceEvalRequest supported = make_force_eval_request(
      {component_pe, ForceObservable::GlobalScalarVirial});
  if (!request.is_subset_of(supported)) {
    throw std::invalid_argument(
        std::string(component_label) +
        " does not support requested force observables.");
  }
  ForceEvalObservableLayout layout;
  if (request.has(component_pe)) {
    layout_partial_count(layout, component_pe) = partial_count;
  }
  if (request.has(ForceObservable::GlobalScalarVirial)) {
    layout.global_virial_partial_count = partial_count;
  }
  return layout;
}

ForceObservableBuffers slice_component_observable_buffers(
    const ForceObservableBuffers& buffers,
    const ForceEvalObservableLayout& layout,
    ForceObservable component_pe,
    index_t global_virial_offset) {
  ForceObservableBuffers result;
  const index_t component_count = layout_partial_count(layout, component_pe);
  if (component_count != 0) {
    buffer_partials(result, component_pe) =
        buffer_partials(buffers, component_pe);
    buffer_partial_count(result, component_pe) = component_count;
  }
  if (layout.global_virial_partial_count != 0) {
    result.global_virial_partials =
        buffers.global_virial_partials == nullptr
            ? nullptr
            : buffers.global_virial_partials + global_virial_offset;
    result.global_virial_partial_count = layout.global_virial_partial_count;
  }
  return result;
}

void require_observable_buffer_shape(
    const ForceObservableBuffers& buffers,
    ForceObservable observable,
    index_t expected_count,
    const char* error_message) {
  if (buffer_partials(buffers, observable) == nullptr ||
      buffer_partial_count(buffers, observable) != expected_count) {
    throw std::logic_error(error_message);
  }
}

void ForceEvalWorkspace::prepare(
    const ForceFieldModel& force_field,
    index_t n_particles,
    const ForceEvalRequest& max_request) {
  // Observable capability validation happens while preparing the run workspace;
  // per-step evaluation assumes requests are subsets of this max request.
  layout_ = force_field.observable_layout(n_particles, max_request);
  if (max_request.has(ForceObservable::PairPotentialEnergy)) {
    if (layout_.pair_pe_partial_count == 0) {
      throw std::logic_error(
          "pair potential energy observable requested without partial capacity.");
    }
    pair_pe_partials_.resize(
        layout_.pair_pe_partial_count);
  } else {
    pair_pe_partials_.resize(0);
  }
  if (max_request.has(ForceObservable::BondPotentialEnergy) &&
      layout_.bond_pe_partial_count != 0) {
    bond_pe_partials_.resize(
        layout_.bond_pe_partial_count);
  } else {
    bond_pe_partials_.resize(0);
  }
  if (max_request.has(ForceObservable::AnglePotentialEnergy) &&
      layout_.angle_pe_partial_count != 0) {
    angle_pe_partials_.resize(
        layout_.angle_pe_partial_count);
  } else {
    angle_pe_partials_.resize(0);
  }
  if (max_request.has(ForceObservable::DihedralPotentialEnergy) &&
      layout_.dihedral_pe_partial_count != 0) {
    dihedral_pe_partials_.resize(
        layout_.dihedral_pe_partial_count);
  } else {
    dihedral_pe_partials_.resize(0);
  }
  if (max_request.has(ForceObservable::GlobalScalarVirial)) {
    if (layout_.global_virial_partial_count == 0) {
      throw std::logic_error(
          "global scalar virial observable requested without partial capacity.");
    }
    global_virial_partials_.resize(
        layout_.global_virial_partial_count);
  } else {
    global_virial_partials_.resize(0);
  }
}

ForceObservableBuffers ForceEvalWorkspace::buffers_for(
    const ForceEvalRequest& request) {
  ForceObservableBuffers buffers;
  if (request.has(ForceObservable::PairPotentialEnergy)) {
    if (pair_pe_partials_.empty()) {
      throw std::logic_error(
          "pair potential energy observable workspace was not prepared.");
    }
    buffers.pair_pe_partials =
        pair_pe_partials_.data();
    buffers.pair_pe_partial_count =
        layout_.pair_pe_partial_count;
  }
  if (request.has(ForceObservable::BondPotentialEnergy)) {
    buffers.bond_pe_partials =
        bond_pe_partials_.data();
    buffers.bond_pe_partial_count =
        layout_.bond_pe_partial_count;
  }
  if (request.has(ForceObservable::AnglePotentialEnergy)) {
    buffers.angle_pe_partials =
        angle_pe_partials_.data();
    buffers.angle_pe_partial_count =
        layout_.angle_pe_partial_count;
  }
  if (request.has(ForceObservable::DihedralPotentialEnergy)) {
    buffers.dihedral_pe_partials =
        dihedral_pe_partials_.data();
    buffers.dihedral_pe_partial_count =
        layout_.dihedral_pe_partial_count;
  }
  if (request.has(ForceObservable::GlobalScalarVirial)) {
    if (global_virial_partials_.empty()) {
      throw std::logic_error(
          "global scalar virial observable workspace was not prepared.");
    }
    buffers.global_virial_partials =
        global_virial_partials_.data();
    buffers.global_virial_partial_count =
        layout_.global_virial_partial_count;
  }
  return buffers;
}

ForceEvaluator::ForceEvaluator(
    const ForceFieldModel& force_field,
    index_t n_particles,
    const ForceEvalRequest& max_request)
    : force_field_(&force_field) {
  workspace_.prepare(force_field, n_particles, max_request);
}

ForceEvalResult ForceEvaluator::evaluate(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const simulation::neighbor::NeighborList& neighbor_list,
    const system::geometry::BoxGeometry& box,
    const ForceEvalRequest& request,
    const system::state::TagToSlotMap* tag_to_slot_map,
    cudaStream_t stream) {
  if (request.empty()) {
    force_field_->compute_forces(
        particles,
        forces,
        neighbor_list,
        box,
        stream,
        tag_to_slot_map);
    return {};
  }

  const ForceObservableBuffers buffers = workspace_.buffers_for(request);
  force_field_->evaluate_forces(
      particles,
      forces,
      neighbor_list,
      box,
      request,
      buffers,
      stream,
      tag_to_slot_map);

  ForceEvalResult result;
  result.pair_pe = present_partials(
      buffers.pair_pe_partials,
      buffers.pair_pe_partial_count);
  result.bond_pe = present_partials(
      buffers.bond_pe_partials,
      buffers.bond_pe_partial_count);
  result.angle_pe = present_partials(
      buffers.angle_pe_partials,
      buffers.angle_pe_partial_count);
  result.dihedral_pe = present_partials(
      buffers.dihedral_pe_partials,
      buffers.dihedral_pe_partial_count);
  result.global_virial = present_partials(
      buffers.global_virial_partials,
      buffers.global_virial_partial_count);
  return result;
}

}  // namespace forcefield
}  // namespace beads
