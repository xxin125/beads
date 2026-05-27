#include "force_field_model.hpp"

#include <forcefield/angle/angle_factory.hpp>
#include <forcefield/bond/bond_factory.hpp>
#include <forcefield/dihedral/dihedral_factory.hpp>
#include <forcefield/pair/pair_factory.hpp>
#include <system/state/host_state.hpp>
#include <system/state/tag_to_slot_map.cuh>

namespace beads {
namespace forcefield {
namespace {

type_id_t count_active_types(const system::state::HostState& host_state) {
  type_id_t max_type = 0;
  const auto& particles = host_state.particles();
  for (index_t particle = 0; particle < particles.n_particles; ++particle) {
    if (particles.types[particle] > max_type) {
      max_type = particles.types[particle];
    }
  }
  return max_type;
}

}  // namespace

ForceFieldModel::ForceFieldModel(
    const input::ForceFieldSpec& forcefield,
    const input::SystemSpec& system)
    : ForceFieldModel(forcefield, system::state::HostState(system)) {}

ForceFieldModel::ForceFieldModel(
    const input::ForceFieldSpec& forcefield,
    const system::state::HostState& host_state)
    : active_type_count_(count_active_types(host_state)),
      pair_(pair::create_pair_model(forcefield, active_type_count_)),
      bond_(bond::create_bond_model(forcefield, host_state)),
      angle_(angle::create_angle_model(forcefield, host_state)),
      dihedral_(dihedral::create_dihedral_model(forcefield, host_state)) {}

ForceEvalObservableLayout ForceFieldModel::observable_layout(
    index_t n_particles,
    const ForceEvalRequest& request) const {
  ForceEvalObservableLayout layout =
      pair_->observable_layout(
          n_particles,
          component_force_request(
              request,
              ForceObservable::PairPotentialEnergy));
  if (bond_) {
    const ForceEvalObservableLayout bond_layout =
        bond_->observable_layout(
            component_force_request(
                request,
                ForceObservable::BondPotentialEnergy));
    layout.bond_pe_partial_count += bond_layout.bond_pe_partial_count;
    layout.global_virial_partial_count +=
        bond_layout.global_virial_partial_count;
  }
  if (angle_) {
    const ForceEvalObservableLayout angle_layout =
        angle_->observable_layout(
            component_force_request(
                request,
                ForceObservable::AnglePotentialEnergy));
    layout.angle_pe_partial_count += angle_layout.angle_pe_partial_count;
    layout.global_virial_partial_count +=
        angle_layout.global_virial_partial_count;
  }
  if (dihedral_) {
    const ForceEvalObservableLayout dihedral_layout =
        dihedral_->observable_layout(
            component_force_request(
                request,
                ForceObservable::DihedralPotentialEnergy));
    layout.dihedral_pe_partial_count +=
        dihedral_layout.dihedral_pe_partial_count;
    layout.global_virial_partial_count +=
        dihedral_layout.global_virial_partial_count;
  }
  return layout;
}

void ForceFieldModel::compute_forces(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const simulation::neighbor::NeighborList& neighbor_list,
    const system::geometry::BoxGeometry& box,
    cudaStream_t stream,
    const system::state::TagToSlotMap* tag_to_slot_map) const {
  pair_->compute_forces(particles, forces, neighbor_list, box, stream);
  if (bond_) {
    bond_->add_forces(
        particles,
        forces,
        require_tag_to_slot_map(tag_to_slot_map),
        box,
        stream);
  }
  if (angle_) {
    angle_->add_forces(
        particles,
        forces,
        require_tag_to_slot_map(tag_to_slot_map),
        box,
        stream);
  }
  if (dihedral_) {
    dihedral_->add_forces(
        particles,
        forces,
        require_tag_to_slot_map(tag_to_slot_map),
        box,
        stream);
  }
}

void ForceFieldModel::evaluate_forces(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const simulation::neighbor::NeighborList& neighbor_list,
    const system::geometry::BoxGeometry& box,
    const ForceEvalRequest& request,
    const ForceObservableBuffers& buffers,
    cudaStream_t stream,
    const system::state::TagToSlotMap* tag_to_slot_map) const {
  const ForceEvalObservableLayout pair_layout =
      pair_->observable_layout(
          particles.n_particles(),
          component_force_request(
              request,
              ForceObservable::PairPotentialEnergy));
  ForceEvalObservableLayout bond_layout;
  if (bond_) {
    bond_layout = bond_->observable_layout(
        component_force_request(
            request,
            ForceObservable::BondPotentialEnergy));
  }
  ForceEvalObservableLayout angle_layout;
  if (angle_) {
    angle_layout = angle_->observable_layout(
        component_force_request(
            request,
            ForceObservable::AnglePotentialEnergy));
  }
  ForceEvalObservableLayout dihedral_layout;
  if (dihedral_) {
    dihedral_layout =
        dihedral_->observable_layout(
            component_force_request(
                request,
                ForceObservable::DihedralPotentialEnergy));
  }

  const ForceObservableBuffers pair_buffers =
      slice_component_observable_buffers(
          buffers,
          pair_layout,
          ForceObservable::PairPotentialEnergy);
  const ForceObservableBuffers bond_buffers =
      slice_component_observable_buffers(
          buffers,
          bond_layout,
          ForceObservable::BondPotentialEnergy,
          pair_layout.global_virial_partial_count);
  const index_t angle_virial_offset =
      pair_layout.global_virial_partial_count +
      bond_layout.global_virial_partial_count;
  const ForceObservableBuffers angle_buffers =
      slice_component_observable_buffers(
          buffers,
          angle_layout,
          ForceObservable::AnglePotentialEnergy,
          angle_virial_offset);
  const index_t dihedral_virial_offset =
      angle_virial_offset + angle_layout.global_virial_partial_count;
  const ForceObservableBuffers dihedral_buffers =
      slice_component_observable_buffers(
          buffers,
          dihedral_layout,
          ForceObservable::DihedralPotentialEnergy,
          dihedral_virial_offset);

  pair_->evaluate_forces(
      particles,
      forces,
      neighbor_list,
      box,
      component_force_request(
          request,
          ForceObservable::PairPotentialEnergy),
      pair_buffers,
      stream);
  if (bond_) {
    bond_->add_forces_and_observables(
        particles,
        forces,
        require_tag_to_slot_map(tag_to_slot_map),
        box,
        component_force_request(
            request,
            ForceObservable::BondPotentialEnergy),
        bond_buffers,
        stream);
  }
  if (angle_) {
    angle_->add_forces_and_observables(
        particles,
        forces,
        require_tag_to_slot_map(tag_to_slot_map),
        box,
        component_force_request(
            request,
            ForceObservable::AnglePotentialEnergy),
        angle_buffers,
        stream);
  }
  if (dihedral_) {
    dihedral_->add_forces_and_observables(
        particles,
        forces,
        require_tag_to_slot_map(tag_to_slot_map),
        box,
        component_force_request(
            request,
            ForceObservable::DihedralPotentialEnergy),
        dihedral_buffers,
        stream);
  }
}

const system::state::TagToSlotMap& ForceFieldModel::require_tag_to_slot_map(
    const system::state::TagToSlotMap* tag_to_slot_map) const {
  if (tag_to_slot_map == nullptr || !tag_to_slot_map->is_current()) {
    throw std::logic_error(
        "Listed force evaluation requires a current TagToSlotMap.");
  }
  return *tag_to_slot_map;
}

}  // namespace forcefield
}  // namespace beads
