#include "neighbor_system.hpp"

#include <simulation/neighbor/full/direct_neighbor_builder.cuh>
#include <simulation/neighbor/full/regular_cell_neighbor_builder.cuh>
#include <simulation/neighbor/full/small_dimension_neighbor_builder.cuh>
#include <simulation/neighbor/cell_list_build.cuh>

#include <sstream>
#include <stdexcept>

namespace beads {
namespace simulation::neighbor {

NeighborSystem::NeighborSystem(
    const input::NeighborSpec& neighbor,
    real_t force_cutoff,
    const system::state::HostState& host_state,
    std::optional<index_t> bonded_exclusion_distance)
    : NeighborSystem(
          neighbor,
          force_cutoff,
          host_state,
          bonded_exclusion_distance.has_value(),
          bonded_exclusion_distance) {}

NeighborSystem::NeighborSystem(
    const input::NeighborSpec& neighbor,
    real_t force_cutoff,
    const system::state::HostState& host_state,
    bool requires_tag_to_slot_map,
    std::optional<index_t> bonded_exclusion_distance)
    : plan_(
          neighbor,
          force_cutoff,
          host_state.particles().n_particles,
          host_state.box()),
      neighbor_list_(plan_.n_particles, plan_.max_neighbors),
      uses_tag_to_slot_map_(
          requires_tag_to_slot_map || bonded_exclusion_distance.has_value()) {
  if (bonded_exclusion_distance) {
    exclusion_source_.emplace(system::topology::compile_bond_graph_exclusion_source(
        host_state.topology(),
        host_state.particles().n_particles,
        *bonded_exclusion_distance));
  }
  if (uses_tag_to_slot_map_) {
    tag_to_slot_map_.prepare(host_state.particles().n_particles);
  }
  if (plan_.uses_cell_list) {
    cell_list_data_.resize(plan_.n_particles, plan_.cell_geometry);
    reorder_scratch_.resize(plan_.n_particles);
  }
}

const system::state::TagToSlotMap* NeighborSystem::current_tag_to_slot_map()
    const noexcept {
  if (!uses_tag_to_slot_map_ || !tag_to_slot_map_.is_current()) {
    return nullptr;
  }
  return &tag_to_slot_map_;
}

index_t NeighborSystem::excluded_pair_count() const noexcept {
  return exclusion_source_ ? exclusion_source_->unique_pair_count() : index_t{0};
}

const char* NeighborSystem::path_name() const noexcept {
  return neighbor_build_path_name(plan_.path_kind);
}

void NeighborSystem::initialize_for_run(
    system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    cudaStream_t stream) {
  rebuild_tracker_.resize(particles.n_particles(), stream);
  rebuild_now(particles, box, stream);
}

bool NeighborSystem::update_after_position_change(
    system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    runstep_t physical_step,
    cudaStream_t stream) {
  if (rebuild_after_every_position_update()) {
    rebuild_now(particles, box, stream);
    return true;
  }

  if (!should_check_rebuild(physical_step)) {
    return false;
  }

  const bool first_scheduled_check = !scheduled_check_seen_since_build_;
  scheduled_check_seen_since_build_ = true;
  if (!rebuild_tracker_.exceeds_displacement_threshold(
          particles, box, rebuild_threshold_sq(), stream)) {
    return false;
  }

  if (plan_.rebuild_check_every > 1 && first_scheduled_check) {
    ++dangerous_rebuild_count_;
  }
  rebuild_now(particles, box, stream);
  return true;
}

void NeighborSystem::rebuild_now(
    system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    cudaStream_t stream) {
  ++build_count_;

  if (plan_.path_kind == NeighborBuildPathKind::Direct) {
    // Direct builds do not reorder, but the initial build still materializes
    // slot identity for listed forces and bonded exclusions.
    const auto exclusions =
        refresh_slot_identity_and_exclusions_if_needed(particles, stream);
    full::build_direct_neighbor_list(
        particles, box, plan_, neighbor_list_, exclusions, stream);
  } else {
    build_cell_list(cell_list_data_, particles, box, plan_.cell_geometry, stream);

    if (should_reorder_on_rebuild(build_count_)) {
      reorder_particles_by_cell_list(
          particles, reorder_scratch_, cell_list_data_, stream);
      tag_to_slot_map_.mark_dirty();
    }

    const auto exclusions =
        refresh_slot_identity_and_exclusions_if_needed(particles, stream);
    switch (plan_.path_kind) {
      case NeighborBuildPathKind::RegularCell:
        full::build_regular_cell_neighbor_list_from_cell_list(
            particles, box, plan_, cell_list_data_, neighbor_list_, exclusions, stream);
        break;
      case NeighborBuildPathKind::SmallDimension:
        full::build_small_dimension_neighbor_list_from_cell_list(
            particles, box, plan_, cell_list_data_, neighbor_list_, exclusions, stream);
        break;
    }
  }

  require_no_overflow(neighbor_list_.download_overflow_flag(stream));
  rebuild_tracker_.capture_reference(particles, box, stream);
  scheduled_check_seen_since_build_ = false;
}

system::topology::SlotExclusionConstView
NeighborSystem::refresh_slot_identity_and_exclusions_if_needed(
    system::state::DeviceParticles& particles,
    cudaStream_t stream) {
  refresh_slot_identity_if_needed(particles, stream);
  return refresh_slot_exclusion_view_if_needed(particles, stream);
}

void NeighborSystem::refresh_slot_identity_if_needed(
    system::state::DeviceParticles& particles,
    cudaStream_t stream) {
  if (uses_tag_to_slot_map_) {
    tag_to_slot_map_.rebuild_if_dirty(particles, stream);
  }
}

system::topology::SlotExclusionConstView
NeighborSystem::refresh_slot_exclusion_view_if_needed(
    system::state::DeviceParticles& particles,
    cudaStream_t stream) {
  if (!exclusion_source_) {
    return {};
  }

  if (slot_exclusions_.slot_generation() != tag_to_slot_map_.generation()) {
    slot_exclusions_.rebuild_from_source(
        *exclusion_source_,
        particles,
        tag_to_slot_map_,
        stream,
        tag_to_slot_map_.generation());
  }
  return slot_exclusions_.view();
}

void NeighborSystem::require_no_overflow(int overflow_flag) const {
  if (overflow_flag != 0) {
    std::ostringstream message;
    message << "NeighborList overflow"
            << "\npath=" << neighbor_build_path_name(plan_.path_kind)
            << "\nn_particles=" << plan_.n_particles
            << "\nmax_neighbors=" << plan_.max_neighbors
            << "\nforce_cutoff=" << plan_.force_cutoff
            << "\ncutoff_buffer=" << plan_.cutoff_buffer
            << "\nsearch_cutoff=" << plan_.search_cutoff
            << ". Increase max_neighbors before consuming this neighbor list.";
    throw std::runtime_error(message.str());
  }
}

bool NeighborSystem::should_reorder_on_rebuild(runstep_t build_count) const noexcept {
  return build_count > 0 &&
      plan_.reorder_every_rebuild > 0 &&
      (build_count % plan_.reorder_every_rebuild) == 0;
}

bool NeighborSystem::should_check_rebuild(runstep_t physical_step) const noexcept {
  return plan_.rebuild_check_every > 0 &&
      (physical_step % plan_.rebuild_check_every) == 0;
}

bool NeighborSystem::rebuild_after_every_position_update() const noexcept {
  return plan_.cutoff_buffer == real_t{0};
}

real_t NeighborSystem::rebuild_threshold_sq() const noexcept {
  const real_t half_buffer = real_t{0.5} * plan_.cutoff_buffer;
  return half_buffer * half_buffer;
}

}  // namespace simulation::neighbor
}  // namespace beads
