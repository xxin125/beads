#pragma once

#include <cuda_runtime.h>

#include <optional>

#include <beads/core/types.hpp>
#include <input/native_spec.hpp>
#include <system/geometry/box_geometry.hpp>
#include <simulation/neighbor/neighbor_list.cuh>
#include <simulation/neighbor/neighbor_plan.hpp>
#include <simulation/neighbor/rebuild_tracker.cuh>
#include <simulation/neighbor/cell_list_build.cuh>
#include <system/state/device_particles.cuh>
#include <system/state/host_state.hpp>
#include <system/state/tag_to_slot_map.cuh>
#include <system/topology/exclusions.cuh>

namespace beads {
namespace simulation::neighbor {

class NeighborSystem {
 public:
  NeighborSystem(
      const input::NeighborSpec& neighbor,
      real_t force_cutoff,
      const system::state::HostState& host_state,
      std::optional<index_t> bonded_exclusion_distance = std::nullopt);
  NeighborSystem(
      const input::NeighborSpec& neighbor,
      real_t force_cutoff,
      const system::state::HostState& host_state,
      bool requires_tag_to_slot_map,
      std::optional<index_t> bonded_exclusion_distance);

  const NeighborList& neighbor_list() const noexcept { return neighbor_list_; }
  runstep_t build_count() const noexcept { return build_count_; }
  runstep_t dangerous_rebuild_count() const noexcept {
    return dangerous_rebuild_count_;
  }
  const char* path_name() const noexcept;
  index_t excluded_pair_count() const noexcept;
  const system::state::TagToSlotMap* current_tag_to_slot_map() const noexcept;

  void initialize_for_run(
      system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream = nullptr);
  bool update_after_position_change(
      system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box,
      runstep_t physical_step,
      cudaStream_t stream = nullptr);

 private:
  void rebuild_now(
      system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box,
      cudaStream_t stream);
  system::topology::SlotExclusionConstView
  refresh_slot_identity_and_exclusions_if_needed(
      system::state::DeviceParticles& particles,
      cudaStream_t stream);
  void refresh_slot_identity_if_needed(
      system::state::DeviceParticles& particles,
      cudaStream_t stream);
  system::topology::SlotExclusionConstView
  refresh_slot_exclusion_view_if_needed(
      system::state::DeviceParticles& particles,
      cudaStream_t stream);
  void require_no_overflow(int overflow_flag) const;
  bool should_reorder_on_rebuild(runstep_t build_count) const noexcept;
  bool should_check_rebuild(runstep_t physical_step) const noexcept;
  bool rebuild_after_every_position_update() const noexcept;
  real_t rebuild_threshold_sq() const noexcept;

  NeighborPlan plan_;
  NeighborList neighbor_list_;
  CellListBuildData cell_list_data_;
  system::state::DeviceParticleReorderScratch reorder_scratch_;
  NeighborRebuildTracker rebuild_tracker_;
  system::state::TagToSlotMap tag_to_slot_map_;
  bool uses_tag_to_slot_map_ = false;
  std::optional<system::topology::ExclusionSourceRuntime> exclusion_source_;
  system::topology::SlotExclusionRuntime slot_exclusions_;
  runstep_t build_count_ = 0;
  runstep_t dangerous_rebuild_count_ = 0;
  bool scheduled_check_seen_since_build_ = false;
};

}  // namespace simulation::neighbor
}  // namespace beads
