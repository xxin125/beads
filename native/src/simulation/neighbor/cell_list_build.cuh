#pragma once

#include <beads/core/device_buffer.cuh>
#include <beads/core/types.hpp>
#include <simulation/neighbor/neighbor_plan.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace beads {
namespace system::geometry {
struct BoxGeometry;
}  // namespace system::geometry
namespace system::state {
class DeviceParticles;
class DeviceParticleReorderScratch;
}  // namespace system::state
namespace simulation::neighbor {

struct CellListBuildDataView {
  index_t n_particles = 0;
  index_t n_cells = 0;
  index_t* particle_cell = nullptr;
  index_t* cell_count = nullptr;
  index_t* cell_offset = nullptr;
  index_t* cell_particle = nullptr;
};

struct CellListBuildDataConstView {
  index_t n_particles = 0;
  index_t n_cells = 0;
  const index_t* particle_cell = nullptr;
  const index_t* cell_count = nullptr;
  const index_t* cell_offset = nullptr;
  const index_t* cell_particle = nullptr;
};

class CellListBuildData {
 public:
  CellListBuildData() = default;
  CellListBuildData(
      index_t n_particles,
      const NeighborCellGeometry& cell_geometry,
      cudaStream_t stream = nullptr);

  CellListBuildData(const CellListBuildData&) = delete;
  CellListBuildData& operator=(const CellListBuildData&) = delete;
  CellListBuildData(CellListBuildData&& other) noexcept;
  CellListBuildData& operator=(CellListBuildData&& other) noexcept;

  void resize(
      index_t n_particles,
      const NeighborCellGeometry& cell_geometry,
      cudaStream_t stream = nullptr);

  index_t n_particles() const noexcept { return n_particles_; }
  index_t n_cells() const noexcept { return n_cells_; }

  CellListBuildDataView view() noexcept;
  CellListBuildDataConstView view() const noexcept;

 private:
  void configure_scan_workspace(cudaStream_t stream = nullptr);
  void clear_counts(cudaStream_t stream = nullptr);

  friend void build_cell_list(
      CellListBuildData& cell_list_data,
      const system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box,
      const NeighborCellGeometry& cell_geometry,
      cudaStream_t stream);
  friend void reorder_particles_by_cell_list(
      system::state::DeviceParticles& particles,
      system::state::DeviceParticleReorderScratch& particle_scratch,
      CellListBuildData& cell_list_data,
      cudaStream_t stream);

  index_t n_particles_ = 0;
  index_t n_cells_ = 0;
  DeviceBuffer<index_t> particle_cell_;
  DeviceBuffer<index_t> cell_count_;
  DeviceBuffer<index_t> cell_offset_;
  DeviceBuffer<index_t> cell_write_offset_;
  DeviceBuffer<index_t> cell_particle_;
  DeviceBuffer<index_t> particle_cell_scratch_;
  DeviceBuffer<std::byte> scan_workspace_;
  std::size_t scan_workspace_bytes_ = 0;
};

void build_cell_list(
    CellListBuildData& cell_list_data,
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    const NeighborCellGeometry& cell_geometry,
    cudaStream_t stream = nullptr);

void reorder_particles_by_cell_list(
    system::state::DeviceParticles& particles,
    system::state::DeviceParticleReorderScratch& particle_scratch,
    CellListBuildData& cell_list_data,
    cudaStream_t stream = nullptr);

}  // namespace simulation::neighbor
}  // namespace beads
