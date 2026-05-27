#include "cell_list_build.cuh"

#include <beads/core/cuda_check.cuh>
#include <system/geometry/box_geometry_view.cuh>
#include <system/state/device_particles.cuh>

#include <cub/cub.cuh>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <utility>

namespace beads {
namespace simulation::neighbor {
namespace {

constexpr int kCellListBuildBlockSize = 256;
constexpr int kParticleReorderBlockSize = 256;

std::size_t checked_buffer_size(index_t value, const char* context) {
  const auto size = static_cast<std::size_t>(value);
  if (size > std::numeric_limits<std::size_t>::max() / sizeof(index_t)) {
    throw std::overflow_error(context);
  }
  return size;
}

void validate_shape(index_t n_particles, index_t n_cells) {
  if (n_particles == 0) {
    throw std::invalid_argument("CellListBuildData n_particles must be positive.");
  }
  if (n_cells == 0) {
    throw std::invalid_argument(
        "CellListBuildData n_cells must be positive.");
  }
}

int require_cub_int_count(index_t value, const char* context) {
  if (value > static_cast<index_t>(std::numeric_limits<int>::max())) {
    std::ostringstream message;
    message << context << " exceeds CUB int-count capacity.";
    throw std::overflow_error(message.str());
  }
  return static_cast<int>(value);
}

struct ParticlePositionConstView {
  index_t n_particles = 0;
  const real_t* position_x = nullptr;
  const real_t* position_y = nullptr;
  const real_t* position_z = nullptr;
};

struct CellListScatterView {
  index_t n_particles = 0;
  const index_t* particle_cell = nullptr;
  index_t* cell_write_offset = nullptr;
  index_t* cell_particle = nullptr;
};

struct ParticleReorderSourceView {
  index_t n_particles = 0;

  const real_t* position_x = nullptr;
  const real_t* position_y = nullptr;
  const real_t* position_z = nullptr;

  const real_t* velocity_x = nullptr;
  const real_t* velocity_y = nullptr;
  const real_t* velocity_z = nullptr;

  const real_t* mass = nullptr;
  const type_id_t* type = nullptr;
  const index_t* tag = nullptr;
  const index_t* molecule_id = nullptr;

  const image_t* image_x = nullptr;
  const image_t* image_y = nullptr;
  const image_t* image_z = nullptr;
};

ParticlePositionConstView position_view(
    const system::state::DeviceParticles& particles) noexcept {
  return ParticlePositionConstView{
      particles.n_particles(),
      particles.position_x().data(),
      particles.position_y().data(),
      particles.position_z().data()};
}

int cell_list_build_grid_size(index_t n_particles, int block_size) {
  if (block_size <= 0) {
    throw std::invalid_argument("CUDA block size must be positive.");
  }
  const auto items = static_cast<std::size_t>(n_particles);
  const auto block = static_cast<std::size_t>(block_size);
  const std::size_t block_count = (items + block - 1) / block;
  if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("CUDA grid size exceeds launch capacity.");
  }
  return static_cast<int>(block_count);
}

int particle_reorder_grid_size(index_t n_particles, int block_size) {
  return cell_list_build_grid_size(n_particles, block_size);
}

__device__ index_t axis_cell_index(
    real_t position,
    real_t lower,
    real_t inverse_cell_size,
    index_t cell_count
) noexcept
{
  const real_t scaled = (position - lower) * inverse_cell_size;
  index_t index = scaled <= real_t{0} ? 0 : static_cast<index_t>(scaled);
  if (index >= cell_count) {
    index = cell_count - 1;
  }
  return index;
}

__device__ index_t flatten_cell(
    index_t ix,
    index_t iy,
    index_t iz,
    const NeighborCellGeometry& geometry
) noexcept
{
  return ix + geometry.nx * (iy + geometry.ny * iz);
}

__global__ void compute_cell_ids_and_counts_kernel(
    ParticlePositionConstView particles,
    system::geometry::BoxGeometryView box,
    NeighborCellGeometry cell_geometry,
    CellListBuildDataView cell_list_data
)
{
  const index_t particle = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (particle >= cell_list_data.n_particles) {
    return;
  }

  const index_t ix = axis_cell_index(
      particles.position_x[particle],
      box.lower_x,
      cell_geometry.inv_cell_size_x,
      cell_geometry.nx);
  const index_t iy = axis_cell_index(
      particles.position_y[particle],
      box.lower_y,
      cell_geometry.inv_cell_size_y,
      cell_geometry.ny);
  const index_t iz = axis_cell_index(
      particles.position_z[particle],
      box.lower_z,
      cell_geometry.inv_cell_size_z,
      cell_geometry.nz);
  const index_t cell = flatten_cell(ix, iy, iz, cell_geometry);
  cell_list_data.particle_cell[particle] = cell;
  atomicAdd(&cell_list_data.cell_count[cell], index_t{1});
}

__global__ void scatter_cell_particles_kernel(
    CellListScatterView cell_list_data
)
{
  const index_t particle = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (particle >= cell_list_data.n_particles) {
    return;
  }

  const index_t cell = cell_list_data.particle_cell[particle];
  const index_t slot = atomicAdd(&cell_list_data.cell_write_offset[cell], index_t{1});
  cell_list_data.cell_particle[slot] = particle;
}

__global__ void reorder_particles_by_cell_list_kernel(
    ParticleReorderSourceView source,
    system::state::DeviceParticleReorderScratchView destination,
    const index_t* cell_particle,
    const index_t* source_particle_cell,
    index_t* destination_particle_cell,
    index_t* destination_cell_particle
)
{
  const index_t new_slot = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (new_slot >= source.n_particles) {
    return;
  }

  const index_t old_slot = cell_particle[new_slot];

  destination.position_x[new_slot] = source.position_x[old_slot];
  destination.position_y[new_slot] = source.position_y[old_slot];
  destination.position_z[new_slot] = source.position_z[old_slot];

  destination.velocity_x[new_slot] = source.velocity_x[old_slot];
  destination.velocity_y[new_slot] = source.velocity_y[old_slot];
  destination.velocity_z[new_slot] = source.velocity_z[old_slot];

  destination.mass[new_slot] = source.mass[old_slot];
  destination.type[new_slot] = source.type[old_slot];
  destination.tag[new_slot] = source.tag[old_slot];
  destination.molecule_id[new_slot] = source.molecule_id[old_slot];

  destination.image_x[new_slot] = source.image_x[old_slot];
  destination.image_y[new_slot] = source.image_y[old_slot];
  destination.image_z[new_slot] = source.image_z[old_slot];

  destination_particle_cell[new_slot] = source_particle_cell[old_slot];
  destination_cell_particle[new_slot] = new_slot;
}

ParticleReorderSourceView source_view(
    const system::state::DeviceParticles& particles) noexcept {
  return ParticleReorderSourceView{
      particles.n_particles(),
      particles.position_x().data(),
      particles.position_y().data(),
      particles.position_z().data(),
      particles.velocity_x().data(),
      particles.velocity_y().data(),
      particles.velocity_z().data(),
      particles.mass().data(),
      particles.type().data(),
      particles.tag().data(),
      particles.molecule_id().data(),
      particles.image_x().data(),
      particles.image_y().data(),
      particles.image_z().data()};
}

}  // namespace

CellListBuildData::CellListBuildData(
    index_t n_particles,
    const NeighborCellGeometry& cell_geometry,
    cudaStream_t stream) {
  resize(n_particles, cell_geometry, stream);
}

CellListBuildData::CellListBuildData(CellListBuildData&& other) noexcept
    : n_particles_(std::exchange(other.n_particles_, 0)),
      n_cells_(std::exchange(other.n_cells_, 0)),
      particle_cell_(std::move(other.particle_cell_)),
      cell_count_(std::move(other.cell_count_)),
      cell_offset_(std::move(other.cell_offset_)),
      cell_write_offset_(std::move(other.cell_write_offset_)),
      cell_particle_(std::move(other.cell_particle_)),
      particle_cell_scratch_(std::move(other.particle_cell_scratch_)),
      scan_workspace_(std::move(other.scan_workspace_)),
      scan_workspace_bytes_(std::exchange(other.scan_workspace_bytes_, 0)) {}

CellListBuildData& CellListBuildData::operator=(CellListBuildData&& other) noexcept {
  if (this != &other) {
    n_particles_ = std::exchange(other.n_particles_, 0);
    n_cells_ = std::exchange(other.n_cells_, 0);
    particle_cell_ = std::move(other.particle_cell_);
    cell_count_ = std::move(other.cell_count_);
    cell_offset_ = std::move(other.cell_offset_);
    cell_write_offset_ = std::move(other.cell_write_offset_);
    cell_particle_ = std::move(other.cell_particle_);
    particle_cell_scratch_ = std::move(other.particle_cell_scratch_);
    scan_workspace_ = std::move(other.scan_workspace_);
    scan_workspace_bytes_ = std::exchange(other.scan_workspace_bytes_, 0);
  }
  return *this;
}

void CellListBuildData::resize(
    index_t n_particles,
    const NeighborCellGeometry& cell_geometry,
    cudaStream_t stream) {
  validate_shape(n_particles, cell_geometry.n_cells);

  const std::size_t particle_count =
      checked_buffer_size(n_particles, "CellListBuildData particle size overflows.");
  const std::size_t cell_count =
      checked_buffer_size(cell_geometry.n_cells, "CellListBuildData cell size overflows.");

  particle_cell_.resize(particle_count);
  cell_particle_.resize(particle_count);
  particle_cell_scratch_.resize(particle_count);
  cell_count_.resize(cell_count);
  cell_offset_.resize(cell_count);
  cell_write_offset_.resize(cell_count);
  n_particles_ = n_particles;
  n_cells_ = cell_geometry.n_cells;
  configure_scan_workspace(stream);
}

void CellListBuildData::clear_counts(cudaStream_t stream) {
  BEADS_CUDA_CHECK(cudaMemsetAsync(
      cell_count_.data(),
      0,
      static_cast<std::size_t>(n_cells_) * sizeof(index_t),
      stream));
}

void CellListBuildData::configure_scan_workspace(cudaStream_t stream) {
  if (n_cells_ == 0) {
    return;
  }

  auto mutable_view = view();
  std::size_t required_bytes = 0;
  BEADS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      nullptr,
      required_bytes,
      mutable_view.cell_count,
      mutable_view.cell_offset,
      require_cub_int_count(n_cells_, "Cell-list cell count"),
      stream));
  if (required_bytes > scan_workspace_bytes_) {
    scan_workspace_.resize(required_bytes);
    scan_workspace_bytes_ = required_bytes;
  }
}

CellListBuildDataView CellListBuildData::view() noexcept {
  return CellListBuildDataView{
      n_particles_,
      n_cells_,
      particle_cell_.data(),
      cell_count_.data(),
      cell_offset_.data(),
      cell_particle_.data()};
}

CellListBuildDataConstView CellListBuildData::view() const noexcept {
  return CellListBuildDataConstView{
      n_particles_,
      n_cells_,
      particle_cell_.data(),
      cell_count_.data(),
      cell_offset_.data(),
      cell_particle_.data()};
}

void build_cell_list(
    CellListBuildData& cell_list_data,
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    const NeighborCellGeometry& cell_geometry,
    cudaStream_t stream) {
  cell_list_data.clear_counts(stream);
  compute_cell_ids_and_counts_kernel<<<
      cell_list_build_grid_size(cell_list_data.n_particles(), kCellListBuildBlockSize),
      kCellListBuildBlockSize,
      0,
      stream>>>(
          position_view(particles),
          system::geometry::make_box_geometry_view(box),
          cell_geometry,
          cell_list_data.view());
  BEADS_CUDA_CHECK(cudaGetLastError());

  auto view = cell_list_data.view();
  BEADS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      cell_list_data.scan_workspace_.data(),
      cell_list_data.scan_workspace_bytes_,
      view.cell_count,
      view.cell_offset,
      require_cub_int_count(cell_list_data.n_cells(), "Cell-list cell count"),
      stream));

  BEADS_CUDA_CHECK(cudaMemcpyAsync(
      cell_list_data.cell_write_offset_.data(),
      cell_list_data.cell_offset_.data(),
      static_cast<std::size_t>(cell_list_data.n_cells()) * sizeof(index_t),
      cudaMemcpyDeviceToDevice,
      stream));

  scatter_cell_particles_kernel<<<
      cell_list_build_grid_size(cell_list_data.n_particles(), kCellListBuildBlockSize),
      kCellListBuildBlockSize,
      0,
      stream>>>(
          CellListScatterView{
              cell_list_data.n_particles_,
              cell_list_data.particle_cell_.data(),
              cell_list_data.cell_write_offset_.data(),
              cell_list_data.cell_particle_.data()});
  BEADS_CUDA_CHECK(cudaGetLastError());
}

void reorder_particles_by_cell_list(
    system::state::DeviceParticles& particles,
    system::state::DeviceParticleReorderScratch& particle_scratch,
    CellListBuildData& cell_list_data,
    cudaStream_t stream) {
  reorder_particles_by_cell_list_kernel<<<
      particle_reorder_grid_size(particles.n_particles(), kParticleReorderBlockSize),
      kParticleReorderBlockSize,
      0,
      stream>>>(
          source_view(particles),
          particle_scratch.view(),
          cell_list_data.cell_particle_.data(),
          cell_list_data.particle_cell_.data(),
          cell_list_data.particle_cell_scratch_.data(),
          cell_list_data.cell_particle_.data());
  BEADS_CUDA_CHECK(cudaGetLastError());

  particles.swap_reorderable_fields(particle_scratch);
  cell_list_data.particle_cell_.swap(cell_list_data.particle_cell_scratch_);
}

}  // namespace simulation::neighbor
}  // namespace beads
