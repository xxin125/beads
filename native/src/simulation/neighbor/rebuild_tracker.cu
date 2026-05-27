#include "rebuild_tracker.cuh"

#include <beads/core/cuda_check.cuh>
#include <system/geometry/box_geometry_view.cuh>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace beads {
namespace simulation::neighbor {
namespace {

constexpr int kRebuildTrackerBlockSize = 256;

int rebuild_tracker_grid_size(index_t n_particles) {
  const auto items = static_cast<std::size_t>(n_particles);
  const auto block = static_cast<std::size_t>(kRebuildTrackerBlockSize);
  const std::size_t block_count = (items + block - 1) / block;
  if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("CUDA grid size exceeds launch capacity.");
  }
  return static_cast<int>(block_count);
}

__global__ void capture_reference_kernel(
    index_t n_particles,
    const real_t* position_x,
    const real_t* position_y,
    const real_t* position_z,
    const image_t* image_x,
    const image_t* image_y,
    const image_t* image_z,
    real_t* reference_position_x,
    real_t* reference_position_y,
    real_t* reference_position_z,
    image_t* reference_image_x,
    image_t* reference_image_y,
    image_t* reference_image_z
)
{
  const index_t i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_particles) {
    return;
  }

  reference_position_x[i] = position_x[i];
  reference_position_y[i] = position_y[i];
  reference_position_z[i] = position_z[i];
  reference_image_x[i] = image_x[i];
  reference_image_y[i] = image_y[i];
  reference_image_z[i] = image_z[i];
}

__global__ void check_displacement_threshold_kernel(
    index_t n_particles,
    const real_t* position_x,
    const real_t* position_y,
    const real_t* position_z,
    const image_t* image_x,
    const image_t* image_y,
    const image_t* image_z,
    system::geometry::BoxGeometryView box,
    const real_t* reference_position_x,
    const real_t* reference_position_y,
    const real_t* reference_position_z,
    const image_t* reference_image_x,
    const image_t* reference_image_y,
    const image_t* reference_image_z,
    real_t threshold_sq,
    int* threshold_exceeded
)
{
  const index_t i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_particles || *threshold_exceeded != 0) {
    return;
  }

  const long long image_dx = static_cast<long long>(image_x[i]) -
      static_cast<long long>(reference_image_x[i]);
  const long long image_dy = static_cast<long long>(image_y[i]) -
      static_cast<long long>(reference_image_y[i]);
  const long long image_dz = static_cast<long long>(image_z[i]) -
      static_cast<long long>(reference_image_z[i]);
  const real_t dx = (position_x[i] - reference_position_x[i]) +
      static_cast<real_t>(image_dx) * box.length_x;
  const real_t dy = (position_y[i] - reference_position_y[i]) +
      static_cast<real_t>(image_dy) * box.length_y;
  const real_t dz = (position_z[i] - reference_position_z[i]) +
      static_cast<real_t>(image_dz) * box.length_z;
  if (dx * dx + dy * dy + dz * dz > threshold_sq) {
    atomicExch(threshold_exceeded, 1);
  }
}

}  // namespace

NeighborRebuildTracker::~NeighborRebuildTracker() {
  release_threshold_storage_noexcept();
}

void NeighborRebuildTracker::resize(index_t n_particles, cudaStream_t stream) {
  if (n_particles == 0) {
    throw std::invalid_argument(
        "NeighborRebuildTracker n_particles must be positive.");
  }
  n_particles_ = n_particles;
  const auto count = static_cast<std::size_t>(n_particles_);
  reference_position_x_.resize(count);
  reference_position_y_.resize(count);
  reference_position_z_.resize(count);
  reference_image_x_.resize(count);
  reference_image_y_.resize(count);
  reference_image_z_.resize(count);
  threshold_exceeded_.resize(1);
  (void)stream;
  has_reference_ = false;
}

void NeighborRebuildTracker::ensure_threshold_storage() {
  if (threshold_exceeded_host_ == nullptr) {
    BEADS_CUDA_CHECK(cudaHostAlloc(
        reinterpret_cast<void**>(&threshold_exceeded_host_),
        sizeof(int),
        cudaHostAllocDefault));
    *threshold_exceeded_host_ = 0;
  }
}

void NeighborRebuildTracker::release_threshold_storage_noexcept() noexcept {
  if (threshold_exceeded_host_ != nullptr) {
    cudaFreeHost(threshold_exceeded_host_);
    threshold_exceeded_host_ = nullptr;
  }
}

void NeighborRebuildTracker::capture_reference(
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    cudaStream_t stream) {
  (void)box;
  if (particles.n_particles() != n_particles_) {
    throw std::logic_error(
        "NeighborRebuildTracker must be resized before capture_reference.");
  }

  capture_reference_kernel<<<
      rebuild_tracker_grid_size(n_particles_),
      kRebuildTrackerBlockSize,
      0,
      stream>>>(
      n_particles_,
      particles.position_x().data(),
      particles.position_y().data(),
      particles.position_z().data(),
      particles.image_x().data(),
      particles.image_y().data(),
      particles.image_z().data(),
      reference_position_x_.data(),
      reference_position_y_.data(),
      reference_position_z_.data(),
      reference_image_x_.data(),
      reference_image_y_.data(),
      reference_image_z_.data());
  BEADS_CUDA_CHECK(cudaGetLastError());
  has_reference_ = true;
}

bool NeighborRebuildTracker::exceeds_displacement_threshold(
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    real_t threshold_sq,
    cudaStream_t stream) {
  if (!has_reference_) {
    throw std::logic_error(
        "NeighborRebuildTracker requires capture_reference before threshold sampling.");
  }
  if (particles.n_particles() != n_particles_) {
    throw std::logic_error(
        "NeighborRebuildTracker particle count does not match its reference.");
  }

  ensure_threshold_storage();
  *threshold_exceeded_host_ = 0;
  BEADS_CUDA_CHECK(cudaMemsetAsync(
      threshold_exceeded_.data(),
      0,
      sizeof(int),
      stream));

  check_displacement_threshold_kernel<<<
      rebuild_tracker_grid_size(n_particles_),
      kRebuildTrackerBlockSize,
      0,
      stream>>>(
      n_particles_,
      particles.position_x().data(),
      particles.position_y().data(),
      particles.position_z().data(),
      particles.image_x().data(),
      particles.image_y().data(),
      particles.image_z().data(),
      system::geometry::make_box_geometry_view(box),
      reference_position_x_.data(),
      reference_position_y_.data(),
      reference_position_z_.data(),
      reference_image_x_.data(),
      reference_image_y_.data(),
      reference_image_z_.data(),
      threshold_sq,
      threshold_exceeded_.data());
  BEADS_CUDA_CHECK(cudaGetLastError());
  BEADS_CUDA_CHECK(cudaMemcpyAsync(
      threshold_exceeded_host_,
      threshold_exceeded_.data(),
      sizeof(int),
      cudaMemcpyDeviceToHost,
      stream));
  BEADS_CUDA_CHECK(cudaStreamSynchronize(stream));
  return *threshold_exceeded_host_ != 0;
}

}  // namespace simulation::neighbor
}  // namespace beads
