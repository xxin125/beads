#include "neighbor_list.cuh"

#include <beads/core/cuda_check.cuh>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace beads {
namespace simulation::neighbor {
namespace {

std::size_t checked_neighbor_slot_count(
    index_t n_particles,
    index_t max_neighbors) {
  const auto particles = static_cast<std::size_t>(n_particles);
  const auto neighbors = static_cast<std::size_t>(max_neighbors);
  if (neighbors != 0 &&
      particles > std::numeric_limits<std::size_t>::max() / neighbors) {
    throw std::overflow_error("NeighborList slot count overflows.");
  }
  const std::size_t slot_count = particles * neighbors;
  if (slot_count > static_cast<std::size_t>(
                       std::numeric_limits<index_t>::max())) {
    throw std::overflow_error("NeighborList slot count exceeds index_t range.");
  }
  return slot_count;
}

void validate_shape(index_t n_particles, index_t max_neighbors) {
  if (n_particles == 0) {
    throw std::invalid_argument("NeighborList n_particles must be positive.");
  }
  if (max_neighbors == 0) {
    throw std::invalid_argument("NeighborList max_neighbors must be positive.");
  }
}

}  // namespace

NeighborList::NeighborList(index_t n_particles, index_t max_neighbors) {
  resize(n_particles, max_neighbors);
  clear_build_state();
}

NeighborList::~NeighborList() {
  release_overflow_host_storage_noexcept();
}

void NeighborList::resize(index_t n_particles, index_t max_neighbors) {
  validate_shape(n_particles, max_neighbors);

  const std::size_t slot_count =
      checked_neighbor_slot_count(n_particles, max_neighbors);
  neighbor_count_.resize(static_cast<std::size_t>(n_particles));
  neighbor_index_.resize(slot_count);
  overflow_flag_.resize(1);
  n_particles_ = n_particles;
  max_neighbors_ = max_neighbors;
}

void NeighborList::ensure_overflow_host_storage() const {
  if (overflow_flag_host_ != nullptr) {
    return;
  }
  BEADS_CUDA_CHECK(cudaHostAlloc(
      reinterpret_cast<void**>(&overflow_flag_host_),
      sizeof(int),
      cudaHostAllocDefault));
  *overflow_flag_host_ = 0;
}

void NeighborList::release_overflow_host_storage_noexcept() noexcept {
  if (overflow_flag_host_ != nullptr) {
    cudaFreeHost(overflow_flag_host_);
    overflow_flag_host_ = nullptr;
  }
}

void NeighborList::clear_counts(cudaStream_t stream) {
  BEADS_CUDA_CHECK(cudaMemsetAsync(
      neighbor_count_.data(),
      0,
      static_cast<std::size_t>(n_particles_) * sizeof(index_t),
      stream));
}

void NeighborList::clear_build_state(cudaStream_t stream) {
  if (overflow_flag_.data() == nullptr) {
    return;
  }
  clear_counts(stream);
  BEADS_CUDA_CHECK(cudaMemsetAsync(
      overflow_flag_.data(), 0, sizeof(int), stream));
}

int NeighborList::download_overflow_flag(cudaStream_t stream) const {
  if (overflow_flag_.data() == nullptr) {
    return 0;
  }
  ensure_overflow_host_storage();
  *overflow_flag_host_ = 0;
  BEADS_CUDA_CHECK(cudaMemcpyAsync(
      overflow_flag_host_,
      overflow_flag_.data(),
      sizeof(int),
      cudaMemcpyDeviceToHost,
      stream));
  BEADS_CUDA_CHECK(cudaStreamSynchronize(stream));
  return *overflow_flag_host_;
}

NeighborListView NeighborList::view() noexcept {
  return NeighborListView{
      n_particles_,
      max_neighbors_,
      neighbor_count_.data(),
      neighbor_index_.data(),
      overflow_flag_.data()};
}

NeighborListConstView NeighborList::view() const noexcept {
  return NeighborListConstView{
      n_particles_,
      max_neighbors_,
      neighbor_count_.data(),
      neighbor_index_.data(),
      overflow_flag_.data()};
}

}  // namespace simulation::neighbor
}  // namespace beads
