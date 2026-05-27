#include "tag_to_slot_map.cuh"

#include <beads/core/cuda_check.cuh>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace beads {
namespace system::state {
namespace {

constexpr int kTagToSlotBlockSize = 256;

int grid_size(index_t n_particles) {
  const auto count = static_cast<std::size_t>(n_particles);
  const auto block = static_cast<std::size_t>(kTagToSlotBlockSize);
  const std::size_t blocks = (count + block - 1) / block;
  if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("TagToSlotMap grid size exceeds int capacity.");
  }
  return static_cast<int>(blocks);
}

__global__ void rebuild_tag_to_slot_map_kernel(
    index_t n_particles,
    const index_t* tags,
    index_t* slots_by_tag) {
  const index_t slot = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (slot >= n_particles) {
    return;
  }
  slots_by_tag[tags[slot]] = slot;
}

}  // namespace

void TagToSlotMap::prepare(index_t n_particles) {
  if (n_particles == std::numeric_limits<index_t>::max()) {
    throw std::overflow_error(
        "TagToSlotMap cannot allocate one-based tag map for index_t::max particles.");
  }
  if (is_prepared()) {
    if (particle_count_ != n_particles) {
      throw std::logic_error(
          "TagToSlotMap cannot be prepared for a different particle count.");
    }
    return;
  }
  particle_count_ = n_particles;
  slots_by_tag_.resize(static_cast<std::size_t>(n_particles) + 1u);
  dirty_ = true;
}

bool TagToSlotMap::is_current() const noexcept {
  return is_prepared() && !dirty_;
}

bool TagToSlotMap::rebuild_if_dirty(
    const DeviceParticles& particles,
    cudaStream_t stream) {
  if (!dirty_) {
    return false;
  }
  rebuild(particles, stream);
  return true;
}

void TagToSlotMap::rebuild(
    const DeviceParticles& particles,
    cudaStream_t stream) {
  require_prepared();

  // System tags are validated as a one-based 1..N permutation, so the rebuild
  // kernel writes slots_by_tag[1..N] exactly once. Slot 0 is an unused sentinel.
  BEADS_CUDA_CHECK(cudaMemsetAsync(
      slots_by_tag_.data(),
      0xff,
      sizeof(index_t),
      stream));
  rebuild_tag_to_slot_map_kernel<<<
      grid_size(particle_count_),
      kTagToSlotBlockSize,
      0,
      stream>>>(
          particle_count_,
          particles.tag().data(),
          slots_by_tag_.data());
  BEADS_CUDA_CHECK(cudaGetLastError());

  dirty_ = false;
  ++generation_;
}

bool TagToSlotMap::is_prepared() const noexcept {
  return particle_count_ > 0 &&
      slots_by_tag_.size() == static_cast<std::size_t>(particle_count_) + 1u;
}

void TagToSlotMap::require_prepared() const {
  if (!is_prepared()) {
    throw std::logic_error("TagToSlotMap must be prepared before rebuild.");
  }
}

void TagToSlotMap::download_slots_by_tag(std::vector<index_t>& host) const {
  host.resize(slots_by_tag_.size());
  if (!host.empty()) {
    BEADS_CUDA_CHECK(cudaMemcpy(
        host.data(),
        slots_by_tag_.data(),
        host.size() * sizeof(index_t),
        cudaMemcpyDeviceToHost));
  }
}

}  // namespace system::state
}  // namespace beads
