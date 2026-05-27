#include "scalar_reduction.cuh"

#include <beads/core/cuda_check.cuh>

#include <cub/cub.cuh>

#include <limits>
#include <stdexcept>
#include <string>

namespace beads {
namespace simulation::output {
namespace {

int checked_cub_count(index_t count, const char* label) {
  if (count <= 0) {
    throw std::logic_error(std::string(label) + " reduction count must be positive.");
  }
  if (count > static_cast<index_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string(label) + " reduction count exceeds CUB int capacity.");
  }
  return static_cast<int>(count);
}

}  // namespace

void ScalarReductionWorkspace::prepare_sum(
    index_t input_count,
    index_t scalar_slot_count) {
  prepared_input_count_ = input_count;
  prepared_cub_count_ = checked_cub_count(input_count, "scalar sum");

  if (scalar_slot_count <= 0) {
    throw std::logic_error("scalar sum requires at least one output slot.");
  }
  device_scalars_.resize(static_cast<std::size_t>(scalar_slot_count));

  std::size_t workspace_bytes = 0;
  BEADS_CUDA_CHECK(cub::DeviceReduce::Sum(
      nullptr,
      workspace_bytes,
      static_cast<const real_t*>(nullptr),
      device_scalars_.data(),
      prepared_cub_count_));
  sum_workspace_.resize(workspace_bytes);
  sum_workspace_bytes_ = workspace_bytes;
}

real_t* ScalarReductionWorkspace::device_scalar(index_t slot_index) noexcept {
  return device_scalars_.data() + slot_index;
}

const real_t* ScalarReductionWorkspace::device_scalar(
    index_t slot_index) const noexcept {
  return device_scalars_.data() + slot_index;
}

void ScalarReductionWorkspace::enqueue_sum(
    const real_t* input,
    index_t input_count,
    index_t slot_index,
    cudaStream_t stream) {
  if (input == nullptr) {
    throw std::logic_error("scalar sum input pointer must not be null.");
  }
  if (input_count != prepared_input_count_ || prepared_cub_count_ == 0) {
    throw std::logic_error("scalar sum workspace was not prepared for this input shape.");
  }
  if (static_cast<std::size_t>(slot_index) >= device_scalars_.size()) {
    throw std::out_of_range("scalar sum output slot is out of range.");
  }

  std::size_t workspace_bytes = sum_workspace_bytes_;
  BEADS_CUDA_CHECK(cub::DeviceReduce::Sum(
      static_cast<void*>(sum_workspace_.data()),
      workspace_bytes,
      input,
      device_scalar(slot_index),
      prepared_cub_count_,
      stream));
}

}  // namespace simulation::output
}  // namespace beads
