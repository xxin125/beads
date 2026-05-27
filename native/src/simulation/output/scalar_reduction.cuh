#pragma once

#include <beads/core/device_buffer.cuh>
#include <beads/core/types.hpp>

#include <cuda_runtime.h>

#include <cstddef>

namespace beads {
namespace simulation::output {

class ScalarReductionWorkspace {
 public:
  void prepare_sum(index_t input_count, index_t scalar_slot_count);

  real_t* device_scalar(index_t slot_index) noexcept;
  const real_t* device_scalar(index_t slot_index) const noexcept;

  void enqueue_sum(
      const real_t* input,
      index_t input_count,
      index_t slot_index,
      cudaStream_t stream);

 private:
  index_t prepared_input_count_ = 0;
  int prepared_cub_count_ = 0;
  DeviceBuffer<std::byte> sum_workspace_;
  std::size_t sum_workspace_bytes_ = 0;
  DeviceBuffer<real_t> device_scalars_;
};

}  // namespace simulation::output
}  // namespace beads
