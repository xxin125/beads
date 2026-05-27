#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace beads {
namespace cuda_check {

inline void check(
    cudaError_t error,
    const char* expression,
    const char* file,
    int line) {
  if (error == cudaSuccess) {
    return;
  }

  std::ostringstream message;
  message << "CUDA call failed: " << expression
          << " at " << file << ":" << line
          << " with error " << static_cast<int>(error)
          << " (" << cudaGetErrorString(error) << ").";
  throw std::runtime_error(message.str());
}

}  // namespace cuda_check
}  // namespace beads

#define BEADS_CUDA_CHECK(call) \
  ::beads::cuda_check::check((call), #call, __FILE__, __LINE__)
