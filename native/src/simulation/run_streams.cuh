#pragma once

#include <beads/core/cuda_check.cuh>

#include <cuda_runtime.h>

#include <stdexcept>
#include <utility>

namespace beads {
namespace simulation {

class RunStreams {
 public:
  RunStreams() {
    try {
      BEADS_CUDA_CHECK(cudaStreamCreateWithFlags(
          &dynamics_stream_,
          cudaStreamNonBlocking));
      BEADS_CUDA_CHECK(cudaStreamCreateWithFlags(
          &transfer_stream_,
          cudaStreamNonBlocking));
      if (dynamics_stream_ == transfer_stream_) {
        throw std::logic_error("run streams must be distinct.");
      }
    } catch (...) {
      release_noexcept();
      throw;
    }
  }

  ~RunStreams() {
    release_noexcept();
  }

  RunStreams(const RunStreams&) = delete;
  RunStreams& operator=(const RunStreams&) = delete;

  RunStreams(RunStreams&& other) noexcept
      : dynamics_stream_(std::exchange(other.dynamics_stream_, nullptr)),
        transfer_stream_(std::exchange(other.transfer_stream_, nullptr)) {}

  RunStreams& operator=(RunStreams&& other) noexcept {
    if (this != &other) {
      release_noexcept();
      dynamics_stream_ = std::exchange(other.dynamics_stream_, nullptr);
      transfer_stream_ = std::exchange(other.transfer_stream_, nullptr);
    }
    return *this;
  }

  cudaStream_t dynamics_stream() const noexcept { return dynamics_stream_; }
  cudaStream_t transfer_stream() const noexcept { return transfer_stream_; }

 private:
  void release_noexcept() noexcept {
    if (transfer_stream_ != nullptr) {
      cudaStreamDestroy(transfer_stream_);
    }
    if (dynamics_stream_ != nullptr) {
      cudaStreamDestroy(dynamics_stream_);
    }
    dynamics_stream_ = nullptr;
    transfer_stream_ = nullptr;
  }

  cudaStream_t dynamics_stream_ = nullptr;
  cudaStream_t transfer_stream_ = nullptr;
};

}  // namespace simulation
}  // namespace beads
