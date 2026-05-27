#pragma once

#include <cuda_runtime.h>

namespace beads {
namespace simulation::output {

struct StagedTransferState {
  cudaStream_t producer_stream = nullptr;
  cudaStream_t transfer_stream = nullptr;
  bool pending = false;
  bool producer_work_enqueued = false;
  bool transfer_stream_work_enqueued = false;
  bool producer_event_recorded = false;
  bool transfer_event_recorded = false;
  bool commit_ready = false;

  void reset() noexcept {
    producer_stream = nullptr;
    transfer_stream = nullptr;
    pending = false;
    producer_work_enqueued = false;
    transfer_stream_work_enqueued = false;
    producer_event_recorded = false;
    transfer_event_recorded = false;
    commit_ready = false;
  }
};

inline void wait_for_staged_transfer_noexcept(
    const StagedTransferState& state,
    cudaEvent_t producer_ready,
    cudaEvent_t transfer_ready) noexcept {
  // Stream synchronizes here are cleanup fallbacks for failures before a
  // completion event could be recorded; normal staging never synchronizes.
  if (state.transfer_event_recorded && transfer_ready != nullptr) {
    cudaEventSynchronize(transfer_ready);
  } else if (state.transfer_stream_work_enqueued &&
             state.transfer_stream != nullptr) {
    cudaStreamSynchronize(state.transfer_stream);
  } else if (state.producer_event_recorded && producer_ready != nullptr) {
    cudaEventSynchronize(producer_ready);
  } else if (state.producer_work_enqueued &&
             state.producer_stream != nullptr) {
    cudaStreamSynchronize(state.producer_stream);
  }
}

}  // namespace simulation::output
}  // namespace beads
