#include "trajectory_output.cuh"

#include <beads/core/cuda_check.cuh>
#include <beads/core/device_buffer.cuh>
#include <simulation/output/host_writer.hpp>
#include <simulation/output/staged_transfer.cuh>
#include <system/geometry/box_geometry.hpp>
#include <system/state/device_forces.cuh>
#include <system/state/device_particles.cuh>
#include <system/units/unit_system.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace beads {
namespace simulation::output::trajectory {
namespace {

constexpr char kMagic[] = "BEADS_TRAJECTORY";
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kEndianMarker = 0x01020304u;
constexpr int kPackBlockSize = 256;
constexpr int kRingSlotCount = 4;

enum TrajectoryFieldBits : std::uint32_t {
  kFieldTag = 1u << 0,
  kFieldType = 1u << 1,
  kFieldPosition = 1u << 2,
  kFieldImage = 1u << 3,
  kFieldVelocity = 1u << 4,
  kFieldForce = 1u << 5,
};

std::string trajectory_path_from_prefix(const std::string& prefix) {
  if (prefix.empty()) {
    throw std::invalid_argument("output.trajectory.prefix must not be empty.");
  }
  return prefix + ".trajectory.beadsbin";
}

bool contains_field(
    const std::vector<std::string>& fields,
    const char* name) {
  return std::find(fields.begin(), fields.end(), name) != fields.end();
}

std::uint32_t trajectory_field_mask(const std::vector<std::string>& fields) {
  std::uint32_t mask = 0;
  if (contains_field(fields, "tag")) {
    mask |= kFieldTag;
  }
  if (contains_field(fields, "type")) {
    mask |= kFieldType;
  }
  if (contains_field(fields, "position")) {
    mask |= kFieldPosition;
  }
  if (contains_field(fields, "image")) {
    mask |= kFieldImage;
  }
  if (contains_field(fields, "velocity")) {
    mask |= kFieldVelocity;
  }
  if (contains_field(fields, "force")) {
    mask |= kFieldForce;
  }
  return mask;
}

bool is_supported_field(const std::string& field) {
  return field == "tag" ||
         field == "type" ||
         field == "position" ||
         field == "image" ||
         field == "velocity" ||
         field == "force";
}

void validate_fields(const std::vector<std::string>& fields) {
  if (fields.empty()) {
    throw std::invalid_argument("output.trajectory.fields must not be empty.");
  }
  if (fields.front() != "tag") {
    throw std::invalid_argument(
        "output.trajectory.fields must include tag as the first field.");
  }

  const std::vector<std::string> canonical{
      "tag",
      "type",
      "position",
      "image",
      "velocity",
      "force"};
  std::vector<std::string> seen;
  std::size_t canonical_index = 0;
  for (const std::string& field : fields) {
    if (!is_supported_field(field)) {
      throw std::invalid_argument(
          "output.trajectory.fields contains unsupported field \"" + field + "\".");
    }
    if (std::find(seen.begin(), seen.end(), field) != seen.end()) {
      throw std::invalid_argument(
          "output.trajectory.fields must not contain duplicates.");
    }
    seen.push_back(field);
    while (canonical_index < canonical.size() &&
           canonical[canonical_index] != field) {
      ++canonical_index;
    }
    if (canonical_index == canonical.size()) {
      throw std::invalid_argument(
          "output.trajectory.fields must follow canonical order: "
          "tag,type,position,image,velocity,force.");
    }
    ++canonical_index;
  }
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* context) {
  if (a > std::numeric_limits<std::size_t>::max() - b) {
    throw std::overflow_error(std::string(context) + " size overflows.");
  }
  return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* context) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::overflow_error(std::string(context) + " size overflows.");
  }
  return a * b;
}

std::size_t record_bytes_from_mask(std::uint32_t mask) {
  std::size_t bytes = sizeof(index_t);
  if (mask & kFieldType) {
    bytes = checked_add(bytes, sizeof(type_id_t), "trajectory record");
  }
  if (mask & kFieldPosition) {
    bytes = checked_add(bytes, 3 * sizeof(real_t), "trajectory record");
  }
  if (mask & kFieldImage) {
    bytes = checked_add(bytes, 3 * sizeof(image_t), "trajectory record");
  }
  if (mask & kFieldVelocity) {
    bytes = checked_add(bytes, 3 * sizeof(real_t), "trajectory record");
  }
  if (mask & kFieldForce) {
    bytes = checked_add(bytes, 3 * sizeof(real_t), "trajectory record");
  }
  return bytes;
}

int pack_grid_size(index_t n_particles) {
  const std::size_t items = static_cast<std::size_t>(n_particles);
  const std::size_t block = static_cast<std::size_t>(kPackBlockSize);
  const std::size_t block_count = (items + block - 1) / block;
  if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("trajectory pack grid size exceeds launch capacity.");
  }
  return static_cast<int>(block_count);
}

template <typename T>
__device__ void write_pod(unsigned char* destination, const T& value) {
  memcpy(destination, &value, sizeof(T));
}

__global__ void pack_trajectory_records_kernel(
    index_t n_particles,
    const index_t* tag,
    const type_id_t* type,
    const real_t* position_x,
    const real_t* position_y,
    const real_t* position_z,
    const image_t* image_x,
    const image_t* image_y,
    const image_t* image_z,
    const real_t* velocity_x,
    const real_t* velocity_y,
    const real_t* velocity_z,
    const real_t* force_x,
    const real_t* force_y,
    const real_t* force_z,
    std::uint32_t field_mask,
    std::size_t record_bytes,
    unsigned char* records) {
  const index_t particle =
      static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (particle >= n_particles) {
    return;
  }

  unsigned char* record =
      records + static_cast<std::size_t>(particle) * record_bytes;
  std::size_t offset = 0;

  write_pod(record + offset, tag[particle]);
  offset += sizeof(index_t);

  if (field_mask & kFieldType) {
    write_pod(record + offset, type[particle]);
    offset += sizeof(type_id_t);
  }
  if (field_mask & kFieldPosition) {
    write_pod(record + offset, position_x[particle]);
    offset += sizeof(real_t);
    write_pod(record + offset, position_y[particle]);
    offset += sizeof(real_t);
    write_pod(record + offset, position_z[particle]);
    offset += sizeof(real_t);
  }
  if (field_mask & kFieldImage) {
    write_pod(record + offset, image_x[particle]);
    offset += sizeof(image_t);
    write_pod(record + offset, image_y[particle]);
    offset += sizeof(image_t);
    write_pod(record + offset, image_z[particle]);
    offset += sizeof(image_t);
  }
  if (field_mask & kFieldVelocity) {
    write_pod(record + offset, velocity_x[particle]);
    offset += sizeof(real_t);
    write_pod(record + offset, velocity_y[particle]);
    offset += sizeof(real_t);
    write_pod(record + offset, velocity_z[particle]);
    offset += sizeof(real_t);
  }
  if (field_mask & kFieldForce) {
    write_pod(record + offset, force_x[particle]);
    offset += sizeof(real_t);
    write_pod(record + offset, force_y[particle]);
    offset += sizeof(real_t);
    write_pod(record + offset, force_z[particle]);
  }
}

template <typename T>
void write_scalar(std::ofstream& stream, const T& value, const char* label) {
  stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!stream) {
    throw std::runtime_error(std::string("failed to write ") + label + ".");
  }
}

void write_bytes(
    std::ofstream& stream,
    const void* data,
    std::size_t byte_count,
    const char* label) {
  if (byte_count == 0) {
    return;
  }
  stream.write(reinterpret_cast<const char*>(data), byte_count);
  if (!stream) {
    throw std::runtime_error(std::string("failed to write ") + label + ".");
  }
}

}  // namespace

class TrajectoryOutput::Impl {
 public:
  Impl(TrajectoryOutputConfig config, HostWriter& writer)
      : every_(config.every),
        fields_(std::move(config.fields)),
        path_(trajectory_path_from_prefix(config.prefix)),
        field_mask_(trajectory_field_mask(fields_)),
        record_bytes_(record_bytes_from_mask(field_mask_)),
        writer_(writer) {
    if (every_ == 0) {
      throw std::invalid_argument("output.trajectory.every must be positive.");
    }
    validate_fields(fields_);
  }

  ~Impl() {
    release_noexcept();
  }

  bool has_output() const noexcept { return true; }

  void prepare(
      index_t n_particles,
      const system::units::UnitSystem& units) {
    n_particles_ = n_particles;
    units_ = units;
    record_buffer_bytes_ = checked_mul(
        static_cast<std::size_t>(n_particles_),
        record_bytes_,
        "trajectory record buffer");
    for (TrajectorySlot& slot : slots_) {
      slot.device_records.resize(record_buffer_bytes_);
    }
    ensure_slots();
    prepared_ = true;
  }

  void stage_if_due(
      runstep_t step,
      const system::state::DeviceParticles& particles,
      const system::state::DeviceForces& forces,
      const system::geometry::BoxGeometry& box,
      cudaStream_t dynamics_stream,
      cudaStream_t transfer_stream) {
    if (!should_write(step)) {
      return;
    }
    if (!prepared_) {
      throw std::logic_error("trajectory output was not prepared before staging.");
    }
    if (dynamics_stream == nullptr) {
      throw std::logic_error("trajectory output requires a non-null dynamics stream.");
    }
    if (transfer_stream == nullptr) {
      throw std::logic_error("trajectory output requires a non-null transfer stream.");
    }
    if (dynamics_stream == transfer_stream) {
      throw std::logic_error("trajectory output requires distinct dynamics and transfer streams.");
    }
    if (particles.n_particles() != n_particles_) {
      throw std::logic_error("trajectory output received unexpected particle count.");
    }
    if ((field_mask_ & kFieldForce) && forces.n_particles() != n_particles_) {
      throw std::logic_error("trajectory output received unexpected force count.");
    }

    poll_ready();
    const int slot_index = acquire_slot();
    TrajectorySlot& slot = slots_[slot_index];
    reset_slot_transient(slot);
    slot.step = step;
    slot.lower = box.lower;
    slot.upper = box.upper;
    slot.transfer.producer_stream = dynamics_stream;
    slot.transfer.transfer_stream = transfer_stream;
    slot.transfer.pending = true;
    pending_fifo_.push_back(slot_index);

    try {
      pack_trajectory_records_kernel<<<
          pack_grid_size(n_particles_),
          kPackBlockSize,
          0,
          dynamics_stream>>>(
          n_particles_,
          particles.tag().data(),
          particles.type().data(),
          particles.position_x().data(),
          particles.position_y().data(),
          particles.position_z().data(),
          particles.image_x().data(),
          particles.image_y().data(),
          particles.image_z().data(),
          particles.velocity_x().data(),
          particles.velocity_y().data(),
          particles.velocity_z().data(),
          forces.force_x().data(),
          forces.force_y().data(),
          forces.force_z().data(),
          field_mask_,
          record_bytes_,
          slot.device_records.data());
      BEADS_CUDA_CHECK(cudaGetLastError());
      slot.transfer.producer_work_enqueued = true;
      BEADS_CUDA_CHECK(cudaEventRecord(slot.pack_ready, dynamics_stream));
      slot.transfer.producer_event_recorded = true;
      BEADS_CUDA_CHECK(cudaStreamWaitEvent(
          transfer_stream,
          slot.pack_ready,
          0));
      slot.transfer.transfer_stream_work_enqueued = true;
      BEADS_CUDA_CHECK(cudaMemcpyAsync(
          slot.host_records,
          slot.device_records.data(),
          record_buffer_bytes_,
          cudaMemcpyDeviceToHost,
          transfer_stream));
      BEADS_CUDA_CHECK(cudaEventRecord(slot.transfer_ready, transfer_stream));
      slot.transfer.transfer_event_recorded = true;
    } catch (...) {
      cleanup_failed_stage(slot_index);
      throw;
    }
  }

  void poll_ready() {
    while (!pending_fifo_.empty()) {
      TrajectorySlot& slot = slots_[pending_fifo_.front()];
      if (!slot.transfer.transfer_event_recorded) {
        return;
      }
      const cudaError_t ready = cudaEventQuery(slot.transfer_ready);
      if (ready == cudaErrorNotReady) {
        return;
      }
      BEADS_CUDA_CHECK(ready);
      slot.transfer.commit_ready = true;
      submit_front();
    }
  }

  void drain_pending() {
    while (!pending_fifo_.empty()) {
      TrajectorySlot& slot = slots_[pending_fifo_.front()];
      if (!slot.transfer.transfer_event_recorded) {
        throw std::logic_error("trajectory output pending slot has no transfer event.");
      }
      BEADS_CUDA_CHECK(cudaEventSynchronize(slot.transfer_ready));
      slot.transfer.commit_ready = true;
      submit_front();
    }
  }

  void flush_file() {
    if (stream_.is_open()) {
      stream_.flush();
      if (!stream_) {
        throw std::runtime_error("failed to flush trajectory output file " + path_ + ".");
      }
    }
  }

  bool needs_force_state(runstep_t step) const noexcept {
    return should_write(step) && ((field_mask_ & kFieldForce) != 0);
  }

 private:
  struct TrajectorySlot {
    DeviceBuffer<unsigned char> device_records;
    unsigned char* host_records = nullptr;
    cudaEvent_t pack_ready = nullptr;
    cudaEvent_t transfer_ready = nullptr;
    runstep_t step = 0;
    std::array<real_t, 3> lower{};
    std::array<real_t, 3> upper{};
    StagedTransferState transfer;
    bool busy = false;
  };

  bool should_write(runstep_t step) const noexcept {
    return step % every_ == 0;
  }

  void ensure_slots() {
    for (TrajectorySlot& slot : slots_) {
      if (slot.host_records == nullptr && record_buffer_bytes_ != 0) {
        BEADS_CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&slot.host_records),
            record_buffer_bytes_,
            cudaHostAllocDefault));
      }
      if (slot.pack_ready == nullptr) {
        BEADS_CUDA_CHECK(cudaEventCreateWithFlags(
            &slot.pack_ready,
            cudaEventDisableTiming));
      }
      if (slot.transfer_ready == nullptr) {
        BEADS_CUDA_CHECK(cudaEventCreateWithFlags(
            &slot.transfer_ready,
            cudaEventDisableTiming));
      }
    }
  }

  int acquire_slot() {
    for (;;) {
      writer_.rethrow_if_failed();
      poll_ready();
      {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        for (int index = 0; index < kRingSlotCount; ++index) {
          const int slot_index = (next_slot_ + index) % kRingSlotCount;
          if (!slots_[slot_index].busy) {
            slots_[slot_index].busy = true;
            next_slot_ = (slot_index + 1) % kRingSlotCount;
            return slot_index;
          }
        }
      }

      if (!pending_fifo_.empty()) {
        TrajectorySlot& slot = slots_[pending_fifo_.front()];
        if (!slot.transfer.transfer_event_recorded) {
          throw std::logic_error("trajectory output pending slot has no transfer event.");
        }
        BEADS_CUDA_CHECK(cudaEventSynchronize(slot.transfer_ready));
        slot.transfer.commit_ready = true;
        submit_front();
        continue;
      }

      std::unique_lock<std::mutex> lock(slot_mutex_);
      slot_cv_.wait(lock, [&]() {
        return any_free_slot_locked();
      });
    }
  }

  bool any_free_slot_locked() const noexcept {
    for (const TrajectorySlot& slot : slots_) {
      if (!slot.busy) {
        return true;
      }
    }
    return false;
  }

  void submit_front() {
    if (pending_fifo_.empty()) {
      throw std::logic_error("trajectory output has no pending slot to submit.");
    }
    const int slot_index = pending_fifo_.front();
    TrajectorySlot& slot = slots_[slot_index];
    if (!slot.transfer.commit_ready) {
      throw std::logic_error("trajectory output tried to submit an unready slot.");
    }

    pending_fifo_.pop_front();
    slot.transfer.pending = false;
    const runstep_t step = slot.step;
    const std::array<real_t, 3> lower = slot.lower;
    const std::array<real_t, 3> upper = slot.upper;
    const std::size_t byte_count = record_buffer_bytes_;
    try {
      writer_.submit([this, slot_index, step, lower, upper, byte_count]() {
        try {
          write_frame(step, lower, upper, slots_[slot_index].host_records, byte_count);
        } catch (...) {
          mark_slot_free(slot_index);
          throw;
        }
        mark_slot_free(slot_index);
      });
    } catch (...) {
      mark_slot_free(slot_index);
      throw;
    }
  }

  void write_file_header() {
    write_bytes(stream_, kMagic, sizeof(kMagic) - 1, "trajectory magic");
    write_scalar(stream_, kVersion, "trajectory version");
    write_scalar(stream_, kEndianMarker, "trajectory endian marker");
    write_scalar(stream_, static_cast<std::uint32_t>(sizeof(real_t)), "trajectory real size");
    write_scalar(stream_, static_cast<std::uint32_t>(sizeof(index_t)), "trajectory index size");
    write_scalar(stream_, static_cast<std::uint32_t>(sizeof(image_t)), "trajectory image size");
    write_scalar(stream_, static_cast<std::uint32_t>(sizeof(type_id_t)), "trajectory type size");
    write_scalar(stream_, static_cast<std::uint32_t>(sizeof(index_t)), "trajectory tag size");
    const std::string units_name = units_.public_name;
    const auto units_length = static_cast<std::uint64_t>(units_name.size());
    write_scalar(stream_, units_length, "trajectory units length");
    write_bytes(stream_, units_name.data(), units_name.size(), "trajectory units");
    write_scalar(stream_, field_mask_, "trajectory field mask");
    write_scalar(stream_, static_cast<std::uint64_t>(n_particles_), "trajectory particle count");
    write_scalar(stream_, static_cast<std::uint64_t>(record_bytes_), "trajectory record bytes");
  }

  void open_if_needed() {
    if (stream_.is_open()) {
      return;
    }
    stream_.open(path_, std::ios::binary | std::ios::trunc);
    if (!stream_) {
      throw std::runtime_error("failed to open trajectory output file " + path_ + ".");
    }
    write_file_header();
  }

  void write_frame(
      runstep_t step,
      const std::array<real_t, 3>& lower,
      const std::array<real_t, 3>& upper,
      const unsigned char* records,
      std::size_t byte_count) {
    open_if_needed();
    write_scalar(stream_, static_cast<std::uint64_t>(step), "trajectory frame step");
    write_bytes(stream_, lower.data(), 3 * sizeof(real_t), "trajectory box lower");
    write_bytes(stream_, upper.data(), 3 * sizeof(real_t), "trajectory box upper");
    write_bytes(stream_, records, byte_count, "trajectory records");
  }

  void mark_slot_free(int slot_index) noexcept {
    {
      std::lock_guard<std::mutex> lock(slot_mutex_);
      reset_slot_transient(slots_[slot_index]);
      slots_[slot_index].busy = false;
    }
    slot_cv_.notify_all();
  }

  void reset_slot_transient(TrajectorySlot& slot) noexcept {
    slot.step = 0;
    slot.lower = {};
    slot.upper = {};
    slot.transfer.reset();
  }

  void cleanup_failed_stage(int slot_index) noexcept {
    TrajectorySlot& slot = slots_[slot_index];
    wait_for_staged_transfer_noexcept(
        slot.transfer,
        slot.pack_ready,
        slot.transfer_ready);
    remove_from_fifo(slot_index);
    mark_slot_free(slot_index);
  }

  void remove_from_fifo(int slot_index) {
    const auto it = std::find(
        pending_fifo_.begin(),
        pending_fifo_.end(),
        slot_index);
    if (it != pending_fifo_.end()) {
      pending_fifo_.erase(it);
    }
  }

  void release_noexcept() noexcept {
    try {
      for (const int slot_index : pending_fifo_) {
        TrajectorySlot& slot = slots_[slot_index];
        wait_for_staged_transfer_noexcept(
            slot.transfer,
            slot.pack_ready,
            slot.transfer_ready);
      }
      pending_fifo_.clear();
      writer_.flush();
    } catch (...) {
    }

    for (TrajectorySlot& slot : slots_) {
      if (slot.pack_ready != nullptr) {
        cudaEventDestroy(slot.pack_ready);
      }
      if (slot.transfer_ready != nullptr) {
        cudaEventDestroy(slot.transfer_ready);
      }
      if (slot.host_records != nullptr) {
        cudaFreeHost(slot.host_records);
      }
      slot.pack_ready = nullptr;
      slot.transfer_ready = nullptr;
      slot.host_records = nullptr;
      slot.busy = false;
    }
  }

  runstep_t every_ = 0;
  std::vector<std::string> fields_;
  std::string path_;
  std::uint32_t field_mask_ = 0;
  std::size_t record_bytes_ = 0;
  HostWriter& writer_;
  std::size_t record_buffer_bytes_ = 0;
  system::units::UnitSystem units_;
  index_t n_particles_ = 0;
  bool prepared_ = false;
  std::array<TrajectorySlot, kRingSlotCount> slots_;
  std::deque<int> pending_fifo_;
  int next_slot_ = 0;
  std::mutex slot_mutex_;
  std::condition_variable slot_cv_;
  std::ofstream stream_;
};

TrajectoryOutput::TrajectoryOutput(TrajectoryOutputConfig config, HostWriter& writer)
    : impl_(std::make_unique<Impl>(std::move(config), writer)) {}

TrajectoryOutput::~TrajectoryOutput() = default;
TrajectoryOutput::TrajectoryOutput(TrajectoryOutput&&) noexcept = default;
TrajectoryOutput& TrajectoryOutput::operator=(TrajectoryOutput&&) noexcept = default;

bool TrajectoryOutput::has_output() const noexcept {
  return impl_->has_output();
}

void TrajectoryOutput::prepare(
    index_t n_particles,
    const system::units::UnitSystem& units) {
  impl_->prepare(n_particles, units);
}

void TrajectoryOutput::stage_if_due(
    runstep_t step,
    const system::state::DeviceParticles& particles,
    const system::state::DeviceForces& forces,
    const system::geometry::BoxGeometry& box,
    cudaStream_t dynamics_stream,
    cudaStream_t transfer_stream) {
  impl_->stage_if_due(
      step,
      particles,
      forces,
      box,
      dynamics_stream,
      transfer_stream);
}

bool TrajectoryOutput::needs_force_state(runstep_t step) const noexcept {
  return impl_->needs_force_state(step);
}

void TrajectoryOutput::poll_ready() {
  impl_->poll_ready();
}

void TrajectoryOutput::drain_pending() {
  impl_->drain_pending();
}

void TrajectoryOutput::flush_file() {
  impl_->flush_file();
}

}  // namespace simulation::output::trajectory
}  // namespace beads
