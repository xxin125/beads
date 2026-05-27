#include "thermo_output.cuh"

#include <beads/core/cuda_check.cuh>
#include <simulation/output/host_writer.hpp>
#include <simulation/output/scalar_reduction.cuh>
#include <simulation/output/staged_transfer.cuh>
#include <system/geometry/box_geometry.hpp>
#include <system/state/device_particles.cuh>
#include <system/units/unit_system.hpp>

#include <cub/block/block_reduce.cuh>

#include <array>
#include <cstddef>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

namespace beads {
namespace simulation::output::thermo {
namespace {

constexpr int kRingSlotCount = 2;
constexpr int kKineticBlockSize = 256;

std::string thermo_path_from_prefix(const std::string& prefix) {
  if (prefix.empty()) {
    throw std::invalid_argument("output.thermo.prefix must not be empty.");
  }
  return prefix + ".thermo.csv";
}

int kinetic_grid_size(index_t n_particles) {
  const auto items = static_cast<std::size_t>(n_particles);
  const auto block = static_cast<std::size_t>(kKineticBlockSize);
  const std::size_t block_count = (items + block - 1) / block;
  if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("kinetic energy grid size exceeds launch capacity.");
  }
  return static_cast<int>(block_count);
}

real_t temperature_dof(index_t n_particles) noexcept {
  const real_t dof = real_t{3} * static_cast<real_t>(n_particles) - real_t{3};
  return dof > real_t{0} ? dof : real_t{0};
}

ThermoSchema schema_from_layout(
    const forcefield::ForceEvalObservableLayout& layout) noexcept {
  ThermoSchema schema;
  schema.include_bond_pe = layout.bond_pe_partial_count != 0;
  schema.include_angle_pe = layout.angle_pe_partial_count != 0;
  schema.include_dihedral_pe = layout.dihedral_pe_partial_count != 0;
  return schema;
}

__global__ void compute_kinetic_energy_partials_kernel(
    index_t n_particles,
    const real_t* velocity_x,
    const real_t* velocity_y,
    const real_t* velocity_z,
    const real_t* mass,
    real_t* kinetic_partials)
{
  using BlockReduce = cub::BlockReduce<real_t, kKineticBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const index_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  const index_t stride = blockDim.x * gridDim.x;
  real_t thread_sum = real_t{0};
  for (index_t particle = thread_index; particle < n_particles; particle += stride) {
    const real_t vx = velocity_x[particle];
    const real_t vy = velocity_y[particle];
    const real_t vz = velocity_z[particle];
    thread_sum += real_t{0.5} * mass[particle] * (vx * vx + vy * vy + vz * vz);
  }

  const real_t block_sum = BlockReduce(temp_storage).Sum(thread_sum);
  if (threadIdx.x == 0) {
    kinetic_partials[blockIdx.x] = block_sum;
  }
}

}  // namespace

class ThermoOutput::Impl {
 public:
  Impl(ThermoOutputConfig config, HostWriter& writer)
      : every_(config.every),
        path_(thermo_path_from_prefix(config.prefix)),
        writer_(writer),
        log_sink_(std::move(config.log_sink)) {
    if (every_ == 0) {
      throw std::invalid_argument("output.thermo.every must be positive.");
    }
  }

  ~Impl() {
    release_noexcept();
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  bool has_output() const noexcept { return true; }

  void prepare(
      index_t n_particles,
      const system::units::UnitSystem& units,
      const forcefield::ForceEvalObservableLayout& layout) {
    n_particles_ = n_particles;
    units_ = units;
    pair_pe_partial_count_ = layout.pair_pe_partial_count;
    bond_pe_partial_count_ = layout.bond_pe_partial_count;
    angle_pe_partial_count_ = layout.angle_pe_partial_count;
    dihedral_pe_partial_count_ = layout.dihedral_pe_partial_count;
    virial_partial_count_ = layout.global_virial_partial_count;
    schema_ = schema_from_layout(layout);
    if (pair_pe_partial_count_ == 0) {
      throw std::logic_error(
          "thermo output requires pair potential energy partials.");
    }
    if (virial_partial_count_ == 0) {
      throw std::logic_error(
          "thermo output requires global scalar virial partials.");
    }

    kinetic_partial_count_ = static_cast<index_t>(kinetic_grid_size(n_particles_));
    kinetic_partials_.resize(kinetic_partial_count_);
    pair_pe_reduction_.prepare_sum(pair_pe_partial_count_, kRingSlotCount);
    if (schema_.include_bond_pe) {
      bond_pe_reduction_.prepare_sum(bond_pe_partial_count_, kRingSlotCount);
    }
    if (schema_.include_angle_pe) {
      angle_pe_reduction_.prepare_sum(angle_pe_partial_count_, kRingSlotCount);
    }
    if (schema_.include_dihedral_pe) {
      dihedral_pe_reduction_.prepare_sum(
          dihedral_pe_partial_count_,
          kRingSlotCount);
    }
    kinetic_reduction_.prepare_sum(kinetic_partial_count_, kRingSlotCount);
    virial_reduction_.prepare_sum(virial_partial_count_, kRingSlotCount);
    ensure_slots();
    prepared_ = true;
  }

  void stage_if_due(
      runstep_t step,
      const system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box,
      const forcefield::ForceEvalResult& force_result,
      cudaStream_t dynamics_stream,
      cudaStream_t transfer_stream) {
    if (!should_write(step)) {
      return;
    }
    if (!prepared_) {
      throw std::logic_error("thermo output was not prepared before staging.");
    }
    if (!force_result.pair_pe.present() ||
        force_result.pair_pe.count != pair_pe_partial_count_) {
      throw std::logic_error(
          "thermo output due step did not receive pair potential energy partials.");
    }
    if (schema_.include_bond_pe &&
        (!force_result.bond_pe.present() ||
         force_result.bond_pe.count != bond_pe_partial_count_)) {
      throw std::logic_error(
          "thermo output due step did not receive bond potential energy partials.");
    }
    if (schema_.include_angle_pe &&
        (!force_result.angle_pe.present() ||
         force_result.angle_pe.count != angle_pe_partial_count_)) {
      throw std::logic_error(
          "thermo output due step did not receive angle potential energy partials.");
    }
    if (schema_.include_dihedral_pe &&
        (!force_result.dihedral_pe.present() ||
         force_result.dihedral_pe.count != dihedral_pe_partial_count_)) {
      throw std::logic_error(
          "thermo output due step did not receive dihedral potential energy partials.");
    }
    if (!force_result.global_virial.present() ||
        force_result.global_virial.count != virial_partial_count_) {
      throw std::logic_error(
          "thermo output due step did not receive global scalar virial partials.");
    }
    if (particles.n_particles() != n_particles_) {
      throw std::logic_error("thermo output received unexpected particle count.");
    }

    poll_ready();
    const int slot_index = next_slot_;
    make_slot_available(slot_index);

    ThermoSlot& slot = slots_[slot_index];
    reset_slot_state(slot);
    slot.step = step;
    slot.volume = system::geometry::volume(box);
    slot.host->pair_pe = real_t{0};
    slot.host->bond_pe = real_t{0};
    slot.host->angle_pe = real_t{0};
    slot.host->dihedral_pe = real_t{0};
    slot.host->ke = real_t{0};
    slot.host->virial = real_t{0};
    pending_fifo_.push_back(slot_index);
    slot.transfer.pending = true;
    slot.transfer.producer_stream = dynamics_stream;
    slot.transfer.transfer_stream = transfer_stream;

    try {
      pair_pe_reduction_.enqueue_sum(
          force_result.pair_pe.data,
          pair_pe_partial_count_,
          slot_index,
          dynamics_stream);
      slot.transfer.producer_work_enqueued = true;
      if (schema_.include_bond_pe) {
        bond_pe_reduction_.enqueue_sum(
            force_result.bond_pe.data,
            bond_pe_partial_count_,
            slot_index,
            dynamics_stream);
      }
      if (schema_.include_angle_pe) {
        angle_pe_reduction_.enqueue_sum(
            force_result.angle_pe.data,
            angle_pe_partial_count_,
            slot_index,
            dynamics_stream);
      }
      if (schema_.include_dihedral_pe) {
        dihedral_pe_reduction_.enqueue_sum(
            force_result.dihedral_pe.data,
            dihedral_pe_partial_count_,
            slot_index,
            dynamics_stream);
      }

      compute_kinetic_energy_partials_kernel<<<
          kinetic_partial_count_,
          kKineticBlockSize,
          0,
          dynamics_stream>>>(
          particles.n_particles(),
          particles.velocity_x().data(),
          particles.velocity_y().data(),
          particles.velocity_z().data(),
          particles.mass().data(),
          kinetic_partials_.data());
      BEADS_CUDA_CHECK(cudaGetLastError());
      kinetic_reduction_.enqueue_sum(
          kinetic_partials_.data(),
          kinetic_partial_count_,
          slot_index,
          dynamics_stream);
      virial_reduction_.enqueue_sum(
          force_result.global_virial.data,
          virial_partial_count_,
          slot_index,
          dynamics_stream);
      BEADS_CUDA_CHECK(cudaEventRecord(slot.reduction_ready, dynamics_stream));
      slot.transfer.producer_event_recorded = true;

      BEADS_CUDA_CHECK(cudaStreamWaitEvent(
          transfer_stream,
          slot.reduction_ready,
          0));
      slot.transfer.transfer_stream_work_enqueued = true;
      BEADS_CUDA_CHECK(cudaMemcpyAsync(
          &slot.host->pair_pe,
          pair_pe_reduction_.device_scalar(slot_index),
          sizeof(real_t),
          cudaMemcpyDeviceToHost,
          transfer_stream));
      if (schema_.include_bond_pe) {
        BEADS_CUDA_CHECK(cudaMemcpyAsync(
            &slot.host->bond_pe,
            bond_pe_reduction_.device_scalar(slot_index),
            sizeof(real_t),
            cudaMemcpyDeviceToHost,
            transfer_stream));
      }
      if (schema_.include_angle_pe) {
        BEADS_CUDA_CHECK(cudaMemcpyAsync(
            &slot.host->angle_pe,
            angle_pe_reduction_.device_scalar(slot_index),
            sizeof(real_t),
            cudaMemcpyDeviceToHost,
            transfer_stream));
      }
      if (schema_.include_dihedral_pe) {
        BEADS_CUDA_CHECK(cudaMemcpyAsync(
            &slot.host->dihedral_pe,
            dihedral_pe_reduction_.device_scalar(slot_index),
            sizeof(real_t),
            cudaMemcpyDeviceToHost,
            transfer_stream));
      }
      BEADS_CUDA_CHECK(cudaMemcpyAsync(
          &slot.host->ke,
          kinetic_reduction_.device_scalar(slot_index),
          sizeof(real_t),
          cudaMemcpyDeviceToHost,
          transfer_stream));
      BEADS_CUDA_CHECK(cudaMemcpyAsync(
          &slot.host->virial,
          virial_reduction_.device_scalar(slot_index),
          sizeof(real_t),
          cudaMemcpyDeviceToHost,
          transfer_stream));
      BEADS_CUDA_CHECK(cudaEventRecord(slot.transfer_ready, transfer_stream));
      slot.transfer.transfer_event_recorded = true;
    } catch (...) {
      cleanup_failed_stage(slot_index);
      throw;
    }

    next_slot_ = (next_slot_ + 1) % kRingSlotCount;
  }

  void poll_ready() {
    while (!pending_fifo_.empty()) {
      ThermoSlot& slot = slots_[pending_fifo_.front()];
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
      ThermoSlot& slot = slots_[pending_fifo_.front()];
      if (!slot.transfer.transfer_event_recorded) {
        throw std::logic_error("thermo output pending slot has no transfer event.");
      }
      BEADS_CUDA_CHECK(cudaEventSynchronize(slot.transfer_ready));
      slot.transfer.commit_ready = true;
      submit_front();
    }
  }

  void flush_file() {
    if (!stream_.is_open()) {
      return;
    }
    stream_.flush();
    if (!stream_) {
      throw std::runtime_error("failed to flush thermo output file " + path_ + ".");
    }
  }

 private:
  struct ThermoHostScalars {
    real_t pair_pe = real_t{0};
    real_t bond_pe = real_t{0};
    real_t angle_pe = real_t{0};
    real_t dihedral_pe = real_t{0};
    real_t ke = real_t{0};
    real_t virial = real_t{0};
  };

  struct ThermoSlot {
    ThermoHostScalars* host = nullptr;
    cudaEvent_t reduction_ready = nullptr;
    cudaEvent_t transfer_ready = nullptr;
    StagedTransferState transfer;
    runstep_t step = 0;
    real_t volume = real_t{0};
  };

  bool should_write(runstep_t step) const noexcept {
    return step % every_ == 0;
  }

  void ensure_slots() {
    for (ThermoSlot& slot : slots_) {
      if (slot.host == nullptr) {
        BEADS_CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&slot.host),
            sizeof(ThermoHostScalars),
            cudaHostAllocDefault));
      }
      if (slot.reduction_ready == nullptr) {
        BEADS_CUDA_CHECK(cudaEventCreateWithFlags(
            &slot.reduction_ready,
            cudaEventDisableTiming));
      }
      if (slot.transfer_ready == nullptr) {
        BEADS_CUDA_CHECK(cudaEventCreateWithFlags(
            &slot.transfer_ready,
            cudaEventDisableTiming));
      }
    }
  }

  void make_slot_available(int slot_index) {
    while (slots_[slot_index].transfer.pending) {
      ThermoSlot& oldest = slots_[pending_fifo_.front()];
      if (!oldest.transfer.transfer_event_recorded) {
        throw std::logic_error("thermo output pending slot has no transfer event.");
      }
      BEADS_CUDA_CHECK(cudaEventSynchronize(oldest.transfer_ready));
      oldest.transfer.commit_ready = true;
      submit_front();
    }
  }

  void submit_front() {
    const int slot_index = pending_fifo_.front();
    ThermoSlot& slot = slots_[slot_index];
    if (!slot.transfer.commit_ready) {
      throw std::logic_error("thermo output slot is not ready to commit.");
    }
    const ThermoLogRow row = materialize_row(slot);
    writer_.submit([this, row]() {
      write_row(row);
      if (log_sink_) {
        log_sink_(row);
      }
    });
    pending_fifo_.pop_front();
    reset_slot_state(slot);
  }

  ThermoLogRow materialize_row(const ThermoSlot& slot) const {
    const real_t pair_pe = slot.host->pair_pe;
    const real_t bond_pe =
        schema_.include_bond_pe ? slot.host->bond_pe : real_t{0};
    const real_t angle_pe =
        schema_.include_angle_pe ? slot.host->angle_pe : real_t{0};
    const real_t dihedral_pe =
        schema_.include_dihedral_pe ? slot.host->dihedral_pe : real_t{0};
    const real_t pe = pair_pe + bond_pe + angle_pe + dihedral_pe;
    const real_t ke = slot.host->ke;
    const real_t virial = slot.host->virial;
    const real_t temp = units_.temperature_from_kinetic_energy_dof(
        ke,
        temperature_dof(n_particles_));
    const real_t press = units_.pressure_from_kinetic_energy_and_virial(
        ke,
        virial,
        slot.volume);
    return ThermoLogRow{
        slot.step,
        pe,
        pair_pe,
        bond_pe,
        angle_pe,
        dihedral_pe,
        ke,
        temp,
        press,
        schema_};
  }

  void write_row(const ThermoLogRow& row) {
    ensure_open();
    stream_ << row.step << ',' << std::setprecision(9)
            << row.pe << ',' << row.pair_pe;
    if (row.schema.include_bond_pe) {
      stream_ << ',' << row.bond_pe;
    }
    if (row.schema.include_angle_pe) {
      stream_ << ',' << row.angle_pe;
    }
    if (row.schema.include_dihedral_pe) {
      stream_ << ',' << row.dihedral_pe;
    }
    stream_ << ',' << row.ke << ',' << row.temp << ',' << row.press << '\n';
    if (!stream_) {
      throw std::runtime_error("failed to write thermo output file " + path_ + ".");
    }
  }

  void ensure_open() {
    if (stream_.is_open()) {
      return;
    }
    stream_.open(path_);
    if (!stream_) {
      throw std::runtime_error("failed to open thermo output file " + path_ + ".");
    }
    stream_ << "step,pe,pair_pe";
    if (schema_.include_bond_pe) {
      stream_ << ",bond_pe";
    }
    if (schema_.include_angle_pe) {
      stream_ << ",angle_pe";
    }
    if (schema_.include_dihedral_pe) {
      stream_ << ",dihedral_pe";
    }
    stream_ << ",ke,temp,press\n";
    if (!stream_) {
      throw std::runtime_error("failed to write thermo output file " + path_ + ".");
    }
  }

  void release_noexcept() noexcept {
    // drain_pending()/flush_file() own ordered CSV commit; destructor cleanup
    // only waits so CUDA resources can be released without throwing.
    for (int slot_index : pending_fifo_) {
      wait_for_slot_work_noexcept(slots_[slot_index]);
      reset_slot_state(slots_[slot_index]);
    }
    pending_fifo_.clear();
    try {
      writer_.flush();
    } catch (...) {
    }

    for (ThermoSlot& slot : slots_) {
      if (slot.transfer_ready != nullptr) {
        cudaEventDestroy(slot.transfer_ready);
      }
      if (slot.reduction_ready != nullptr) {
        cudaEventDestroy(slot.reduction_ready);
      }
      if (slot.host != nullptr) {
        cudaFreeHost(slot.host);
      }
      slot = {};
    }
  }

  void cleanup_failed_stage(int slot_index) noexcept {
    ThermoSlot& slot = slots_[slot_index];
    wait_for_slot_work_noexcept(slot);
    remove_pending_slot(slot_index);
    reset_slot_state(slot);
  }

  void wait_for_slot_work_noexcept(ThermoSlot& slot) noexcept {
    wait_for_staged_transfer_noexcept(
        slot.transfer,
        slot.reduction_ready,
        slot.transfer_ready);
  }

  void remove_pending_slot(int slot_index) noexcept {
    for (auto iter = pending_fifo_.begin(); iter != pending_fifo_.end(); ++iter) {
      if (*iter == slot_index) {
        pending_fifo_.erase(iter);
        return;
      }
    }
  }

  void reset_slot_state(ThermoSlot& slot) noexcept {
    slot.step = 0;
    slot.volume = real_t{0};
    slot.transfer.reset();
  }

  runstep_t every_ = 0;
  std::string path_;
  HostWriter& writer_;
  std::function<void(const ThermoLogRow&)> log_sink_;
  system::units::UnitSystem units_;
  index_t n_particles_ = 0;
  index_t pair_pe_partial_count_ = 0;
  index_t bond_pe_partial_count_ = 0;
  index_t angle_pe_partial_count_ = 0;
  index_t dihedral_pe_partial_count_ = 0;
  index_t virial_partial_count_ = 0;
  index_t kinetic_partial_count_ = 0;
  ThermoSchema schema_;
  bool prepared_ = false;
  DeviceBuffer<real_t> kinetic_partials_;
  ScalarReductionWorkspace pair_pe_reduction_;
  ScalarReductionWorkspace bond_pe_reduction_;
  ScalarReductionWorkspace angle_pe_reduction_;
  ScalarReductionWorkspace dihedral_pe_reduction_;
  ScalarReductionWorkspace kinetic_reduction_;
  ScalarReductionWorkspace virial_reduction_;
  std::array<ThermoSlot, kRingSlotCount> slots_;
  std::deque<int> pending_fifo_;
  int next_slot_ = 0;
  std::ofstream stream_;
};

ThermoOutput::ThermoOutput(ThermoOutputConfig config, HostWriter& writer)
    : impl_(std::make_unique<Impl>(std::move(config), writer)) {}

ThermoOutput::~ThermoOutput() = default;
ThermoOutput::ThermoOutput(ThermoOutput&&) noexcept = default;
ThermoOutput& ThermoOutput::operator=(ThermoOutput&&) noexcept = default;

bool ThermoOutput::has_output() const noexcept {
  return impl_->has_output();
}

void ThermoOutput::prepare(
    index_t n_particles,
    const system::units::UnitSystem& units,
    const forcefield::ForceEvalObservableLayout& layout) {
  impl_->prepare(n_particles, units, layout);
}

void ThermoOutput::stage_if_due(
    runstep_t step,
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box,
    const forcefield::ForceEvalResult& force_result,
    cudaStream_t dynamics_stream,
    cudaStream_t transfer_stream) {
  impl_->stage_if_due(
      step,
      particles,
      box,
      force_result,
      dynamics_stream,
      transfer_stream);
}

void ThermoOutput::poll_ready() {
  impl_->poll_ready();
}

void ThermoOutput::drain_pending() {
  impl_->drain_pending();
}

void ThermoOutput::flush_file() {
  impl_->flush_file();
}

}  // namespace simulation::output::thermo
}  // namespace beads
