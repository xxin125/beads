#include "angle_harmonic.hpp"

#include <beads/core/cuda_check.cuh>
#include <system/geometry/box_geometry_view.cuh>
#include <system/state/device_forces.cuh>
#include <system/state/device_particles.cuh>
#include <system/state/host_state.hpp>
#include <system/state/tag_to_slot_map.cuh>

#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace beads {
namespace forcefield {
namespace angle {
namespace {

inline constexpr int kAngleForceBlockSize = 256;
inline constexpr real_t kPi =
    real_t{3.141592653589793238462643383279502884};
inline constexpr real_t kSinFloor = real_t{0.001};

struct Vec3 {
  real_t x = real_t{0};
  real_t y = real_t{0};
  real_t z = real_t{0};
};

__device__ real_t dot(Vec3 lhs, Vec3 rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__device__ real_t clamp_cosine(real_t value) noexcept {
  if (value > real_t{1}) {
    return real_t{1};
  }
  if (value < real_t{-1}) {
    return real_t{-1};
  }
  return value;
}

__device__ real_t harmonic_angle_energy(
    const HarmonicAngleModel::DeviceCoeff& coeff,
    real_t theta) noexcept {
  const real_t dtheta = theta - coeff.theta0_rad;
  return real_t{0.5} * coeff.k * dtheta * dtheta;
}

int angle_force_grid_size(index_t angle_count) {
  const auto items = static_cast<std::size_t>(angle_count);
  const auto block = static_cast<std::size_t>(kAngleForceBlockSize);
  const std::size_t block_count = (items + block - 1) / block;
  if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Harmonic angle CUDA grid size exceeds launch capacity.");
  }
  return static_cast<int>(block_count);
}

template <bool NeedAnglePotentialEnergy, bool NeedGlobalScalarVirial>
__global__ void compute_harmonic_angle_forces_kernel(
    index_t angle_count,
    const HarmonicAngleModel::AngleEntry* angles,
    const HarmonicAngleModel::DeviceCoeff* coeffs,
    const index_t* slots_by_tag,
    system::state::DeviceParticlesConstView particles,
    system::state::DeviceForcesView forces,
    system::geometry::BoxGeometryView box,
    real_t* angle_pe_partials,
    real_t* global_virial_partials)
{
  real_t thread_energy = real_t{0};
  real_t thread_virial = real_t{0};
  const index_t thread = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  const index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);
  for (index_t angle_index = thread;
       angle_index < angle_count;
       angle_index += stride) {
    const HarmonicAngleModel::AngleEntry entry = angles[angle_index];
    const index_t slot_i = slots_by_tag[entry.tag_i];
    const index_t slot_j = slots_by_tag[entry.tag_j];
    const index_t slot_k = slots_by_tag[entry.tag_k];

    Vec3 arm_ij{
        particles.position_x[slot_i] - particles.position_x[slot_j],
        particles.position_y[slot_i] - particles.position_y[slot_j],
        particles.position_z[slot_i] - particles.position_z[slot_j]};
    Vec3 arm_kj{
        particles.position_x[slot_k] - particles.position_x[slot_j],
        particles.position_y[slot_k] - particles.position_y[slot_j],
        particles.position_z[slot_k] - particles.position_z[slot_j]};
    arm_ij.x = system::geometry::minimum_image_delta_x_for_wrapped_positions(
        box, arm_ij.x);
    arm_ij.y = system::geometry::minimum_image_delta_y_for_wrapped_positions(
        box, arm_ij.y);
    arm_ij.z = system::geometry::minimum_image_delta_z_for_wrapped_positions(
        box, arm_ij.z);
    arm_kj.x = system::geometry::minimum_image_delta_x_for_wrapped_positions(
        box, arm_kj.x);
    arm_kj.y = system::geometry::minimum_image_delta_y_for_wrapped_positions(
        box, arm_kj.y);
    arm_kj.z = system::geometry::minimum_image_delta_z_for_wrapped_positions(
        box, arm_kj.z);

    const auto coeff = coeffs[entry.type];
    const real_t rsq_ij = dot(arm_ij, arm_ij);
    const real_t rsq_kj = dot(arm_kj, arm_kj);
    if (rsq_ij == real_t{0} || rsq_kj == real_t{0}) {
      // Exact zero-length arms have undefined angle derivatives. Keep runtime
      // values finite by using theta=0 for energy and skipping force/virial.
      if constexpr (NeedAnglePotentialEnergy) {
        thread_energy += harmonic_angle_energy(coeff, real_t{0});
      }
      continue;
    }
    const real_t r_ij = sqrt(rsq_ij);
    const real_t r_kj = sqrt(rsq_kj);
    real_t cosine = dot(arm_ij, arm_kj) / (r_ij * r_kj);
    cosine = clamp_cosine(cosine);
    const real_t theta =
        (cosine == real_t{1})
            ? real_t{0}
            : ((cosine == real_t{-1}) ? kPi : acos(cosine));

    if constexpr (NeedAnglePotentialEnergy) {
      thread_energy += harmonic_angle_energy(coeff, theta);
    }

    const real_t sin_sq = real_t{1} - cosine * cosine;
    const real_t sin_theta =
        sin_sq <= kSinFloor * kSinFloor ? kSinFloor : sqrt(sin_sq);
    const real_t a = -coeff.k * (theta - coeff.theta0_rad) / sin_theta;
    const real_t a11 = a * cosine / rsq_ij;
    const real_t a12 = -a / (r_ij * r_kj);
    const real_t a22 = a * cosine / rsq_kj;

    const Vec3 force_i{
        a11 * arm_ij.x + a12 * arm_kj.x,
        a11 * arm_ij.y + a12 * arm_kj.y,
        a11 * arm_ij.z + a12 * arm_kj.z};
    const Vec3 force_k{
        a22 * arm_kj.x + a12 * arm_ij.x,
        a22 * arm_kj.y + a12 * arm_ij.y,
        a22 * arm_kj.z + a12 * arm_ij.z};
    const Vec3 force_j{
        -(force_i.x + force_k.x),
        -(force_i.y + force_k.y),
        -(force_i.z + force_k.z)};

    atomicAdd(&forces.force_x[slot_i], force_i.x);
    atomicAdd(&forces.force_y[slot_i], force_i.y);
    atomicAdd(&forces.force_z[slot_i], force_i.z);
    atomicAdd(&forces.force_x[slot_j], force_j.x);
    atomicAdd(&forces.force_y[slot_j], force_j.y);
    atomicAdd(&forces.force_z[slot_j], force_j.z);
    atomicAdd(&forces.force_x[slot_k], force_k.x);
    atomicAdd(&forces.force_y[slot_k], force_k.y);
    atomicAdd(&forces.force_z[slot_k], force_k.z);

    if constexpr (NeedGlobalScalarVirial) {
      thread_virial += dot(arm_ij, force_i) + dot(arm_kj, force_k);
    }
  }

  if constexpr (NeedAnglePotentialEnergy) {
    using BlockReduce = cub::BlockReduce<real_t, kAngleForceBlockSize>;
    __shared__ typename BlockReduce::TempStorage energy_reduce_storage;
    const real_t block_energy =
        BlockReduce(energy_reduce_storage).Sum(thread_energy);
    if (threadIdx.x == 0) {
      angle_pe_partials[blockIdx.x] = block_energy;
    }
  }
  if constexpr (NeedGlobalScalarVirial) {
    using BlockReduce = cub::BlockReduce<real_t, kAngleForceBlockSize>;
    __shared__ typename BlockReduce::TempStorage virial_reduce_storage;
    const real_t block_virial =
        BlockReduce(virial_reduce_storage).Sum(thread_virial);
    if (threadIdx.x == 0) {
      global_virial_partials[blockIdx.x] = block_virial;
    }
  }
}

}  // namespace

HarmonicAngleModel::HarmonicAngleModel()
    : AngleModel(kStyleName) {}

void HarmonicAngleModel::read_settings(
    const input::AngleStyleSpec& angle_style) {
  require_exact_parameter_keys(angle_style.params, {}, "angle_style");
}

void HarmonicAngleModel::begin_topology(
    const system::state::HostState& host_state) {
  const auto& host_angles = host_state.topology().angles();

  type_id_t max_type = 0;
  host_angles_.clear();
  host_angles_.reserve(host_angles.size());
  for (const auto& angle : host_angles) {
    host_angles_.push_back({angle.tag_i, angle.tag_j, angle.tag_k, angle.type});
    if (angle.type > max_type) {
      max_type = angle.type;
    }
  }

  host_coeffs_.assign(static_cast<std::size_t>(max_type) + 1u, DeviceCoeff{});
  coeff_filled_.assign(static_cast<std::size_t>(max_type) + 1u, false);
}

void HarmonicAngleModel::read_coeff(const input::AngleCoeffSpec& coeff) {
  if (coeff.type < 1 ||
      static_cast<std::size_t>(coeff.type) >= host_coeffs_.size()) {
    throw std::invalid_argument(
        "Angle(\"harmonic\") angle_coeff type is not active in topology angles.");
  }
  const auto index = static_cast<std::size_t>(coeff.type);
  if (coeff_filled_[index]) {
    throw std::invalid_argument(
        "Angle(\"harmonic\") angle_coeffs must not contain duplicates.");
  }
  host_coeffs_[index] = make_coeff(coeff);
  coeff_filled_[index] = true;
}

void HarmonicAngleModel::finish_configuration() {
  for (std::size_t type = 1; type < coeff_filled_.size(); ++type) {
    if (!coeff_filled_[type]) {
      throw std::invalid_argument(
          "Angle(\"harmonic\") angle_coeffs must cover all active topology angle types.");
    }
  }

  upload_entries(host_angles_);
  upload_coeffs(host_coeffs_);
  std::vector<AngleEntry>().swap(host_angles_);
  std::vector<DeviceCoeff>().swap(host_coeffs_);
  std::vector<bool>().swap(coeff_filled_);
}

auto HarmonicAngleModel::make_coeff(
    const input::AngleCoeffSpec& coeff) const -> DeviceCoeff {
  require_exact_parameter_keys(coeff.params, {"k", "theta0"}, "angle_coeff");
  const real_t theta0_degrees = require_nonnegative_real_parameter(
      coeff.params,
      "theta0",
      "angle_coeff");
  if (theta0_degrees > real_t{180}) {
    throw std::invalid_argument(
        "Angle(\"harmonic\") angle_coeff.theta0 must be between 0 and 180 degrees.");
  }
  return DeviceCoeff{
      require_nonnegative_real_parameter(
          coeff.params,
          "k",
          "angle_coeff"),
      theta0_degrees * kPi / real_t{180}};
}

void HarmonicAngleModel::upload_entries(
    const std::vector<AngleEntry>& entries) {
  angle_count_ = static_cast<index_t>(entries.size());
  angles_.resize(entries.size());
  if (!entries.empty()) {
    BEADS_CUDA_CHECK(cudaMemcpy(
        angles_.data(),
        entries.data(),
        entries.size() * sizeof(AngleEntry),
        cudaMemcpyHostToDevice));
  }
}

void HarmonicAngleModel::upload_coeffs(
    const std::vector<DeviceCoeff>& coeffs) {
  coeffs_.resize(coeffs.size());
  if (!coeffs.empty()) {
    BEADS_CUDA_CHECK(cudaMemcpy(
        coeffs_.data(),
        coeffs.data(),
        coeffs.size() * sizeof(DeviceCoeff),
        cudaMemcpyHostToDevice));
  }
}

ForceEvalObservableLayout HarmonicAngleModel::observable_layout(
    const ForceEvalRequest& request) const {
  return make_component_observable_layout(
      request,
      ForceObservable::AnglePotentialEnergy,
      static_cast<index_t>(angle_force_grid_size(angle_count_)),
      "Angle(\"harmonic\")");
}

void HarmonicAngleModel::add_forces(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const system::state::TagToSlotMap& tag_to_slot_map,
    const system::geometry::BoxGeometry& box,
    cudaStream_t stream) const {
  const int grid_size = angle_force_grid_size(angle_count_);
  compute_harmonic_angle_forces_kernel<false, false><<<
      grid_size,
      kAngleForceBlockSize,
      0,
      stream>>>(
      angle_count_,
      angles_.data(),
      coeffs_.data(),
      tag_to_slot_map.slots_by_tag().data(),
      particles.view(),
      forces.view(),
      system::geometry::make_box_geometry_view(box),
      nullptr,
      nullptr);
  BEADS_CUDA_CHECK(cudaGetLastError());
}

void HarmonicAngleModel::add_forces_and_observables(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const system::state::TagToSlotMap& tag_to_slot_map,
    const system::geometry::BoxGeometry& box,
    const ForceEvalRequest& request,
    const ForceObservableBuffers& buffers,
    cudaStream_t stream) const {
  const bool need_angle_pe =
      request.has(ForceObservable::AnglePotentialEnergy);
  const bool need_global_virial =
      request.has(ForceObservable::GlobalScalarVirial);
  if (!need_angle_pe && !need_global_virial) {
    add_forces(particles, forces, tag_to_slot_map, box, stream);
    return;
  }

  const int grid_size = angle_force_grid_size(angle_count_);
  if (need_angle_pe) {
    require_observable_buffer_shape(
        buffers,
        ForceObservable::AnglePotentialEnergy,
        static_cast<index_t>(grid_size),
        "Harmonic angle potential energy buffer has wrong shape.");
  }
  if (need_global_virial) {
    require_observable_buffer_shape(
        buffers,
        ForceObservable::GlobalScalarVirial,
        static_cast<index_t>(grid_size),
        "Harmonic angle global scalar virial buffer has wrong shape.");
  }

  if (need_angle_pe && need_global_virial) {
    compute_harmonic_angle_forces_kernel<true, true><<<
        grid_size,
        kAngleForceBlockSize,
        0,
        stream>>>(
        angle_count_,
        angles_.data(),
        coeffs_.data(),
        tag_to_slot_map.slots_by_tag().data(),
        particles.view(),
        forces.view(),
        system::geometry::make_box_geometry_view(box),
        buffers.angle_pe_partials,
        buffers.global_virial_partials);
  } else if (need_angle_pe) {
    compute_harmonic_angle_forces_kernel<true, false><<<
        grid_size,
        kAngleForceBlockSize,
        0,
        stream>>>(
        angle_count_,
        angles_.data(),
        coeffs_.data(),
        tag_to_slot_map.slots_by_tag().data(),
        particles.view(),
        forces.view(),
        system::geometry::make_box_geometry_view(box),
        buffers.angle_pe_partials,
        nullptr);
  } else {
    compute_harmonic_angle_forces_kernel<false, true><<<
        grid_size,
        kAngleForceBlockSize,
        0,
        stream>>>(
        angle_count_,
        angles_.data(),
        coeffs_.data(),
        tag_to_slot_map.slots_by_tag().data(),
        particles.view(),
        forces.view(),
        system::geometry::make_box_geometry_view(box),
        nullptr,
        buffers.global_virial_partials);
  }
  BEADS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace angle
}  // namespace forcefield
}  // namespace beads
