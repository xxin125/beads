#include "dihedral_harmonic.hpp"

#include <beads/core/cuda_check.cuh>
#include <forcefield/force_launch.cuh>
#include <system/geometry/box_geometry_view.cuh>
#include <system/state/device_forces.cuh>
#include <system/state/device_particles.cuh>
#include <system/state/host_state.hpp>
#include <system/state/tag_to_slot_map.cuh>

#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace beads {
namespace forcefield {
namespace dihedral {
namespace {

inline constexpr int kDihedralForceBlockSize = 256;
inline constexpr real_t kPi =
    real_t{3.141592653589793238462643383279502884};

struct Vec3 {
  real_t x = real_t{0};
  real_t y = real_t{0};
  real_t z = real_t{0};
};

__device__ Vec3 add(Vec3 lhs, Vec3 rhs) noexcept {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

__device__ Vec3 subtract(Vec3 lhs, Vec3 rhs) noexcept {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

__device__ Vec3 scale(Vec3 value, real_t factor) noexcept {
  return {factor * value.x, factor * value.y, factor * value.z};
}

__device__ real_t dot(Vec3 lhs, Vec3 rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__device__ Vec3 cross(Vec3 lhs, Vec3 rhs) noexcept {
  return {
      lhs.y * rhs.z - lhs.z * rhs.y,
      lhs.z * rhs.x - lhs.x * rhs.z,
      lhs.x * rhs.y - lhs.y * rhs.x};
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

template <bool NeedDihedralPotentialEnergy, bool NeedGlobalScalarVirial>
__global__ void compute_harmonic_dihedral_forces_kernel(
    index_t dihedral_count,
    const HarmonicDihedralModel::DihedralEntry* dihedrals,
    const HarmonicDihedralModel::DeviceCoeff* coeffs,
    const index_t* slots_by_tag,
    system::state::DeviceParticlesConstView particles,
    system::state::DeviceForcesView forces,
    system::geometry::BoxGeometryView box,
    real_t* dihedral_pe_partials,
    real_t* global_virial_partials)
{
  real_t thread_energy = real_t{0};
  real_t thread_virial = real_t{0};
  const index_t thread = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  const index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);
  for (index_t dihedral_index = thread;
       dihedral_index < dihedral_count;
       dihedral_index += stride) {
    const HarmonicDihedralModel::DihedralEntry entry =
        dihedrals[dihedral_index];
    const index_t slot_i = slots_by_tag[entry.tag_i];
    const index_t slot_j = slots_by_tag[entry.tag_j];
    const index_t slot_k = slots_by_tag[entry.tag_k];
    const index_t slot_l = slots_by_tag[entry.tag_l];

    Vec3 v0{
        particles.position_x[slot_i] - particles.position_x[slot_j],
        particles.position_y[slot_i] - particles.position_y[slot_j],
        particles.position_z[slot_i] - particles.position_z[slot_j]};
    Vec3 v1{
        particles.position_x[slot_k] - particles.position_x[slot_j],
        particles.position_y[slot_k] - particles.position_y[slot_j],
        particles.position_z[slot_k] - particles.position_z[slot_j]};
    Vec3 v2{
        particles.position_x[slot_k] - particles.position_x[slot_l],
        particles.position_y[slot_k] - particles.position_y[slot_l],
        particles.position_z[slot_k] - particles.position_z[slot_l]};
    v0.x = system::geometry::minimum_image_delta_x_for_wrapped_positions(
        box, v0.x);
    v0.y = system::geometry::minimum_image_delta_y_for_wrapped_positions(
        box, v0.y);
    v0.z = system::geometry::minimum_image_delta_z_for_wrapped_positions(
        box, v0.z);
    v1.x = system::geometry::minimum_image_delta_x_for_wrapped_positions(
        box, v1.x);
    v1.y = system::geometry::minimum_image_delta_y_for_wrapped_positions(
        box, v1.y);
    v1.z = system::geometry::minimum_image_delta_z_for_wrapped_positions(
        box, v1.z);
    v2.x = system::geometry::minimum_image_delta_x_for_wrapped_positions(
        box, v2.x);
    v2.y = system::geometry::minimum_image_delta_y_for_wrapped_positions(
        box, v2.y);
    v2.z = system::geometry::minimum_image_delta_z_for_wrapped_positions(
        box, v2.z);

    const Vec3 cp0 = cross(v0, v1);
    const Vec3 cp1 = cross(v1, v2);
    const real_t norm_bc_sq = dot(v1, v1);
    const real_t norm_cross0_sq = dot(cp0, cp0);
    const real_t norm_cross1_sq = dot(cp1, cp1);
    const bool singular =
        norm_bc_sq == real_t{0} ||
        norm_cross0_sq == real_t{0} ||
        norm_cross1_sq == real_t{0};

    real_t theta = real_t{0};
    const auto coeff = coeffs[entry.type];
    if (!singular) {
      const real_t inv_norm_cross =
          real_t{1} / sqrt(norm_cross0_sq * norm_cross1_sq);
      real_t cosine = dot(cp0, cp1) * inv_norm_cross;
      cosine = clamp_cosine(cosine);
      theta =
          (cosine == real_t{1})
              ? real_t{0}
              : ((cosine == real_t{-1}) ? kPi : acos(cosine));
      if (dot(v0, cp1) < real_t{0}) {
        theta = -theta;
      }
    }

    real_t energy = real_t{0};
    real_t dE_dtheta = real_t{0};
    if (singular) {
      energy =
          coeff.n == 0
              ? coeff.k * (real_t{1} + static_cast<real_t>(coeff.d))
              : coeff.k;
    } else {
      const real_t delta = static_cast<real_t>(coeff.n) * theta;
      energy =
          coeff.k *
          (real_t{1} + static_cast<real_t>(coeff.d) * cos(delta));
      dE_dtheta =
          -coeff.k *
          static_cast<real_t>(coeff.d) *
          static_cast<real_t>(coeff.n) *
          sin(delta);
    }
    if constexpr (NeedDihedralPotentialEnergy) {
      thread_energy += energy;
    }

    if (singular) {
      continue;
    }

    const real_t norm_bc = sqrt(norm_bc_sq);
    const real_t force_i_scale =
        -dE_dtheta * norm_bc / norm_cross0_sq;
    const real_t force_l_scale =
        dE_dtheta * norm_bc / norm_cross1_sq;
    const Vec3 force_i = scale(cp0, force_i_scale);
    const Vec3 force_l = scale(cp1, force_l_scale);
    const real_t inv_norm_bc_sq = real_t{1} / norm_bc_sq;
    const Vec3 s = subtract(
        scale(force_i, dot(v0, v1) * inv_norm_bc_sq),
        scale(force_l, dot(v2, v1) * inv_norm_bc_sq));
    const Vec3 force_j = subtract(s, force_i);
    const Vec3 force_k = scale(add(s, force_l), real_t{-1});

    atomicAdd(&forces.force_x[slot_i], force_i.x);
    atomicAdd(&forces.force_y[slot_i], force_i.y);
    atomicAdd(&forces.force_z[slot_i], force_i.z);
    atomicAdd(&forces.force_x[slot_j], force_j.x);
    atomicAdd(&forces.force_y[slot_j], force_j.y);
    atomicAdd(&forces.force_z[slot_j], force_j.z);
    atomicAdd(&forces.force_x[slot_k], force_k.x);
    atomicAdd(&forces.force_y[slot_k], force_k.y);
    atomicAdd(&forces.force_z[slot_k], force_k.z);
    atomicAdd(&forces.force_x[slot_l], force_l.x);
    atomicAdd(&forces.force_y[slot_l], force_l.y);
    atomicAdd(&forces.force_z[slot_l], force_l.z);

    if constexpr (NeedGlobalScalarVirial) {
      const Vec3 arm_lj = subtract(v1, v2);
      thread_virial +=
          dot(v0, force_i) + dot(v1, force_k) + dot(arm_lj, force_l);
    }
  }

  if constexpr (NeedDihedralPotentialEnergy) {
    using BlockReduce = cub::BlockReduce<real_t, kDihedralForceBlockSize>;
    __shared__ typename BlockReduce::TempStorage energy_reduce_storage;
    const real_t block_energy =
        BlockReduce(energy_reduce_storage).Sum(thread_energy);
    if (threadIdx.x == 0) {
      dihedral_pe_partials[blockIdx.x] = block_energy;
    }
  }
  if constexpr (NeedGlobalScalarVirial) {
    using BlockReduce = cub::BlockReduce<real_t, kDihedralForceBlockSize>;
    __shared__ typename BlockReduce::TempStorage virial_reduce_storage;
    const real_t block_virial =
        BlockReduce(virial_reduce_storage).Sum(thread_virial);
    if (threadIdx.x == 0) {
      global_virial_partials[blockIdx.x] = block_virial;
    }
  }
}

}  // namespace

HarmonicDihedralModel::HarmonicDihedralModel()
    : DihedralModel(kStyleName) {}

void HarmonicDihedralModel::read_settings(
    const input::DihedralStyleSpec& dihedral_style) {
  require_exact_parameter_keys(dihedral_style.params, {}, "dihedral_style");
}

void HarmonicDihedralModel::begin_topology(
    const system::state::HostState& host_state) {
  const auto& host_dihedrals = host_state.topology().dihedrals();

  type_id_t max_type = 0;
  host_dihedrals_.clear();
  host_dihedrals_.reserve(host_dihedrals.size());
  for (const auto& dihedral : host_dihedrals) {
    host_dihedrals_.push_back(
        {dihedral.tag_i,
         dihedral.tag_j,
         dihedral.tag_k,
         dihedral.tag_l,
         dihedral.type});
    if (dihedral.type > max_type) {
      max_type = dihedral.type;
    }
  }

  host_coeffs_.assign(static_cast<std::size_t>(max_type) + 1u, DeviceCoeff{});
  coeff_filled_.assign(static_cast<std::size_t>(max_type) + 1u, false);
}

void HarmonicDihedralModel::read_coeff(
    const input::DihedralCoeffSpec& coeff) {
  if (coeff.type < 1 ||
      static_cast<std::size_t>(coeff.type) >= host_coeffs_.size()) {
    throw std::invalid_argument(
        "Dihedral(\"harmonic\") dihedral_coeff type is not active in topology dihedrals.");
  }
  const auto index = static_cast<std::size_t>(coeff.type);
  if (coeff_filled_[index]) {
    throw std::invalid_argument(
        "Dihedral(\"harmonic\") dihedral_coeffs must not contain duplicates.");
  }
  host_coeffs_[index] = make_coeff(coeff);
  coeff_filled_[index] = true;
}

void HarmonicDihedralModel::finish_configuration() {
  for (std::size_t type = 1; type < coeff_filled_.size(); ++type) {
    if (!coeff_filled_[type]) {
      throw std::invalid_argument(
          "Dihedral(\"harmonic\") dihedral_coeffs must cover all active topology dihedral types.");
    }
  }

  upload_entries(host_dihedrals_);
  upload_coeffs(host_coeffs_);
  std::vector<DihedralEntry>().swap(host_dihedrals_);
  std::vector<DeviceCoeff>().swap(host_coeffs_);
  std::vector<bool>().swap(coeff_filled_);
}

auto HarmonicDihedralModel::make_coeff(
    const input::DihedralCoeffSpec& coeff) const -> DeviceCoeff {
  require_exact_parameter_keys(coeff.params, {"k", "d", "n"}, "dihedral_coeff");
  const std::int64_t d = require_integer_parameter(
      coeff.params,
      "d",
      "dihedral_coeff");
  if (d != -1 && d != 1) {
    throw std::invalid_argument(
        "Dihedral(\"harmonic\") dihedral_coeff.d must be +1 or -1.");
  }
  const std::int64_t n = require_integer_parameter(
      coeff.params,
      "n",
      "dihedral_coeff");
  if (n < 0 || n > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "Dihedral(\"harmonic\") dihedral_coeff.n must be non-negative.");
  }
  return DeviceCoeff{
      require_nonnegative_real_parameter(
          coeff.params,
          "k",
          "dihedral_coeff"),
      static_cast<int>(d),
      static_cast<int>(n)};
}

void HarmonicDihedralModel::upload_entries(
    const std::vector<DihedralEntry>& entries) {
  dihedral_count_ = static_cast<index_t>(entries.size());
  dihedrals_.resize(entries.size());
  if (!entries.empty()) {
    BEADS_CUDA_CHECK(cudaMemcpy(
        dihedrals_.data(),
        entries.data(),
        entries.size() * sizeof(DihedralEntry),
        cudaMemcpyHostToDevice));
  }
}

void HarmonicDihedralModel::upload_coeffs(
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

ForceEvalObservableLayout HarmonicDihedralModel::observable_layout(
    const ForceEvalRequest& request) const {
  return make_component_observable_layout(
      request,
      ForceObservable::DihedralPotentialEnergy,
      static_cast<index_t>(force_grid_size(dihedral_count_, kDihedralForceBlockSize)),
      "Dihedral(\"harmonic\")");
}

void HarmonicDihedralModel::add_forces(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const system::state::TagToSlotMap& tag_to_slot_map,
    const system::geometry::BoxGeometry& box,
    cudaStream_t stream) const {
  const int grid_size = force_grid_size(dihedral_count_, kDihedralForceBlockSize);
  compute_harmonic_dihedral_forces_kernel<false, false><<<
      grid_size,
      kDihedralForceBlockSize,
      0,
      stream>>>(
      dihedral_count_,
      dihedrals_.data(),
      coeffs_.data(),
      tag_to_slot_map.slots_by_tag().data(),
      particles.view(),
      forces.view(),
      system::geometry::make_box_geometry_view(box),
      nullptr,
      nullptr);
  BEADS_CUDA_CHECK(cudaGetLastError());
}

void HarmonicDihedralModel::add_forces_and_observables(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const system::state::TagToSlotMap& tag_to_slot_map,
    const system::geometry::BoxGeometry& box,
    const ForceEvalRequest& request,
    const ForceObservableBuffers& buffers,
    cudaStream_t stream) const {
  const bool need_dihedral_pe =
      request.has(ForceObservable::DihedralPotentialEnergy);
  const bool need_global_virial =
      request.has(ForceObservable::GlobalScalarVirial);
  if (!need_dihedral_pe && !need_global_virial) {
    add_forces(particles, forces, tag_to_slot_map, box, stream);
    return;
  }

  const int grid_size = force_grid_size(dihedral_count_, kDihedralForceBlockSize);
  if (need_dihedral_pe) {
    require_observable_buffer_shape(
        buffers,
        ForceObservable::DihedralPotentialEnergy,
        static_cast<index_t>(grid_size),
        "Harmonic dihedral potential energy buffer has wrong shape.");
  }
  if (need_global_virial) {
    require_observable_buffer_shape(
        buffers,
        ForceObservable::GlobalScalarVirial,
        static_cast<index_t>(grid_size),
        "Harmonic dihedral global scalar virial buffer has wrong shape.");
  }

  if (need_dihedral_pe && need_global_virial) {
    compute_harmonic_dihedral_forces_kernel<true, true><<<
        grid_size,
        kDihedralForceBlockSize,
        0,
        stream>>>(
        dihedral_count_,
        dihedrals_.data(),
        coeffs_.data(),
        tag_to_slot_map.slots_by_tag().data(),
        particles.view(),
        forces.view(),
        system::geometry::make_box_geometry_view(box),
        buffers.dihedral_pe_partials,
        buffers.global_virial_partials);
  } else if (need_dihedral_pe) {
    compute_harmonic_dihedral_forces_kernel<true, false><<<
        grid_size,
        kDihedralForceBlockSize,
        0,
        stream>>>(
        dihedral_count_,
        dihedrals_.data(),
        coeffs_.data(),
        tag_to_slot_map.slots_by_tag().data(),
        particles.view(),
        forces.view(),
        system::geometry::make_box_geometry_view(box),
        buffers.dihedral_pe_partials,
        nullptr);
  } else {
    compute_harmonic_dihedral_forces_kernel<false, true><<<
        grid_size,
        kDihedralForceBlockSize,
        0,
        stream>>>(
        dihedral_count_,
        dihedrals_.data(),
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

}  // namespace dihedral
}  // namespace forcefield
}  // namespace beads
