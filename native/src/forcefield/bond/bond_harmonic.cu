#include "bond_harmonic.hpp"

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
namespace bond {
namespace {

inline constexpr int kBondForceBlockSize = 256;

int bond_force_grid_size(index_t bond_count) {
  const auto items = static_cast<std::size_t>(bond_count);
  const auto block = static_cast<std::size_t>(kBondForceBlockSize);
  const std::size_t block_count = (items + block - 1) / block;
  if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Harmonic bond CUDA grid size exceeds launch capacity.");
  }
  return static_cast<int>(block_count);
}

template <bool NeedBondPotentialEnergy, bool NeedGlobalScalarVirial>
__global__ void compute_harmonic_bond_forces_kernel(
    index_t bond_count,
    const HarmonicBondModel::BondEntry* bonds,
    const HarmonicBondModel::DeviceCoeff* coeffs,
    const index_t* slots_by_tag,
    system::state::DeviceParticlesConstView particles,
    system::state::DeviceForcesView forces,
    system::geometry::BoxGeometryView box,
    real_t* bond_pe_partials,
    real_t* global_virial_partials)
{
  real_t thread_energy = real_t{0};
  real_t thread_virial = real_t{0};
  const index_t thread = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  const index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);
  for (index_t bond_index = thread; bond_index < bond_count; bond_index += stride) {
    const HarmonicBondModel::BondEntry entry = bonds[bond_index];
    const index_t slot_i = slots_by_tag[entry.tag_i];
    const index_t slot_j = slots_by_tag[entry.tag_j];

    const real_t dx = system::geometry::minimum_image_delta_x_for_wrapped_positions(
        box, particles.position_x[slot_i] - particles.position_x[slot_j]);
    const real_t dy = system::geometry::minimum_image_delta_y_for_wrapped_positions(
        box, particles.position_y[slot_i] - particles.position_y[slot_j]);
    const real_t dz = system::geometry::minimum_image_delta_z_for_wrapped_positions(
        box, particles.position_z[slot_i] - particles.position_z[slot_j]);
    const real_t r2 = dx * dx + dy * dy + dz * dz;
    const real_t r = sqrt(r2);
    const auto coeff = coeffs[entry.type];
    const real_t stretch = r - coeff.r0;

    if constexpr (NeedBondPotentialEnergy) {
      thread_energy += real_t{0.5} * coeff.k * stretch * stretch;
    }

    if (r == real_t{0}) {
      continue;
    }

    const real_t scale = -coeff.k * stretch / r;
    const real_t fx = scale * dx;
    const real_t fy = scale * dy;
    const real_t fz = scale * dz;
    atomicAdd(&forces.force_x[slot_i], fx);
    atomicAdd(&forces.force_y[slot_i], fy);
    atomicAdd(&forces.force_z[slot_i], fz);
    atomicAdd(&forces.force_x[slot_j], -fx);
    atomicAdd(&forces.force_y[slot_j], -fy);
    atomicAdd(&forces.force_z[slot_j], -fz);

    if constexpr (NeedGlobalScalarVirial) {
      thread_virial += dx * fx + dy * fy + dz * fz;
    }
  }

  if constexpr (NeedBondPotentialEnergy) {
    using BlockReduce = cub::BlockReduce<real_t, kBondForceBlockSize>;
    __shared__ typename BlockReduce::TempStorage energy_reduce_storage;
    const real_t block_energy =
        BlockReduce(energy_reduce_storage).Sum(thread_energy);
    if (threadIdx.x == 0) {
      bond_pe_partials[blockIdx.x] = block_energy;
    }
  }
  if constexpr (NeedGlobalScalarVirial) {
    using BlockReduce = cub::BlockReduce<real_t, kBondForceBlockSize>;
    __shared__ typename BlockReduce::TempStorage virial_reduce_storage;
    const real_t block_virial =
        BlockReduce(virial_reduce_storage).Sum(thread_virial);
    if (threadIdx.x == 0) {
      global_virial_partials[blockIdx.x] = block_virial;
    }
  }
}

}  // namespace

HarmonicBondModel::HarmonicBondModel()
    : BondModel(kStyleName) {}

void HarmonicBondModel::read_settings(
    const input::BondStyleSpec& bond_style) {
  require_exact_parameter_keys(bond_style.params, {}, "bond_style");
}

void HarmonicBondModel::begin_topology(
    const system::state::HostState& host_state) {
  const auto& host_bonds = host_state.topology().bonds();

  type_id_t max_type = 0;
  host_bonds_.clear();
  host_bonds_.reserve(host_bonds.size());
  for (const auto& bond : host_bonds) {
    host_bonds_.push_back({bond.tag_i, bond.tag_j, bond.type});
    if (bond.type > max_type) {
      max_type = bond.type;
    }
  }

  host_coeffs_.assign(static_cast<std::size_t>(max_type) + 1u, DeviceCoeff{});
  coeff_filled_.assign(static_cast<std::size_t>(max_type) + 1u, false);
}

void HarmonicBondModel::read_coeff(const input::BondCoeffSpec& coeff) {
  if (coeff.type < 1 ||
      static_cast<std::size_t>(coeff.type) >= host_coeffs_.size()) {
    throw std::invalid_argument(
        "Bond(\"harmonic\") bond_coeff type is not active in topology bonds.");
  }
  const auto index = static_cast<std::size_t>(coeff.type);
  if (coeff_filled_[index]) {
    throw std::invalid_argument(
        "Bond(\"harmonic\") bond_coeffs must not contain duplicates.");
  }
  host_coeffs_[index] = make_coeff(coeff);
  coeff_filled_[index] = true;
}

void HarmonicBondModel::finish_configuration() {
  for (std::size_t type = 1; type < coeff_filled_.size(); ++type) {
    if (!coeff_filled_[type]) {
      throw std::invalid_argument(
          "Bond(\"harmonic\") bond_coeffs must cover all active topology bond types.");
    }
  }

  upload_entries(host_bonds_);
  upload_coeffs(host_coeffs_);
  std::vector<BondEntry>().swap(host_bonds_);
  std::vector<DeviceCoeff>().swap(host_coeffs_);
  std::vector<bool>().swap(coeff_filled_);
}

auto HarmonicBondModel::make_coeff(
    const input::BondCoeffSpec& coeff) const -> DeviceCoeff {
  require_exact_parameter_keys(coeff.params, {"k", "r0"}, "bond_coeff");
  return DeviceCoeff{
      require_nonnegative_real_parameter(
          coeff.params,
          "k",
          "bond_coeff"),
      require_nonnegative_real_parameter(
          coeff.params,
          "r0",
          "bond_coeff")};
}

void HarmonicBondModel::upload_entries(
    const std::vector<BondEntry>& entries) {
  bond_count_ = static_cast<index_t>(entries.size());
  bonds_.resize(entries.size());
  if (!entries.empty()) {
    BEADS_CUDA_CHECK(cudaMemcpy(
        bonds_.data(),
        entries.data(),
        entries.size() * sizeof(BondEntry),
        cudaMemcpyHostToDevice));
  }
}

void HarmonicBondModel::upload_coeffs(
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

ForceEvalObservableLayout HarmonicBondModel::observable_layout(
    const ForceEvalRequest& request) const {
  return make_component_observable_layout(
      request,
      ForceObservable::BondPotentialEnergy,
      static_cast<index_t>(bond_force_grid_size(bond_count_)),
      "Bond(\"harmonic\")");
}

void HarmonicBondModel::add_forces(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const system::state::TagToSlotMap& tag_to_slot_map,
    const system::geometry::BoxGeometry& box,
    cudaStream_t stream) const {
  const int grid_size = bond_force_grid_size(bond_count_);
  compute_harmonic_bond_forces_kernel<false, false><<<
      grid_size,
      kBondForceBlockSize,
      0,
      stream>>>(
      bond_count_,
      bonds_.data(),
      coeffs_.data(),
      tag_to_slot_map.slots_by_tag().data(),
      particles.view(),
      forces.view(),
      system::geometry::make_box_geometry_view(box),
      nullptr,
      nullptr);
  BEADS_CUDA_CHECK(cudaGetLastError());
}

void HarmonicBondModel::add_forces_and_observables(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const system::state::TagToSlotMap& tag_to_slot_map,
    const system::geometry::BoxGeometry& box,
    const ForceEvalRequest& request,
    const ForceObservableBuffers& buffers,
    cudaStream_t stream) const {
  const bool need_bond_pe =
      request.has(ForceObservable::BondPotentialEnergy);
  const bool need_global_virial =
      request.has(ForceObservable::GlobalScalarVirial);
  if (!need_bond_pe && !need_global_virial) {
    add_forces(particles, forces, tag_to_slot_map, box, stream);
    return;
  }

  const int grid_size = bond_force_grid_size(bond_count_);
  if (need_bond_pe) {
    require_observable_buffer_shape(
        buffers,
        ForceObservable::BondPotentialEnergy,
        static_cast<index_t>(grid_size),
        "Harmonic bond potential energy buffer has wrong shape.");
  }
  if (need_global_virial) {
    require_observable_buffer_shape(
        buffers,
        ForceObservable::GlobalScalarVirial,
        static_cast<index_t>(grid_size),
        "Harmonic bond global scalar virial buffer has wrong shape.");
  }

  if (need_bond_pe && need_global_virial) {
    compute_harmonic_bond_forces_kernel<true, true><<<
        grid_size,
        kBondForceBlockSize,
        0,
        stream>>>(
        bond_count_,
        bonds_.data(),
        coeffs_.data(),
        tag_to_slot_map.slots_by_tag().data(),
        particles.view(),
        forces.view(),
        system::geometry::make_box_geometry_view(box),
        buffers.bond_pe_partials,
        buffers.global_virial_partials);
  } else if (need_bond_pe) {
    compute_harmonic_bond_forces_kernel<true, false><<<
        grid_size,
        kBondForceBlockSize,
        0,
        stream>>>(
        bond_count_,
        bonds_.data(),
        coeffs_.data(),
        tag_to_slot_map.slots_by_tag().data(),
        particles.view(),
        forces.view(),
        system::geometry::make_box_geometry_view(box),
        buffers.bond_pe_partials,
        nullptr);
  } else {
    compute_harmonic_bond_forces_kernel<false, true><<<
        grid_size,
        kBondForceBlockSize,
        0,
        stream>>>(
        bond_count_,
        bonds_.data(),
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

}  // namespace bond
}  // namespace forcefield
}  // namespace beads
