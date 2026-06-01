#include "pair_lj.hpp"

#include <beads/core/cuda_check.cuh>
#include <system/geometry/box_geometry_view.cuh>
#include <simulation/neighbor/neighbor_list.cuh>
#include <system/state/device_forces.cuh>
#include <system/state/device_particles.cuh>

#include <cub/block/block_reduce.cuh>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace beads {
namespace forcefield {
namespace pair {
namespace {

inline constexpr int kPairForceBlockSize = 256;

int pair_force_grid_size(index_t n_particles, int block_size) {
  if (block_size <= 0) {
    throw std::invalid_argument("CUDA block size must be positive.");
  }
  const auto items = static_cast<std::size_t>(n_particles);
  const auto block = static_cast<std::size_t>(block_size);
  const std::size_t block_count = (items + block - 1) / block;
  if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("CUDA grid size exceeds launch capacity.");
  }
  return static_cast<int>(block_count);
}

__global__ void compute_lj_pair_forces_kernel(
    system::state::DeviceParticlesConstView particles,
    system::state::DeviceForcesView forces,
    simulation::neighbor::NeighborListConstView neighbors,
    system::geometry::BoxGeometryView box,
    DenseTypePairTableDeviceView<LjPairModel::DeviceCoeff> coeffs
)
{
  const index_t particle = blockIdx.x * blockDim.x + threadIdx.x;
  const index_t stride = blockDim.x * gridDim.x;
  for (index_t i = particle; i < particles.n_particles; i += stride) {
    const real_t xi = particles.position_x[i];
    const real_t yi = particles.position_y[i];
    const real_t zi = particles.position_z[i];
    const type_id_t type_i = particles.type[i];

    real_t fx = real_t{0};
    real_t fy = real_t{0};
    real_t fz = real_t{0};

    const index_t neighbor_count = neighbors.neighbor_count[i];
    for (index_t slot = 0; slot < neighbor_count; ++slot) {
      const index_t j = neighbors.neighbor_index[simulation::neighbor::neighbor_index_slot(
          slot, neighbors.n_particles, i)];

      const real_t dx = system::geometry::minimum_image_delta_x_for_wrapped_positions(
          box, xi - particles.position_x[j]);
      const real_t dy = system::geometry::minimum_image_delta_y_for_wrapped_positions(
          box, yi - particles.position_y[j]);
      const real_t dz = system::geometry::minimum_image_delta_z_for_wrapped_positions(
          box, zi - particles.position_z[j]);
      const real_t r2 = dx * dx + dy * dy + dz * dz;
      const auto coeff = coeffs.at(type_i, particles.type[j]);
      if (r2 >= coeff.cutoff_sq) {
        continue;
      }

      const real_t sr2 = coeff.sigma2 / r2;
      const real_t sr6 = sr2 * sr2 * sr2;
      const real_t sr12 = sr6 * sr6;
      const real_t scale = coeff.epsilon24 / r2 *
          (real_t{2} * sr12 - sr6);
      fx += scale * dx;
      fy += scale * dy;
      fz += scale * dz;
    }

    forces.force_x[i] = fx;
    forces.force_y[i] = fy;
    forces.force_z[i] = fz;
  }
}

template <bool NeedPairPotentialEnergy, bool NeedGlobalScalarVirial>
__global__ void compute_lj_pair_forces_with_observables_kernel(
    system::state::DeviceParticlesConstView particles,
    system::state::DeviceForcesView forces,
    simulation::neighbor::NeighborListConstView neighbors,
    system::geometry::BoxGeometryView box,
    DenseTypePairTableDeviceView<LjPairModel::DeviceCoeff> coeffs,
    real_t* pair_pe_partials,
    real_t* global_virial_partials
)
{
  using BlockReduce = cub::BlockReduce<real_t, kPairForceBlockSize>;
  __shared__ typename BlockReduce::TempStorage energy_reduce_storage;
  __shared__ typename BlockReduce::TempStorage virial_reduce_storage;

  const index_t particle = blockIdx.x * blockDim.x + threadIdx.x;
  const index_t stride = blockDim.x * gridDim.x;
  real_t thread_energy = real_t{0};
  real_t thread_virial = real_t{0};

  for (index_t i = particle; i < particles.n_particles; i += stride) {
    const real_t xi = particles.position_x[i];
    const real_t yi = particles.position_y[i];
    const real_t zi = particles.position_z[i];
    const type_id_t type_i = particles.type[i];

    real_t fx = real_t{0};
    real_t fy = real_t{0};
    real_t fz = real_t{0};
    real_t particle_energy = real_t{0};
    real_t particle_virial = real_t{0};

    const index_t neighbor_count = neighbors.neighbor_count[i];
    for (index_t slot = 0; slot < neighbor_count; ++slot) {
      const index_t j = neighbors.neighbor_index[simulation::neighbor::neighbor_index_slot(
          slot, neighbors.n_particles, i)];

      const real_t dx = system::geometry::minimum_image_delta_x_for_wrapped_positions(
          box, xi - particles.position_x[j]);
      const real_t dy = system::geometry::minimum_image_delta_y_for_wrapped_positions(
          box, yi - particles.position_y[j]);
      const real_t dz = system::geometry::minimum_image_delta_z_for_wrapped_positions(
          box, zi - particles.position_z[j]);
      const real_t r2 = dx * dx + dy * dy + dz * dz;
      const auto coeff = coeffs.at(type_i, particles.type[j]);
      if (r2 >= coeff.cutoff_sq) {
        continue;
      }

      const real_t sr2 = coeff.sigma2 / r2;
      const real_t sr6 = sr2 * sr2 * sr2;
      const real_t sr12 = sr6 * sr6;
      const real_t scale = coeff.epsilon24 / r2 *
          (real_t{2} * sr12 - sr6);
      const real_t pair_fx = scale * dx;
      const real_t pair_fy = scale * dy;
      const real_t pair_fz = scale * dz;
      fx += pair_fx;
      fy += pair_fy;
      fz += pair_fz;
      if constexpr (NeedPairPotentialEnergy) {
        particle_energy += coeff.epsilon4 * (sr12 - sr6) - coeff.shift_energy;
      }
      if constexpr (NeedGlobalScalarVirial) {
        particle_virial += dx * pair_fx + dy * pair_fy + dz * pair_fz;
      }
    }

    forces.force_x[i] = fx;
    forces.force_y[i] = fy;
    forces.force_z[i] = fz;
    if constexpr (NeedPairPotentialEnergy) {
      thread_energy += real_t{0.5} * particle_energy;
    }
    if constexpr (NeedGlobalScalarVirial) {
      thread_virial += real_t{0.5} * particle_virial;
    }
  }

  if constexpr (NeedPairPotentialEnergy) {
    const real_t block_energy =
        BlockReduce(energy_reduce_storage).Sum(thread_energy);
    if (threadIdx.x == 0) {
      pair_pe_partials[blockIdx.x] = block_energy;
    }
  }
  if constexpr (NeedGlobalScalarVirial) {
    const real_t block_virial =
        BlockReduce(virial_reduce_storage).Sum(thread_virial);
    if (threadIdx.x == 0) {
      global_virial_partials[blockIdx.x] = block_virial;
    }
  }
}

}  // namespace

auto LjPairModel::pack_device_coeff(
    real_t epsilon,
    real_t sigma,
    real_t cutoff,
    bool shift) -> DeviceCoeff {
  const real_t sigma2 = sigma * sigma;
  const real_t epsilon4 = real_t{4} * epsilon;

  real_t shift_energy = real_t{0};
  if (shift) {
    const real_t sr2 = sigma2 / (cutoff * cutoff);
    const real_t sr6 = sr2 * sr2 * sr2;
    const real_t sr12 = sr6 * sr6;
    shift_energy = epsilon4 * (sr12 - sr6);
  }
  return DeviceCoeff{
      sigma2,
      epsilon4,
      real_t{6} * epsilon4,
      cutoff * cutoff,
      shift_energy};
}

LjPairModel::LjPairModel() : PairModel(kStyleName) {}

void LjPairModel::read_settings(const input::PairStyleSpec& pair_style) {
  require_allowed_parameter_keys(pair_style.params, {"cutoff", "shift"}, "pair_style");
  cutoff_ = require_positive_real_parameter(
      pair_style.params,
      "cutoff",
      "pair_style");
  shift_ = optional_boolean_parameter(
      pair_style.params,
      "shift",
      false,
      "pair_style");
}

void LjPairModel::begin_coeffs(type_id_t active_type_count) {
  coeffs_.resize(active_type_count);
}

void LjPairModel::read_coeff(const input::PairCoeffSpec& pair_coeff) {
  require_exact_parameter_keys(
      pair_coeff.params,
      {"epsilon", "sigma"},
      "pair_coeff");
  const real_t epsilon = require_positive_real_parameter(
      pair_coeff.params,
      "epsilon",
      "pair_coeff");
  const real_t sigma = require_positive_real_parameter(
      pair_coeff.params,
      "sigma",
      "pair_coeff");

  coeffs_.set_symmetric(
      pair_coeff.type_i,
      pair_coeff.type_j,
      pack_device_coeff(epsilon, sigma, cutoff_, shift_),
      "Pair(\"lj\") pair_coeff");
}

void LjPairModel::finish_configuration() {
  coeffs_.upload_and_release_host("Pair(\"lj\") pair_coeff");
}

ForceEvalObservableLayout LjPairModel::observable_layout(
    index_t n_particles,
    const ForceEvalRequest& request) const {
  return make_component_observable_layout(
      request,
      ForceObservable::PairPotentialEnergy,
      static_cast<index_t>(pair_force_grid_size(n_particles, kPairForceBlockSize)),
      "Pair(\"lj\")");
}

void LjPairModel::compute_forces(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const simulation::neighbor::NeighborList& neighbor_list,
    const system::geometry::BoxGeometry& box,
    cudaStream_t stream) const {
  const int grid_size = pair_force_grid_size(
      particles.n_particles(),
      kPairForceBlockSize);
  compute_lj_pair_forces_kernel<<<grid_size, kPairForceBlockSize, 0, stream>>>(
      particles.view(),
      forces.view(),
      neighbor_list.view(),
      system::geometry::make_box_geometry_view(box),
      coeffs_.device_view());
  BEADS_CUDA_CHECK(cudaGetLastError());
}

void LjPairModel::evaluate_forces(
    const system::state::DeviceParticles& particles,
    system::state::DeviceForces& forces,
    const simulation::neighbor::NeighborList& neighbor_list,
    const system::geometry::BoxGeometry& box,
    const ForceEvalRequest& request,
    const ForceObservableBuffers& buffers,
    cudaStream_t stream) const {
  const bool need_pair_pe =
      request.has(ForceObservable::PairPotentialEnergy);
  const bool need_global_virial =
      request.has(ForceObservable::GlobalScalarVirial);
  if (!need_pair_pe && !need_global_virial) {
    compute_forces(particles, forces, neighbor_list, box, stream);
    return;
  }

  const int grid_size = pair_force_grid_size(
      particles.n_particles(),
      kPairForceBlockSize);
  if (need_pair_pe) {
    require_observable_buffer_shape(
        buffers,
        ForceObservable::PairPotentialEnergy,
        static_cast<index_t>(grid_size),
        "LJ pair potential energy buffer has wrong shape.");
  }
  if (need_global_virial) {
    require_observable_buffer_shape(
        buffers,
        ForceObservable::GlobalScalarVirial,
        static_cast<index_t>(grid_size),
        "LJ global scalar virial buffer has wrong shape.");
  }

  if (need_pair_pe && need_global_virial) {
    compute_lj_pair_forces_with_observables_kernel<true, true><<<
        grid_size,
        kPairForceBlockSize,
        0,
        stream>>>(
        particles.view(),
        forces.view(),
        neighbor_list.view(),
        system::geometry::make_box_geometry_view(box),
        coeffs_.device_view(),
        buffers.pair_pe_partials,
        buffers.global_virial_partials);
  } else if (need_pair_pe) {
    compute_lj_pair_forces_with_observables_kernel<true, false><<<
        grid_size,
        kPairForceBlockSize,
        0,
        stream>>>(
        particles.view(),
        forces.view(),
        neighbor_list.view(),
        system::geometry::make_box_geometry_view(box),
        coeffs_.device_view(),
        buffers.pair_pe_partials,
        nullptr);
  } else {
    compute_lj_pair_forces_with_observables_kernel<false, true><<<
        grid_size,
        kPairForceBlockSize,
        0,
        stream>>>(
        particles.view(),
        forces.view(),
        neighbor_list.view(),
        system::geometry::make_box_geometry_view(box),
        coeffs_.device_view(),
        nullptr,
        buffers.global_virial_partials);
  }
  BEADS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace pair
}  // namespace forcefield
}  // namespace beads
