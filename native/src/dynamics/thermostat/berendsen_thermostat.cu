#include "berendsen_thermostat.cuh"

#include <beads/core/cuda_check.cuh>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>

namespace beads {
namespace dynamics::thermostat {
namespace {

constexpr int kBerendsenBlockSize = 256;

bool has_param(
    const input::StyleParamMap& params,
    const char* key) {
  return params.find(key) != params.end();
}

double require_real_parameter(
    const input::StyleParamMap& params,
    const char* key,
    const char* path) {
  const auto iter = params.find(key);
  if (iter == params.end()) {
    throw std::invalid_argument(std::string(path) + "." + key + " is required.");
  }
  if (std::holds_alternative<double>(iter->second)) {
    return std::get<double>(iter->second);
  }
  if (std::holds_alternative<std::int64_t>(iter->second)) {
    return static_cast<double>(std::get<std::int64_t>(iter->second));
  }
  throw std::invalid_argument(std::string(path) + "." + key + " must be numeric.");
}

void require_exact_berendsen_keys(const input::ThermostatSpec& thermostat) {
  if (thermostat.params.size() != 2 ||
      !has_param(thermostat.params, "temperature") ||
      !has_param(thermostat.params, "tau")) {
    throw std::invalid_argument(
        "Thermostat(\"berendsen\") requires exactly parameters "
        "\"temperature\" and \"tau\".");
  }
}

real_t require_nonnegative_real_parameter(
    const input::StyleParamMap& params,
    const char* key,
    const char* path) {
  const double value = require_real_parameter(params, key, path);
  const real_t cast = static_cast<real_t>(value);
  if (!std::isfinite(value) ||
      !std::isfinite(static_cast<double>(cast)) ||
      cast < real_t{0}) {
    throw std::invalid_argument(
        std::string(path) + "." + key + " must be finite and non-negative.");
  }
  return cast;
}

real_t require_positive_real_parameter(
    const input::StyleParamMap& params,
    const char* key,
    const char* path) {
  const double value = require_real_parameter(params, key, path);
  const real_t cast = static_cast<real_t>(value);
  if (!std::isfinite(value) ||
      !std::isfinite(static_cast<double>(cast)) ||
      cast <= real_t{0}) {
    throw std::invalid_argument(
        std::string(path) + "." + key + " must be finite and positive.");
  }
  return cast;
}

int checked_grid_size(index_t n_particles) {
  const auto items = static_cast<std::size_t>(n_particles);
  const auto block = static_cast<std::size_t>(kBerendsenBlockSize);
  const std::size_t block_count = (items + block - 1) / block;
  if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Berendsen thermostat grid size exceeds launch capacity.");
  }
  return static_cast<int>(block_count);
}

int checked_cub_count(index_t count) {
  if (count == 0) {
    throw std::logic_error("Berendsen thermostat reduction count must be positive.");
  }
  if (count > static_cast<index_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Berendsen thermostat reduction count exceeds CUB capacity.");
  }
  return static_cast<int>(count);
}

real_t temperature_dof(index_t n_particles) noexcept {
  const real_t dof = real_t{3} * static_cast<real_t>(n_particles) - real_t{3};
  return dof > real_t{0} ? dof : real_t{0};
}

__global__ void compute_kinetic_partials_kernel(
    system::state::DeviceParticlesView particles,
    real_t* kinetic_partials)
{
  using BlockReduce = cub::BlockReduce<real_t, kBerendsenBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const index_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  const index_t stride = blockDim.x * gridDim.x;
  real_t thread_sum = real_t{0};
  for (index_t particle = thread_index;
       particle < particles.n_particles;
       particle += stride) {
    const real_t vx = particles.velocity_x[particle];
    const real_t vy = particles.velocity_y[particle];
    const real_t vz = particles.velocity_z[particle];
    thread_sum += real_t{0.5} * particles.mass[particle] *
        (vx * vx + vy * vy + vz * vz);
  }

  const real_t block_sum = BlockReduce(temp_storage).Sum(thread_sum);
  if (threadIdx.x == 0) {
    kinetic_partials[blockIdx.x] = block_sum;
  }
}

__global__ void scale_velocities_kernel(
    system::state::DeviceParticlesView particles,
    const real_t* kinetic_total,
    real_t target_thermal_energy,
    real_t tau,
    real_t dt,
    real_t dof)
{
  const real_t kinetic = kinetic_total[0];
  real_t scale = real_t{1};
  if (kinetic > real_t{0} && dof > real_t{0}) {
    const real_t current_thermal_energy = (real_t{2} * kinetic) / dof;
    const real_t argument =
        real_t{1} +
        (dt / tau) *
            ((target_thermal_energy / current_thermal_energy) - real_t{1});
    const real_t clamped_argument =
        argument > real_t{0} ? argument : real_t{0};
    scale = sqrt(clamped_argument);
  }

  const index_t particle = blockIdx.x * blockDim.x + threadIdx.x;
  const index_t stride = blockDim.x * gridDim.x;
  for (index_t i = particle; i < particles.n_particles; i += stride) {
    particles.velocity_x[i] *= scale;
    particles.velocity_y[i] *= scale;
    particles.velocity_z[i] *= scale;
  }
}

}  // namespace

BerendsenThermostat::BerendsenThermostat(
    const input::ThermostatSpec& thermostat,
    const system::units::UnitSystem& units) {
  if (thermostat.style != "berendsen") {
    throw std::invalid_argument(
        "BerendsenThermostat expected thermostat style \"berendsen\".");
  }
  require_exact_berendsen_keys(thermostat);
  const real_t public_temperature = require_nonnegative_real_parameter(
      thermostat.params,
      "temperature",
      "Thermostat(\"berendsen\")");
  target_thermal_energy_ =
      units.thermal_energy_from_temperature(public_temperature);
  if (!std::isfinite(static_cast<double>(target_thermal_energy_)) ||
      target_thermal_energy_ < real_t{0}) {
    throw std::invalid_argument(
        "Thermostat(\"berendsen\").temperature must produce finite thermal energy.");
  }
  tau_ = require_positive_real_parameter(
      thermostat.params,
      "tau",
      "Thermostat(\"berendsen\")");
}

void BerendsenThermostat::ensure_workspace(index_t n_particles) const {
  if (prepared_particle_count_ == n_particles) {
    return;
  }

  const int grid_size = checked_grid_size(n_particles);
  kinetic_partial_count_ = static_cast<index_t>(grid_size);
  cub_partial_count_ = checked_cub_count(kinetic_partial_count_);
  kinetic_partials_.resize(static_cast<std::size_t>(kinetic_partial_count_));
  kinetic_total_.resize(1);

  std::size_t workspace_bytes = 0;
  BEADS_CUDA_CHECK(cub::DeviceReduce::Sum(
      nullptr,
      workspace_bytes,
      kinetic_partials_.data(),
      kinetic_total_.data(),
      cub_partial_count_));
  sum_workspace_.resize(workspace_bytes);
  sum_workspace_bytes_ = workspace_bytes;
  prepared_particle_count_ = n_particles;
}

void BerendsenThermostat::apply(
    system::state::DeviceParticles& particles,
    real_t dt,
    cudaStream_t stream) const {
  ensure_workspace(particles.n_particles());
  const int grid_size = checked_grid_size(particles.n_particles());
  compute_kinetic_partials_kernel<<<
      grid_size,
      kBerendsenBlockSize,
      0,
      stream>>>(
      particles.view(),
      kinetic_partials_.data());
  BEADS_CUDA_CHECK(cudaGetLastError());

  std::size_t workspace_bytes = sum_workspace_bytes_;
  BEADS_CUDA_CHECK(cub::DeviceReduce::Sum(
      static_cast<void*>(sum_workspace_.data()),
      workspace_bytes,
      kinetic_partials_.data(),
      kinetic_total_.data(),
      cub_partial_count_,
      stream));

  scale_velocities_kernel<<<
      grid_size,
      kBerendsenBlockSize,
      0,
      stream>>>(
      particles.view(),
      kinetic_total_.data(),
      target_thermal_energy_,
      tau_,
      dt,
      temperature_dof(particles.n_particles()));
  BEADS_CUDA_CHECK(cudaGetLastError());
}

std::unique_ptr<Thermostat> make_berendsen_thermostat(
    const input::ThermostatSpec& thermostat,
    const system::units::UnitSystem& units) {
  return std::make_unique<BerendsenThermostat>(thermostat, units);
}

}  // namespace dynamics::thermostat
}  // namespace beads
