#include "velocity_verlet.cuh"

#include <beads/core/cuda_check.cuh>
#include <input/native_spec.hpp>
#include <system/geometry/box_geometry_view.cuh>

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <variant>

namespace beads {
namespace dynamics {
namespace integrator {
namespace {

inline constexpr int kVelocityVerletBlockSize = 256;

real_t require_velocity_verlet_dt(const input::DynamicsSpec& dynamics) {
  const auto found = dynamics.params.find("dt");
  if (dynamics.params.size() != 1 || found == dynamics.params.end()) {
    throw std::invalid_argument(
        "dynamics.params for style \"velocity_verlet\" requires exactly parameter \"dt\".");
  }

  double value = 0.0;
  if (std::holds_alternative<double>(found->second)) {
    value = std::get<double>(found->second);
  } else if (std::holds_alternative<std::int64_t>(found->second)) {
    value = static_cast<double>(std::get<std::int64_t>(found->second));
  } else {
    throw std::invalid_argument("dynamics.params.dt must be numeric.");
  }

  const real_t dt = static_cast<real_t>(value);
  if (!std::isfinite(value) || !std::isfinite(static_cast<double>(dt)) ||
      dt <= real_t{0}) {
    throw std::invalid_argument("dynamics.params.dt must be finite and positive.");
  }
  return dt;
}

int velocity_verlet_grid_size(index_t n_particles, int block_size) {
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

__global__ void velocity_verlet_pre_force_kernel(
    system::state::DeviceParticlesView particles,
    system::state::DeviceForcesConstView forces,
    system::geometry::BoxGeometryView box,
    real_t dt
)
{
  const index_t particle = blockIdx.x * blockDim.x + threadIdx.x;
  const index_t stride = blockDim.x * gridDim.x;
  for (index_t i = particle; i < particles.n_particles; i += stride) {
    const real_t half_dt_over_mass = real_t{0.5} * dt / particles.mass[i];
    const real_t vx = particles.velocity_x[i] +
        half_dt_over_mass * forces.force_x[i];
    const real_t vy = particles.velocity_y[i] +
        half_dt_over_mass * forces.force_y[i];
    const real_t vz = particles.velocity_z[i] +
        half_dt_over_mass * forces.force_z[i];

    real_t x = particles.position_x[i] + dt * vx;
    real_t y = particles.position_y[i] + dt * vy;
    real_t z = particles.position_z[i] + dt * vz;

    const image_t dx = system::geometry::wrap_position_x_with_delta(box, x);
    const image_t dy = system::geometry::wrap_position_y_with_delta(box, y);
    const image_t dz = system::geometry::wrap_position_z_with_delta(box, z);

    particles.velocity_x[i] = vx;
    particles.velocity_y[i] = vy;
    particles.velocity_z[i] = vz;
    particles.position_x[i] = x;
    particles.position_y[i] = y;
    particles.position_z[i] = z;
    particles.image_x[i] += dx;
    particles.image_y[i] += dy;
    particles.image_z[i] += dz;
  }
}

__global__ void velocity_verlet_post_force_kernel(
    system::state::DeviceParticlesView particles,
    system::state::DeviceForcesConstView forces,
    real_t dt
)
{
  const index_t particle = blockIdx.x * blockDim.x + threadIdx.x;
  const index_t stride = blockDim.x * gridDim.x;
  for (index_t i = particle; i < particles.n_particles; i += stride) {
    const real_t half_dt_over_mass = real_t{0.5} * dt / particles.mass[i];
    particles.velocity_x[i] += half_dt_over_mass * forces.force_x[i];
    particles.velocity_y[i] += half_dt_over_mass * forces.force_y[i];
    particles.velocity_z[i] += half_dt_over_mass * forces.force_z[i];
  }
}

}  // namespace

VelocityVerletIntegrator::VelocityVerletIntegrator(real_t dt) : dt_(dt) {
  if (!std::isfinite(static_cast<double>(dt_)) || dt_ <= real_t{0}) {
    throw std::invalid_argument(
        "VelocityVerletIntegrator dt must be finite and positive.");
  }
}

std::unique_ptr<Integrator> make_velocity_verlet_integrator(
    const input::DynamicsSpec& dynamics) {
  return std::make_unique<VelocityVerletIntegrator>(
      require_velocity_verlet_dt(dynamics));
}

void VelocityVerletIntegrator::pre_force(
    system::state::DeviceParticles& particles,
    const system::state::DeviceForces& forces,
    const system::geometry::BoxGeometry& box,
    cudaStream_t stream) const {
  const int grid_size = velocity_verlet_grid_size(
      particles.n_particles(),
      kVelocityVerletBlockSize);

  velocity_verlet_pre_force_kernel<<<
      grid_size,
      kVelocityVerletBlockSize,
      0,
      stream>>>(
      particles.view(),
      forces.view(),
      system::geometry::make_box_geometry_view(box),
      dt_);
  BEADS_CUDA_CHECK(cudaGetLastError());
}

void VelocityVerletIntegrator::post_force(
    system::state::DeviceParticles& particles,
    const system::state::DeviceForces& forces,
    cudaStream_t stream) const {
  const int grid_size = velocity_verlet_grid_size(
      particles.n_particles(),
      kVelocityVerletBlockSize);

  velocity_verlet_post_force_kernel<<<
      grid_size,
      kVelocityVerletBlockSize,
      0,
      stream>>>(
      particles.view(),
      forces.view(),
      dt_);
  BEADS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace integrator
}  // namespace dynamics
}  // namespace beads
