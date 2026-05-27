#pragma once

#include <beads/core/types.hpp>

#include <vector>

namespace beads {
namespace system::state {

class HostForces {
 public:
  HostForces() = default;
  explicit HostForces(index_t n_particles);

  index_t n_particles() const noexcept { return n_particles_; }
  const std::vector<real_t>& force_x() const noexcept { return force_x_; }
  const std::vector<real_t>& force_y() const noexcept { return force_y_; }
  const std::vector<real_t>& force_z() const noexcept { return force_z_; }

 private:
  index_t n_particles_ = 0;
  std::vector<real_t> force_x_;
  std::vector<real_t> force_y_;
  std::vector<real_t> force_z_;
};

}  // namespace system::state
}  // namespace beads
