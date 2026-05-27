#include "host_forces.hpp"

#include <cstddef>

namespace beads {
namespace system::state {

HostForces::HostForces(index_t n_particles)
    : n_particles_(n_particles),
      force_x_(static_cast<std::size_t>(n_particles), real_t{0}),
      force_y_(static_cast<std::size_t>(n_particles), real_t{0}),
      force_z_(static_cast<std::size_t>(n_particles), real_t{0}) {}

}  // namespace system::state
}  // namespace beads
