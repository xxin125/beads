#pragma once

#include <beads/core/types.hpp>
#include <input/native_spec.hpp>

#include <vector>

namespace beads {
namespace system::state {

struct HostParticles {
  HostParticles() = default;
  explicit HostParticles(const input::SystemSpec& system);

  index_t n_particles = 0;
  std::vector<real_t> position_x;
  std::vector<real_t> position_y;
  std::vector<real_t> position_z;
  std::vector<real_t> velocity_x;
  std::vector<real_t> velocity_y;
  std::vector<real_t> velocity_z;
  std::vector<real_t> masses;
  std::vector<type_id_t> types;
  std::vector<index_t> tags;
  std::vector<index_t> molecule_ids;
  std::vector<image_t> image_x;
  std::vector<image_t> image_y;
  std::vector<image_t> image_z;
};

}  // namespace system::state
}  // namespace beads
