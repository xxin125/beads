#include "topology.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace beads {
namespace system::topology {
namespace {

void require_count_fits(std::size_t count, const char* label) {
  if (count > static_cast<std::size_t>(std::numeric_limits<index_t>::max())) {
    throw std::invalid_argument(
        std::string(label) + " exceeds supported index_t range.");
  }
}

}  // namespace

HostTopology::HostTopology(const input::TopologySpec& topology) {
  require_count_fits(topology.bonds.size(), "system.topology.bonds count");
  require_count_fits(topology.angles.size(), "system.topology.angles count");
  require_count_fits(
      topology.dihedrals.size(), "system.topology.dihedrals count");

  bonds_.reserve(topology.bonds.size());
  for (const input::BondTopologySpec& bond : topology.bonds) {
    bonds_.push_back({bond.tag_i, bond.tag_j, bond.type});
  }

  angles_.reserve(topology.angles.size());
  for (const input::AngleTopologySpec& angle : topology.angles) {
    angles_.push_back({angle.tag_i, angle.tag_j, angle.tag_k, angle.type});
  }

  dihedrals_.reserve(topology.dihedrals.size());
  for (const input::DihedralTopologySpec& dihedral : topology.dihedrals) {
    dihedrals_.push_back(
        {dihedral.tag_i,
         dihedral.tag_j,
         dihedral.tag_k,
         dihedral.tag_l,
         dihedral.type});
  }
}

index_t HostTopology::bond_count() const noexcept {
  return static_cast<index_t>(bonds_.size());
}

index_t HostTopology::angle_count() const noexcept {
  return static_cast<index_t>(angles_.size());
}

index_t HostTopology::dihedral_count() const noexcept {
  return static_cast<index_t>(dihedrals_.size());
}

bool HostTopology::empty() const noexcept {
  return bonds_.empty() && angles_.empty() && dihedrals_.empty();
}

}  // namespace system::topology
}  // namespace beads
