#pragma once

#include <beads/core/types.hpp>
#include <input/native_spec.hpp>

#include <vector>

namespace beads {
namespace system::topology {

struct BondRecord {
  index_t tag_i = 0;
  index_t tag_j = 0;
  type_id_t type = 0;
};

struct AngleRecord {
  index_t tag_i = 0;
  index_t tag_j = 0;
  index_t tag_k = 0;
  type_id_t type = 0;
};

struct DihedralRecord {
  index_t tag_i = 0;
  index_t tag_j = 0;
  index_t tag_k = 0;
  index_t tag_l = 0;
  type_id_t type = 0;
};

class HostTopology {
 public:
  HostTopology() = default;
  explicit HostTopology(const input::TopologySpec& topology);

  const std::vector<BondRecord>& bonds() const noexcept { return bonds_; }
  const std::vector<AngleRecord>& angles() const noexcept { return angles_; }
  const std::vector<DihedralRecord>& dihedrals() const noexcept {
    return dihedrals_;
  }

  index_t bond_count() const noexcept;
  index_t angle_count() const noexcept;
  index_t dihedral_count() const noexcept;
  bool empty() const noexcept;

 private:
  std::vector<BondRecord> bonds_;
  std::vector<AngleRecord> angles_;
  std::vector<DihedralRecord> dihedrals_;
};

}  // namespace system::topology
}  // namespace beads
