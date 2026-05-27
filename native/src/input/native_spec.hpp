#pragma once

#include <beads/core/types.hpp>

#include <cstdint>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace beads {
namespace input {

using StyleParamValue = std::variant<bool, std::int64_t, double, std::string>;
using StyleParamMap = std::unordered_map<std::string, StyleParamValue>;

template <typename T>
struct ArrayViewSpec {
  const T* data = nullptr;
  std::vector<std::size_t> shape;
};

struct BondTopologySpec {
  index_t tag_i = 0;
  index_t tag_j = 0;
  type_id_t type = 0;
};

struct AngleTopologySpec {
  index_t tag_i = 0;
  index_t tag_j = 0;
  index_t tag_k = 0;
  type_id_t type = 0;
};

struct DihedralTopologySpec {
  index_t tag_i = 0;
  index_t tag_j = 0;
  index_t tag_k = 0;
  index_t tag_l = 0;
  type_id_t type = 0;
};

struct TopologySpec {
  std::vector<BondTopologySpec> bonds;
  std::vector<AngleTopologySpec> angles;
  std::vector<DihedralTopologySpec> dihedrals;
};

struct SystemSpec {
  std::string units;
  index_t n_particles = 0;
  ArrayViewSpec<real_t> box_bound;
  ArrayViewSpec<real_t> positions;
  ArrayViewSpec<real_t> velocities;
  ArrayViewSpec<real_t> masses;
  ArrayViewSpec<type_id_t> types;
  ArrayViewSpec<index_t> tags;
  ArrayViewSpec<index_t> molecule_ids;
  ArrayViewSpec<image_t> images;
  TopologySpec topology;
};

struct PairStyleSpec {
  std::string style;
  StyleParamMap params;
};

struct PairCoeffSpec {
  std::string style;
  type_id_t type_i = 0;
  type_id_t type_j = 0;
  StyleParamMap params;
};

struct BondStyleSpec {
  std::string style;
  StyleParamMap params;
};

struct BondCoeffSpec {
  std::string style;
  type_id_t type = 0;
  StyleParamMap params;
};

struct AngleStyleSpec {
  std::string style;
  StyleParamMap params;
};

struct AngleCoeffSpec {
  std::string style;
  type_id_t type = 0;
  StyleParamMap params;
};

struct DihedralStyleSpec {
  std::string style;
  StyleParamMap params;
};

struct DihedralCoeffSpec {
  std::string style;
  type_id_t type = 0;
  StyleParamMap params;
};

struct ForceFieldSpec {
  PairStyleSpec pair_style;
  std::vector<PairCoeffSpec> pair_coeffs;
  std::optional<BondStyleSpec> bond_style;
  std::vector<BondCoeffSpec> bond_coeffs;
  std::optional<AngleStyleSpec> angle_style;
  std::vector<AngleCoeffSpec> angle_coeffs;
  std::optional<DihedralStyleSpec> dihedral_style;
  std::vector<DihedralCoeffSpec> dihedral_coeffs;
  std::optional<index_t> bonded_exclusion_distance;
  std::string bonded_exclusion_policy = "none";
};

struct ThermostatSpec {
  std::string style;
  StyleParamMap params;
};

struct DynamicsSpec {
  std::string style;
  StyleParamMap params;
  std::optional<ThermostatSpec> thermostat;
};

struct NeighborSpec {
  real_t cutoff_buffer = real_t{0};
  runstep_t rebuild_check_every = 0;
  runstep_t sort_every_rebuild = 0;
  index_t max_neighbors = 0;
};

struct ThermoOutputSpec {
  runstep_t every = 0;
  std::string prefix;
};

struct FinalStateOutputSpec {
  std::string prefix;
};

struct TrajectoryOutputSpec {
  runstep_t every = 0;
  std::string prefix;
  std::vector<std::string> fields;
};

struct LogOutputSpec {
  std::string echo;
  std::optional<std::string> prefix;
};

struct OutputSpec {
  std::optional<ThermoOutputSpec> thermo;
  std::optional<FinalStateOutputSpec> final_state;
  std::optional<TrajectoryOutputSpec> trajectory;
  std::optional<LogOutputSpec> log;
};

struct SimulationSpec {
  SystemSpec system;
  ForceFieldSpec forcefield;
  DynamicsSpec dynamics;
  NeighborSpec neighbor;
  OutputSpec output;
  runstep_t runsteps = 0;
};

}  // namespace input
}  // namespace beads
