#pragma once

#include <input/native_spec.hpp>
#include <system/units/unit_system.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace beads {
namespace input {
namespace validation {

inline void require_finite(
    real_t value,
    const char* path) {
  if (!std::isfinite(static_cast<double>(value))) {
    throw std::invalid_argument(std::string(path) + " must be finite.");
  }
}

inline void validate_system_units(const SystemSpec& system) {
  static_cast<void>(
      system::units::unit_system_from_public_name(system.units));
}

inline void validate_box_bound(const SystemSpec& system) {
  const real_t* box = system.box_bound.data;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const real_t lo = box[axis];
    const real_t hi = box[3 + axis];
    require_finite(lo, "system.box_bound");
    require_finite(hi, "system.box_bound");
    if (!(hi > lo)) {
      throw std::invalid_argument(
          "system.box_bound upper row must be greater than lower row.");
    }
  }
}

inline void validate_real_particle_fields(const SystemSpec& system) {
  const std::size_t n_particles = system.n_particles;
  for (std::size_t particle = 0; particle < n_particles; ++particle) {
    const real_t* box = system.box_bound.data;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const std::size_t coord = 3 * particle + axis;
      const real_t position = system.positions.data[coord];
      const real_t velocity = system.velocities.data[coord];
      require_finite(position, "system.positions");
      require_finite(velocity, "system.velocities");
      if (position < box[axis] || position >= box[3 + axis]) {
        throw std::invalid_argument("system.positions must be inside box_bound.");
      }
    }

    const real_t mass = system.masses.data[particle];
    require_finite(mass, "system.masses");
    if (!(mass > real_t{0})) {
      throw std::invalid_argument("system.masses must be finite and positive.");
    }
  }
}

inline void validate_dense_one_based_types(const SystemSpec& system) {
  std::vector<bool> seen(system.n_particles + 1, false);
  type_id_t max_type = 0;
  for (std::size_t particle = 0; particle < system.n_particles; ++particle) {
    const type_id_t type = system.types.data[particle];
    if (type < 1 || static_cast<std::size_t>(type) > system.n_particles) {
      throw std::invalid_argument(
          "system.types must be dense one-based ids without gaps.");
    }
    seen[static_cast<std::size_t>(type)] = true;
    max_type = std::max(max_type, type);
  }

  for (type_id_t type = 1; type <= max_type; ++type) {
    if (!seen[static_cast<std::size_t>(type)]) {
      throw std::invalid_argument(
          "system.types must be dense one-based ids without gaps.");
    }
  }
}

inline void validate_tag_permutation(const SystemSpec& system) {
  std::vector<bool> seen(system.n_particles + 1, false);
  for (std::size_t particle = 0; particle < system.n_particles; ++particle) {
    const index_t tag = system.tags.data[particle];
    if (tag < 1 || static_cast<std::size_t>(tag) > system.n_particles ||
        seen[static_cast<std::size_t>(tag)]) {
      throw std::invalid_argument("system.tags must be a permutation of 1..N.");
    }
    seen[static_cast<std::size_t>(tag)] = true;
  }
}

inline void validate_contiguous_molecule_ids(const SystemSpec& system) {
  std::vector<bool> seen(system.n_particles + 1, false);
  index_t max_molecule_id = 0;
  for (std::size_t particle = 0; particle < system.n_particles; ++particle) {
    const index_t molecule_id = system.molecule_ids.data[particle];
    if (molecule_id < 1 ||
        static_cast<std::size_t>(molecule_id) > system.n_particles) {
      throw std::invalid_argument(
          "system.molecule_ids must form a contiguous 1-based set starting at 1.");
    }
    seen[static_cast<std::size_t>(molecule_id)] = true;
    max_molecule_id = std::max(max_molecule_id, molecule_id);
  }

  for (index_t molecule_id = 1; molecule_id <= max_molecule_id; ++molecule_id) {
    if (!seen[static_cast<std::size_t>(molecule_id)]) {
      throw std::invalid_argument(
          "system.molecule_ids must form a contiguous 1-based set starting at 1.");
    }
  }
}

inline std::unordered_set<index_t> active_tag_set(const SystemSpec& system) {
  std::unordered_set<index_t> result;
  result.reserve(system.n_particles);
  for (std::size_t particle = 0; particle < system.n_particles; ++particle) {
    result.insert(system.tags.data[particle]);
  }
  return result;
}

inline void require_topology_tag(
    index_t tag,
    const std::unordered_set<index_t>& active_tags,
    const char* path) {
  if (tag == 0 || active_tags.count(tag) == 0) {
    throw std::invalid_argument(std::string(path) + " must reference active System tags.");
  }
}

inline void require_distinct_tags(
    const std::vector<index_t>& tags,
    const char* path) {
  std::unordered_set<index_t> seen;
  for (const index_t tag : tags) {
    if (!seen.insert(tag).second) {
      throw std::invalid_argument(std::string(path) + " must reference distinct tags.");
    }
  }
}

inline void require_dense_topology_types(
    const std::vector<type_id_t>& type_ids,
    const char* path) {
  if (type_ids.empty()) {
    return;
  }
  std::vector<type_id_t> unique = type_ids;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());

  type_id_t expected_type = 1;
  for (const type_id_t type_id : unique) {
    if (type_id < 1) {
      throw std::invalid_argument(std::string(path) + " types must be positive.");
    }
    if (type_id != expected_type) {
      throw std::invalid_argument(
          std::string(path) + " types must be dense one-based ids without gaps.");
    }
    ++expected_type;
  }
}

inline std::uint64_t topology_pair_key(
    index_t first,
    index_t second,
    index_t n_particles) {
  if (second < first) {
    std::swap(first, second);
  }
  return static_cast<std::uint64_t>(first) *
             static_cast<std::uint64_t>(n_particles + 1) +
         static_cast<std::uint64_t>(second);
}

inline std::string topology_record_key(
    const std::vector<index_t>& tags) {
  std::string result;
  for (const index_t tag : tags) {
    result += std::to_string(tag);
    result += ":";
  }
  return result;
}

inline void validate_topology_spec(const SystemSpec& system) {
  const std::unordered_set<index_t> active_tags = active_tag_set(system);

  std::unordered_set<std::uint64_t> observed_bonds;
  std::vector<type_id_t> bond_types;
  bond_types.reserve(system.topology.bonds.size());
  for (const BondTopologySpec& bond : system.topology.bonds) {
    require_topology_tag(bond.tag_i, active_tags, "system.topology.bonds tag_i");
    require_topology_tag(bond.tag_j, active_tags, "system.topology.bonds tag_j");
    require_distinct_tags({bond.tag_i, bond.tag_j}, "system.topology.bonds");
    if (bond.tag_i >= bond.tag_j) {
      throw std::invalid_argument(
          "system.topology.bonds must be canonical with tag_i < tag_j.");
    }
    const std::uint64_t key =
        topology_pair_key(bond.tag_i, bond.tag_j, system.n_particles);
    if (!observed_bonds.insert(key).second) {
      throw std::invalid_argument("system.topology.bonds must not contain duplicates.");
    }
    bond_types.push_back(bond.type);
  }
  require_dense_topology_types(bond_types, "system.topology.bonds");

  std::unordered_set<std::string> observed_angles;
  std::vector<type_id_t> angle_types;
  angle_types.reserve(system.topology.angles.size());
  for (const AngleTopologySpec& angle : system.topology.angles) {
    require_topology_tag(angle.tag_i, active_tags, "system.topology.angles tag_i");
    require_topology_tag(angle.tag_j, active_tags, "system.topology.angles tag_j");
    require_topology_tag(angle.tag_k, active_tags, "system.topology.angles tag_k");
    require_distinct_tags(
        {angle.tag_i, angle.tag_j, angle.tag_k},
        "system.topology.angles");
    if (angle.tag_i >= angle.tag_k) {
      throw std::invalid_argument(
          "system.topology.angles must be canonical with tag_i < tag_k.");
    }
    const std::string key =
        topology_record_key({angle.tag_i, angle.tag_j, angle.tag_k});
    if (!observed_angles.insert(key).second) {
      throw std::invalid_argument("system.topology.angles must not contain duplicates.");
    }
    if (observed_bonds.count(
            topology_pair_key(angle.tag_i, angle.tag_j, system.n_particles)) == 0 ||
        observed_bonds.count(
            topology_pair_key(angle.tag_j, angle.tag_k, system.n_particles)) == 0) {
      throw std::invalid_argument(
          "system.topology.angles require supporting topology bonds.");
    }
    angle_types.push_back(angle.type);
  }
  require_dense_topology_types(angle_types, "system.topology.angles");

  std::unordered_set<std::string> observed_dihedrals;
  std::vector<type_id_t> dihedral_types;
  dihedral_types.reserve(system.topology.dihedrals.size());
  for (const DihedralTopologySpec& dihedral : system.topology.dihedrals) {
    require_topology_tag(dihedral.tag_i, active_tags, "system.topology.dihedrals tag_i");
    require_topology_tag(dihedral.tag_j, active_tags, "system.topology.dihedrals tag_j");
    require_topology_tag(dihedral.tag_k, active_tags, "system.topology.dihedrals tag_k");
    require_topology_tag(dihedral.tag_l, active_tags, "system.topology.dihedrals tag_l");
    require_distinct_tags(
        {dihedral.tag_i, dihedral.tag_j, dihedral.tag_k, dihedral.tag_l},
        "system.topology.dihedrals");
    const std::array<index_t, 4> forward{
        dihedral.tag_i,
        dihedral.tag_j,
        dihedral.tag_k,
        dihedral.tag_l};
    const std::array<index_t, 4> reverse{
        dihedral.tag_l,
        dihedral.tag_k,
        dihedral.tag_j,
        dihedral.tag_i};
    if (reverse < forward) {
      throw std::invalid_argument(
          "system.topology.dihedrals must be canonical under endpoint reversal.");
    }
    const std::string key = topology_record_key(
        {dihedral.tag_i, dihedral.tag_j, dihedral.tag_k, dihedral.tag_l});
    if (!observed_dihedrals.insert(key).second) {
      throw std::invalid_argument(
          "system.topology.dihedrals must not contain duplicates.");
    }
    if (observed_bonds.count(topology_pair_key(
            dihedral.tag_i,
            dihedral.tag_j,
            system.n_particles)) == 0 ||
        observed_bonds.count(topology_pair_key(
            dihedral.tag_j,
            dihedral.tag_k,
            system.n_particles)) == 0 ||
        observed_bonds.count(topology_pair_key(
            dihedral.tag_k,
            dihedral.tag_l,
            system.n_particles)) == 0) {
      throw std::invalid_argument(
          "system.topology.dihedrals require supporting topology bonds.");
    }
    dihedral_types.push_back(dihedral.type);
  }
  require_dense_topology_types(dihedral_types, "system.topology.dihedrals");
}

inline void validate_system_spec(const SystemSpec& system) {
  if (system.n_particles == 0) {
    throw std::invalid_argument("system.n_particles must be positive.");
  }
  validate_system_units(system);
  validate_box_bound(system);
  validate_real_particle_fields(system);
  validate_dense_one_based_types(system);
  validate_tag_permutation(system);
  validate_contiguous_molecule_ids(system);
  validate_topology_spec(system);
}

inline type_id_t active_type_count(const SystemSpec& system) {
  type_id_t max_type = 0;
  for (std::size_t particle = 0; particle < system.n_particles; ++particle) {
    max_type = std::max(max_type, system.types.data[particle]);
  }
  return max_type;
}

inline std::uint64_t pair_key(
    type_id_t type_i,
    type_id_t type_j,
    type_id_t active_types) {
  return static_cast<std::uint64_t>(type_i) *
             static_cast<std::uint64_t>(active_types + 1) +
         static_cast<std::uint64_t>(type_j);
}

inline bool has_param(
    const StyleParamMap& params,
    const char* key) {
  return params.find(key) != params.end();
}

inline double require_real_parameter(
    const StyleParamMap& params,
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
  throw std::invalid_argument(std::string(path) + "." + key + " must be a real value.");
}

inline void require_exact_parameter_keys(
    const StyleParamMap& params,
    const std::vector<const char*>& keys,
    const char* path) {
  for (const char* key : keys) {
    if (!has_param(params, key)) {
      throw std::invalid_argument(std::string(path) + "." + key + " is required.");
    }
  }
  for (const auto& [key, _] : params) {
    const bool expected =
        std::find(keys.begin(), keys.end(), key) != keys.end();
    if (!expected) {
      throw std::invalid_argument(
          std::string(path) + " has unsupported parameter \"" + key + "\".");
    }
  }
}

inline std::vector<type_id_t> active_bond_types(const SystemSpec& system) {
  std::vector<type_id_t> result;
  result.reserve(system.topology.bonds.size());
  for (const BondTopologySpec& bond : system.topology.bonds) {
    result.push_back(bond.type);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

inline std::vector<type_id_t> active_angle_types(const SystemSpec& system) {
  std::vector<type_id_t> result;
  result.reserve(system.topology.angles.size());
  for (const AngleTopologySpec& angle : system.topology.angles) {
    result.push_back(angle.type);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

inline std::vector<type_id_t> active_dihedral_types(const SystemSpec& system) {
  std::vector<type_id_t> result;
  result.reserve(system.topology.dihedrals.size());
  for (const DihedralTopologySpec& dihedral : system.topology.dihedrals) {
    result.push_back(dihedral.type);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

template <typename CoeffSpec>
inline void validate_listed_coeff_coverage(
    const std::vector<type_id_t>& active_types,
    const std::vector<CoeffSpec>& coeffs,
    const std::string& active_style,
    const char* style_label,
    const char* coeff_label,
    const char* topology_label) {
  std::unordered_set<type_id_t> active_type_set(
      active_types.begin(),
      active_types.end());
  std::unordered_set<type_id_t> observed_coeffs;
  for (const CoeffSpec& coeff : coeffs) {
    if (coeff.style != active_style) {
      throw std::invalid_argument(
          std::string(coeff_label) + " style must match active " + style_label + ".");
    }
    if (active_type_set.count(coeff.type) == 0) {
      throw std::invalid_argument(
          std::string(coeff_label) + " type ids must be active " + topology_label + " types.");
    }
    if (!observed_coeffs.insert(coeff.type).second) {
      throw std::invalid_argument(
          std::string(coeff_label) + "s must not contain duplicates.");
    }
  }
  for (const type_id_t active_type : active_types) {
    if (observed_coeffs.count(active_type) == 0) {
      throw std::invalid_argument(
          std::string(coeff_label) + "s must cover all active " + topology_label + " types.");
    }
  }
}

inline void validate_forcefield_spec(
    const ForceFieldSpec& forcefield,
    const SystemSpec& system) {
  const bool known_exclusion_policy =
      forcefield.bonded_exclusion_policy == "none" ||
      forcefield.bonded_exclusion_policy == "default" ||
      forcefield.bonded_exclusion_policy == "explicit";
  if (!known_exclusion_policy) {
    throw std::invalid_argument(
        "forcefield.bonded_exclusion_policy must be none, default, or explicit.");
  }
  if (forcefield.bonded_exclusion_distance) {
    const index_t distance = *forcefield.bonded_exclusion_distance;
    if (distance < 1 || distance > 3) {
      throw std::invalid_argument(
          "forcefield.bonded_exclusion_distance must be 1, 2, or 3.");
    }
    if (system.topology.bonds.empty()) {
      throw std::invalid_argument(
          "forcefield.bonded_exclusion_distance requires system.topology.bonds.");
    }
    if (forcefield.bonded_exclusion_policy != "default" &&
        forcefield.bonded_exclusion_policy != "explicit") {
      throw std::invalid_argument(
          "forcefield.bonded_exclusion_policy must be default or explicit when bonded_exclusion_distance is set.");
    }
  } else if (forcefield.bonded_exclusion_policy == "default" ||
             forcefield.bonded_exclusion_policy == "explicit") {
    throw std::invalid_argument(
        "forcefield.bonded_exclusion_policy requires bonded_exclusion_distance.");
  }

  if (forcefield.pair_coeffs.empty()) {
    throw std::invalid_argument(
        "forcefield.pair_coeffs must contain at least one pair_coeff.");
  }

  const type_id_t active_types = active_type_count(system);
  std::unordered_set<std::uint64_t> observed_pairs;
  for (const PairCoeffSpec& pair_coeff : forcefield.pair_coeffs) {
    if (pair_coeff.style != forcefield.pair_style.style) {
      throw std::invalid_argument(
          "forcefield.pair_coeff style must match active pair_style.");
    }
    if (pair_coeff.type_i < 1 || pair_coeff.type_i > active_types ||
        pair_coeff.type_j < 1 || pair_coeff.type_j > active_types) {
      throw std::invalid_argument(
          "forcefield.pair_coeff type ids must be active system types.");
    }
    if (pair_coeff.type_i > pair_coeff.type_j) {
      throw std::invalid_argument(
          "forcefield.pair_coeff type ids must be normalized with type_i <= type_j.");
    }

    const std::uint64_t key =
        pair_key(pair_coeff.type_i, pair_coeff.type_j, active_types);
    if (!observed_pairs.insert(key).second) {
      throw std::invalid_argument("forcefield.pair_coeffs must not contain duplicates.");
    }
  }

  for (type_id_t type_i = 1; type_i <= active_types; ++type_i) {
    for (type_id_t type_j = type_i; type_j <= active_types; ++type_j) {
      if (observed_pairs.count(pair_key(type_i, type_j, active_types)) == 0) {
        throw std::invalid_argument(
            "forcefield.pair_coeffs must cover all active system type pairs.");
      }
    }
  }

  if (!forcefield.bond_style) {
    if (!system.topology.bonds.empty()) {
      throw std::invalid_argument(
          "system.topology.bonds require forcefield.bond_style.");
    }
    if (!forcefield.bond_coeffs.empty()) {
      throw std::invalid_argument(
          "forcefield.bond_coeffs require active bond_style.");
    }
  } else {
    if (system.topology.bonds.empty()) {
      throw std::invalid_argument(
          "forcefield.bond_style requires system.topology.bonds.");
    }

    validate_listed_coeff_coverage(
        active_bond_types(system),
        forcefield.bond_coeffs,
        forcefield.bond_style->style,
        "bond_style",
        "forcefield.bond_coeff",
        "topology bond");
  }

  if (!forcefield.angle_style) {
    if (!system.topology.angles.empty()) {
      throw std::invalid_argument(
          "system.topology.angles require forcefield.angle_style.");
    }
    if (!forcefield.angle_coeffs.empty()) {
      throw std::invalid_argument(
          "forcefield.angle_coeffs require active angle_style.");
    }
  } else {
    if (system.topology.angles.empty()) {
      throw std::invalid_argument(
          "forcefield.angle_style requires system.topology.angles.");
    }
    validate_listed_coeff_coverage(
        active_angle_types(system),
        forcefield.angle_coeffs,
        forcefield.angle_style->style,
        "angle_style",
        "forcefield.angle_coeff",
        "topology angle");
  }

  if (!forcefield.dihedral_style) {
    if (!system.topology.dihedrals.empty()) {
      throw std::invalid_argument(
          "system.topology.dihedrals require forcefield.dihedral_style.");
    }
    if (!forcefield.dihedral_coeffs.empty()) {
      throw std::invalid_argument(
          "forcefield.dihedral_coeffs require active dihedral_style.");
    }
  } else {
    if (system.topology.dihedrals.empty()) {
      throw std::invalid_argument(
          "forcefield.dihedral_style requires system.topology.dihedrals.");
    }
    validate_listed_coeff_coverage(
        active_dihedral_types(system),
        forcefield.dihedral_coeffs,
        forcefield.dihedral_style->style,
        "dihedral_style",
        "forcefield.dihedral_coeff",
        "topology dihedral");
  }
}

inline void validate_dynamics_spec(const DynamicsSpec& dynamics) {
  if (dynamics.style.empty()) {
    throw std::invalid_argument("dynamics.style must not be empty.");
  }
  if (!dynamics.thermostat) {
    return;
  }
  if (dynamics.style != "velocity_verlet") {
    throw std::invalid_argument(
        "dynamics.thermostat requires dynamics.style velocity_verlet.");
  }
  if (dynamics.thermostat->style != "berendsen") {
    throw std::invalid_argument(
        "dynamics.thermostat.style supports only berendsen.");
  }
  require_exact_parameter_keys(
      dynamics.thermostat->params,
      {"temperature", "tau"},
      "dynamics.thermostat.params");
  const double temperature = require_real_parameter(
      dynamics.thermostat->params,
      "temperature",
      "dynamics.thermostat.params");
  const double tau = require_real_parameter(
      dynamics.thermostat->params,
      "tau",
      "dynamics.thermostat.params");
  if (!std::isfinite(temperature) || temperature < 0.0) {
    throw std::invalid_argument(
        "dynamics.thermostat.params.temperature must be finite and non-negative.");
  }
  if (!std::isfinite(tau) || tau <= 0.0) {
    throw std::invalid_argument(
        "dynamics.thermostat.params.tau must be finite and positive.");
  }
  const real_t temperature_cast = static_cast<real_t>(temperature);
  const real_t tau_cast = static_cast<real_t>(tau);
  if (!std::isfinite(static_cast<double>(temperature_cast)) ||
      temperature_cast < real_t{0}) {
    throw std::invalid_argument(
        "dynamics.thermostat.params.temperature must be finite and non-negative.");
  }
  if (!std::isfinite(static_cast<double>(tau_cast)) || tau_cast <= real_t{0}) {
    throw std::invalid_argument(
        "dynamics.thermostat.params.tau must be finite and positive.");
  }
}

inline void validate_neighbor_spec(const NeighborSpec& neighbor) {
  if (!std::isfinite(static_cast<double>(neighbor.cutoff_buffer))) {
    throw std::invalid_argument("neighbor.cutoff_buffer must be finite.");
  }
  if (neighbor.cutoff_buffer < real_t{0}) {
    throw std::invalid_argument("neighbor.cutoff_buffer must be non-negative.");
  }
  if (neighbor.rebuild_check_every == 0) {
    throw std::invalid_argument("neighbor.rebuild_check_every must be positive.");
  }
  if (neighbor.sort_every_rebuild == 0) {
    throw std::invalid_argument("neighbor.sort_every_rebuild must be positive.");
  }
  if (neighbor.max_neighbors == 0) {
    throw std::invalid_argument("neighbor.max_neighbors must be positive.");
  }
  if (neighbor.cutoff_buffer == real_t{0} &&
      neighbor.rebuild_check_every != 1) {
    throw std::invalid_argument(
        "neighbor.cutoff_buffer=0 requires neighbor.rebuild_check_every=1.");
  }
}

inline void validate_output_spec(const OutputSpec& output) {
  if (output.thermo && output.thermo->every == 0) {
    throw std::invalid_argument("output.thermo.every must be positive.");
  }
  if (output.thermo && output.thermo->prefix.empty()) {
    throw std::invalid_argument("output.thermo.prefix must not be empty.");
  }
  if (output.final_state && output.final_state->prefix.empty()) {
    throw std::invalid_argument("output.final_state.prefix must not be empty.");
  }
  if (output.trajectory && output.trajectory->every == 0) {
    throw std::invalid_argument("output.trajectory.every must be positive.");
  }
  if (output.trajectory && output.trajectory->prefix.empty()) {
    throw std::invalid_argument("output.trajectory.prefix must not be empty.");
  }
  if (output.trajectory && output.trajectory->fields.empty()) {
    throw std::invalid_argument("output.trajectory.fields must not be empty.");
  }
  if (output.log) {
    const std::string& echo = output.log->echo;
    const bool file_echo = echo == "log" || echo == "both";
    const bool non_file_echo = echo == "screen" || echo == "none";
    if (!file_echo && !non_file_echo) {
      throw std::invalid_argument(
          "output.log.echo must be screen, log, both, or none.");
    }
    if (file_echo) {
      if (!output.log->prefix || output.log->prefix->empty()) {
        throw std::invalid_argument(
            "output.log.prefix must be provided for file logging.");
      }
    } else if (output.log->prefix) {
      throw std::invalid_argument(
          "output.log.prefix is only supported for log or both echo.");
    }
  }
}

inline void validate_simulation_spec_shell(const SimulationSpec& spec) {
  validate_system_spec(spec.system);
  validate_forcefield_spec(spec.forcefield, spec.system);
  validate_dynamics_spec(spec.dynamics);
  validate_neighbor_spec(spec.neighbor);
  validate_output_spec(spec.output);
}

}  // namespace validation
}  // namespace input
}  // namespace beads
