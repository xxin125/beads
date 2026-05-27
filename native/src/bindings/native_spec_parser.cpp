#include "native_spec_parser.hpp"

#include "ndarray_reader.hpp"
#include "native_spec_value_reader.hpp"
#include "py_object_reader.hpp"
#include <input/native_spec_validation.hpp>

#include <cstddef>
#include <algorithm>
#include <string>
#include <vector>

namespace py = pybind11;

namespace beads {
namespace bindings {
namespace {

namespace py_reader = ::beads::bindings::py_object;
namespace array_reader = ::beads::bindings::ndarray_reader;
namespace value_reader = ::beads::bindings::native_spec_values;

constexpr index_t kDefaultBondedExclusionDistance = 2;

constexpr const char* kRequiredTopLevelSections[] = {
    "system",
    "forcefield",
    "dynamics",
    "neighbor",
    "runsteps",
};

input::BondTopologySpec parse_bond_topology_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(spec, {"tag_i", "tag_j", "type"}, path);

  input::BondTopologySpec result;
  result.tag_i = value_reader::require_positive_index(
      py_reader::require_item(spec, "tag_i", path),
      path + ".tag_i");
  result.tag_j = value_reader::require_positive_index(
      py_reader::require_item(spec, "tag_j", path),
      path + ".tag_j");
  result.type = value_reader::require_positive_type_id(
      py_reader::require_item(spec, "type", path),
      path + ".type");
  return result;
}

input::AngleTopologySpec parse_angle_topology_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(spec, {"tag_i", "tag_j", "tag_k", "type"}, path);

  input::AngleTopologySpec result;
  result.tag_i = value_reader::require_positive_index(
      py_reader::require_item(spec, "tag_i", path),
      path + ".tag_i");
  result.tag_j = value_reader::require_positive_index(
      py_reader::require_item(spec, "tag_j", path),
      path + ".tag_j");
  result.tag_k = value_reader::require_positive_index(
      py_reader::require_item(spec, "tag_k", path),
      path + ".tag_k");
  result.type = value_reader::require_positive_type_id(
      py_reader::require_item(spec, "type", path),
      path + ".type");
  return result;
}

input::DihedralTopologySpec parse_dihedral_topology_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(
      spec,
      {"tag_i", "tag_j", "tag_k", "tag_l", "type"},
      path);

  input::DihedralTopologySpec result;
  result.tag_i = value_reader::require_positive_index(
      py_reader::require_item(spec, "tag_i", path),
      path + ".tag_i");
  result.tag_j = value_reader::require_positive_index(
      py_reader::require_item(spec, "tag_j", path),
      path + ".tag_j");
  result.tag_k = value_reader::require_positive_index(
      py_reader::require_item(spec, "tag_k", path),
      path + ".tag_k");
  result.tag_l = value_reader::require_positive_index(
      py_reader::require_item(spec, "tag_l", path),
      path + ".tag_l");
  result.type = value_reader::require_positive_type_id(
      py_reader::require_item(spec, "type", path),
      path + ".type");
  return result;
}

std::vector<input::BondTopologySpec> parse_bond_topology_list(
    const py::object& value,
    const std::string& path) {
  const py::list records = py_reader::require_list(value, path);
  std::vector<input::BondTopologySpec> result;
  result.reserve(static_cast<std::size_t>(py::len(records)));
  for (py::ssize_t index = 0; index < py::len(records); ++index) {
    result.push_back(parse_bond_topology_spec(
        py::reinterpret_borrow<py::object>(records[index]),
        path + "[" + std::to_string(index) + "]"));
  }
  return result;
}

std::vector<input::AngleTopologySpec> parse_angle_topology_list(
    const py::object& value,
    const std::string& path) {
  const py::list records = py_reader::require_list(value, path);
  std::vector<input::AngleTopologySpec> result;
  result.reserve(static_cast<std::size_t>(py::len(records)));
  for (py::ssize_t index = 0; index < py::len(records); ++index) {
    result.push_back(parse_angle_topology_spec(
        py::reinterpret_borrow<py::object>(records[index]),
        path + "[" + std::to_string(index) + "]"));
  }
  return result;
}

std::vector<input::DihedralTopologySpec> parse_dihedral_topology_list(
    const py::object& value,
    const std::string& path) {
  const py::list records = py_reader::require_list(value, path);
  std::vector<input::DihedralTopologySpec> result;
  result.reserve(static_cast<std::size_t>(py::len(records)));
  for (py::ssize_t index = 0; index < py::len(records); ++index) {
    result.push_back(parse_dihedral_topology_spec(
        py::reinterpret_borrow<py::object>(records[index]),
        path + "[" + std::to_string(index) + "]"));
  }
  return result;
}

input::TopologySpec parse_topology_spec(const py::dict& spec) {
  py_reader::require_only_keys(
      spec,
      {"bonds", "angles", "dihedrals"},
      "system.topology");

  input::TopologySpec result;
  result.bonds = parse_bond_topology_list(
      py_reader::require_item(spec, "bonds", "system.topology"),
      "system.topology.bonds");
  result.angles = parse_angle_topology_list(
      py_reader::require_item(spec, "angles", "system.topology"),
      "system.topology.angles");
  result.dihedrals = parse_dihedral_topology_list(
      py_reader::require_item(spec, "dihedrals", "system.topology"),
      "system.topology.dihedrals");
  return result;
}

input::SystemSpec parse_system_spec(const py::dict& spec) {
  py_reader::require_only_keys(
      spec,
      {
          "units",
          "n_particles",
          "box_bound",
          "positions",
          "velocities",
          "masses",
          "types",
          "tags",
          "molecule_ids",
          "images",
          "topology",
      },
      "system");

  input::SystemSpec result;
  result.units = py_reader::require_string(
      py_reader::require_item(spec, "units", "system"),
      "system.units");
  result.n_particles = value_reader::require_positive_index(
      py_reader::require_item(spec, "n_particles", "system"),
      "system.n_particles");
  const std::size_t n_particles = result.n_particles;
  result.box_bound = array_reader::require_array<real_t>(
      py_reader::require_item(spec, "box_bound", "system"),
      "system.box_bound",
      {2, 3});
  result.positions = array_reader::require_array<real_t>(
      py_reader::require_item(spec, "positions", "system"),
      "system.positions",
      {n_particles, 3});
  result.velocities = array_reader::require_array<real_t>(
      py_reader::require_item(spec, "velocities", "system"),
      "system.velocities",
      {n_particles, 3});
  result.masses = array_reader::require_array<real_t>(
      py_reader::require_item(spec, "masses", "system"),
      "system.masses",
      {n_particles});
  result.types = array_reader::require_array<type_id_t>(
      py_reader::require_item(spec, "types", "system"),
      "system.types",
      {n_particles});
  result.tags = array_reader::require_array<index_t>(
      py_reader::require_item(spec, "tags", "system"),
      "system.tags",
      {n_particles});
  result.molecule_ids = array_reader::require_array<index_t>(
      py_reader::require_item(spec, "molecule_ids", "system"),
      "system.molecule_ids",
      {n_particles});
  result.images = array_reader::require_array<image_t>(
      py_reader::require_item(spec, "images", "system"),
      "system.images",
      {n_particles, 3});
  result.topology = parse_topology_spec(
      py_reader::require_dict_item(spec, "topology", "system"));
  return result;
}

input::PairStyleSpec parse_pair_style_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(spec, {"style", "params"}, path);

  input::PairStyleSpec result;
  result.style = py_reader::require_string(
      py_reader::require_item(spec, "style", path),
      path + ".style");
  result.params = value_reader::parse_style_params(
      py_reader::require_item(spec, "params", path),
      path + ".params");
  return result;
}

input::PairCoeffSpec parse_pair_coeff_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(
      spec,
      {"style", "type_i", "type_j", "params"},
      path);

  input::PairCoeffSpec result;
  result.style = py_reader::require_string(
      py_reader::require_item(spec, "style", path),
      path + ".style");
  result.type_i = value_reader::require_positive_type_id(
      py_reader::require_item(spec, "type_i", path),
      path + ".type_i");
  result.type_j = value_reader::require_positive_type_id(
      py_reader::require_item(spec, "type_j", path),
      path + ".type_j");
  result.params = value_reader::parse_style_params(
      py_reader::require_item(spec, "params", path),
      path + ".params");
  return result;
}

input::BondStyleSpec parse_bond_style_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(spec, {"style", "params"}, path);

  input::BondStyleSpec result;
  result.style = py_reader::require_string(
      py_reader::require_item(spec, "style", path),
      path + ".style");
  result.params = value_reader::parse_style_params(
      py_reader::require_item(spec, "params", path),
      path + ".params");
  return result;
}

input::AngleStyleSpec parse_angle_style_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(spec, {"style", "params"}, path);

  input::AngleStyleSpec result;
  result.style = py_reader::require_string(
      py_reader::require_item(spec, "style", path),
      path + ".style");
  result.params = value_reader::parse_style_params(
      py_reader::require_item(spec, "params", path),
      path + ".params");
  return result;
}

input::DihedralStyleSpec parse_dihedral_style_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(spec, {"style", "params"}, path);

  input::DihedralStyleSpec result;
  result.style = py_reader::require_string(
      py_reader::require_item(spec, "style", path),
      path + ".style");
  result.params = value_reader::parse_style_params(
      py_reader::require_item(spec, "params", path),
      path + ".params");
  return result;
}

input::ThermostatSpec parse_thermostat_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(spec, {"style", "params"}, path);

  input::ThermostatSpec result;
  result.style = py_reader::require_string(
      py_reader::require_item(spec, "style", path),
      path + ".style");
  result.params = value_reader::parse_style_params(
      py_reader::require_item(spec, "params", path),
      path + ".params");
  return result;
}

input::BondCoeffSpec parse_bond_coeff_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(spec, {"style", "type", "params"}, path);

  input::BondCoeffSpec result;
  result.style = py_reader::require_string(
      py_reader::require_item(spec, "style", path),
      path + ".style");
  result.type = value_reader::require_positive_type_id(
      py_reader::require_item(spec, "type", path),
      path + ".type");
  result.params = value_reader::parse_style_params(
      py_reader::require_item(spec, "params", path),
      path + ".params");
  return result;
}

input::AngleCoeffSpec parse_angle_coeff_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(spec, {"style", "type", "params"}, path);

  input::AngleCoeffSpec result;
  result.style = py_reader::require_string(
      py_reader::require_item(spec, "style", path),
      path + ".style");
  result.type = value_reader::require_positive_type_id(
      py_reader::require_item(spec, "type", path),
      path + ".type");
  result.params = value_reader::parse_style_params(
      py_reader::require_item(spec, "params", path),
      path + ".params");
  return result;
}

input::DihedralCoeffSpec parse_dihedral_coeff_spec(
    const py::object& value,
    const std::string& path) {
  const py::dict spec = py_reader::require_dict(value, path);
  py_reader::require_only_keys(spec, {"style", "type", "params"}, path);

  input::DihedralCoeffSpec result;
  result.style = py_reader::require_string(
      py_reader::require_item(spec, "style", path),
      path + ".style");
  result.type = value_reader::require_positive_type_id(
      py_reader::require_item(spec, "type", path),
      path + ".type");
  result.params = value_reader::parse_style_params(
      py_reader::require_item(spec, "params", path),
      path + ".params");
  return result;
}

std::vector<input::PairCoeffSpec> parse_pair_coeffs(
    const py::object& value,
    const std::string& path) {
  const py::list coeffs = py_reader::require_list(value, path);
  std::vector<input::PairCoeffSpec> result;
  result.reserve(static_cast<std::size_t>(py::len(coeffs)));
  for (py::ssize_t index = 0; index < py::len(coeffs); ++index) {
    result.push_back(parse_pair_coeff_spec(
        py::reinterpret_borrow<py::object>(coeffs[index]),
        path + "[" + std::to_string(index) + "]"));
  }
  return result;
}

std::vector<input::BondCoeffSpec> parse_bond_coeffs(
    const py::object& value,
    const std::string& path) {
  const py::list coeffs = py_reader::require_list(value, path);
  std::vector<input::BondCoeffSpec> result;
  result.reserve(static_cast<std::size_t>(py::len(coeffs)));
  for (py::ssize_t index = 0; index < py::len(coeffs); ++index) {
    result.push_back(parse_bond_coeff_spec(
        py::reinterpret_borrow<py::object>(coeffs[index]),
        path + "[" + std::to_string(index) + "]"));
  }
  return result;
}

std::vector<input::AngleCoeffSpec> parse_angle_coeffs(
    const py::object& value,
    const std::string& path) {
  const py::list coeffs = py_reader::require_list(value, path);
  std::vector<input::AngleCoeffSpec> result;
  result.reserve(static_cast<std::size_t>(py::len(coeffs)));
  for (py::ssize_t index = 0; index < py::len(coeffs); ++index) {
    result.push_back(parse_angle_coeff_spec(
        py::reinterpret_borrow<py::object>(coeffs[index]),
        path + "[" + std::to_string(index) + "]"));
  }
  return result;
}

std::vector<input::DihedralCoeffSpec> parse_dihedral_coeffs(
    const py::object& value,
    const std::string& path) {
  const py::list coeffs = py_reader::require_list(value, path);
  std::vector<input::DihedralCoeffSpec> result;
  result.reserve(static_cast<std::size_t>(py::len(coeffs)));
  for (py::ssize_t index = 0; index < py::len(coeffs); ++index) {
    result.push_back(parse_dihedral_coeff_spec(
        py::reinterpret_borrow<py::object>(coeffs[index]),
        path + "[" + std::to_string(index) + "]"));
  }
  return result;
}

input::ForceFieldSpec parse_forcefield_spec(const py::dict& spec) {
  py_reader::require_only_keys(
      spec,
      {
          "pair_style",
          "pair_coeffs",
          "bond_style",
          "bond_coeffs",
          "angle_style",
          "angle_coeffs",
          "dihedral_style",
          "dihedral_coeffs",
          "bonded_exclusion_distance",
          "bonded_exclusion_policy",
      },
      "forcefield");
  input::ForceFieldSpec result;
  result.bonded_exclusion_policy = "auto";
  result.pair_style = parse_pair_style_spec(
      py_reader::require_item(spec, "pair_style", "forcefield"),
      "forcefield.pair_style");
  result.pair_coeffs = parse_pair_coeffs(
      py_reader::require_item(spec, "pair_coeffs", "forcefield"),
      "forcefield.pair_coeffs");
  const py::object bond_style = py_reader::require_item(
      spec,
      "bond_style",
      "forcefield");
  if (!bond_style.is_none()) {
    result.bond_style = parse_bond_style_spec(
        bond_style,
        "forcefield.bond_style");
  }
  result.bond_coeffs = parse_bond_coeffs(
      py_reader::require_item(spec, "bond_coeffs", "forcefield"),
      "forcefield.bond_coeffs");
  const py::object angle_style = py_reader::require_item(
      spec,
      "angle_style",
      "forcefield");
  if (!angle_style.is_none()) {
    result.angle_style = parse_angle_style_spec(
        angle_style,
        "forcefield.angle_style");
  }
  result.angle_coeffs = parse_angle_coeffs(
      py_reader::require_item(spec, "angle_coeffs", "forcefield"),
      "forcefield.angle_coeffs");
  const py::object dihedral_style = py_reader::require_item(
      spec,
      "dihedral_style",
      "forcefield");
  if (!dihedral_style.is_none()) {
    result.dihedral_style = parse_dihedral_style_spec(
        dihedral_style,
        "forcefield.dihedral_style");
  }
  result.dihedral_coeffs = parse_dihedral_coeffs(
      py_reader::require_item(spec, "dihedral_coeffs", "forcefield"),
      "forcefield.dihedral_coeffs");
  const py::object exclusion_distance = py_reader::require_item(
      spec,
      "bonded_exclusion_distance",
      "forcefield");
  if (!exclusion_distance.is_none()) {
    result.bonded_exclusion_distance =
        value_reader::require_positive_index(
            exclusion_distance,
            "forcefield.bonded_exclusion_distance");
  }
  const py::str exclusion_policy_key("bonded_exclusion_policy");
  if (spec.contains(exclusion_policy_key)) {
    result.bonded_exclusion_policy = py_reader::require_string(
        py::reinterpret_borrow<py::object>(spec[exclusion_policy_key]),
        "forcefield.bonded_exclusion_policy");
  }
  return result;
}

input::DynamicsSpec parse_dynamics_spec(const py::dict& spec) {
  py_reader::require_only_keys(spec, {"style", "params", "thermostat"}, "dynamics");

  input::DynamicsSpec result;
  result.style = py_reader::require_string(
      py_reader::require_item(spec, "style", "dynamics"),
      "dynamics.style");
  result.params = value_reader::parse_style_params(
      py_reader::require_item(spec, "params", "dynamics"),
      "dynamics.params");
  const py::str thermostat_key("thermostat");
  if (spec.contains(thermostat_key)) {
    const py::object thermostat = py::reinterpret_borrow<py::object>(
        spec[thermostat_key]);
    if (!thermostat.is_none()) {
      result.thermostat = parse_thermostat_spec(
          thermostat,
          "dynamics.thermostat");
    }
  }
  return result;
}

input::NeighborSpec parse_neighbor_spec(const py::dict& spec) {
  py_reader::require_only_keys(
      spec,
      {
          "cutoff_buffer",
          "rebuild_check_every",
          "sort_every_rebuild",
          "max_neighbors",
      },
      "neighbor");

  input::NeighborSpec result;

  const double cutoff_buffer = value_reader::require_finite_real(
      py_reader::require_item(spec, "cutoff_buffer", "neighbor"),
      "neighbor.cutoff_buffer");
  result.cutoff_buffer = static_cast<real_t>(cutoff_buffer);

  result.rebuild_check_every = value_reader::require_uint64(
      py_reader::require_item(spec, "rebuild_check_every", "neighbor"),
      "neighbor.rebuild_check_every");
  result.sort_every_rebuild = value_reader::require_uint64(
      py_reader::require_item(spec, "sort_every_rebuild", "neighbor"),
      "neighbor.sort_every_rebuild");
  result.max_neighbors = value_reader::require_index(
      py_reader::require_item(spec, "max_neighbors", "neighbor"),
      "neighbor.max_neighbors");

  return result;
}

input::ThermoOutputSpec parse_thermo_output_spec(const py::dict& spec) {
  py_reader::require_only_keys(spec, {"every", "prefix"}, "output.thermo");

  input::ThermoOutputSpec result;
  result.every = value_reader::require_positive_uint64(
      py_reader::require_item(spec, "every", "output.thermo"),
      "output.thermo.every");
  result.prefix = py_reader::require_string(
      py_reader::require_item(spec, "prefix", "output.thermo"),
      "output.thermo.prefix");
  return result;
}

input::FinalStateOutputSpec parse_final_state_output_spec(const py::dict& spec) {
  py_reader::require_only_keys(spec, {"prefix"}, "output.final_state");

  input::FinalStateOutputSpec result;
  result.prefix = py_reader::require_string(
      py_reader::require_item(spec, "prefix", "output.final_state"),
      "output.final_state.prefix");
  return result;
}

std::vector<std::string> parse_string_list(
    const py::object& value,
    const std::string& path) {
  const py::list list = py_reader::require_list(value, path);
  std::vector<std::string> result;
  result.reserve(static_cast<std::size_t>(py::len(list)));
  for (py::ssize_t index = 0; index < py::len(list); ++index) {
    result.push_back(py_reader::require_string(
        py::reinterpret_borrow<py::object>(list[index]),
        path + "[" + std::to_string(index) + "]"));
  }
  return result;
}

bool is_supported_trajectory_field(const std::string& field) {
  return field == "tag" ||
         field == "type" ||
         field == "position" ||
         field == "image" ||
         field == "velocity" ||
         field == "force";
}

void validate_trajectory_fields(const std::vector<std::string>& fields) {
  if (fields.empty()) {
    py_reader::throw_value_error("output.trajectory.fields must not be empty.");
  }
  if (fields.front() != "tag") {
    py_reader::throw_value_error(
        "output.trajectory.fields must include tag as the first field.");
  }

  const std::vector<std::string> canonical{
      "tag",
      "type",
      "position",
      "image",
      "velocity",
      "force"};
  std::vector<std::string> seen;
  std::size_t canonical_index = 0;
  for (const std::string& field : fields) {
    if (!is_supported_trajectory_field(field)) {
      py_reader::throw_value_error(
          "output.trajectory.fields contains unsupported field \"" + field + "\".");
    }
    if (std::find(seen.begin(), seen.end(), field) != seen.end()) {
      py_reader::throw_value_error(
          "output.trajectory.fields must not contain duplicates.");
    }
    seen.push_back(field);
    while (canonical_index < canonical.size() &&
           canonical[canonical_index] != field) {
      ++canonical_index;
    }
    if (canonical_index == canonical.size()) {
      py_reader::throw_value_error(
          "output.trajectory.fields must follow canonical order: "
          "tag,type,position,image,velocity,force.");
    }
    ++canonical_index;
  }
}

input::TrajectoryOutputSpec parse_trajectory_output_spec(const py::dict& spec) {
  py_reader::require_only_keys(
      spec,
      {"every", "prefix", "fields"},
      "output.trajectory");

  input::TrajectoryOutputSpec result;
  result.every = value_reader::require_positive_uint64(
      py_reader::require_item(spec, "every", "output.trajectory"),
      "output.trajectory.every");
  result.prefix = py_reader::require_string(
      py_reader::require_item(spec, "prefix", "output.trajectory"),
      "output.trajectory.prefix");
  result.fields = parse_string_list(
      py_reader::require_item(spec, "fields", "output.trajectory"),
      "output.trajectory.fields");
  validate_trajectory_fields(result.fields);
  return result;
}

input::LogOutputSpec parse_log_output_spec(const py::dict& spec) {
  py_reader::require_only_keys(spec, {"echo", "prefix"}, "output.log");

  input::LogOutputSpec result;
  result.echo = py_reader::require_string(
      py_reader::require_item(spec, "echo", "output.log"),
      "output.log.echo");
  const py::str prefix_key("prefix");
  if (spec.contains(prefix_key)) {
    result.prefix = py_reader::require_string(
        py_reader::require_item(spec, "prefix", "output.log"),
        "output.log.prefix");
  }
  return result;
}

input::OutputSpec parse_output_spec(const py::dict& spec) {
  py_reader::require_only_keys(
      spec,
      {"thermo", "final_state", "trajectory", "log"},
      "output");

  input::OutputSpec result;
  const py::str thermo_key("thermo");
  if (spec.contains(thermo_key)) {
    result.thermo = parse_thermo_output_spec(
        py_reader::require_dict_item(spec, "thermo", "output"));
  }
  const py::str final_state_key("final_state");
  if (spec.contains(final_state_key)) {
    result.final_state = parse_final_state_output_spec(
        py_reader::require_dict_item(spec, "final_state", "output"));
  }
  const py::str trajectory_key("trajectory");
  if (spec.contains(trajectory_key)) {
    result.trajectory = parse_trajectory_output_spec(
        py_reader::require_dict_item(spec, "trajectory", "output"));
  }
  const py::str log_key("log");
  if (spec.contains(log_key)) {
    result.log = parse_log_output_spec(
        py_reader::require_dict_item(spec, "log", "output"));
  }
  return result;
}

void apply_default_bonded_exclusion_policy(input::SimulationSpec& spec) {
  input::ForceFieldSpec& forcefield = spec.forcefield;
  if (forcefield.bonded_exclusion_policy != "auto") {
    if (forcefield.bonded_exclusion_policy == "default" &&
        !forcefield.bonded_exclusion_distance &&
        !spec.system.topology.bonds.empty()) {
      forcefield.bonded_exclusion_distance = kDefaultBondedExclusionDistance;
    }
    return;
  }

  if (forcefield.bonded_exclusion_distance) {
    forcefield.bonded_exclusion_policy = "explicit";
    return;
  }
  if (!spec.system.topology.bonds.empty()) {
    forcefield.bonded_exclusion_distance = kDefaultBondedExclusionDistance;
    forcefield.bonded_exclusion_policy = "default";
    return;
  }
  forcefield.bonded_exclusion_policy = "none";
}

}  // namespace

input::SimulationSpec parse_native_spec(const py::object& spec) {
  const py::dict native_spec = py_reader::require_dict(spec, "native spec");
  for (const char* key : kRequiredTopLevelSections) {
    (void)py_reader::require_item(native_spec, key, "native spec");
  }
  py_reader::require_only_keys(
      native_spec,
      {
          "system",
          "forcefield",
          "dynamics",
          "neighbor",
          "output",
          "runsteps",
      },
      "native spec");

  input::SimulationSpec result;
  result.system = parse_system_spec(
      py_reader::require_dict_item(native_spec, "system", "native spec"));
  result.forcefield = parse_forcefield_spec(
      py_reader::require_dict_item(native_spec, "forcefield", "native spec"));
  result.dynamics = parse_dynamics_spec(
      py_reader::require_dict_item(native_spec, "dynamics", "native spec"));
  result.neighbor = parse_neighbor_spec(
      py_reader::require_dict_item(native_spec, "neighbor", "native spec"));
  const py::str output_key("output");
  if (native_spec.contains(output_key)) {
    result.output = parse_output_spec(
        py_reader::require_dict_item(native_spec, "output", "native spec"));
  }
  result.runsteps = value_reader::require_uint64(
      py_reader::require_item(native_spec, "runsteps", "native spec"),
      "runsteps");
  apply_default_bonded_exclusion_policy(result);
  input::validation::validate_simulation_spec_shell(result);
  return result;
}

}  // namespace bindings
}  // namespace beads
