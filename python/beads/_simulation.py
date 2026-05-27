"""Simulation assembly objects."""

from __future__ import annotations

from ._dynamics import Dynamics
from ._forcefield import ForceField
from ._native_bridge import execute_simulation
from ._neighbor import _NeighborConfig
from ._output import (
    _FinalStateConfig,
    _LogConfig,
    _ThermoConfig,
    _TrajectoryConfig,
    log_config,
    output_prefix,
    trajectory_config,
)
from ._system import System
from ._validation import nonnegative_uint64, positive_uint64


class Simulation:

    def __init__(
        self,
        *,
        system: System,
        forcefield: ForceField,
        dynamics: Dynamics,
    ) -> None:
        if not isinstance(system, System):
            raise TypeError("Simulation requires a System.")
        if not isinstance(forcefield, ForceField):
            raise TypeError("Simulation requires a ForceField.")
        if not isinstance(dynamics, Dynamics):
            raise TypeError("Simulation requires a Dynamics.")

        self._system = system._freeze()
        self._forcefield_config = forcefield._freeze(system)
        self._dynamics_config = dynamics._freeze()
        self._neighbor_config = _NeighborConfig()
        self._thermo_config: _ThermoConfig | None = None
        self._final_state_config: _FinalStateConfig | None = None
        self._trajectory_config: _TrajectoryConfig | None = None
        self._log_config = _LogConfig(echo="screen")
        self._runsteps = 0
        self._executed = False

    def _require_not_executed(
        self,
        action: str = "modified",
    ) -> None:
        if self._executed:
            if action == "execute":
                raise RuntimeError(
                    "Simulation.execute() cannot be called after execute(); "
                    "create a new Simulation for another run."
                )
            raise RuntimeError(
                "Simulation cannot be modified after execute(); "
                "create a new Simulation for another run."
            )

    def set_runsteps(
        self,
        runsteps: object,
    ) -> "Simulation":
        self._require_not_executed()
        self._runsteps = nonnegative_uint64("runsteps", runsteps)
        return self

    def set_neighbor(
        self,
        *,
        cutoff_buffer: object = 0.3,
        rebuild_check_every: object = 1,
        sort_every_rebuild: object = 10,
        max_neighbors: object = 512,
    ) -> "Simulation":
        self._require_not_executed()
        self._neighbor_config = _NeighborConfig(
            cutoff_buffer=cutoff_buffer,
            rebuild_check_every=rebuild_check_every,
            sort_every_rebuild=sort_every_rebuild,
            max_neighbors=max_neighbors,
        )
        return self

    def set_thermo(
        self,
        *,
        every: object,
        prefix: object,
    ) -> "Simulation":
        self._require_not_executed()
        self._thermo_config = _ThermoConfig(
            every=positive_uint64("thermo every", every),
            prefix=output_prefix("thermo", prefix),
        )
        return self

    def save_final_state(
        self,
        *,
        prefix: object,
    ) -> "Simulation":
        self._require_not_executed()
        self._final_state_config = _FinalStateConfig(
            prefix=output_prefix("final_state", prefix),
        )
        return self

    def set_trajectory(
        self,
        *,
        every: object,
        prefix: object,
        fields: object | None = None,
    ) -> "Simulation":
        self._require_not_executed()
        self._trajectory_config = trajectory_config(
            every=every,
            prefix=prefix,
            fields=fields,
        )
        return self

    def set_logging(
        self,
        *,
        echo: object = "screen",
        prefix: object | None = None,
    ) -> "Simulation":
        self._require_not_executed()
        self._log_config = log_config(echo=echo, prefix=prefix)
        return self

    def _native_spec(self) -> dict[str, object]:
        log_spec: dict[str, object] = {"echo": self._log_config.echo}
        if self._log_config.prefix is not None:
            log_spec["prefix"] = self._log_config.prefix

        output: dict[str, object] = {"log": log_spec}
        if self._thermo_config is not None:
            output["thermo"] = {
                "every": self._thermo_config.every,
                "prefix": self._thermo_config.prefix,
            }
        if self._final_state_config is not None:
            output["final_state"] = {
                "prefix": self._final_state_config.prefix,
            }
        if self._trajectory_config is not None:
            output["trajectory"] = {
                "every": self._trajectory_config.every,
                "prefix": self._trajectory_config.prefix,
                "fields": list(self._trajectory_config.fields),
            }

        dynamics_spec: dict[str, object] = {
            "style": self._dynamics_config.style,
            "params": dict(self._dynamics_config.params),
        }
        if self._dynamics_config.thermostat is not None:
            dynamics_spec["thermostat"] = {
                "style": self._dynamics_config.thermostat.style,
                "params": dict(self._dynamics_config.thermostat.params),
            }

        return {
            "system": {
                "units": self._system.units,
                "n_particles": self._system.n_particles,
                "box_bound": self._system.box_bound,
                "positions": self._system.positions,
                "velocities": self._system.velocities,
                "masses": self._system.masses,
                "types": self._system.types,
                "tags": self._system.tags,
                "molecule_ids": self._system.molecule_ids,
                "images": self._system.images,
                "topology": {
                    "bonds": [
                        {"tag_i": tag_i, "tag_j": tag_j, "type": type_id}
                        for tag_i, tag_j, type_id in self._system.topology.bonds
                    ],
                    "angles": [
                        {
                            "tag_i": tag_i,
                            "tag_j": tag_j,
                            "tag_k": tag_k,
                            "type": type_id,
                        }
                        for tag_i, tag_j, tag_k, type_id in self._system.topology.angles
                    ],
                    "dihedrals": [
                        {
                            "tag_i": tag_i,
                            "tag_j": tag_j,
                            "tag_k": tag_k,
                            "tag_l": tag_l,
                            "type": type_id,
                        }
                        for tag_i, tag_j, tag_k, tag_l, type_id
                        in self._system.topology.dihedrals
                    ],
                },
            },
            "forcefield": {
                "pair_style": {
                    "style": self._forcefield_config.pair_style,
                    "params": dict(self._forcefield_config.pair_style_params),
                },
                "pair_coeffs": [
                    {
                        "style": pair_coeff.style,
                        "type_i": pair_coeff.type_i,
                        "type_j": pair_coeff.type_j,
                        "params": dict(pair_coeff.params),
                    }
                    for pair_coeff in self._forcefield_config.pair_coeffs
                ],
                "bond_style": None
                if self._forcefield_config.bond_style is None
                else {
                    "style": self._forcefield_config.bond_style,
                    "params": dict(self._forcefield_config.bond_style_params),
                },
                "bond_coeffs": [
                    {
                        "style": bond_coeff.style,
                        "type": bond_coeff.type,
                        "params": dict(bond_coeff.params),
                    }
                    for bond_coeff in self._forcefield_config.bond_coeffs
                ],
                "angle_style": None
                if self._forcefield_config.angle_style is None
                else {
                    "style": self._forcefield_config.angle_style,
                    "params": dict(self._forcefield_config.angle_style_params),
                },
                "angle_coeffs": [
                    {
                        "style": angle_coeff.style,
                        "type": angle_coeff.type,
                        "params": dict(angle_coeff.params),
                    }
                    for angle_coeff in self._forcefield_config.angle_coeffs
                ],
                "dihedral_style": None
                if self._forcefield_config.dihedral_style is None
                else {
                    "style": self._forcefield_config.dihedral_style,
                    "params": dict(self._forcefield_config.dihedral_style_params),
                },
                "dihedral_coeffs": [
                    {
                        "style": dihedral_coeff.style,
                        "type": dihedral_coeff.type,
                        "params": dict(dihedral_coeff.params),
                    }
                    for dihedral_coeff in self._forcefield_config.dihedral_coeffs
                ],
                "bonded_exclusion_distance":
                    self._forcefield_config.bonded_exclusion_distance,
                "bonded_exclusion_policy":
                    self._forcefield_config.bonded_exclusion_policy,
            },
            "dynamics": dynamics_spec,
            "neighbor": {
                "cutoff_buffer": self._neighbor_config.cutoff_buffer,
                "rebuild_check_every": self._neighbor_config.rebuild_check_every,
                "sort_every_rebuild": self._neighbor_config.sort_every_rebuild,
                "max_neighbors": self._neighbor_config.max_neighbors,
            },
            "output": output,
            "runsteps": self._runsteps,
        }

    def _execute_native(
        self,
        spec: dict[str, object],
    ) -> None:
        execute_simulation(spec)

    def execute(self) -> None:
        self._require_not_executed("execute")
        self._executed = True
        native_spec = self._native_spec()
        self._execute_native(native_spec)
