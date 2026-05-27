from __future__ import annotations

import pathlib

import beads
import beads.io

from _systems import write_lammps_molecular_data


data_path = pathlib.Path("07_imported_lammps.data")
prefix = "07_imported_lammps"
dump_path = pathlib.Path("07_imported_lammps.lammpstrj")

write_lammps_molecular_data(data_path)

system = beads.io.read_lammps_data(data_path, units="nm_kjmol")

forcefield = beads.ForceField()
forcefield.pair_style("lj", cutoff=0.9)
forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=1.0, sigma=0.34)
forcefield.pair_coeff("lj", type_i=1, type_j=2, epsilon=0.8, sigma=0.36)
forcefield.pair_coeff("lj", type_i=2, type_j=2, epsilon=0.6, sigma=0.38)
forcefield.bond_style("harmonic")
forcefield.bond_coeff("harmonic", type=1, k=500.0, r0=0.35)

simulation = beads.Simulation(
    system=system,
    forcefield=forcefield,
    dynamics=beads.Dynamics("velocity_verlet", dt=0.001),
)
simulation.set_runsteps(10)
simulation.set_neighbor(
    cutoff_buffer=0.25,
    rebuild_check_every=1,
    sort_every_rebuild=2,
    max_neighbors=64,
)
simulation.set_thermo(every=5, prefix=prefix)
simulation.set_trajectory(
    every=5,
    prefix=prefix,
    fields=["tag", "type", "position", "image", "velocity", "force"],
)
simulation.set_logging(echo="log", prefix=prefix)
simulation.execute()

trajectory_path = pathlib.Path(f"{prefix}.trajectory.beadsbin")
beads.io.write_lammps_dump(trajectory_path, dump_path)

print(f"wrote {data_path}, {prefix}.*, and {dump_path}")
