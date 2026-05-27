from __future__ import annotations

import pathlib

import beads

from _systems import cubic_lattice_system


first_prefix = "06_first_run"
continued_prefix = "06_continued"

fixture = cubic_lattice_system(
    n_side=4,
    spacing=0.55,
    margin=0.5,
    units="nm_kjmol",
    mass=39.948,
    velocity_scale=0.02,
)
system = beads.System(
    positions=fixture.positions,
    velocities=fixture.velocities,
    masses=fixture.masses,
    types=fixture.types,
    box_bound=fixture.box_bound,
    units=fixture.units,
)

first_forcefield = beads.ForceField()
first_forcefield.pair_style("lj", cutoff=0.9)
first_forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=1.0, sigma=0.34)

first = beads.Simulation(
    system=system,
    forcefield=first_forcefield,
    dynamics=beads.Dynamics("velocity_verlet", dt=0.002),
)
first.set_runsteps(5)
first.set_thermo(every=5, prefix=first_prefix)
first.save_final_state(prefix=first_prefix)
first.set_logging(echo="log", prefix=first_prefix)
first.execute()

state_path = pathlib.Path(f"{first_prefix}.state.beadsbin")
restored = beads.System.from_state(state_path)

continued_forcefield = beads.ForceField()
continued_forcefield.pair_style("lj", cutoff=0.9)
continued_forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=1.0, sigma=0.34)

continued = beads.Simulation(
    system=restored,
    forcefield=continued_forcefield,
    dynamics=beads.Dynamics("velocity_verlet", dt=0.002),
)
continued.set_runsteps(5)
continued.set_thermo(every=5, prefix=continued_prefix)
continued.save_final_state(prefix=continued_prefix)
continued.set_logging(echo="log", prefix=continued_prefix)
continued.execute()

print(f"wrote {first_prefix}.* and {continued_prefix}.*")
