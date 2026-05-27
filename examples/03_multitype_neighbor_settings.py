from __future__ import annotations

import beads

from _systems import cubic_lattice_system


prefix = "03_multitype_neighbor_settings"

fixture = cubic_lattice_system(
    n_side=5,
    spacing=0.45,
    margin=0.3,
    units="nm_kjmol",
    n_types=2,
    mass=39.948,
    velocity_scale=0.015,
)
system = beads.System(
    positions=fixture.positions,
    velocities=fixture.velocities,
    masses=fixture.masses,
    types=fixture.types,
    box_bound=fixture.box_bound,
    units=fixture.units,
)
forcefield = beads.ForceField()
forcefield.pair_style("lj", cutoff=0.9)
forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=1.0, sigma=0.34)
forcefield.pair_coeff("lj", type_i=1, type_j=2, epsilon=0.8, sigma=0.36)
forcefield.pair_coeff("lj", type_i=2, type_j=2, epsilon=0.6, sigma=0.38)

dynamics = beads.Dynamics("velocity_verlet", dt=0.001)

simulation = beads.Simulation(
    system=system,
    forcefield=forcefield,
    dynamics=dynamics,
)
simulation.set_runsteps(10)
simulation.set_neighbor(
    cutoff_buffer=0.2,
    rebuild_check_every=1,
    sort_every_rebuild=2,
    max_neighbors=160,
)
simulation.set_thermo(every=5, prefix=prefix)
simulation.set_logging(echo="log", prefix=prefix)
simulation.execute()

print(f"wrote {prefix}.*")
