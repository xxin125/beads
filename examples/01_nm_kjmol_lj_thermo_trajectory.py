from __future__ import annotations

import beads

from _systems import cubic_lattice_system


prefix = "01_nm_kjmol_lj"

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
forcefield = beads.ForceField()

# Pair LJ (default): U = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
forcefield.pair_style("lj", cutoff=0.9)
forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=1.0, sigma=0.34)

# Pair LJ (shifted): U = 4 * epsilon * { [(sigma/r)^12 - (sigma/r)^6]
#                                      - [(sigma/rc)^12 - (sigma/rc)^6] }
# forcefield.pair_style("lj", cutoff=0.9, shift=True)
# forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=1.0, sigma=0.34)

dynamics = beads.Dynamics("velocity_verlet", dt=0.001)

simulation = beads.Simulation(
    system=system,
    forcefield=forcefield,
    dynamics=dynamics,
)
simulation.set_runsteps(20)
simulation.set_neighbor(
    cutoff_buffer=0.3,
    rebuild_check_every=1,
    sort_every_rebuild=5,
    max_neighbors=128,
)
simulation.set_thermo(every=5, prefix=prefix)
simulation.set_trajectory(
    every=10,
    prefix=prefix,
    fields=["tag", "type", "position", "image", "velocity", "force"],
)
simulation.save_final_state(prefix=prefix)
simulation.set_logging(echo="log", prefix=prefix)
simulation.execute()

print(f"wrote {prefix}.*")
