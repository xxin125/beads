from __future__ import annotations

import beads

from _systems import cubic_lattice_system


prefix = "02_nm_kjmol_berendsen"

fixture = cubic_lattice_system(
    n_side=4,
    spacing=0.55,
    margin=0.3,
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
forcefield.pair_style("lj", cutoff=0.75)
forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=1.0, sigma=0.34)

dynamics = beads.Dynamics("velocity_verlet", dt=0.001)
dynamics.set_thermostat("berendsen", temperature=120.0, tau=0.1)

simulation = beads.Simulation(
    system=system,
    forcefield=forcefield,
    dynamics=dynamics,
)
simulation.set_runsteps(20)
simulation.set_thermo(every=5, prefix=prefix)
simulation.save_final_state(prefix=prefix)
simulation.set_logging(echo="log", prefix=prefix)
simulation.execute()

print(f"wrote {prefix}.*")
