from __future__ import annotations

import beads

from _systems import polymer_chain_system


prefix = "04_bonded_chain_topology"

fixture = polymer_chain_system(
    n_chains=4,
    beads_per_chain=16,
    bond_length=0.35,
    chain_spacing=1.0,
    units="nm_kjmol",
    include_angles=True,
    include_dihedrals=True,
)
system = beads.System(
    positions=fixture.positions,
    velocities=fixture.velocities,
    masses=fixture.masses,
    types=fixture.types,
    tags=fixture.tags,
    molecule_ids=fixture.molecule_ids,
    box_bound=fixture.box_bound,
    units=fixture.units,
)
system.set_topology(
    bonds=fixture.bonds,
    angles=fixture.angles,
    dihedrals=fixture.dihedrals,
)
forcefield = beads.ForceField()
forcefield.pair_style("lj", cutoff=0.75)
forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=0.5, sigma=0.34)
forcefield.bond_style("harmonic")
forcefield.bond_coeff("harmonic", type=1, k=500.0, r0=0.35)
forcefield.angle_style("harmonic")
forcefield.angle_coeff("harmonic", type=1, k=5.0, theta0=165.0)
forcefield.dihedral_style("harmonic")
forcefield.dihedral_coeff("harmonic", type=1, k=1.0, d=1, n=1)

dynamics = beads.Dynamics("velocity_verlet", dt=0.001)

simulation = beads.Simulation(
    system=system,
    forcefield=forcefield,
    dynamics=dynamics,
)
simulation.set_runsteps(10)
simulation.set_thermo(every=5, prefix=prefix)
simulation.save_final_state(prefix=prefix)
simulation.set_logging(echo="log", prefix=prefix)
simulation.execute()

print(f"wrote {prefix}.*")
