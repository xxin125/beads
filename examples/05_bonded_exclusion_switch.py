from __future__ import annotations

import beads

from _systems import polymer_chain_system


default_prefix = "05_default_distance_two"
default_fixture = polymer_chain_system(
    n_chains=1,
    beads_per_chain=8,
    bond_length=0.35,
    chain_spacing=1.0,
    units="nm_kjmol",
    include_angles=False,
    include_dihedrals=False,
)
default_system = beads.System(
    positions=default_fixture.positions,
    velocities=default_fixture.velocities,
    masses=default_fixture.masses,
    types=default_fixture.types,
    tags=default_fixture.tags,
    molecule_ids=default_fixture.molecule_ids,
    box_bound=default_fixture.box_bound,
    units=default_fixture.units,
)
default_system.set_topology(bonds=default_fixture.bonds)
default_forcefield = beads.ForceField()
default_forcefield.pair_style("lj", cutoff=1.2)
default_forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=0.5, sigma=0.34)
default_forcefield.bond_style("harmonic")
default_forcefield.bond_coeff("harmonic", type=1, k=0.0, r0=0.35)

default_simulation = beads.Simulation(
    system=default_system,
    forcefield=default_forcefield,
    dynamics=beads.Dynamics("none"),
)
default_simulation.set_runsteps(0)
default_simulation.set_thermo(every=1, prefix=default_prefix)
default_simulation.set_logging(echo="log", prefix=default_prefix)
default_simulation.execute()

distance_three_prefix = "05_explicit_distance_three"
distance_three_fixture = polymer_chain_system(
    n_chains=1,
    beads_per_chain=8,
    bond_length=0.35,
    chain_spacing=1.0,
    units="nm_kjmol",
    include_angles=False,
    include_dihedrals=False,
)
distance_three_system = beads.System(
    positions=distance_three_fixture.positions,
    velocities=distance_three_fixture.velocities,
    masses=distance_three_fixture.masses,
    types=distance_three_fixture.types,
    tags=distance_three_fixture.tags,
    molecule_ids=distance_three_fixture.molecule_ids,
    box_bound=distance_three_fixture.box_bound,
    units=distance_three_fixture.units,
)
distance_three_system.set_topology(bonds=distance_three_fixture.bonds)
distance_three_forcefield = beads.ForceField()
distance_three_forcefield.pair_style("lj", cutoff=1.2)
distance_three_forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=0.5, sigma=0.34)
distance_three_forcefield.bond_style("harmonic")
distance_three_forcefield.bond_coeff("harmonic", type=1, k=0.0, r0=0.35)
distance_three_forcefield.exclude_bonded(distance=3)

distance_three_simulation = beads.Simulation(
    system=distance_three_system,
    forcefield=distance_three_forcefield,
    dynamics=beads.Dynamics("none"),
)
distance_three_simulation.set_runsteps(0)
distance_three_simulation.set_thermo(every=1, prefix=distance_three_prefix)
distance_three_simulation.set_logging(echo="log", prefix=distance_three_prefix)
distance_three_simulation.execute()

print(f"wrote {default_prefix}.* and {distance_three_prefix}.*")
