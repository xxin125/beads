# BEADS Examples

These examples demonstrate common BEADS Python workflows. They generate
deterministic systems in code, define force fields explicitly, and write output
files in the current working directory.

Run an example after installing BEADS:

```bash
python beads/examples/01_nm_kjmol_lj_thermo_trajectory.py
```

To keep generated files somewhere else, run the example from that directory or
edit the output prefix in the script.

The examples use one CUDA logical device per process. By default, BEADS uses
the first CUDA device visible to the process. Set `CUDA_VISIBLE_DEVICES` before
launch only when you want to choose a specific physical GPU.

## Current examples

| Example | Units | Shows |
| --- | --- | --- |
| `01_nm_kjmol_lj_thermo_trajectory.py` | `nm_kjmol` | 64-particle LJ Velocity-Verlet run with thermo, trajectory, final state, and file log output. |
| `02_nm_kjmol_berendsen.py` | `nm_kjmol` | Argon-like LJ run using nm, ps, kJ/mol, K, and global Berendsen temperature coupling. |
| `03_multitype_neighbor_settings.py` | `nm_kjmol` | 125-particle two-type LJ mixture with full type-pair coverage and explicit neighbor settings. |
| `04_bonded_chain_topology.py` | `nm_kjmol` | Bond, angle, and dihedral topology with harmonic listed forces and default bonded LJ exclusions. |
| `05_bonded_exclusion_switch.py` | `nm_kjmol` | Side-by-side default distance-2 bonded LJ exclusions vs explicit `exclude_bonded(distance=3)`. |
| `06_state_roundtrip_continue.py` | `nm_kjmol` | Save a final-state snapshot, restore it with `System.from_state(...)`, and continue in a fresh simulation. |
| `07_lammps_data_io.py` | `nm_kjmol` | Import a LAMMPS molecular data file with `beads.io`, run it, and export the BEADS trajectory as a LAMMPS custom dump. |

## Output files

Output methods take a prefix and append their own suffix:

- thermo CSV: `<prefix>.thermo.csv`
- binary trajectory: `<prefix>.trajectory.beadsbin`
- final-state snapshot: `<prefix>.state.beadsbin`
- log file: `<prefix>.log`
- LAMMPS custom dump converted from a BEADS trajectory: `<name>.lammpstrj`
