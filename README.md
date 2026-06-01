# BEADS

BEADS, the **Bead-based Efficient Accelerated Dynamics Simulator**, is a
Python-first, CUDA-backed molecular dynamics runtime for coarse-grained
bead-level systems.

BEADS is built around a small Python API for constructing systems, assigning
force fields, choosing dynamics, and running simulations through a compiled
CUDA/C++ backend. It is not a pure Python package, does not provide distributed
multi-GPU execution, and does not expose a stable C++ SDK.

## Requirements

- Linux or WSL2-style CUDA development environment
- CUDA toolkit with `nvcc`
- CMake 3.24 or newer
- Python 3.10 or newer
- NumPy

`python -m pip install .` builds the BEADS native extension locally. A machine
without a CUDA toolkit, supported host compiler, and CMake cannot install BEADS
as if it were pure Python.

## Install

From this package directory:

```bash
python -m pip install .
```

By default, BEADS builds the native engine in single precision
(`real_t = float`, NumPy real dtype `float32`). To build the double-precision
engine instead, pass the CMake option through scikit-build:

```bash
python -m pip install . --config-settings=cmake.define.BEADS_REAL64=ON
```

`beads.build_info()` reports the precision and NumPy dtypes compiled into the
installed native extension. The Python layer uses that information when
validating arrays and reading/writing BEADS binary outputs.

Check the installed package:

```bash
python - <<'PY'
import beads

print(beads.__version__)
print(dict(beads.build_info()))
PY
```

BEADS uses one CUDA logical device per process. By default, it uses the first
CUDA device visible to the process:

```bash
python your_simulation.py
```

To choose a specific physical GPU, set `CUDA_VISIBLE_DEVICES` before launching
Python, for example `CUDA_VISIBLE_DEVICES=0 python your_simulation.py`.

## Minimal Example

```python
import numpy as np

import beads

positions = np.array(
    [
        [0.25, 0.25, 0.25],
        [0.75, 0.25, 0.25],
    ],
    dtype=np.float32,
)
types = np.array([1, 1], dtype=np.int32)
box_bound = np.array(
    [
        [0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0],
    ],
    dtype=np.float32,
)

system = beads.System(
    positions=positions,
    types=types,
    box_bound=box_bound,
    units="nm_kjmol",
)

forcefield = beads.ForceField()
forcefield.pair_style("lj", cutoff=0.9)
forcefield.pair_coeff("lj", type_i=1, type_j=1, epsilon=1.0, sigma=0.34)

dynamics = beads.Dynamics("velocity_verlet", dt=0.001)

simulation = beads.Simulation(
    system=system,
    forcefield=forcefield,
    dynamics=dynamics,
)
simulation.set_runsteps(10)
simulation.set_thermo(every=5, prefix="minimal")
simulation.set_logging(echo="screen")
simulation.execute()
```

## Core Objects

| Object | Role |
| --- | --- |
| `System` | Stores particle arrays, box bounds, units, and optional topology. |
| `ForceField` | Defines pair interactions, listed interactions, and bonded exclusions. |
| `Dynamics` | Selects the integrator and optional thermostat. |
| `Simulation` | Binds a system, force field, dynamics, neighbor settings, outputs, and run length. |
| `beads.io` | Reads supported LAMMPS data files and BEADS binary trajectories. |

`System`, `ForceField`, and `Dynamics` are builder-style objects. Configure
them before binding them to `Simulation`; after binding, they are frozen.

## Units

Choose the unit style when you create a `System`:

```python
system = beads.System(..., units="reduced")
system = beads.System(..., units="nm_kjmol")
```

`units=` is required. It is a contract for how BEADS interprets every numeric
input and how it reports thermo, state, and trajectory data. BEADS does not
convert arrays or parameters when you choose a unit style; provide all
dimensional values in the selected style.

| Quantity | `reduced` | `nm_kjmol` |
| --- | --- | --- |
| length | reduced length | nm |
| mass | reduced mass | u |
| time | reduced time | ps |
| energy | reduced energy | kJ/mol |
| force | reduced force | kJ mol^-1 nm^-1 |
| temperature | reduced temperature | K |
| pressure | reduced pressure | bar |
| Boltzmann constant | `kB = 1` | `kB = 0.00831446261815324` |
| pressure scale | `1` | `16.60539067` |

In `reduced` units, LJ parameters such as `epsilon=1.0` and `sigma=1.0`,
thermostat targets such as `temperature=1.0`, and time steps such as
`dt=0.001` are reduced quantities.

In `nm_kjmol` units, positions and box bounds are in nm, masses are in u,
velocities are in nm/ps, energies are in kJ/mol, time values are in ps,
thermostat temperatures are in K, and thermo pressure is reported in bar.
For example, `epsilon=1.0` is kJ/mol, `sigma=0.34` and pair cutoffs are nm,
`dt=0.001` is ps, and `temperature=300.0` is K.

Lennard-Jones pair interactions use
`U = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]` by default. Pass
`shift=True` to `pair_style("lj", cutoff=..., shift=True)` to subtract the
potential value at the cutoff, making the pair potential zero at `r = cutoff`
without changing forces.

Harmonic bond `r0` values use length units and harmonic bond `k` values use
energy/length^2. Harmonic angle `theta0` values are degrees; angle `k` values
use energy/radian^2. Harmonic dihedral `k` values use energy units.

BEADS writes the selected unit style into final-state and trajectory files.
`System.from_state(...)` restores that unit style. `beads.io.read_lammps_data`
requires a `units=` argument and does not perform unit conversion.

## Supported Surface

- Units: `reduced` and `nm_kjmol`
- Pair style: Lennard-Jones through `pair_style("lj", cutoff=..., shift=...)`
- Listed forces: harmonic bonds, angles, and dihedrals
- Dynamics: `Dynamics("velocity_verlet", dt=...)` for zero-step evaluation and
  time integration
- Thermostat: global Berendsen thermostat for velocity-Verlet dynamics
- Neighbor controls: cutoff buffer, rebuild cadence, sort cadence, and maximum
  neighbors
- Outputs: thermo CSV, log file or screen log, BEADS binary trajectory, and
  final-state snapshot
- IO: supported LAMMPS molecular data-file import, BEADS trajectory reading,
  and LAMMPS custom dump export
- Precision: single precision by default, or double precision with
  `BEADS_REAL64=ON` at build time

## Topology And Force-Field Rules

Call `System.set_topology(...)` before creating a `Simulation`. Bonds, angles,
and dihedrals reference particle tags, not array slots.

If a topology contains bonds, configure `bond_style("harmonic")` and matching
`bond_coeff(...)` entries. If it contains angles, configure
`angle_style("harmonic")` and matching `angle_coeff(...)` entries. If it
contains dihedrals, configure `dihedral_style("harmonic")` and matching
`dihedral_coeff(...)` entries.

Angles must be supported by bonds between their adjacent particles, and
dihedrals must be supported by bonds along the dihedral chain. This keeps the
molecular graph and listed interactions consistent.

When bonds are present, BEADS applies default bonded Lennard-Jones exclusions.
Use `ForceField.exclude_bonded(distance=...)` to make the exclusion distance
explicit.

## Output Files

Output methods take a prefix and append their own suffix:

- thermo CSV: `<prefix>.thermo.csv`
- binary trajectory: `<prefix>.trajectory.beadsbin`
- final-state snapshot: `<prefix>.state.beadsbin`
- log file: `<prefix>.log`
- LAMMPS custom dump converted from a BEADS trajectory: `<name>.lammpstrj`

A final-state snapshot is a restartable particle state, not a full simulation
checkpoint. Recreate the force field, dynamics, and outputs when continuing
from `System.from_state(...)`.

## Examples

Runnable examples are available under [examples/](examples/). See
[examples/README.md](examples/README.md) for the example catalog and output
file conventions.

- `01_nm_kjmol_lj_thermo_trajectory.py`
- `02_nm_kjmol_berendsen.py`
- `03_multitype_neighbor_settings.py`
- `04_bonded_chain_topology.py`
- `05_bonded_exclusion_switch.py`
- `06_state_roundtrip_continue.py`
- `07_lammps_data_io.py`

Run an example after installation:

```bash
python examples/01_nm_kjmol_lj_thermo_trajectory.py
```

The examples write output files in the current working directory. To keep
generated files somewhere else, run the example from that directory or edit the
output prefix in the script.

## Author

BEADS is developed by Xinxin Deng.

## License

BEADS is distributed under the GNU General Public License version 2. See
`LICENSE` for the full license text.
