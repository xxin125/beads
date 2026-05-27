from __future__ import annotations

from dataclasses import dataclass
import pathlib

import numpy as np


@dataclass(frozen=True)
class SystemArrays:
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    types: np.ndarray
    box_bound: np.ndarray
    units: str
    tags: np.ndarray | None = None
    molecule_ids: np.ndarray | None = None
    bonds: tuple[tuple[int, ...], ...] = ()
    angles: tuple[tuple[int, ...], ...] = ()
    dihedrals: tuple[tuple[int, ...], ...] = ()

def write_lammps_molecular_data(path: pathlib.Path) -> pathlib.Path:
    n_chains = 2
    beads_per_chain = 6
    n_particles = n_chains * beads_per_chain
    box_hi = (4.0, 3.0, 3.0)

    lines = [
        "BEADS example LAMMPS data",
        "",
        f"{n_particles} atoms",
        f"{n_particles - n_chains} bonds",
        "2 atom types",
        "1 bond types",
        "",
        f"0.0 {box_hi[0]:.8g} xlo xhi",
        f"0.0 {box_hi[1]:.8g} ylo yhi",
        f"0.0 {box_hi[2]:.8g} zlo zhi",
        "",
        "Masses",
        "",
        "1 39.948",
        "2 83.798",
        "",
        "Atoms # molecular",
        "",
    ]

    for chain in range(n_chains):
        molecule_id = chain + 1
        y = 0.8 + 0.9 * chain
        for bead in range(beads_per_chain):
            tag = chain * beads_per_chain + bead + 1
            atom_type = 1 if bead % 2 == 0 else 2
            x = 0.5 + 0.35 * bead
            z = 0.8 + 0.03 * ((bead % 3) - 1)
            lines.append(
                f"{tag} {molecule_id} {atom_type} {x:.8g} {y:.8g} {z:.8g} 0 0 0"
            )

    lines.extend(["", "Velocities", ""])
    for tag in range(1, n_particles + 1):
        vx = 0.010 * ((tag % 3) - 1)
        vy = 0.006 * ((tag % 5) - 2)
        vz = -0.004 * ((tag % 4) - 1)
        lines.append(f"{tag} {vx:.8g} {vy:.8g} {vz:.8g}")

    lines.extend(["", "Bonds", ""])
    bond_id = 1
    for chain in range(n_chains):
        first = chain * beads_per_chain + 1
        for bead in range(beads_per_chain - 1):
            lines.append(f"{bond_id} 1 {first + bead} {first + bead + 1}")
            bond_id += 1

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def deterministic_velocities(n_particles: int, scale: float) -> np.ndarray:
    index = np.arange(1, n_particles + 1, dtype=np.float64)
    velocities = np.column_stack(
        (
            np.sin(0.37 * index),
            np.cos(0.23 * index),
            np.sin(0.19 * index + 0.4),
        )
    )
    velocities -= velocities.mean(axis=0, keepdims=True)
    return velocities * scale


def cubic_lattice_system(
    *,
    n_side: int,
    spacing: float,
    margin: float,
    units: str,
    n_types: int = 1,
    mass: float = 1.0,
    velocity_scale: float = 0.02,
) -> SystemArrays:
    grid = np.indices((n_side, n_side, n_side), dtype=np.int64).reshape(3, -1).T
    positions = margin + spacing * grid.astype(np.float64)
    upper = 2.0 * margin + spacing * float(n_side - 1)
    types = (grid.sum(axis=1) % n_types + 1).astype(np.int32)
    masses = np.full(positions.shape[0], mass, dtype=np.float64)
    box_bound = np.asarray(
        [[0.0, 0.0, 0.0], [upper, upper, upper]],
        dtype=np.float64,
    )
    return SystemArrays(
        positions=positions,
        velocities=deterministic_velocities(positions.shape[0], velocity_scale),
        masses=masses,
        types=types,
        box_bound=box_bound,
        units=units,
    )


def polymer_chain_system(
    *,
    n_chains: int,
    beads_per_chain: int,
    bond_length: float,
    chain_spacing: float,
    units: str = "nm_kjmol",
    include_angles: bool = True,
    include_dihedrals: bool = True,
) -> SystemArrays:
    positions: list[list[float]] = []
    molecule_ids: list[int] = []
    for chain in range(n_chains):
        y0 = 1.0 + chain * chain_spacing
        for bead in range(beads_per_chain):
            zig = 0.12 if bead % 2 == 0 else -0.12
            twist = 0.08 * float((bead % 3) - 1)
            positions.append(
                [
                    1.0 + bead * bond_length,
                    y0 + zig,
                    1.0 + twist,
                ]
            )
            molecule_ids.append(chain + 1)

    n_particles = n_chains * beads_per_chain
    tags = np.arange(1, n_particles + 1, dtype=np.uint32)
    bonds: list[tuple[int, int, int]] = []
    angles: list[tuple[int, int, int, int]] = []
    dihedrals: list[tuple[int, int, int, int, int]] = []
    for chain in range(n_chains):
        first = chain * beads_per_chain + 1
        for bead in range(beads_per_chain - 1):
            bonds.append((first + bead, first + bead + 1, 1))
        if include_angles:
            for bead in range(beads_per_chain - 2):
                angles.append((first + bead, first + bead + 1, first + bead + 2, 1))
        if include_dihedrals:
            for bead in range(beads_per_chain - 3):
                dihedrals.append(
                    (
                        first + bead,
                        first + bead + 1,
                        first + bead + 2,
                        first + bead + 3,
                        1,
                    )
                )

    upper_x = 2.0 + bond_length * float(beads_per_chain - 1)
    upper_y = max(4.0, 2.0 + chain_spacing * float(n_chains - 1) + 1.0)
    box_bound = np.asarray(
        [[0.0, 0.0, 0.0], [upper_x, upper_y, 4.0]],
        dtype=np.float64,
    )
    return SystemArrays(
        positions=np.asarray(positions, dtype=np.float64),
        velocities=deterministic_velocities(n_particles, 0.01),
        masses=np.ones(n_particles, dtype=np.float64),
        types=np.ones(n_particles, dtype=np.int32),
        tags=tags,
        molecule_ids=np.asarray(molecule_ids, dtype=np.uint32),
        box_bound=box_bound,
        units=units,
        bonds=tuple(bonds),
        angles=tuple(angles),
        dihedrals=tuple(dihedrals),
    )
