"""Small IO helpers for BEADS systems and trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
import os
import struct

import numpy as np

from ._system import System

__all__ = [
    "read_beads_trajectory",
    "read_lammps_data",
    "read_trajectory",
    "write_lammps_dump",
]

_LAMMPS_DATA_SYSTEM_SECTIONS = frozenset(
    {
        "Masses",
        "Atoms",
        "Velocities",
        "Bonds",
        "Angles",
        "Dihedrals",
        "Impropers",
    }
)
_LAMMPS_DATA_SKIP_SECTIONS = frozenset(
    {
        "Atom Type Labels",
        "Bond Type Labels",
        "Angle Type Labels",
        "Dihedral Type Labels",
        "Improper Type Labels",
        "Pair Coeffs",
        "PairIJ Coeffs",
        "Bond Coeffs",
        "Angle Coeffs",
        "Dihedral Coeffs",
        "Improper Coeffs",
        "BondBond Coeffs",
        "BondAngle Coeffs",
        "MiddleBondTorsion Coeffs",
        "EndBondTorsion Coeffs",
        "AngleTorsion Coeffs",
        "AngleAngleTorsion Coeffs",
        "BondBond13 Coeffs",
        "AngleAngle Coeffs",
    }
)
_LAMMPS_DATA_UNSUPPORTED_SECTIONS = frozenset(
    {
        "Ellipsoids",
        "Lines",
        "Triangles",
        "Bodies",
    }
)
_LAMMPS_DATA_SECTIONS = (
    _LAMMPS_DATA_SYSTEM_SECTIONS
    | _LAMMPS_DATA_SKIP_SECTIONS
    | _LAMMPS_DATA_UNSUPPORTED_SECTIONS
)
_LAMMPS_DATA_SECTION_BY_LOWER = {
    name.lower(): name
    for name in _LAMMPS_DATA_SECTIONS
}
_LAMMPS_DATA_TYPE_LABEL_SECTIONS = frozenset(
    name for name in _LAMMPS_DATA_SKIP_SECTIONS if name.endswith("Type Labels")
)

_TRAJECTORY_MAGIC = b"BEADS_TRAJECTORY"
_TRAJECTORY_VERSION = 1
_TRAJECTORY_ENDIAN_MARKER = 0x01020304
_TRAJECTORY_FIELD_TAG = 1 << 0
_TRAJECTORY_FIELD_TYPE = 1 << 1
_TRAJECTORY_FIELD_POSITION = 1 << 2
_TRAJECTORY_FIELD_IMAGE = 1 << 3
_TRAJECTORY_FIELD_VELOCITY = 1 << 4
_TRAJECTORY_FIELD_FORCE = 1 << 5
_TRAJECTORY_FIELD_BITS = (
    ("tag", _TRAJECTORY_FIELD_TAG),
    ("type", _TRAJECTORY_FIELD_TYPE),
    ("position", _TRAJECTORY_FIELD_POSITION),
    ("image", _TRAJECTORY_FIELD_IMAGE),
    ("velocity", _TRAJECTORY_FIELD_VELOCITY),
    ("force", _TRAJECTORY_FIELD_FORCE),
)
_TRAJECTORY_KNOWN_MASK = sum(bit for _, bit in _TRAJECTORY_FIELD_BITS)


def read_lammps_data(path: object, *, units: str) -> System:
    """Read a minimal LAMMPS data-file subset as a ``beads.System``.

    The importer reads atoms, optional masses and velocities, orthogonal box
    bounds, and Bonds/Angles/Dihedrals topology. It does not parse force-field
    coefficients or perform units conversion; ``units`` is passed directly to
    ``beads.System``.
    """
    if units is None:
        raise TypeError("read_lammps_data(...) requires the units keyword argument.")

    header, sections = _read_lammps_data_sections(path)
    _reject_lammps_data_type_labels(sections)
    _reject_lammps_data_unsupported_sections(sections)
    _require_lammps_data_header(header, sections)

    declared_impropers = int(header.get("impropers", 0))
    impropers_rows = sections.get("Impropers", {"lines": []})["lines"]
    if declared_impropers > 0 or impropers_rows:
        raise ValueError(
            "LAMMPS data import does not support Impropers. Remove the Impropers "
            "section before importing the file into BEADS."
        )

    atoms = _parse_lammps_atoms_section(
        sections["Atoms"]["lines"],
        str(sections["Atoms"]["comment"]),
    )
    expected_atoms = int(header["atoms"])
    if len(atoms) != expected_atoms:
        raise ValueError(
            f"LAMMPS data atoms count is {expected_atoms}, "
            f"but Atoms contains {len(atoms)} rows."
        )

    _validate_lammps_atom_ids_for_beads(atoms)
    _validate_lammps_molecule_ids_for_beads(atoms)
    atom_ids = {int(atom["id"]) for atom in atoms}
    atom_types = {int(atom["type"]) for atom in atoms}
    _validate_lammps_declared_type_count(header, "atom types", atom_types, "atom")
    _require_dense_lammps_type_ids("atom", atom_types)

    velocities_by_id = None
    if "Velocities" in sections:
        velocities_by_id = _parse_lammps_velocities_section(
            sections["Velocities"]["lines"],
            atom_ids,
        )

    masses_by_type = None
    if "Masses" in sections:
        masses_by_type = _parse_lammps_masses_section(sections["Masses"]["lines"])
        _validate_lammps_declared_type_count(
            header,
            "atom types",
            set(masses_by_type),
            "Masses atom",
        )

    bonds = _parse_lammps_topology_section(
        sections.get("Bonds", {"lines": []})["lines"],
        2,
        "Bonds",
        atom_ids,
    )
    angles = _parse_lammps_topology_section(
        sections.get("Angles", {"lines": []})["lines"],
        3,
        "Angles",
        atom_ids,
    )
    dihedrals = _parse_lammps_topology_section(
        sections.get("Dihedrals", {"lines": []})["lines"],
        4,
        "Dihedrals",
        atom_ids,
    )
    _validate_lammps_topology_counts(header, bonds, angles, dihedrals)
    _validate_lammps_declared_type_count(
        header,
        "bond types",
        {int(item[-1]) for item in bonds},
        "bond",
    )
    _validate_lammps_declared_type_count(
        header,
        "angle types",
        {int(item[-1]) for item in angles},
        "angle",
    )
    _validate_lammps_declared_type_count(
        header,
        "dihedral types",
        {int(item[-1]) for item in dihedrals},
        "dihedral",
    )
    _require_dense_lammps_type_ids("bond topology", (item[-1] for item in bonds))
    _require_dense_lammps_type_ids("angle topology", (item[-1] for item in angles))
    _require_dense_lammps_type_ids(
        "dihedral topology",
        (item[-1] for item in dihedrals),
    )

    positions = np.array([atom["position"] for atom in atoms], dtype=np.float64)
    types = np.array([atom["type"] for atom in atoms], dtype=np.int64)
    tags = np.array([atom["id"] for atom in atoms], dtype=np.uint64)
    molecule_ids = np.array([atom["molecule_id"] for atom in atoms], dtype=np.uint64)
    images = np.array([atom["image"] for atom in atoms], dtype=np.int64)
    velocities = _lammps_velocities_for_atoms(atoms, velocities_by_id)
    masses = _lammps_masses_for_atoms(atoms, masses_by_type)
    system = System(
        box_bound=np.array(
            [
                [header["xlo"], header["ylo"], header["zlo"]],
                [header["xhi"], header["yhi"], header["zhi"]],
            ],
            dtype=np.float64,
        ),
        positions=positions,
        types=types,
        velocities=velocities,
        masses=masses,
        tags=tags,
        molecule_ids=molecule_ids,
        images=images,
        units=units,
    )
    if bonds or angles or dihedrals:
        system.set_topology(
            bonds=bonds,
            angles=angles,
            dihedrals=dihedrals,
        )
    return system


def read_beads_trajectory(path: object) -> dict[str, object]:
    """Read a BEADS binary trajectory into memory.

    ``records`` preserves the writer's particle slot order for each frame. Use
    the ``tag`` field to sort or join particle identities across frames.
    """
    with open(os.fspath(path), "rb") as handle:
        header = _trajectory_header_from_handle(handle)
        frames = np.fromfile(
            handle,
            dtype=header.frame_dtype,
            count=header.frame_count,
        )
        if frames.shape[0] != header.frame_count:
            raise ValueError("Trajectory payload ended before all frames were read.")

    boxes = np.array(frames["box_bound"], copy=True)
    return {
        "version": header.version,
        "endian": header.endian,
        "real_size": header.real_size,
        "index_size": header.index_size,
        "image_size": header.image_size,
        "type_size": header.type_size,
        "tag_size": header.tag_size,
        "units": header.units,
        "field_mask": header.field_mask,
        "fields": list(header.fields),
        "n_particles": header.n_particles,
        "particle_count": header.n_particles,
        "record_bytes": header.record_bytes,
        "frame_count": header.frame_count,
        "steps": np.array(frames["step"], copy=True),
        "box_bounds": boxes,
        "box_lo": boxes[:, 0, :],
        "box_hi": boxes[:, 1, :],
        "records": np.array(frames["records"], copy=True),
    }


def read_trajectory(path: object) -> dict[str, object]:
    """Alias for ``read_beads_trajectory``."""
    return read_beads_trajectory(path)


def write_lammps_dump(
    trajectory_path: object,
    dump_path: object,
    *,
    sort_by_tag: bool = True,
    precision: int = 17,
) -> None:
    """Convert a BEADS binary trajectory to LAMMPS custom dump text."""
    if not isinstance(sort_by_tag, bool):
        raise TypeError("write_lammps_dump(...) sort_by_tag must be a bool.")
    precision = _checked_lammps_dump_precision(precision)
    with open(os.fspath(trajectory_path), "rb") as source:
        header = _trajectory_header_from_handle(source)
        with open(os.fspath(dump_path), "w", encoding="utf-8") as target:
            _write_lammps_dump_frames(
                source,
                target,
                header=header,
                sort_by_tag=sort_by_tag,
                precision=precision,
            )


def _split_lammps_data_comment(raw_line: str) -> tuple[str, str]:
    content, separator, comment = raw_line.partition("#")
    return content.strip(), comment.strip() if separator else ""


def _lammps_data_section_name(content: str) -> str | None:
    normalized = " ".join(content.split())
    known = _LAMMPS_DATA_SECTION_BY_LOWER.get(normalized.lower())
    if known is not None:
        return known
    lower = normalized.lower()
    if lower.endswith(" coeffs") or lower.endswith(" type labels"):
        return normalized
    return None


def _lammps_data_int(token: str, context: str) -> int:
    try:
        return int(token, 10)
    except ValueError as exc:
        raise ValueError(f"LAMMPS data {context} must be an integer.") from exc


def _lammps_data_positive_int(token: str, context: str) -> int:
    value = _lammps_data_int(token, context)
    if value <= 0:
        raise ValueError(f"LAMMPS data {context} must be positive.")
    return value


def _lammps_data_nonnegative_int(token: str, context: str) -> int:
    value = _lammps_data_int(token, context)
    if value < 0:
        raise ValueError(f"LAMMPS data {context} must be non-negative.")
    return value


def _lammps_data_float(token: str, context: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"LAMMPS data {context} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"LAMMPS data {context} must be finite.")
    return value


def _require_lammps_data_width(
    tokens: list[str],
    width: int,
    context: str,
    *,
    layout: str | None = None,
) -> None:
    if len(tokens) != width:
        layout_suffix = f" ({layout})" if layout else ""
        raise ValueError(
            f"LAMMPS data {context} expected {width} columns{layout_suffix}, "
            f"got {len(tokens)}."
        )


def _read_lammps_data_sections(
    path: object,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    header: dict[str, object] = {}
    sections: dict[str, dict[str, object]] = {}
    current_section: str | None = None

    with open(os.fspath(path), "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            content, comment = _split_lammps_data_comment(raw_line)
            if not content:
                continue
            section_name = _lammps_data_section_name(content)
            if section_name is not None:
                if section_name in sections:
                    raise ValueError(
                        f"LAMMPS data section {section_name!r} is duplicated."
                    )
                current_section = section_name
                sections[section_name] = {
                    "comment": comment,
                    "lines": [],
                }
                continue
            if current_section is not None:
                sections[current_section]["lines"].append((line_number, content))
                continue

            tokens = content.split()
            if len(tokens) >= 2 and tokens[-2:] == ["xlo", "xhi"]:
                _require_lammps_data_width(tokens, 4, "x bounds")
                header["xlo"] = _lammps_data_float(tokens[0], "xlo")
                header["xhi"] = _lammps_data_float(tokens[1], "xhi")
                continue
            if len(tokens) >= 2 and tokens[-2:] == ["ylo", "yhi"]:
                _require_lammps_data_width(tokens, 4, "y bounds")
                header["ylo"] = _lammps_data_float(tokens[0], "ylo")
                header["yhi"] = _lammps_data_float(tokens[1], "yhi")
                continue
            if len(tokens) >= 2 and tokens[-2:] == ["zlo", "zhi"]:
                _require_lammps_data_width(tokens, 4, "z bounds")
                header["zlo"] = _lammps_data_float(tokens[0], "zlo")
                header["zhi"] = _lammps_data_float(tokens[1], "zhi")
                continue
            if len(tokens) >= 3 and tokens[-3:] == ["xy", "xz", "yz"]:
                raise ValueError(
                    "LAMMPS data import supports only orthogonal box bounds; "
                    "triclinic tilt factors are unsupported."
                )
            if tokens[0] in {"avec", "bvec", "cvec"} or tokens[:2] == ["abc", "origin"]:
                raise ValueError(
                    "LAMMPS data import does not support general triclinic boxes."
                )
            if len(tokens) >= 2:
                count_key = " ".join(tokens[1:])
                if count_key in {
                    "atoms",
                    "bonds",
                    "angles",
                    "dihedrals",
                    "impropers",
                    "atom types",
                    "bond types",
                    "angle types",
                    "dihedral types",
                    "improper types",
                }:
                    header[count_key] = _lammps_data_nonnegative_int(
                        tokens[0],
                        count_key,
                    )
                    continue
            header.setdefault("title", content)
    return header, sections


def _reject_lammps_data_type_labels(sections: dict[str, dict[str, object]]) -> None:
    type_label_sections = sorted(
        name
        for name in sections
        if name in _LAMMPS_DATA_TYPE_LABEL_SECTIONS or name.endswith("Type Labels")
    )
    if type_label_sections:
        joined = ", ".join(type_label_sections)
        raise ValueError(f"LAMMPS data type labels are not supported: {joined}.")


def _reject_lammps_data_unsupported_sections(
    sections: dict[str, dict[str, object]],
) -> None:
    unsupported_sections = sorted(
        name for name in sections if name in _LAMMPS_DATA_UNSUPPORTED_SECTIONS
    )
    if unsupported_sections:
        joined = ", ".join(unsupported_sections)
        raise ValueError(f"LAMMPS data import does not support sections: {joined}.")


def _require_lammps_data_header(
    header: dict[str, object],
    sections: dict[str, dict[str, object]],
) -> None:
    missing_bounds = [
        key
        for key in ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")
        if key not in header
    ]
    if missing_bounds:
        raise ValueError("LAMMPS data file is missing required orthogonal box bounds.")
    for axis in ("x", "y", "z"):
        if float(header[f"{axis}hi"]) <= float(header[f"{axis}lo"]):
            raise ValueError(f"LAMMPS data {axis}hi must be greater than {axis}lo.")
    if "atoms" not in header:
        raise ValueError("LAMMPS data file is missing the atoms count.")
    if "Atoms" not in sections:
        raise ValueError("LAMMPS data file is missing the Atoms section.")


def _parse_lammps_atoms_section(
    lines: object,
    comment: str,
) -> list[dict[str, object]]:
    if comment:
        atom_style = comment.split()[0].lower()
        if atom_style != "molecular":
            raise ValueError("LAMMPS data import supports only Atoms # molecular.")

    atoms: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    has_images: bool | None = None
    for line_number, content in lines:
        tokens = str(content).split()
        if len(tokens) not in (6, 9):
            raise ValueError(
                f"LAMMPS data Atoms line {line_number} must use molecular format: "
                "atom-ID molecule-ID atom-type x y z [ix iy iz]."
            )
        line_has_images = len(tokens) == 9
        if has_images is None:
            has_images = line_has_images
        elif has_images != line_has_images:
            raise ValueError("LAMMPS data Atoms section must not mix image flags.")
        atom_id = _lammps_data_positive_int(
            tokens[0],
            f"Atoms line {line_number} atom-ID",
        )
        if atom_id in seen_ids:
            raise ValueError(f"LAMMPS data Atoms section duplicates atom ID {atom_id}.")
        seen_ids.add(atom_id)
        molecule_id = _lammps_data_int(
            tokens[1],
            f"Atoms line {line_number} molecule-ID",
        )
        if molecule_id < 0:
            raise ValueError("LAMMPS data molecule-ID values must be non-negative.")
        atom_type = _lammps_data_positive_int(
            tokens[2],
            f"Atoms line {line_number} atom-type",
        )
        position = tuple(
            _lammps_data_float(tokens[index], f"Atoms line {line_number} coordinate")
            for index in (3, 4, 5)
        )
        image = (0, 0, 0)
        if line_has_images:
            image = tuple(
                _lammps_data_int(tokens[index], f"Atoms line {line_number} image flag")
                for index in (6, 7, 8)
            )
        atoms.append(
            {
                "id": atom_id,
                "molecule_id": molecule_id,
                "type": atom_type,
                "position": position,
                "image": image,
            }
        )
    atoms.sort(key=lambda atom: int(atom["id"]))
    return atoms


def _parse_lammps_masses_section(lines: object) -> dict[int, float]:
    masses: dict[int, float] = {}
    for line_number, content in lines:
        tokens = str(content).split()
        _require_lammps_data_width(
            tokens,
            2,
            f"Masses line {line_number}",
            layout="atom-type mass",
        )
        atom_type = _lammps_data_positive_int(
            tokens[0],
            f"Masses line {line_number} atom type",
        )
        if atom_type in masses:
            raise ValueError(
                f"LAMMPS data Masses section duplicates atom type {atom_type}."
            )
        mass = _lammps_data_float(tokens[1], f"Masses line {line_number} mass")
        if mass <= 0.0:
            raise ValueError("LAMMPS data masses must be positive.")
        masses[atom_type] = mass
    return masses


def _parse_lammps_velocities_section(
    lines: object,
    atom_ids: set[int],
) -> dict[int, tuple[float, float, float]]:
    velocities: dict[int, tuple[float, float, float]] = {}
    for line_number, content in lines:
        tokens = str(content).split()
        _require_lammps_data_width(tokens, 4, f"Velocities line {line_number}")
        atom_id = _lammps_data_positive_int(
            tokens[0],
            f"Velocities line {line_number} atom-ID",
        )
        if atom_id not in atom_ids:
            raise ValueError(
                f"LAMMPS data Velocities section references unknown atom ID {atom_id}."
            )
        if atom_id in velocities:
            raise ValueError(
                f"LAMMPS data Velocities section duplicates atom ID {atom_id}."
            )
        velocities[atom_id] = tuple(
            _lammps_data_float(
                tokens[index],
                f"Velocities line {line_number} component",
            )
            for index in (1, 2, 3)
        )
    if set(velocities) != atom_ids:
        raise ValueError(
            "LAMMPS data Velocities section must include every atom exactly once."
        )
    return velocities


def _parse_lammps_topology_section(
    lines: object,
    width: int,
    label: str,
    atom_ids: set[int],
) -> tuple[tuple[int, ...], ...]:
    entries: list[tuple[int, ...]] = []
    seen_term_ids: set[int] = set()
    layout_by_width = {
        2: "id type atom1 atom2",
        3: "id type atom1 atom2 atom3",
        4: "id type atom1 atom2 atom3 atom4",
    }
    for line_number, content in lines:
        tokens = str(content).split()
        _require_lammps_data_width(
            tokens,
            width + 2,
            f"{label} line {line_number}",
            layout=layout_by_width[width],
        )
        term_id = _lammps_data_positive_int(
            tokens[0],
            f"{label} line {line_number} id",
        )
        if term_id in seen_term_ids:
            raise ValueError(f"LAMMPS data {label} section duplicates id {term_id}.")
        seen_term_ids.add(term_id)
        term_type = _lammps_data_positive_int(
            tokens[1],
            f"{label} line {line_number} type",
        )
        refs = tuple(
            _lammps_data_positive_int(
                tokens[index],
                f"{label} line {line_number} atom id",
            )
            for index in range(2, 2 + width)
        )
        for atom_id in refs:
            if atom_id not in atom_ids:
                raise ValueError(
                    f"LAMMPS data {label} section references unknown atom ID {atom_id}."
                )
        entries.append((*refs, term_type))
    return tuple(entries)


def _validate_lammps_atom_ids_for_beads(atoms: list[dict[str, object]]) -> None:
    n_particles = len(atoms)
    atom_ids = [int(atom["id"]) for atom in atoms]
    if set(atom_ids) != set(range(1, n_particles + 1)):
        raise ValueError(
            "LAMMPS data atom-ID values must satisfy BEADS System.tags: "
            "a unique 1..N permutation."
        )


def _validate_lammps_molecule_ids_for_beads(atoms: list[dict[str, object]]) -> None:
    n_particles = len(atoms)
    molecule_ids = [int(atom["molecule_id"]) for atom in atoms]
    if any(molecule_id <= 0 for molecule_id in molecule_ids):
        raise ValueError(
            "LAMMPS data molecule-ID values must satisfy BEADS System.molecule_ids: "
            "positive dense 1-based ids."
        )
    unique_ids = set(molecule_ids)
    expected = set(range(1, len(unique_ids) + 1))
    if unique_ids != expected or max(unique_ids, default=0) > n_particles:
        raise ValueError(
            "LAMMPS data molecule-ID values must satisfy BEADS System.molecule_ids: "
            "positive dense 1-based ids that do not exceed the particle count."
        )


def _validate_lammps_declared_type_count(
    header: dict[str, object],
    header_key: str,
    active_types: object,
    context: str,
) -> None:
    if header_key not in header:
        return
    active = [int(item) for item in active_types]
    highest = max(active, default=0)
    declared = int(header[header_key])
    if highest > declared:
        raise ValueError(
            f"LAMMPS data {context} type id {highest} exceeds declared "
            f"{header_key} count {declared}."
        )


def _validate_lammps_topology_counts(
    header: dict[str, object],
    bonds: tuple[tuple[int, ...], ...],
    angles: tuple[tuple[int, ...], ...],
    dihedrals: tuple[tuple[int, ...], ...],
) -> None:
    for section_name, parsed in (
        ("bonds", bonds),
        ("angles", angles),
        ("dihedrals", dihedrals),
    ):
        if section_name in header and int(header[section_name]) != len(parsed):
            raise ValueError(
                f"LAMMPS data {section_name} count is {header[section_name]}, "
                f"but the section contains {len(parsed)} rows."
            )


def _require_dense_lammps_type_ids(label: str, type_ids: object) -> None:
    unique = sorted({int(item) for item in type_ids})
    if not unique:
        return
    expected = list(range(1, unique[-1] + 1))
    if unique != expected:
        raise ValueError(
            f"LAMMPS data {label} type ids must be dense one-based ids without gaps."
        )


def _lammps_velocities_for_atoms(
    atoms: list[dict[str, object]],
    velocities_by_id: dict[int, tuple[float, float, float]] | None,
) -> np.ndarray | None:
    if velocities_by_id is None:
        return None
    return np.array(
        [velocities_by_id[int(atom["id"])] for atom in atoms],
        dtype=np.float64,
    )


def _lammps_masses_for_atoms(
    atoms: list[dict[str, object]],
    masses_by_type: dict[int, float] | None,
) -> np.ndarray | None:
    if masses_by_type is None:
        return None
    active_types = {int(atom["type"]) for atom in atoms}
    missing = sorted(active_types - masses_by_type.keys())
    if missing:
        joined = ", ".join(str(item) for item in missing)
        raise ValueError(f"LAMMPS data Masses section is missing atom types: {joined}.")
    return np.array(
        [masses_by_type[int(atom["type"])] for atom in atoms],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class _TrajectoryHeader:
    version: int
    endian: str
    real_size: int
    index_size: int
    image_size: int
    type_size: int
    tag_size: int
    units: str
    field_mask: int
    fields: tuple[str, ...]
    n_particles: int
    record_bytes: int
    frame_dtype: np.dtype
    frame_count: int


def _read_exact(handle, byte_count: int, label: str) -> bytes:
    data = handle.read(byte_count)
    if len(data) != byte_count:
        raise ValueError(f"Trajectory file is truncated while reading {label}.")
    return data


def _read_u32(handle, endian: str, label: str) -> int:
    return struct.unpack(f"{endian}I", _read_exact(handle, 4, label))[0]


def _read_u64(handle, endian: str, label: str) -> int:
    return struct.unpack(f"{endian}Q", _read_exact(handle, 8, label))[0]


def _trajectory_header_from_handle(handle) -> _TrajectoryHeader:
    magic = _read_exact(handle, len(_TRAJECTORY_MAGIC), "magic")
    if magic != _TRAJECTORY_MAGIC:
        raise ValueError(f"Unsupported BEADS trajectory magic: {magic!r}.")

    version_bytes = _read_exact(handle, 4, "version")
    marker_bytes = _read_exact(handle, 4, "endian marker")
    little_marker = struct.unpack("<I", marker_bytes)[0]
    big_marker = struct.unpack(">I", marker_bytes)[0]
    if little_marker == _TRAJECTORY_ENDIAN_MARKER:
        endian = "<"
    elif big_marker == _TRAJECTORY_ENDIAN_MARKER:
        endian = ">"
    else:
        raise ValueError("Unsupported BEADS trajectory endian marker.")

    version = struct.unpack(f"{endian}I", version_bytes)[0]
    if version != _TRAJECTORY_VERSION:
        raise ValueError(
            f"Unsupported BEADS trajectory version: {version}. "
            f"This reader supports version {_TRAJECTORY_VERSION}."
        )

    real_size = _read_u32(handle, endian, "real size")
    index_size = _read_u32(handle, endian, "index size")
    image_size = _read_u32(handle, endian, "image size")
    type_size = _read_u32(handle, endian, "type size")
    tag_size = _read_u32(handle, endian, "tag size")
    units_length = _read_u64(handle, endian, "units length")
    if units_length > 4096:
        raise ValueError("BEADS trajectory units string is unreasonably long.")
    units = _read_exact(handle, units_length, "units").decode("ascii")
    field_mask = _read_u32(handle, endian, "field mask")
    n_particles = _read_u64(handle, endian, "particle count")
    record_bytes = _read_u64(handle, endian, "record bytes")

    if field_mask & ~_TRAJECTORY_KNOWN_MASK:
        raise ValueError("BEADS trajectory field mask contains unsupported bits.")
    if not (field_mask & _TRAJECTORY_FIELD_TAG):
        raise ValueError("BEADS trajectory field mask must include tag.")
    if n_particles <= 0:
        raise ValueError("BEADS trajectory particle count must be positive.")

    fields = tuple(name for name, bit in _TRAJECTORY_FIELD_BITS if field_mask & bit)
    real_dtype = _trajectory_real_dtype(real_size, endian)
    tag_dtype = _trajectory_unsigned_dtype(tag_size, endian, "tag")
    image_dtype = _trajectory_signed_dtype(image_size, endian, "image")
    type_dtype = _trajectory_signed_dtype(type_size, endian, "type")
    expected_record_bytes = _trajectory_record_bytes(
        field_mask,
        real_size=real_size,
        tag_size=tag_size,
        image_size=image_size,
        type_size=type_size,
    )
    if record_bytes != expected_record_bytes:
        raise ValueError(
            "BEADS trajectory record byte count is inconsistent with its field mask."
        )

    record_dtype_fields: list[tuple[str, np.dtype] | tuple[str, np.dtype, tuple[int, ...]]] = [
        ("tag", tag_dtype)
    ]
    if field_mask & _TRAJECTORY_FIELD_TYPE:
        record_dtype_fields.append(("type", type_dtype))
    if field_mask & _TRAJECTORY_FIELD_POSITION:
        record_dtype_fields.append(("position", real_dtype, (3,)))
    if field_mask & _TRAJECTORY_FIELD_IMAGE:
        record_dtype_fields.append(("image", image_dtype, (3,)))
    if field_mask & _TRAJECTORY_FIELD_VELOCITY:
        record_dtype_fields.append(("velocity", real_dtype, (3,)))
    if field_mask & _TRAJECTORY_FIELD_FORCE:
        record_dtype_fields.append(("force", real_dtype, (3,)))
    record_dtype = np.dtype(record_dtype_fields)
    frame_dtype = np.dtype(
        [
            ("step", np.dtype(f"{endian}u8")),
            ("box_bound", real_dtype, (2, 3)),
            ("records", record_dtype, (n_particles,)),
        ]
    )

    header_bytes = handle.tell()
    frame_bytes = 8 + 6 * real_size + n_particles * record_bytes
    handle.seek(0, os.SEEK_END)
    file_bytes = handle.tell()
    data_bytes = file_bytes - header_bytes
    if data_bytes < 0 or data_bytes % frame_bytes != 0:
        raise ValueError("BEADS trajectory file size is inconsistent with its header.")
    frame_count = data_bytes // frame_bytes
    handle.seek(header_bytes, os.SEEK_SET)
    return _TrajectoryHeader(
        version=version,
        endian=endian,
        real_size=real_size,
        index_size=index_size,
        image_size=image_size,
        type_size=type_size,
        tag_size=tag_size,
        units=units,
        field_mask=field_mask,
        fields=fields,
        n_particles=n_particles,
        record_bytes=record_bytes,
        frame_dtype=frame_dtype,
        frame_count=int(frame_count),
    )


def _trajectory_real_dtype(size: int, endian: str) -> np.dtype:
    if size == 4:
        return np.dtype(f"{endian}f4")
    if size == 8:
        return np.dtype(f"{endian}f8")
    raise ValueError(f"Unsupported BEADS trajectory real size: {size}.")


def _trajectory_signed_dtype(size: int, endian: str, label: str) -> np.dtype:
    if size == 4:
        return np.dtype(f"{endian}i4")
    if size == 8:
        return np.dtype(f"{endian}i8")
    raise ValueError(f"Unsupported BEADS trajectory {label} size: {size}.")


def _trajectory_unsigned_dtype(size: int, endian: str, label: str) -> np.dtype:
    if size == 4:
        return np.dtype(f"{endian}u4")
    if size == 8:
        return np.dtype(f"{endian}u8")
    raise ValueError(f"Unsupported BEADS trajectory {label} size: {size}.")


def _trajectory_record_bytes(
    field_mask: int,
    *,
    real_size: int,
    tag_size: int,
    image_size: int,
    type_size: int,
) -> int:
    byte_count = tag_size
    if field_mask & _TRAJECTORY_FIELD_TYPE:
        byte_count += type_size
    if field_mask & _TRAJECTORY_FIELD_POSITION:
        byte_count += 3 * real_size
    if field_mask & _TRAJECTORY_FIELD_IMAGE:
        byte_count += 3 * image_size
    if field_mask & _TRAJECTORY_FIELD_VELOCITY:
        byte_count += 3 * real_size
    if field_mask & _TRAJECTORY_FIELD_FORCE:
        byte_count += 3 * real_size
    return byte_count


def _checked_lammps_dump_precision(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("write_lammps_dump(...) precision must be an integer, not bool.")
    try:
        precision = int(operator.index(value))
    except TypeError as exc:
        raise TypeError("write_lammps_dump(...) precision must be an integer.") from exc
    if precision < 1 or precision > 17:
        raise ValueError("write_lammps_dump(...) precision must be between 1 and 17.")
    return precision


def _format_lammps_dump_float(value: object, precision: int) -> str:
    return f"{float(value):.{precision}g}"


def _write_lammps_dump_frames(
    source,
    target,
    *,
    header: _TrajectoryHeader,
    sort_by_tag: bool,
    precision: int,
) -> None:
    fields = set(header.fields)
    columns = ["id"]
    if "type" in fields:
        columns.append("type")
    if "position" in fields:
        columns.extend(["x", "y", "z"])
    if "image" in fields:
        columns.extend(["ix", "iy", "iz"])
    if "velocity" in fields:
        columns.extend(["vx", "vy", "vz"])
    if "force" in fields:
        columns.extend(["fx", "fy", "fz"])

    for _ in range(header.frame_count):
        payload = np.fromfile(source, dtype=header.frame_dtype, count=1)
        if payload.shape[0] != 1:
            raise ValueError(
                "Trajectory payload ended before all LAMMPS dump frames were converted."
            )
        frame_record = payload[0]
        records = frame_record["records"]
        box_bound = frame_record["box_bound"]
        if sort_by_tag:
            order = np.argsort(records["tag"], kind="stable")
        else:
            order = np.arange(header.n_particles)

        target.write("ITEM: TIMESTEP\n")
        target.write(f"{int(frame_record['step'])}\n")
        target.write("ITEM: NUMBER OF ATOMS\n")
        target.write(f"{header.n_particles}\n")
        target.write("ITEM: BOX BOUNDS pp pp pp\n")
        for axis in range(3):
            target.write(
                f"{_format_lammps_dump_float(box_bound[0, axis], precision)} "
                f"{_format_lammps_dump_float(box_bound[1, axis], precision)}\n"
            )
        target.write(f"ITEM: ATOMS {' '.join(columns)}\n")
        for row_index in order:
            row = records[row_index]
            values = [str(int(row["tag"]))]
            if "type" in fields:
                values.append(str(int(row["type"])))
            if "position" in fields:
                values.extend(
                    _format_lammps_dump_float(row["position"][axis], precision)
                    for axis in range(3)
                )
            if "image" in fields:
                values.extend(str(int(row["image"][axis])) for axis in range(3))
            if "velocity" in fields:
                values.extend(
                    _format_lammps_dump_float(row["velocity"][axis], precision)
                    for axis in range(3)
                )
            if "force" in fields:
                values.extend(
                    _format_lammps_dump_float(row["force"][axis], precision)
                    for axis in range(3)
                )
            target.write(" ".join(values))
            target.write("\n")
