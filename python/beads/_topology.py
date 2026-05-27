"""System-owned topology identity records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ._validation import positive_index, positive_type_id


@dataclass(frozen=True)
class _TopologyConfig:
    bonds: tuple[tuple[int, int, int], ...]
    angles: tuple[tuple[int, int, int, int], ...]
    dihedrals: tuple[tuple[int, int, int, int, int], ...]


def empty_topology() -> _TopologyConfig:
    return _TopologyConfig(bonds=(), angles=(), dihedrals=())


def topology_config(
    *,
    bonds: object | None = None,
    angles: object | None = None,
    dihedrals: object | None = None,
    active_tags: Iterable[int],
) -> _TopologyConfig:
    active_tag_set = frozenset(int(tag) for tag in active_tags)
    canonical_bonds = _canonical_bonds(bonds, active_tag_set)
    bond_pairs = frozenset((tag_i, tag_j) for tag_i, tag_j, _ in canonical_bonds)
    canonical_angles = _canonical_angles(angles, active_tag_set)
    _require_angle_bond_support(canonical_angles, bond_pairs)
    canonical_dihedrals = _canonical_dihedrals(dihedrals, active_tag_set)
    _require_dihedral_bond_support(canonical_dihedrals, bond_pairs)
    return _TopologyConfig(
        bonds=canonical_bonds,
        angles=canonical_angles,
        dihedrals=canonical_dihedrals,
    )


def validate_topology(config: _TopologyConfig, *, active_tags: Iterable[int]) -> None:
    active_tag_set = frozenset(int(tag) for tag in active_tags)
    canonical_bonds = _canonical_bonds(config.bonds, active_tag_set)
    bond_pairs = frozenset((tag_i, tag_j) for tag_i, tag_j, _ in canonical_bonds)
    canonical_angles = _canonical_angles(config.angles, active_tag_set)
    _require_angle_bond_support(canonical_angles, bond_pairs)
    canonical_dihedrals = _canonical_dihedrals(config.dihedrals, active_tag_set)
    _require_dihedral_bond_support(canonical_dihedrals, bond_pairs)


def _canonical_bonds(
    records: object | None,
    active_tags: frozenset[int],
) -> tuple[tuple[int, int, int], ...]:
    result: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, record in enumerate(_record_iterable("bonds", records)):
        values = _record_tuple("bonds", index, record, width=3)
        tag_i = _topology_tag("bonds", index, 0, values[0], active_tags)
        tag_j = _topology_tag("bonds", index, 1, values[1], active_tags)
        type_id = _topology_type("bonds", index, values[2])
        if tag_i == tag_j:
            raise ValueError("bonds entries must reference distinct tags")
        if tag_j < tag_i:
            tag_i, tag_j = tag_j, tag_i
        canonical = (tag_i, tag_j, type_id)
        duplicate_key = (tag_i, tag_j)
        if duplicate_key in seen:
            raise ValueError("bonds must not contain duplicate records")
        seen.add(duplicate_key)
        result.append(canonical)
    _require_dense_topology_types("bond", (record[2] for record in result))
    return tuple(result)


def _canonical_angles(
    records: object | None,
    active_tags: frozenset[int],
) -> tuple[tuple[int, int, int, int], ...]:
    result: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for index, record in enumerate(_record_iterable("angles", records)):
        values = _record_tuple("angles", index, record, width=4)
        tag_i = _topology_tag("angles", index, 0, values[0], active_tags)
        tag_j = _topology_tag("angles", index, 1, values[1], active_tags)
        tag_k = _topology_tag("angles", index, 2, values[2], active_tags)
        type_id = _topology_type("angles", index, values[3])
        if len({tag_i, tag_j, tag_k}) != 3:
            raise ValueError("angles entries must reference three distinct tags")
        if tag_k < tag_i:
            tag_i, tag_k = tag_k, tag_i
        canonical = (tag_i, tag_j, tag_k, type_id)
        duplicate_key = (tag_i, tag_j, tag_k)
        if duplicate_key in seen:
            raise ValueError("angles must not contain duplicate records")
        seen.add(duplicate_key)
        result.append(canonical)
    _require_dense_topology_types("angle", (record[3] for record in result))
    return tuple(result)


def _canonical_dihedrals(
    records: object | None,
    active_tags: frozenset[int],
) -> tuple[tuple[int, int, int, int, int], ...]:
    result: list[tuple[int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for index, record in enumerate(_record_iterable("dihedrals", records)):
        values = _record_tuple("dihedrals", index, record, width=5)
        tags = (
            _topology_tag("dihedrals", index, 0, values[0], active_tags),
            _topology_tag("dihedrals", index, 1, values[1], active_tags),
            _topology_tag("dihedrals", index, 2, values[2], active_tags),
            _topology_tag("dihedrals", index, 3, values[3], active_tags),
        )
        type_id = _topology_type("dihedrals", index, values[4])
        if len(set(tags)) != 4:
            raise ValueError("dihedrals entries must reference four distinct tags")
        reverse = tuple(reversed(tags))
        canonical_tags = min(tags, reverse)
        canonical = (*canonical_tags, type_id)
        if canonical_tags in seen:
            raise ValueError("dihedrals must not contain duplicate records")
        seen.add(canonical_tags)
        result.append(canonical)
    _require_dense_topology_types("dihedral", (record[4] for record in result))
    return tuple(result)


def _record_iterable(
    name: str,
    records: object | None,
):
    if records is None:
        return ()
    if isinstance(records, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of topology records")
    try:
        return tuple(records)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of topology records") from exc


def _record_tuple(
    name: str,
    index: int,
    record: object,
    *,
    width: int,
) -> tuple[object, ...]:
    if isinstance(record, (str, bytes)):
        raise TypeError(f"{name}[{index}] must be a topology record")
    try:
        values = tuple(record)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name}[{index}] must be a topology record") from exc
    if len(values) != width:
        raise ValueError(f"{name}[{index}] must contain exactly {width} values")
    return values


def _topology_tag(
    name: str,
    index: int,
    offset: int,
    value: object,
    active_tags: frozenset[int],
) -> int:
    tag = positive_index(f"{name}[{index}][{offset}]", value)
    if tag not in active_tags:
        raise ValueError(f"{name}[{index}] references unknown System tag {tag}")
    return tag


def _topology_type(
    name: str,
    index: int,
    value: object,
) -> int:
    return positive_type_id(f"{name}[{index}] type", value)


def _require_dense_topology_types(
    label: str,
    type_ids: Iterable[int],
) -> None:
    unique = sorted(set(type_ids))
    if not unique:
        return
    for expected, observed in enumerate(unique, start=1):
        if observed != expected:
            raise ValueError(
                f"{label} topology types must be dense one-based ids without gaps"
            )


def _bond_pair(tag_i: int, tag_j: int) -> tuple[int, int]:
    return (tag_i, tag_j) if tag_i < tag_j else (tag_j, tag_i)


def _require_angle_bond_support(
    angles: Iterable[tuple[int, int, int, int]],
    bond_pairs: frozenset[tuple[int, int]],
) -> None:
    for tag_i, tag_j, tag_k, _ in angles:
        for first, second in ((tag_i, tag_j), (tag_j, tag_k)):
            if _bond_pair(first, second) not in bond_pairs:
                raise ValueError(
                    "angles require supporting topology bonds for each edge"
                )


def _require_dihedral_bond_support(
    dihedrals: Iterable[tuple[int, int, int, int, int]],
    bond_pairs: frozenset[tuple[int, int]],
) -> None:
    for tag_i, tag_j, tag_k, tag_l, _ in dihedrals:
        for first, second in ((tag_i, tag_j), (tag_j, tag_k), (tag_k, tag_l)):
            if _bond_pair(first, second) not in bond_pairs:
                raise ValueError(
                    "dihedrals require supporting topology bonds for each edge"
                )
