"""Force-field builder objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ._params import freeze_scalar_params, immutable_mapping
from ._validation import positive_int, positive_type_id

_DEFAULT_BONDED_EXCLUSION_DISTANCE = 2


@dataclass(frozen=True)
class _PairCoeffConfig:
    style: str
    type_i: int
    type_j: int
    params: Mapping[str, Any]


@dataclass(frozen=True)
class _BondCoeffConfig:
    style: str
    type: int
    params: Mapping[str, Any]


@dataclass(frozen=True)
class _AngleCoeffConfig:
    style: str
    type: int
    params: Mapping[str, Any]


@dataclass(frozen=True)
class _DihedralCoeffConfig:
    style: str
    type: int
    params: Mapping[str, Any]


@dataclass(frozen=True)
class _ForceFieldConfig:
    pair_style: str | None
    pair_style_params: Mapping[str, Any]
    pair_coeffs: tuple[_PairCoeffConfig, ...]
    bond_style: str | None
    bond_style_params: Mapping[str, Any]
    bond_coeffs: tuple[_BondCoeffConfig, ...]
    angle_style: str | None
    angle_style_params: Mapping[str, Any]
    angle_coeffs: tuple[_AngleCoeffConfig, ...]
    dihedral_style: str | None
    dihedral_style_params: Mapping[str, Any]
    dihedral_coeffs: tuple[_DihedralCoeffConfig, ...]
    bonded_exclusion_distance: int | None
    bonded_exclusion_policy: str


class ForceField:

    def __init__(self) -> None:
        self._pair_style: tuple[str, dict[str, Any]] | None = None
        self._pair_coeffs: dict[tuple[int, int], tuple[str, dict[str, Any]]] = {}
        self._bond_style: tuple[str, dict[str, Any]] | None = None
        self._bond_coeffs: dict[int, tuple[str, dict[str, Any]]] = {}
        self._angle_style: tuple[str, dict[str, Any]] | None = None
        self._angle_coeffs: dict[int, tuple[str, dict[str, Any]]] = {}
        self._dihedral_style: tuple[str, dict[str, Any]] | None = None
        self._dihedral_coeffs: dict[int, tuple[str, dict[str, Any]]] = {}
        self._bonded_exclusion_distance: int | None = None
        self._bonded_exclusion_policy: str | None = None
        self._frozen = False

    def pair_style(
        self,
        style: str,
        /,
        **params: object,
    ) -> "ForceField":
        self._require_mutable()
        if self._pair_style is not None:
            raise ValueError("ForceField supports one active pair_style.")
        self._pair_style = (
            _style_name("pair_style", style),
            freeze_scalar_params("pair parameter", params, allow_bool=True),
        )
        return self

    def pair_coeff(
        self,
        style: str,
        /,
        *,
        type_i: object,
        type_j: object,
        **params: object,
    ) -> "ForceField":
        self._require_mutable()
        if self._pair_style is None:
            raise ValueError("pair_style must be set before pair_coeff.")
        style_name = _style_name("pair_coeff style", style)
        if style_name != self._pair_style[0]:
            raise ValueError("pair_coeff style must match active pair_style.")
        left = positive_type_id("pair_coeff type_i", type_i)
        right = positive_type_id("pair_coeff type_j", type_j)
        if right < left:
            left, right = right, left
        key = (left, right)
        if key in self._pair_coeffs:
            raise ValueError(f"pair_coeff for types {left} {right} is already set.")
        self._pair_coeffs[key] = (
            style_name,
            freeze_scalar_params("pair parameter", params, allow_bool=True),
        )
        return self

    def bond_style(
        self,
        style: str,
        /,
        **params: object,
    ) -> "ForceField":
        self._require_mutable()
        if self._bond_style is not None:
            raise ValueError("ForceField supports one active bond_style.")
        self._bond_style = (
            _style_name("bond_style", style),
            freeze_scalar_params("bond parameter", params, allow_bool=True),
        )
        return self

    def bond_coeff(
        self,
        style: str,
        /,
        *,
        type: object,
        **params: object,
    ) -> "ForceField":
        self._require_mutable()
        if self._bond_style is None:
            raise ValueError("bond_style must be set before bond_coeff.")
        style_name = _style_name("bond_coeff style", style)
        if style_name != self._bond_style[0]:
            raise ValueError("bond_coeff style must match active bond_style.")
        type_id = positive_type_id("bond_coeff type", type)
        if type_id in self._bond_coeffs:
            raise ValueError(f"bond_coeff for type {type_id} is already set.")
        self._bond_coeffs[type_id] = (
            style_name,
            freeze_scalar_params("bond parameter", params, allow_bool=True),
        )
        return self

    def angle_style(
        self,
        style: str,
        /,
        **params: object,
    ) -> "ForceField":
        self._require_mutable()
        if self._angle_style is not None:
            raise ValueError("ForceField supports one active angle_style.")
        self._angle_style = (
            _style_name("angle_style", style),
            freeze_scalar_params("angle parameter", params, allow_bool=True),
        )
        return self

    def angle_coeff(
        self,
        style: str,
        /,
        *,
        type: object,
        **params: object,
    ) -> "ForceField":
        self._require_mutable()
        if self._angle_style is None:
            raise ValueError("angle_style must be set before angle_coeff.")
        style_name = _style_name("angle_coeff style", style)
        if style_name != self._angle_style[0]:
            raise ValueError("angle_coeff style must match active angle_style.")
        type_id = positive_type_id("angle_coeff type", type)
        if type_id in self._angle_coeffs:
            raise ValueError(f"angle_coeff for type {type_id} is already set.")
        self._angle_coeffs[type_id] = (
            style_name,
            freeze_scalar_params("angle parameter", params, allow_bool=True),
        )
        return self

    def dihedral_style(
        self,
        style: str,
        /,
        **params: object,
    ) -> "ForceField":
        self._require_mutable()
        if self._dihedral_style is not None:
            raise ValueError("ForceField supports one active dihedral_style.")
        self._dihedral_style = (
            _style_name("dihedral_style", style),
            freeze_scalar_params("dihedral parameter", params, allow_bool=True),
        )
        return self

    def dihedral_coeff(
        self,
        style: str,
        /,
        *,
        type: object,
        **params: object,
    ) -> "ForceField":
        self._require_mutable()
        if self._dihedral_style is None:
            raise ValueError("dihedral_style must be set before dihedral_coeff.")
        style_name = _style_name("dihedral_coeff style", style)
        if style_name != self._dihedral_style[0]:
            raise ValueError("dihedral_coeff style must match active dihedral_style.")
        type_id = positive_type_id("dihedral_coeff type", type)
        if type_id in self._dihedral_coeffs:
            raise ValueError(f"dihedral_coeff for type {type_id} is already set.")
        self._dihedral_coeffs[type_id] = (
            style_name,
            freeze_scalar_params("dihedral parameter", params, allow_bool=True),
        )
        return self

    def exclude_bonded(
        self,
        *,
        distance: object = _DEFAULT_BONDED_EXCLUSION_DISTANCE,
    ) -> "ForceField":
        self._require_mutable()
        if self._bonded_exclusion_policy is not None:
            raise ValueError("ForceField bonded exclusions are already configured.")
        result = positive_int("bonded exclusion distance", distance)
        if result not in {1, 2, 3}:
            raise ValueError("bonded exclusion distance must be 1, 2, or 3.")
        self._bonded_exclusion_distance = result
        self._bonded_exclusion_policy = "explicit"
        return self

    def _require_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("ForceField is frozen after binding to a Simulation.")

    def _freeze(
        self,
        system: object | None = None,
    ) -> _ForceFieldConfig:
        if self._pair_style is None:
            pair_style = None
            pair_style_params = {}
        else:
            pair_style, pair_style_params = self._pair_style

        bonded_exclusion_distance = self._bonded_exclusion_distance
        bonded_exclusion_policy = self._bonded_exclusion_policy
        if bonded_exclusion_policy is None:
            if system is not None and system.topology.bonds:
                bonded_exclusion_distance = _DEFAULT_BONDED_EXCLUSION_DISTANCE
                bonded_exclusion_policy = "default"
            else:
                bonded_exclusion_policy = "none"

        config = _ForceFieldConfig(
            pair_style=pair_style,
            pair_style_params=immutable_mapping(pair_style_params),
            pair_coeffs=tuple(
                _PairCoeffConfig(
                    style=style,
                    type_i=left,
                    type_j=right,
                    params=immutable_mapping(params),
                )
                for (left, right), (style, params) in sorted(self._pair_coeffs.items())
            ),
            bond_style=None if self._bond_style is None else self._bond_style[0],
            bond_style_params=immutable_mapping(
                {} if self._bond_style is None else self._bond_style[1]
            ),
            bond_coeffs=tuple(
                _BondCoeffConfig(
                    style=style,
                    type=type_id,
                    params=immutable_mapping(params),
                )
                for type_id, (style, params) in sorted(self._bond_coeffs.items())
            ),
            angle_style=None if self._angle_style is None else self._angle_style[0],
            angle_style_params=immutable_mapping(
                {} if self._angle_style is None else self._angle_style[1]
            ),
            angle_coeffs=tuple(
                _AngleCoeffConfig(
                    style=style,
                    type=type_id,
                    params=immutable_mapping(params),
                )
                for type_id, (style, params) in sorted(self._angle_coeffs.items())
            ),
            dihedral_style=None
            if self._dihedral_style is None
            else self._dihedral_style[0],
            dihedral_style_params=immutable_mapping(
                {} if self._dihedral_style is None else self._dihedral_style[1]
            ),
            dihedral_coeffs=tuple(
                _DihedralCoeffConfig(
                    style=style,
                    type=type_id,
                    params=immutable_mapping(params),
                )
                for type_id, (style, params) in sorted(self._dihedral_coeffs.items())
            ),
            bonded_exclusion_distance=bonded_exclusion_distance,
            bonded_exclusion_policy=bonded_exclusion_policy,
        )
        if system is not None:
            _require_pair_coverage(config, system)
            _require_bond_coverage(config, system)
            _require_angle_coverage(config, system)
            _require_dihedral_coverage(config, system)
            _require_exclusion_topology(config, system)
        self._frozen = True
        return config

    @property
    def pair_style_name(self) -> str | None:
        if self._pair_style is None:
            return None
        return self._pair_style[0]

    @property
    def pair_style_params(self) -> dict[str, Any]:
        if self._pair_style is None:
            return {}
        return dict(self._pair_style[1])

    @property
    def pair_coeff_entries(self) -> tuple[tuple[str, int, int, dict[str, Any]], ...]:
        return tuple(
            (style, left, right, dict(params))
            for (left, right), (style, params) in sorted(self._pair_coeffs.items())
        )


def _style_name(context: str, style: object) -> str:
    if not isinstance(style, str):
        raise TypeError(f"{context} name must be a string")
    if not style:
        raise ValueError(f"{context} name must not be empty")
    return style


def _require_pair_coverage(
    config: _ForceFieldConfig,
    system: object,
) -> None:
    if config.pair_style is None:
        raise ValueError("ForceField requires an active pair_style.")

    active_types = sorted(int(x) for x in np.unique(system.types))
    active_type_set = set(active_types)
    expected_pairs = {
        (type_i, type_j)
        for type_i in active_types
        for type_j in active_types
        if type_i <= type_j
    }
    observed_pairs = {
        (entry.type_i, entry.type_j)
        for entry in config.pair_coeffs
    }

    out_of_range_pairs = sorted(
        pair
        for pair in observed_pairs
        if pair[0] not in active_type_set or pair[1] not in active_type_set
    )
    if out_of_range_pairs:
        raise ValueError(
            "ForceField pair_coeff type ids must be active System types; "
            f"got {out_of_range_pairs}."
        )

    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        raise ValueError(
            "ForceField pair_coeffs must cover all active System type pairs; "
            f"missing {missing_pairs}."
        )


def _require_exclusion_topology(
    config: _ForceFieldConfig,
    system: object,
) -> None:
    if config.bonded_exclusion_distance is None:
        return
    if not system.topology.bonds:
        raise ValueError(
            "ForceField.exclude_bonded(...) requires System topology bonds."
        )


def _require_bond_coverage(
    config: _ForceFieldConfig,
    system: object,
) -> None:
    if config.bond_style is None:
        if system.topology.bonds:
            raise ValueError("System topology bonds require an active bond_style.")
        if config.bond_coeffs:
            raise ValueError("bond_coeff requires an active bond_style.")
        return
    if not system.topology.bonds:
        raise ValueError("bond_style requires System topology bonds.")

    active_types = sorted({type_id for _, _, type_id in system.topology.bonds})
    expected = set(active_types)
    observed = {entry.type for entry in config.bond_coeffs}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(
            "bond_coeffs must cover all active bond topology types; "
            f"missing {missing}."
        )
    extra = sorted(observed - expected)
    if extra:
        raise ValueError(
            "bond_coeff types must be active bond topology types; "
            f"got {extra}."
        )
    for entry in config.bond_coeffs:
        if entry.style != config.bond_style:
            raise ValueError("bond_coeff style must match active bond_style.")


def _require_angle_coverage(
    config: _ForceFieldConfig,
    system: object,
) -> None:
    if config.angle_style is None:
        if system.topology.angles:
            raise ValueError("System topology angles require an active angle_style.")
        if config.angle_coeffs:
            raise ValueError("angle_coeff requires an active angle_style.")
        return
    if not system.topology.angles:
        raise ValueError("angle_style requires System topology angles.")

    active_types = sorted({type_id for _, _, _, type_id in system.topology.angles})
    expected = set(active_types)
    observed = {entry.type for entry in config.angle_coeffs}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(
            "angle_coeffs must cover all active angle topology types; "
            f"missing {missing}."
        )
    extra = sorted(observed - expected)
    if extra:
        raise ValueError(
            "angle_coeff types must be active angle topology types; "
            f"got {extra}."
        )
    for entry in config.angle_coeffs:
        if entry.style != config.angle_style:
            raise ValueError("angle_coeff style must match active angle_style.")


def _require_dihedral_coverage(
    config: _ForceFieldConfig,
    system: object,
) -> None:
    if config.dihedral_style is None:
        if system.topology.dihedrals:
            raise ValueError(
                "System topology dihedrals require an active dihedral_style."
            )
        if config.dihedral_coeffs:
            raise ValueError("dihedral_coeff requires an active dihedral_style.")
        return
    if not system.topology.dihedrals:
        raise ValueError("dihedral_style requires System topology dihedrals.")

    active_types = sorted(
        {type_id for _, _, _, _, type_id in system.topology.dihedrals}
    )
    expected = set(active_types)
    observed = {entry.type for entry in config.dihedral_coeffs}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(
            "dihedral_coeffs must cover all active dihedral topology types; "
            f"missing {missing}."
        )
    extra = sorted(observed - expected)
    if extra:
        raise ValueError(
            "dihedral_coeff types must be active dihedral topology types; "
            f"got {extra}."
        )
    for entry in config.dihedral_coeffs:
        if entry.style != config.dihedral_style:
            raise ValueError("dihedral_coeff style must match active dihedral_style.")
