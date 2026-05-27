from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._state_snapshot import read_system_from_state
from ._topology import (
    _TopologyConfig,
    empty_topology,
    topology_config,
    validate_topology,
)
from ._validation import (
    IMAGE_DTYPE,
    INDEX_DTYPE,
    REAL_DTYPE,
    TYPE_DTYPE,
    integer_array,
    real_array,
    real_matrix3,
)


@dataclass(frozen=True, eq=False)
class System:

    positions: np.ndarray
    box_bound: np.ndarray
    types: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    tags: np.ndarray
    molecule_ids: np.ndarray
    images: np.ndarray
    units: str
    topology: _TopologyConfig

    def __init__(
        self,
        *,
        positions: object,
        types: object,
        box_bound: object | None = None,
        velocities: object | None = None,
        masses: object | None = None,
        tags: object | None = None,
        molecule_ids: object | None = None,
        images: object | None = None,
        units: str | None = None,
    ) -> None:
        if units is None:
            raise TypeError('System requires explicit units="reduced" or units="nm_kjmol".')
        if not isinstance(units, str):
            raise TypeError("units must be a string")
        if units not in {"reduced", "nm_kjmol"}:
            raise ValueError('units must be "reduced" or "nm_kjmol"')

        positions_array = _positions_array(positions)
        n_particles = int(positions_array.shape[0])

        box_bound_array = _box_bound_array(box_bound)
        _require_positions_inside_box(positions_array, box_bound_array)

        types_array = _types_array(types, n_particles)

        velocities_array = (
            np.zeros_like(positions_array)
            if velocities is None
            else real_matrix3("velocities", velocities, n_particles)
        )

        masses_array = (
            np.ones(n_particles, dtype=REAL_DTYPE)
            if masses is None
            else real_array("masses", masses, (n_particles,))
        )
        if np.any(masses_array <= 0.0):
            raise ValueError("masses must be finite and positive")

        tags_array = (
            np.arange(1, n_particles + 1, dtype=INDEX_DTYPE)
            if tags is None
            else _tag_permutation("tags", tags, n_particles)
        )

        molecule_ids_array = (
            np.ones(n_particles, dtype=INDEX_DTYPE)
            if molecule_ids is None
            else _dense_one_based_ids("molecule_ids", molecule_ids, n_particles)
        )

        input_images_array = (
            np.zeros((n_particles, 3), dtype=IMAGE_DTYPE)
            if images is None
            else integer_array("images", images, (n_particles, 3), dtype=IMAGE_DTYPE)
        )

        object.__setattr__(self, "positions", _owned_array(positions_array, REAL_DTYPE))
        object.__setattr__(self, "box_bound", _owned_array(box_bound_array, REAL_DTYPE))
        object.__setattr__(self, "types", _owned_array(types_array, TYPE_DTYPE))
        object.__setattr__(self, "velocities", _owned_array(velocities_array, REAL_DTYPE))
        object.__setattr__(self, "masses", _owned_array(masses_array, REAL_DTYPE))
        object.__setattr__(self, "tags", _owned_array(tags_array, INDEX_DTYPE))
        object.__setattr__(self, "molecule_ids", _owned_array(molecule_ids_array, INDEX_DTYPE))
        object.__setattr__(self, "images", _owned_array(input_images_array, IMAGE_DTYPE))
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "topology", empty_topology())
        object.__setattr__(self, "_frozen", False)

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    @classmethod
    def from_state(cls, path: object) -> "System":
        return read_system_from_state(path, system_cls=cls)

    def _require_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("System is frozen after binding to a Simulation.")

    def set_topology(
        self,
        *,
        bonds: object | None = None,
        angles: object | None = None,
        dihedrals: object | None = None,
    ) -> "System":
        self._require_mutable()
        object.__setattr__(
            self,
            "topology",
            topology_config(
                bonds=bonds,
                angles=angles,
                dihedrals=dihedrals,
                active_tags=self.tags.tolist(),
            ),
        )
        return self

    def _freeze(self) -> "System":
        _validate_owned_state(self)
        for name, dtype in (
            ("positions", REAL_DTYPE),
            ("box_bound", REAL_DTYPE),
            ("types", TYPE_DTYPE),
            ("velocities", REAL_DTYPE),
            ("masses", REAL_DTYPE),
            ("tags", INDEX_DTYPE),
            ("molecule_ids", INDEX_DTYPE),
            ("images", IMAGE_DTYPE),
        ):
            object.__setattr__(self, name, _frozen_array(getattr(self, name), dtype))
        object.__setattr__(
            self,
            "topology",
            topology_config(
                bonds=self.topology.bonds,
                angles=self.topology.angles,
                dihedrals=self.topology.dihedrals,
                active_tags=self.tags.tolist(),
            ),
        )
        object.__setattr__(self, "_frozen", True)
        return self


def _box_bound_array(box_bound: object | None) -> np.ndarray:
    if box_bound is None:
        raise TypeError("System requires box_bound=")
    bounds = real_array("box_bound", box_bound, (2, 3))
    lengths = bounds[1] - bounds[0]
    if np.any(lengths <= 0.0):
        raise ValueError("box_bound upper row must be greater than lower row")
    return bounds


def _positions_array(value: object) -> np.ndarray:
    array = real_array("positions", value)
    if array.ndim == 1 and array.shape[0] == 0:
        raise ValueError("positions must contain at least one particle")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("positions must have shape (n_particles, 3)")
    if array.shape[0] <= 0:
        raise ValueError("positions must contain at least one particle")
    return array


def _types_array(
    value: object,
    n_particles: int,
) -> np.ndarray:
    array = integer_array("types", value, (n_particles,), dtype=TYPE_DTYPE)
    if np.any(array < 1):
        raise ValueError("types must be one-based positive integers")
    unique = sorted(int(x) for x in np.unique(array))
    expected = list(range(1, unique[-1] + 1))
    if unique != expected:
        raise ValueError("types must be dense one-based ids without gaps")
    return array


def _tag_permutation(
    name: str,
    value: object,
    n_particles: int,
) -> np.ndarray:
    array = integer_array(name, value, (n_particles,), dtype=INDEX_DTYPE)
    expected = set(range(1, n_particles + 1))
    observed = set(int(x) for x in array.tolist())
    if observed != expected:
        raise ValueError(f"{name} must be a permutation of 1..N")
    if len(set(int(x) for x in array.tolist())) != n_particles:
        raise ValueError(f"{name} must be unique")
    return array


def _dense_one_based_ids(
    name: str,
    value: object,
    n_particles: int,
) -> np.ndarray:
    array = integer_array(name, value, (n_particles,), dtype=INDEX_DTYPE)
    if np.any(array < 1):
        raise ValueError(f"{name} must be one-based positive integers")
    unique = sorted(int(x) for x in np.unique(array))
    if unique[-1] > n_particles or unique != list(range(1, unique[-1] + 1)):
        raise ValueError(f"{name} must form a contiguous 1-based set starting at 1")
    return array


def _require_positions_inside_box(
    positions: np.ndarray,
    box_bound: np.ndarray,
) -> None:
    if np.any(positions < box_bound[0]) or np.any(positions >= box_bound[1]):
        raise ValueError("positions must be inside box_bound")


def _owned_array(
    value: np.ndarray,
    dtype,
) -> np.ndarray:
    return np.array(value, dtype=dtype, copy=True, order="C")


def _frozen_array(
    value: np.ndarray,
    dtype,
) -> np.ndarray:
    result = _owned_array(value, dtype)
    result.setflags(write=False)
    return result


def _validate_owned_state(system: System) -> None:
    if system.units not in {"reduced", "nm_kjmol"}:
        raise ValueError('units must be "reduced" or "nm_kjmol"')

    _require_owned_array(
        "positions",
        system.positions,
        dtype=REAL_DTYPE,
        shape=(system.n_particles, 3),
    )
    if system.n_particles <= 0:
        raise ValueError("positions must contain at least one particle")
    _require_finite("positions", system.positions)
    _require_owned_array("box_bound", system.box_bound, dtype=REAL_DTYPE, shape=(2, 3))
    _require_finite("box_bound", system.box_bound)
    lengths = system.box_bound[1] - system.box_bound[0]
    if np.any(lengths <= 0.0):
        raise ValueError("box_bound upper row must be greater than lower row")
    _require_positions_inside_box(system.positions, system.box_bound)

    _require_owned_array(
        "velocities",
        system.velocities,
        dtype=REAL_DTYPE,
        shape=(system.n_particles, 3),
    )
    _require_finite("velocities", system.velocities)

    _require_owned_array("masses", system.masses, dtype=REAL_DTYPE, shape=(system.n_particles,))
    _require_finite("masses", system.masses)
    if np.any(system.masses <= 0.0):
        raise ValueError("masses must be finite and positive")

    _require_owned_array("types", system.types, dtype=TYPE_DTYPE, shape=(system.n_particles,))
    _require_dense_types(system.types)
    _require_owned_array("tags", system.tags, dtype=INDEX_DTYPE, shape=(system.n_particles,))
    _require_tag_permutation("tags", system.tags, system.n_particles)
    _require_owned_array(
        "molecule_ids",
        system.molecule_ids,
        dtype=INDEX_DTYPE,
        shape=(system.n_particles,),
    )
    _require_dense_one_based_array("molecule_ids", system.molecule_ids, system.n_particles)
    _require_owned_array(
        "images",
        system.images,
        dtype=IMAGE_DTYPE,
        shape=(system.n_particles, 3),
    )
    validate_topology(system.topology, active_tags=system.tags.tolist())


def _require_owned_array(
    name: str,
    array: np.ndarray,
    *,
    dtype,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype).name}")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")


def _require_dense_types(array: np.ndarray) -> None:
    if array.size == 0:
        raise ValueError("types must be dense one-based ids without gaps")
    if np.any(array < 1):
        raise ValueError("types must be one-based positive integers")
    unique = sorted(int(x) for x in np.unique(array))
    expected = list(range(1, unique[-1] + 1))
    if unique != expected:
        raise ValueError("types must be dense one-based ids without gaps")


def _require_tag_permutation(
    name: str,
    array: np.ndarray,
    n_particles: int,
) -> None:
    expected = set(range(1, n_particles + 1))
    observed = set(int(x) for x in array.tolist())
    if observed != expected:
        raise ValueError(f"{name} must be a permutation of 1..N")
    if len(set(int(x) for x in array.tolist())) != n_particles:
        raise ValueError(f"{name} must be unique")


def _require_dense_one_based_array(
    name: str,
    array: np.ndarray,
    n_particles: int,
) -> None:
    if array.size == 0:
        raise ValueError(f"{name} must form a contiguous 1-based set starting at 1")
    if np.any(array < 1):
        raise ValueError(f"{name} must be one-based positive integers")
    unique = sorted(int(x) for x in np.unique(array))
    if unique[-1] > n_particles or unique != list(range(1, unique[-1] + 1)):
        raise ValueError(f"{name} must form a contiguous 1-based set starting at 1")
