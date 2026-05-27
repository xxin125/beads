"""Shared public-input validation helpers."""

from __future__ import annotations

import numpy as np

from ._build_config import dtype_for

REAL_DTYPE = dtype_for("real_dtype")
TYPE_DTYPE = dtype_for("type_dtype")
INDEX_DTYPE = dtype_for("index_dtype")
IMAGE_DTYPE = dtype_for("image_dtype")
RUNSTEP_DTYPE = dtype_for("runstep_dtype")


def real_array(
    name: str,
    value: object,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise TypeError(f"{name} must contain real numeric values")
    result = np.array(value, dtype=REAL_DTYPE, copy=True, order="C")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def real_matrix3(
    name: str,
    value: object,
    n_rows: int | None = None,
) -> np.ndarray:
    result = real_array(name, value)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_particles, 3)")
    if n_rows is not None and result.shape[0] != n_rows:
        raise ValueError(f"{name} must have one row per particle")
    return result


def integer_array(
    name: str,
    value: object,
    shape: tuple[int, ...] | None = None,
    *,
    dtype=TYPE_DTYPE,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind in {"b", "O", "S", "U", "f", "c"}:
        raise TypeError(f"{name} must contain integer values")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    bounds = np.iinfo(dtype)
    if array.size:
        minimum = int(np.min(array))
        maximum = int(np.max(array))
        if minimum < bounds.min or maximum > bounds.max:
            raise ValueError(
                f"{name} contains values outside the supported {np.dtype(dtype).name} range"
            )
    result = np.array(value, dtype=dtype, copy=True, order="C")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def positive_int(
    name: str,
    value: object,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def positive_index(
    name: str,
    value: object,
) -> int:
    result = positive_int(name, value)
    if result > np.iinfo(INDEX_DTYPE).max:
        raise ValueError(
            f"{name} contains values outside the supported "
            f"{np.dtype(INDEX_DTYPE).name} range"
        )
    return result


def nonnegative_uint64(
    name: str,
    value: object,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    if result > np.iinfo(RUNSTEP_DTYPE).max:
        raise ValueError(
            f"{name} contains values outside the supported "
            f"{np.dtype(RUNSTEP_DTYPE).name} range"
        )
    return result


def positive_uint64(
    name: str,
    value: object,
) -> int:
    result = nonnegative_uint64(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def nonnegative_real(
    name: str,
    value: object,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def positive_type_id(
    name: str,
    value: object,
) -> int:
    result = positive_int(name, value)
    if result > np.iinfo(TYPE_DTYPE).max:
        raise ValueError(
            f"{name} contains values outside the supported {np.dtype(TYPE_DTYPE).name} range"
        )
    return result
