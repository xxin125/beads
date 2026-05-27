"""Shared scalar parameter helpers."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ._build_config import dtype_for

REAL_DTYPE = dtype_for("real_dtype")
_INT64_INFO = np.iinfo(np.int64)


def freeze_scalar_params(
    context: str,
    params: Mapping[str, object],
    *,
    allow_bool: bool,
) -> dict[str, Any]:
    return {
        _param_name(context, key): _scalar_param_value(
            context,
            key,
            value,
            allow_bool=allow_bool,
        )
        for key, value in params.items()
    }


def immutable_mapping(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(params))


def _param_name(
    context: str,
    name: object,
) -> str:
    if not isinstance(name, str) or not name:
        raise TypeError(f"{context} names must be non-empty strings.")
    return name


def _scalar_param_value(
    context: str,
    name: str,
    value: object,
    *,
    allow_bool: bool,
) -> Any:
    if isinstance(value, (bool, np.bool_)):
        if allow_bool:
            return bool(value)
        raise TypeError(f"{context} {name} must not be bool.")
    if isinstance(value, (int, np.integer)):
        result = int(value)
        if result < _INT64_INFO.min or result > _INT64_INFO.max:
            raise ValueError(
                f"{context} {name} contains values outside the supported int64 range."
            )
        return result
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{context} {name} must be finite.")
        with np.errstate(over="ignore", under="ignore"):
            native_value = np.asarray(result, dtype=REAL_DTYPE).item()
        if not np.isfinite(float(native_value)):
            raise ValueError(
                f"{context} {name} must be representable as {REAL_DTYPE.name}."
            )
        return result
    if isinstance(value, str):
        return value

    allowed_types = (
        "bool, int, float, or string"
        if allow_bool
        else "int, float, or string"
    )
    raise TypeError(f"{context} {name} must be {allowed_types}.")
