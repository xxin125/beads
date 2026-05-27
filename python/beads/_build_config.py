"""Build-time dtype configuration shared by Python and native code."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType

import numpy as np

_SOURCE_DEFAULT_BUILD_INFO = {
    "real_precision": "single",
    "real_dtype": "float32",
    "type_dtype": "int32",
    "index_dtype": "uint32",
    "image_dtype": "int32",
    "runstep_dtype": "uint64",
}

_DTYPES = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "uint64": np.dtype(np.uint64),
}


def build_info() -> Mapping[str, str]:
    return MappingProxyType(dict(_BUILD_INFO))


def ensure_native_build_info_compatible(native: object) -> None:
    native_info = _build_info_from_native(native)
    if native_info != _BUILD_INFO:
        raise RuntimeError(
            "BEADS Python/native dtype configuration mismatch: "
            f"Python uses {_BUILD_INFO}, native reports {native_info}."
        )


def dtype_for(name: str) -> np.dtype:
    return _DTYPES[_BUILD_INFO[name]]


def _initial_build_info() -> dict[str, str]:
    try:
        native = import_module("beads._native")
    except ModuleNotFoundError as exc:
        if exc.name != "beads._native":
            raise
        return _normalize_build_info(_SOURCE_DEFAULT_BUILD_INFO)
    return _build_info_from_native(native)


def _build_info_from_native(native: object) -> dict[str, str]:
    build_info_func = getattr(native, "build_info", None)
    if build_info_func is None:
        raise RuntimeError("BEADS native extension does not provide build_info.")
    return _normalize_build_info(build_info_func())


def _normalize_build_info(info: object) -> dict[str, str]:
    if not isinstance(info, Mapping):
        raise RuntimeError("BEADS native build_info must be a mapping.")

    result = {
        key: _required_string(info, key)
        for key in _SOURCE_DEFAULT_BUILD_INFO
    }

    if result["real_precision"] not in {"single", "double"}:
        raise RuntimeError('BEADS real_precision must be "single" or "double".')
    expected_real_dtype = (
        "float32"
        if result["real_precision"] == "single"
        else "float64"
    )
    if result["real_dtype"] != expected_real_dtype:
        raise RuntimeError(
            "BEADS real_dtype must match real_precision "
            f"({result['real_precision']} -> {expected_real_dtype})."
        )

    for key in (
        "real_dtype",
        "type_dtype",
        "index_dtype",
        "image_dtype",
        "runstep_dtype",
    ):
        if result[key] not in _DTYPES:
            raise RuntimeError(f"BEADS {key} is unsupported: {result[key]!r}.")

    return result


def _required_string(
    info: Mapping[str, object],
    key: str,
) -> str:
    value = info.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"BEADS native build_info requires string field {key!r}.")
    return value


_BUILD_INFO = _initial_build_info()
