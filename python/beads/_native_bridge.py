"""Bridge to the BEADS native extension."""

from __future__ import annotations

from importlib import import_module
from typing import Mapping

from ._build_config import ensure_native_build_info_compatible


def execute_simulation(spec: Mapping[str, object]) -> None:
    native = _load_native_extension()
    ensure_native_build_info_compatible(native)
    execute = getattr(native, "execute_simulation", None)
    if execute is None:
        raise RuntimeError(
            "BEADS native extension does not provide execute_simulation."
        )
    execute(spec)


def _load_native_extension() -> object:
    try:
        return import_module("beads._native")
    except ModuleNotFoundError as exc:
        if exc.name != "beads._native":
            raise
        raise RuntimeError(
            "BEADS native extension is not built. Build/install BEADS with native "
            "support before executing a simulation."
        ) from exc
