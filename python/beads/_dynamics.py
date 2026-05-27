"""Dynamics configuration builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._params import freeze_scalar_params, immutable_mapping


@dataclass(frozen=True)
class _DynamicsConfig:
    style: str
    params: Mapping[str, Any]
    thermostat: "_ThermostatConfig | None" = None


@dataclass(frozen=True)
class _ThermostatConfig:
    style: str
    params: Mapping[str, Any]


class Dynamics:

    def __init__(
        self,
        style: str,
        /,
        **params: object,
    ) -> None:
        self._style = _style_name(style)
        self._params = freeze_scalar_params(
            "Dynamics parameter",
            params,
            allow_bool=False,
        )
        _validate_dynamics_contract(self._style, self._params)
        self._thermostat: _ThermostatConfig | None = None
        self._frozen = False

    @property
    def style(self) -> str:
        return self._style

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def set_thermostat(
        self,
        style: str,
        /,
        **params: object,
    ) -> "Dynamics":
        self._require_mutable()
        thermostat_style = _thermostat_style_name(style)
        thermostat_params = freeze_scalar_params(
            "Thermostat parameter",
            params,
            allow_bool=False,
        )
        _validate_thermostat_contract(
            self._style,
            thermostat_style,
            thermostat_params,
        )
        self._thermostat = _ThermostatConfig(
            style=thermostat_style,
            params=immutable_mapping(thermostat_params),
        )
        return self

    def _require_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("Dynamics is frozen after binding to a Simulation.")

    def _freeze(self) -> _DynamicsConfig:
        self._frozen = True
        return _DynamicsConfig(
            style=self._style,
            params=immutable_mapping(self._params),
            thermostat=self._thermostat,
        )


def _style_name(style: object) -> str:
    if not isinstance(style, str):
        raise TypeError("Dynamics style must be a string.")
    if not style:
        raise ValueError("Dynamics style must not be empty.")
    if style not in {"none", "velocity_verlet"}:
        raise ValueError(f"Unknown Dynamics style {style!r}.")
    return style


def _validate_dynamics_contract(
    style: str,
    params: Mapping[str, Any],
) -> None:
    if style == "none":
        if params:
            raise ValueError('Dynamics("none") does not accept parameters.')
        return

    if style == "velocity_verlet":
        if set(params) != {"dt"}:
            raise ValueError('Dynamics("velocity_verlet") requires exactly parameter "dt".')
        dt = params["dt"]
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError("Dynamics dt must be positive.")
        return

    raise ValueError(f"Unknown Dynamics style {style!r}.")


def _thermostat_style_name(style: object) -> str:
    if not isinstance(style, str):
        raise TypeError("Thermostat style must be a string.")
    if not style:
        raise ValueError("Thermostat style must not be empty.")
    if style != "berendsen":
        raise ValueError(f"Unknown Thermostat style {style!r}.")
    return style


def _require_real(
    context: str,
    params: Mapping[str, Any],
    key: str,
) -> float:
    value = params[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context} {key} must be a real number.")
    return float(value)


def _validate_thermostat_contract(
    dynamics_style: str,
    thermostat_style: str,
    params: Mapping[str, Any],
) -> None:
    if dynamics_style != "velocity_verlet":
        raise ValueError("Thermostat requires Dynamics(\"velocity_verlet\").")
    if thermostat_style != "berendsen":
        raise ValueError(f"Unknown Thermostat style {thermostat_style!r}.")
    if set(params) != {"temperature", "tau"}:
        raise ValueError(
            'Thermostat("berendsen") requires exactly parameters '
            '"temperature" and "tau".'
        )
    temperature = _require_real("Thermostat parameter", params, "temperature")
    tau = _require_real("Thermostat parameter", params, "tau")
    if temperature < 0.0:
        raise ValueError("Thermostat temperature must be non-negative.")
    if tau <= 0.0:
        raise ValueError("Thermostat tau must be positive.")
