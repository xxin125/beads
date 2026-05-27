"""Output configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os

from ._validation import positive_uint64


_TRAJECTORY_CANONICAL_FIELDS = (
    "tag",
    "type",
    "position",
    "image",
    "velocity",
    "force",
)
_TRAJECTORY_DEFAULT_FIELDS = ("tag", "type", "position", "image")


@dataclass(frozen=True)
class _ThermoConfig:
    every: int
    prefix: str


@dataclass(frozen=True)
class _FinalStateConfig:
    prefix: str


@dataclass(frozen=True)
class _TrajectoryConfig:
    every: int
    prefix: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class _LogConfig:
    echo: str
    prefix: str | None = None


def output_prefix(label: str, prefix: object) -> str:
    try:
        value = os.fspath(prefix)
    except TypeError as error:
        raise TypeError(
            f"{label} prefix must be a string or path-like object"
        ) from error
    if not isinstance(value, str):
        raise TypeError(f"{label} prefix must be a string or path-like object")
    if not value:
        raise ValueError(f"{label} prefix must not be empty")
    return value


def trajectory_fields(fields: object | None) -> tuple[str, ...]:
    if fields is None:
        return _TRAJECTORY_DEFAULT_FIELDS
    if isinstance(fields, (str, bytes)):
        raise TypeError("trajectory fields must be an iterable of strings")

    try:
        values = tuple(fields)
    except TypeError as error:
        raise TypeError("trajectory fields must be an iterable of strings") from error
    if not values:
        raise ValueError("trajectory fields must not be empty")
    for field in values:
        if not isinstance(field, str):
            raise TypeError("trajectory fields must contain only strings")
        if field not in _TRAJECTORY_CANONICAL_FIELDS:
            raise ValueError(f"trajectory field {field!r} is not supported")
    if values[0] != "tag":
        raise ValueError("trajectory fields must include tag as the first field")
    if len(set(values)) != len(values):
        raise ValueError("trajectory fields must not contain duplicates")

    canonical_positions = {
        field: index
        for index, field in enumerate(_TRAJECTORY_CANONICAL_FIELDS)
    }
    observed = [canonical_positions[field] for field in values]
    if observed != sorted(observed):
        raise ValueError(
            "trajectory fields must follow canonical order: "
            "tag,type,position,image,velocity,force"
        )
    return values


def trajectory_config(
    *,
    every: object,
    prefix: object,
    fields: object | None,
) -> _TrajectoryConfig:
    return _TrajectoryConfig(
        every=positive_uint64("trajectory every", every),
        prefix=output_prefix("trajectory", prefix),
        fields=trajectory_fields(fields),
    )


def log_config(
    *,
    echo: object = "screen",
    prefix: object | None = None,
) -> _LogConfig:
    if not isinstance(echo, str):
        raise TypeError("logging echo must be a string")
    if echo not in {"screen", "log", "both", "none"}:
        raise ValueError(
            "logging echo must be one of 'screen', 'log', 'both', or 'none'"
        )
    if echo in {"log", "both"}:
        if prefix is None:
            raise ValueError(f"logging echo={echo!r} requires a prefix")
        return _LogConfig(
            echo=echo,
            prefix=output_prefix("log", prefix),
        )
    if prefix is not None:
        raise ValueError(f"logging echo={echo!r} does not accept a prefix")
    return _LogConfig(echo=echo)
