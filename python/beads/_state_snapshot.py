"""Python reader for BEADS final-state snapshot files."""

from __future__ import annotations

import os
import struct
import sys
from typing import BinaryIO, TypeVar

from ._validation import IMAGE_DTYPE, INDEX_DTYPE, TYPE_DTYPE

_MAGIC = b"BEADS_SNAPSHOT"
_VERSION = 1
_ENDIAN_MARKER = 0x01020304
_SUPPORTED_UNITS = frozenset({"reduced", "nm_kjmol"})

_SystemT = TypeVar("_SystemT")


def read_system_from_state(path: object, *, system_cls: type[_SystemT]) -> _SystemT:
    path_string = _path_string(path)
    try:
        with open(path_string, "rb") as stream:
            payload = _read_snapshot_payload(stream)
    except OSError as exc:
        raise OSError(f"failed to open state snapshot file: {path_string}") from exc
    return system_cls(**payload)


def _path_string(path: object) -> str:
    try:
        value = os.fspath(path)
    except TypeError as exc:
        raise TypeError(
            "state snapshot path must be a string or path-like object"
        ) from exc
    if not isinstance(value, str):
        raise TypeError("state snapshot path must be a string or path-like object")
    if not value:
        raise ValueError("state snapshot path must not be empty")
    return value


def _read_snapshot_payload(stream: BinaryIO) -> dict[str, object]:
    magic = _read_exact(stream, len(_MAGIC), "magic")
    if magic != _MAGIC:
        raise ValueError("state snapshot magic is not supported")

    version = _read_u32(stream, "version")
    if version != _VERSION:
        raise ValueError(f"state snapshot version {version} is not supported")

    real_size = _read_u32(stream, "real size")
    index_size = _read_u32(stream, "index size")
    image_size = _read_u32(stream, "image size")
    type_size = _read_u32(stream, "type size")
    tag_size = _read_u32(stream, "tag size")
    endian_marker = _read_u32(stream, "endian marker")
    if endian_marker != _ENDIAN_MARKER:
        raise ValueError("state snapshot endian marker is not supported")

    real_reader = _real_array_reader(real_size)
    _require_size(index_size, INDEX_DTYPE.itemsize, "index")
    _require_size(image_size, IMAGE_DTYPE.itemsize, "image")
    _require_size(type_size, TYPE_DTYPE.itemsize, "type")
    _require_size(tag_size, INDEX_DTYPE.itemsize, "tag")
    index_reader = _integer_array_reader(INDEX_DTYPE, "index")
    type_reader = _integer_array_reader(TYPE_DTYPE, "type")
    image_reader = _integer_array_reader(IMAGE_DTYPE, "image")

    units_length = _read_u64(stream, "units length")
    units_bytes = _read_exact(
        stream,
        _checked_count(units_length, "units"),
        "units",
    )
    try:
        units = units_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("state snapshot units are not ASCII") from exc
    if units not in _SUPPORTED_UNITS:
        raise ValueError(f"state snapshot units {units!r} are not supported")

    # Step is part of the file compatibility contract but not part of System.
    _read_u64(stream, "step")
    n_particles = _checked_count(_read_u64(stream, "particle count"), "particle count")
    if n_particles <= 0:
        raise ValueError("state snapshot particle count must be positive")

    lower = real_reader(stream, 3, "box lower")
    upper = real_reader(stream, 3, "box upper")
    tags = index_reader(stream, n_particles, "tag")
    types = type_reader(stream, n_particles, "type")
    molecule_ids = index_reader(stream, n_particles, "molecule_id")
    masses = real_reader(stream, n_particles, "mass")
    position_x = real_reader(stream, n_particles, "position_x")
    position_y = real_reader(stream, n_particles, "position_y")
    position_z = real_reader(stream, n_particles, "position_z")
    velocity_x = real_reader(stream, n_particles, "velocity_x")
    velocity_y = real_reader(stream, n_particles, "velocity_y")
    velocity_z = real_reader(stream, n_particles, "velocity_z")
    image_x = image_reader(stream, n_particles, "image_x")
    image_y = image_reader(stream, n_particles, "image_y")
    image_z = image_reader(stream, n_particles, "image_z")

    trailing = stream.read(1)
    if trailing:
        raise ValueError("state snapshot contains trailing bytes")

    return {
        "units": units,
        "box_bound": [lower, upper],
        "positions": _matrix3(position_x, position_y, position_z),
        "velocities": _matrix3(velocity_x, velocity_y, velocity_z),
        "masses": masses,
        "types": types,
        "tags": tags,
        "molecule_ids": molecule_ids,
        "images": _matrix3(image_x, image_y, image_z),
    }


def _read_exact(stream: BinaryIO, byte_count: int, label: str) -> bytes:
    data = stream.read(byte_count)
    if len(data) != byte_count:
        raise ValueError(f"state snapshot is truncated while reading {label}")
    return data


def _read_u32(stream: BinaryIO, label: str) -> int:
    return struct.unpack("=I", _read_exact(stream, 4, label))[0]


def _read_u64(stream: BinaryIO, label: str) -> int:
    return struct.unpack("=Q", _read_exact(stream, 8, label))[0]


def _require_size(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"state snapshot {label} size {actual} is not supported; "
            f"expected {expected}"
        )


def _checked_count(value: int, label: str) -> int:
    if value > sys.maxsize:
        raise ValueError(f"state snapshot {label} exceeds supported host size")
    return int(value)


def _real_array_reader(real_size: int):
    if real_size == 4:
        return _read_float32_array
    if real_size == 8:
        return _read_float64_array
    raise ValueError(
        f"state snapshot real size {real_size} is not supported; expected 4 or 8"
    )


def _read_float32_array(stream: BinaryIO, count: int, label: str) -> list[float]:
    return _read_real_array(stream, count, label, "f", 4)


def _read_float64_array(stream: BinaryIO, count: int, label: str) -> list[float]:
    return _read_real_array(stream, count, label, "d", 8)


def _read_real_array(
    stream: BinaryIO,
    count: int,
    label: str,
    format_char: str,
    item_size: int,
) -> list[float]:
    return list(
        struct.unpack(
            f"={count}{format_char}",
            _read_exact(stream, count * item_size, label),
        )
    )


def _integer_array_reader(dtype, label: str):
    format_char = _integer_struct_format(dtype, label)
    item_size = dtype.itemsize

    def read_array(stream: BinaryIO, count: int, field_label: str) -> list[int]:
        return list(
            struct.unpack(
                f"={count}{format_char}",
                _read_exact(stream, count * item_size, field_label),
            )
        )

    return read_array


def _integer_struct_format(dtype, label: str) -> str:
    if dtype.kind == "u" and dtype.itemsize == 4:
        return "I"
    if dtype.kind == "u" and dtype.itemsize == 8:
        return "Q"
    if dtype.kind == "i" and dtype.itemsize == 4:
        return "i"
    if dtype.kind == "i" and dtype.itemsize == 8:
        return "q"
    raise RuntimeError(
        f"state snapshot reader does not support {label} dtype {dtype}"
    )


def _matrix3(
    x_values: list[float] | list[int],
    y_values: list[float] | list[int],
    z_values: list[float] | list[int],
) -> list[list[float]] | list[list[int]]:
    return [
        [x, y, z]
        for x, y, z in zip(x_values, y_values, z_values, strict=True)
    ]
