"""Neighbor-list configuration objects."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import nonnegative_real, positive_index, positive_uint64


@dataclass(frozen=True)
class _NeighborConfig:
    cutoff_buffer: float = 0.3
    rebuild_check_every: int = 1
    sort_every_rebuild: int = 10
    max_neighbors: int = 512

    def __post_init__(self) -> None:
        cutoff_buffer = nonnegative_real(
            "neighbor cutoff_buffer",
            self.cutoff_buffer,
        )
        rebuild_check_every = positive_uint64(
            "neighbor rebuild_check_every",
            self.rebuild_check_every,
        )
        sort_every_rebuild = positive_uint64(
            "neighbor sort_every_rebuild",
            self.sort_every_rebuild,
        )
        max_neighbors = positive_index(
            "neighbor max_neighbors",
            self.max_neighbors,
        )

        if cutoff_buffer == 0.0 and rebuild_check_every != 1:
            raise ValueError(
                "neighbor cutoff_buffer=0 requires rebuild_check_every=1"
            )

        object.__setattr__(self, "cutoff_buffer", cutoff_buffer)
        object.__setattr__(self, "rebuild_check_every", rebuild_check_every)
        object.__setattr__(self, "sort_every_rebuild", sort_every_rebuild)
        object.__setattr__(self, "max_neighbors", max_neighbors)
