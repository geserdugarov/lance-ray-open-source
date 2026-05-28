"""Partition-id → actor-index routing for the sharded IVF cache."""

from __future__ import annotations


class PartitionRouter:
    """Assigns IVF partition ids to actors via ``partition_id % num_actors``.

    The v1 broadcast-to-all-actors search path only ever asks the router
    "which partition ids does actor *i* own" at actor-startup time; it
    never asks "which actor owns this id" on the per-query hot path.
    The companion ``owner`` / ``owners_for`` accessors described in the
    plan's §4.1 are intentionally deferred to Phase 2 along with the
    centroid-aware coordinator routing variant (§4.5).
    """

    def __init__(self, num_actors: int) -> None:
        if not isinstance(num_actors, int) or isinstance(num_actors, bool):
            raise TypeError(
                f"num_actors must be an int, got {type(num_actors).__name__}"
            )
        if num_actors <= 0:
            raise ValueError(f"num_actors must be positive, got {num_actors}")
        self._num_actors = num_actors

    @property
    def num_actors(self) -> int:
        return self._num_actors

    def owned_by(self, actor_index: int, num_partitions: int) -> list[int]:
        """Return the sorted owned-partition slice for ``actor_index``.

        With ``num_actors=N`` the owned slice is the arithmetic progression
        ``[actor_index, actor_index + N, actor_index + 2N, ...]`` clipped
        to ``[0, num_partitions)``. The result is always sorted ascending
        because callers (the scanner ``partition_ids=[...]`` selector and
        the actor's invariant checker) rely on stable ordering.
        """
        if not isinstance(actor_index, int) or isinstance(actor_index, bool):
            raise TypeError(
                f"actor_index must be an int, got {type(actor_index).__name__}"
            )
        if not isinstance(num_partitions, int) or isinstance(num_partitions, bool):
            raise TypeError(
                f"num_partitions must be an int, got {type(num_partitions).__name__}"
            )
        if actor_index < 0 or actor_index >= self._num_actors:
            raise ValueError(
                f"actor_index {actor_index} out of range [0, {self._num_actors})"
            )
        if num_partitions < 0:
            raise ValueError(
                f"num_partitions must be non-negative, got {num_partitions}"
            )
        return list(range(actor_index, num_partitions, self._num_actors))
